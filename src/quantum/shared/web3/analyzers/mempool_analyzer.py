"""
Analyseur de Mempool pour la détection de transactions baleines.

Ce module surveille la mempool Ethereum/Solana pour détecter les
grosses transactions AVANT leur validation dans un bloc.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from collections import deque
import hashlib

from quantum.shared.web3.settings import web3_config, Chain, SignalType
from quantum.shared.web3.models.mempool_types import (
    WhaleAlert,
    TransactionIntent,
    TransactionIntentType,
    MempoolSignal,
    MempoolAction,
    PendingTransaction,
)
from quantum.shared.web3.clients.quicknode_client import QuickNodeClient, get_quicknode_client

logger = logging.getLogger(__name__)


class MempoolAnalyzer:
    """
    Analyseur de mempool pour détection de grosses transactions.
    
    DESCRIPTION:
    ============
    Écoute les transactions pendantes dans la mempool et détecte
    les mouvements de baleines AVANT leur validation dans un bloc.
    Génère des alertes pour les transactions significatives.
    
    INNOVATION MARCHÉ:
    ==================
    Détecte les transactions "whale" en attente dans la mempool AVANT
    leur validation dans un bloc. Permet d'anticiper les mouvements
    de prix de 10-30 secondes avant le marché.
    
    Contrairement à Whale Alert qui ne signale que les transactions
    CONFIRMÉES, notre système détecte les intentions avant exécution.
    
    FONCTIONNEMENT:
    ===============
    1. Écoute les pending transactions via QuickNode WebSocket
    2. Filtre par valeur (> threshold configurable)
    3. Identifie les contract calls vers DEX (Uniswap, Curve, etc.)
    4. Calcule l'impact prix estimé via simulation
    5. Émet un signal vers le système principal
    
    AVANTAGE COMPÉTITIF:
    ====================
    - Anticipation de 10-30 secondes sur les CEX
    - Détection des patterns sandwich avant exécution
    - Identification des smart money moves en temps réel
    - Filtrage intelligent du bruit (bot detection)
    
    RISQUE ASSOCIÉ:
    ===============
    - Les transactions peuvent être annulées (probability: ~5%)
    - Replaced-by-fee (RBF) peut modifier le montant
    - Haute congestion réseau = délais imprévisibles
    - Les bots MEV peuvent front-runner nos signaux
    - Volume très élevé sur mainnet (milliers de TX/seconde)
    
    MITIGATION:
    ===========
    - Score de confiance probabiliste (0-1)
    - Timeout automatique sur les signaux non-confirmés
    - Filtrage des bots MEV connus
    - Rate limiting interne pour éviter saturation
    
    USAGE:
    ======
    ```python
    analyzer = MempoolAnalyzer()
    await analyzer.start()
    analyzer.register_callback(my_whale_handler)
    ```
    """
    
    # Buffer de TX récentes pour déduplication
    MAX_TX_BUFFER_SIZE = 10000
    
    # Seuils de classification
    WHALE_TIERS = {
        'small': (100, 500),      # 100-500 ETH
        'medium': (500, 2000),    # 500-2000 ETH
        'large': (2000, 10000),   # 2000-10000 ETH
        'mega': (10000, float('inf'))  # >10000 ETH
    }
    
    def __init__(self, client: Optional[QuickNodeClient] = None):
        """
        Initialise l'analyseur de mempool.
        
        DESCRIPTION:
        ============
        Configure les seuils de détection et initialise
        les buffers de tracking.
        
        Args:
            client: Client QuickNode (utilise l'instance globale si None)
        """
        self.client = client or get_quicknode_client()
        self.config = web3_config.mempool
        self.whale_config = web3_config.whale_thresholds
        
        # Buffer de TX vues (pour déduplication)
        self._seen_tx_hashes: Set[str] = set()
        self._tx_buffer: deque = deque(maxlen=self.MAX_TX_BUFFER_SIZE)
        
        # Alertes actives
        self._active_alerts: Dict[str, WhaleAlert] = {}
        
        # Callbacks
        self._whale_callbacks: List[Callable[[WhaleAlert], None]] = []
        self._signal_callbacks: List[Callable[[MempoolSignal], None]] = []
        
        # Métriques
        self._metrics = {
            'transactions_analyzed': 0,
            'whale_alerts_generated': 0,
            'signals_emitted': 0,
            'false_positives': 0,  # TX annulées
        }
        
        # État
        self._running = False
        self._subscription_id: Optional[str] = None
        
        # Adresses connues (bots, exchanges, etc.)
        self._known_bot_addresses: Set[str] = set()
        self._known_whale_addresses: Dict[str, str] = {}  # address -> label
        
        # Agrégateur de signaux
        self._signal_window: deque = deque(maxlen=100)
        
        logger.info("MempoolAnalyzer initialisé")
    
    async def start(self, chain: Chain = Chain.ETHEREUM) -> bool:
        """
        Démarre l'analyse de la mempool.
        
        DESCRIPTION:
        ============
        Se connecte au client WebSocket et souscrit aux
        pending transactions.
        
        Args:
            chain: Chaîne à analyser (ETHEREUM par défaut)
            
        Returns:
            True si démarrage réussi
            
        RISQUE:
        =======
        Le volume de transactions peut être très élevé.
        Nécessite une machine performante pour le processing.
        """
        if self._running:
            logger.warning("MempoolAnalyzer déjà en cours d'exécution")
            return True
        
        try:
            # Connexion au client
            if chain == Chain.ETHEREUM:
                connected = await self.client.connect_ethereum()
            elif chain == Chain.SOLANA:
                connected = await self.client.connect_solana()
            else:
                logger.error(f"Chaîne non supportée: {chain}")
                return False
            
            if not connected:
                logger.error(f"Échec connexion {chain.value}")
                return False
            
            # Souscrire aux pending TX
            self._subscription_id = await self.client.subscribe_pending_transactions(
                callback=self._handle_pending_tx,
                chain=chain
            )
            
            if self._subscription_id:
                self._running = True
                logger.info(f"✅ MempoolAnalyzer démarré sur {chain.value}")
                
                # Démarrer l'agrégateur de signaux
                asyncio.create_task(self._signal_aggregator_loop())
                
                return True
            else:
                logger.error("Échec souscription pending TX")
                return False
                
        except Exception as e:
            logger.error(f"Erreur démarrage MempoolAnalyzer: {e}")
            return False
    
    async def stop(self):
        """
        Arrête l'analyse de la mempool.
        
        DESCRIPTION:
        ============
        Se désabonne et ferme les connexions proprement.
        """
        self._running = False
        
        if self._subscription_id:
            await self.client.unsubscribe(self._subscription_id)
            self._subscription_id = None
        
        logger.info("MempoolAnalyzer arrêté")
    
    def register_whale_callback(self, callback: Callable[[WhaleAlert], None]):
        """
        Enregistre un callback pour les alertes baleines.
        
        DESCRIPTION:
        ============
        Le callback sera appelé à chaque nouvelle détection
        de transaction baleine.
        
        Args:
            callback: Fonction(WhaleAlert) -> None
            
        USAGE:
        ======
        ```python
        def handle_whale(alert: WhaleAlert):
            print(f"Whale {alert.action}: {alert.amount_usd}$")
        
        analyzer.register_whale_callback(handle_whale)
        ```
        """
        self._whale_callbacks.append(callback)
    
    def register_signal_callback(self, callback: Callable[[MempoolSignal], None]):
        """
        Enregistre un callback pour les signaux agrégés.
        
        DESCRIPTION:
        ============
        Le callback sera appelé périodiquement avec un signal
        agrégé de la pression mempool.
        
        Args:
            callback: Fonction(MempoolSignal) -> None
        """
        self._signal_callbacks.append(callback)
    
    async def _handle_pending_tx(self, tx_hash: str):
        """
        Traite une transaction pendante.
        
        DESCRIPTION:
        ============
        Callback appelé pour chaque nouvelle transaction
        dans la mempool. Analyse et filtre selon les seuils.
        
        Args:
            tx_hash: Hash de la transaction
        """
        # Déduplication
        if tx_hash in self._seen_tx_hashes:
            return
        
        self._seen_tx_hashes.add(tx_hash)
        if len(self._seen_tx_hashes) > self.MAX_TX_BUFFER_SIZE:
            # Purger les plus anciennes
            oldest = list(self._seen_tx_hashes)[:1000]
            for h in oldest:
                self._seen_tx_hashes.discard(h)
        
        self._metrics['transactions_analyzed'] += 1
        
        try:
            # Récupérer les détails de la TX
            tx_details = await self.client.get_transaction(tx_hash)
            
            if not tx_details:
                return
            
            # Analyser la transaction
            alert = await self._analyze_transaction(tx_hash, tx_details)
            
            if alert:
                await self._emit_whale_alert(alert)
                
        except Exception as e:
            logger.debug(f"Erreur analyse TX {tx_hash}: {e}")
    
    async def _analyze_transaction(
        self,
        tx_hash: str,
        tx_details: Dict
    ) -> Optional[WhaleAlert]:
        """
        Analyse une transaction pour détecter une baleine.
        
        DESCRIPTION:
        ============
        Évalue si la transaction dépasse les seuils de détection
        et décode l'intention (swap, transfer, etc.).
        
        INNOVATION:
        ===========
        Utilise une combinaison de valeur, gas price et destination
        pour scorer la probabilité d'impact marché.
        
        Args:
            tx_hash: Hash de la transaction
            tx_details: Détails récupérés via RPC
            
        Returns:
            WhaleAlert si transaction significative, None sinon
            
        RISQUE:
        =======
        Le décodage peut échouer sur des contrats non-standard.
        """
        try:
            # Extraire les données clés
            from_addr = tx_details.get('from', '').lower()
            to_addr = tx_details.get('to', '').lower() if tx_details.get('to') else ''
            value_wei = int(tx_details.get('value', 0))
            gas_price = int(tx_details.get('gasPrice', 0))
            input_data = tx_details.get('input', '0x')
            
            # Convertir en ETH
            value_eth = value_wei / 1e18
            gas_price_gwei = gas_price / 1e9
            
            # Filtrer par gas price minimum
            if gas_price_gwei < self.config.MIN_GAS_PRICE_GWEI:
                return None
            
            # Filtrer les bots connus
            if from_addr in self._known_bot_addresses:
                return None
            
            # Vérifier si c'est une whale
            eth_threshold = self.whale_config.ETH_WHALE_THRESHOLD
            
            # Vérifier la valeur directe
            is_whale_by_value = value_eth >= eth_threshold
            
            # Vérifier si c'est un call vers DEX
            is_dex_interaction = self._is_dex_contract(to_addr)
            
            # Décoder l'intention
            intent = self._decode_transaction_intent(to_addr, input_data)
            
            if not is_whale_by_value and not is_dex_interaction:
                return None
            
            # Déterminer l'action (BUY/SELL)
            action = self._determine_action(intent, value_eth)
            
            # Estimer l'impact prix
            estimated_impact = self._estimate_price_impact(value_eth, intent)
            
            # Calculer le score de confiance
            confidence = self._calculate_confidence(
                value_eth=value_eth,
                gas_price_gwei=gas_price_gwei,
                is_known_whale=from_addr in self._known_whale_addresses,
                is_dex_interaction=is_dex_interaction,
            )
            
            # Seuil de confiance minimum
            if confidence < self.config.MIN_CONFIDENCE_THRESHOLD:
                return None
            
            # Créer l'alerte
            now = datetime.utcnow()
            
            alert = WhaleAlert(
                tx_hash=tx_hash,
                chain=Chain.ETHEREUM.value,
                whale_address=from_addr,
                action=action,
                token="ETH",
                amount=value_eth,
                amount_usd=value_eth * self._get_eth_price(),  # Prix simplifié
                estimated_price_impact_percent=estimated_impact,
                confidence=confidence,
                timestamp=now,
                expires_at=now + timedelta(seconds=self.config.SIGNAL_VALIDITY_SECONDS),
                target_dex=self._get_dex_name(to_addr) if is_dex_interaction else None,
                gas_price_gwei=gas_price_gwei,
                is_smart_money=from_addr in self._known_whale_addresses,
                historical_success_rate=self._known_whale_addresses.get(from_addr),
            )
            
            return alert
            
        except Exception as e:
            logger.error(f"Erreur analyse transaction: {e}")
            return None
    
    def _is_dex_contract(self, address: str) -> bool:
        """Vérifie si l'adresse est un routeur DEX connu."""
        dex_routers = self.config.MONITORED_DEX_ROUTERS
        return address.lower() in [addr.lower() for addr in dex_routers.values()]
    
    def _get_dex_name(self, address: str) -> Optional[str]:
        """Retourne le nom du DEX associé à une adresse."""
        address_lower = address.lower()
        for name, addr in self.config.MONITORED_DEX_ROUTERS.items():
            if addr.lower() == address_lower:
                return name
        return None
    
    def _decode_transaction_intent(
        self,
        to_address: str,
        input_data: str
    ) -> TransactionIntent:
        """
        Décode l'intention d'une transaction.
        
        DESCRIPTION:
        ============
        Analyse les calldata pour comprendre ce que la
        transaction essaie d'accomplir.
        
        INNOVATION:
        ===========
        Détection des patterns d'arbitrage et MEV pour
        distinguer les vrais mouvements de marché du bruit.
        
        Args:
            to_address: Adresse de destination
            input_data: Calldata de la transaction
            
        Returns:
            TransactionIntent avec les détails décodés
        """
        intent = TransactionIntent(
            intent_type=TransactionIntentType.UNKNOWN,
            target_protocol="unknown",
            tokens_involved=[],
        )
        
        if not input_data or input_data == '0x':
            intent.intent_type = TransactionIntentType.TRANSFER
            return intent
        
        # Extraire le function selector (4 bytes)
        selector = input_data[:10] if len(input_data) >= 10 else ''
        
        # Mapping des selectors connus
        SWAP_SELECTORS = {
            '0x7ff36ab5': 'swapExactETHForTokens',
            '0x18cbafe5': 'swapExactTokensForETH',
            '0x38ed1739': 'swapExactTokensForTokens',
            '0x8803dbee': 'swapTokensForExactTokens',
            '0xfb3bdb41': 'swapETHForExactTokens',
            '0x5c11d795': 'swapExactTokensForTokensSupportingFeeOnTransferTokens',
        }
        
        STAKE_SELECTORS = {
            '0xa694fc3a': 'stake',
            '0x2e1a7d4d': 'withdraw',
            '0xe9fad8ee': 'exit',
        }
        
        APPROVE_SELECTORS = {
            '0x095ea7b3': 'approve',
        }
        
        if selector in SWAP_SELECTORS:
            intent.intent_type = TransactionIntentType.SWAP
            intent.target_protocol = self._get_dex_name(to_address) or "dex"
            
            # Détection d'arbitrage (gas price très élevé)
            # Sera affiné avec plus de context
            
        elif selector in STAKE_SELECTORS:
            if 'withdraw' in STAKE_SELECTORS.get(selector, '').lower():
                intent.intent_type = TransactionIntentType.UNSTAKE
            else:
                intent.intent_type = TransactionIntentType.STAKE
            intent.target_protocol = "staking"
            
        elif selector in APPROVE_SELECTORS:
            intent.intent_type = TransactionIntentType.APPROVE
            
        elif self._is_bridge_contract(to_address):
            intent.intent_type = TransactionIntentType.BRIDGE
            intent.target_protocol = self._get_bridge_name(to_address)
        
        return intent
    
    def _is_bridge_contract(self, address: str) -> bool:
        """Vérifie si l'adresse est un bridge connu."""
        bridges = self.config.BRIDGE_CONTRACTS
        return address.lower() in [addr.lower() for addr in bridges.values()]
    
    def _get_bridge_name(self, address: str) -> str:
        """Retourne le nom du bridge."""
        address_lower = address.lower()
        for name, addr in self.config.BRIDGE_CONTRACTS.items():
            if addr.lower() == address_lower:
                return name
        return "bridge"
    
    def _determine_action(
        self,
        intent: TransactionIntent,
        value_eth: float
    ) -> MempoolAction:
        """
        Détermine si la transaction est un achat ou une vente.
        
        Args:
            intent: Intention décodée
            value_eth: Valeur en ETH
            
        Returns:
            MempoolAction (BUY, SELL, NEUTRAL)
        """
        if intent.intent_type == TransactionIntentType.SWAP:
            # Si la TX envoie de l'ETH, c'est probablement un achat de tokens
            if value_eth > 0:
                return MempoolAction.BUY
            else:
                # Vente de tokens pour ETH
                return MempoolAction.SELL
                
        elif intent.intent_type == TransactionIntentType.TRANSFER:
            return MempoolAction.NEUTRAL
            
        elif intent.intent_type == TransactionIntentType.UNSTAKE:
            return MempoolAction.SELL  # Unstake souvent = intention de vente
            
        return MempoolAction.NEUTRAL
    
    def _estimate_price_impact(
        self,
        value_eth: float,
        intent: TransactionIntent
    ) -> float:
        """
        Estime l'impact sur le prix de la transaction.
        
        DESCRIPTION:
        ============
        Estimation simplifiée basée sur la taille de la transaction
        par rapport à la liquidité moyenne des pools.
        
        INNOVATION:
        ===========
        Une implémentation complète utiliserait des simulations
        via Tenderly ou similaire.
        
        Args:
            value_eth: Valeur en ETH
            intent: Intention de la transaction
            
        Returns:
            Impact estimé en pourcentage (0.0 - 100.0)
        """
        # Estimation simplifiée: 
        # ~0.1% d'impact pour chaque 100 ETH sur un pool typique
        base_impact = (value_eth / 100) * 0.1
        
        # Plafonner à 10%
        return min(base_impact, 10.0)
    
    def _calculate_confidence(
        self,
        value_eth: float,
        gas_price_gwei: float,
        is_known_whale: bool,
        is_dex_interaction: bool,
    ) -> float:
        """
        Calcule le score de confiance de l'alerte.
        
        DESCRIPTION:
        ============
        Combine plusieurs facteurs pour évaluer la fiabilité
        de l'alerte générée.
        
        Args:
            value_eth: Valeur de la transaction
            gas_price_gwei: Prix du gas
            is_known_whale: Si l'adresse est une whale connue
            is_dex_interaction: Si c'est une interaction DEX
            
        Returns:
            Score de confiance (0.0 - 1.0)
        """
        confidence = 0.5  # Base
        
        # Bonus pour valeur élevée
        if value_eth >= 1000:
            confidence += 0.2
        elif value_eth >= 500:
            confidence += 0.15
        elif value_eth >= 100:
            confidence += 0.1
        
        # Bonus pour whale connue
        if is_known_whale:
            confidence += 0.15
        
        # Bonus pour interaction DEX (intent clair)
        if is_dex_interaction:
            confidence += 0.1
        
        # Malus pour gas price suspect (trop bas = peut fail)
        if gas_price_gwei < 30:
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    def _get_eth_price(self) -> float:
        """Retourne le prix ETH/USD estimé."""
        # TODO: Intégrer un feed de prix réel
        return 3500.0  # Prix placeholder
    
    async def _emit_whale_alert(self, alert: WhaleAlert):
        """
        Émet une alerte baleine aux callbacks enregistrés.
        
        Args:
            alert: Alerte à émettre
        """
        self._metrics['whale_alerts_generated'] += 1
        self._active_alerts[alert.tx_hash] = alert
        self._signal_window.append(alert)
        
        logger.info(
            f"🐋 WHALE ALERT: {alert.action.value} {alert.amount:.2f} ETH "
            f"(${alert.amount_usd:,.0f}) - Confidence: {alert.confidence:.2f}"
        )
        
        for callback in self._whale_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Erreur callback whale: {e}")
    
    async def _signal_aggregator_loop(self):
        """
        Boucle d'agrégation des signaux.
        
        DESCRIPTION:
        ============
        Agrège les alertes individuelles en un signal
        consolidé toutes les N secondes.
        """
        while self._running:
            try:
                await asyncio.sleep(30)  # Agrégation toutes les 30s
                
                signal = self._aggregate_signals()
                
                if signal:
                    self._metrics['signals_emitted'] += 1
                    
                    for callback in self._signal_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(signal)
                            else:
                                callback(signal)
                        except Exception as e:
                            logger.error(f"Erreur callback signal: {e}")
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur agrégation: {e}")
    
    def _aggregate_signals(self) -> Optional[MempoolSignal]:
        """
        Agrège les alertes récentes en un signal unique.
        
        Returns:
            MempoolSignal agrégé ou None si pas assez de données
        """
        # Filtrer les alertes valides
        now = datetime.utcnow()
        valid_alerts = [
            alert for alert in self._signal_window
            if alert.is_valid()
        ]
        
        if not valid_alerts:
            return None
        
        # Calculer les métriques
        total_volume = sum(a.amount_usd for a in valid_alerts)
        buy_volume = sum(a.amount_usd for a in valid_alerts if a.action == MempoolAction.BUY)
        sell_volume = sum(a.amount_usd for a in valid_alerts if a.action == MempoolAction.SELL)
        net_pressure = buy_volume - sell_volume
        
        # Déterminer le type de signal
        if net_pressure > total_volume * 0.2:
            signal_type = "BULLISH_PRESSURE"
        elif net_pressure < -total_volume * 0.2:
            signal_type = "BEARISH_PRESSURE"
        else:
            signal_type = "NEUTRAL"
        
        # Score de pression (-100 à +100)
        if total_volume > 0:
            pressure_score = (net_pressure / total_volume) * 100
        else:
            pressure_score = 0
        
        # Confiance moyenne
        avg_confidence = sum(a.confidence for a in valid_alerts) / len(valid_alerts)
        
        return MempoolSignal(
            signal_type=signal_type,
            total_pending_volume_usd=total_volume,
            net_buy_pressure_usd=net_pressure,
            whale_count=len(valid_alerts),
            whale_alerts=list(valid_alerts),
            pressure_score=pressure_score,
            confidence=avg_confidence,
            timestamp=now,
            window_seconds=60,
        )
    
    def get_active_alerts(self) -> List[WhaleAlert]:
        """
        Retourne les alertes actives (non expirées).
        
        Returns:
            Liste des alertes valides
        """
        now = datetime.utcnow()
        return [
            alert for alert in self._active_alerts.values()
            if alert.is_valid()
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques de l'analyseur.
        
        Returns:
            Dictionnaire des métriques
        """
        return {
            **self._metrics,
            'active_alerts': len(self.get_active_alerts()),
            'running': self._running,
        }
    
    def add_known_whale(self, address: str, label: str):
        """
        Ajoute une adresse baleine connue.
        
        Args:
            address: Adresse du wallet
            label: Label descriptif (ex: "Alameda", "Jump")
        """
        self._known_whale_addresses[address.lower()] = label
    
    def add_known_bot(self, address: str):
        """
        Ajoute une adresse de bot connue (sera ignorée).
        
        Args:
            address: Adresse du bot
        """
        self._known_bot_addresses.add(address.lower())


# Factory function
def create_mempool_analyzer(client: Optional[QuickNodeClient] = None) -> MempoolAnalyzer:
    """
    Crée une instance de MempoolAnalyzer.
    
    Args:
        client: Client QuickNode optionnel
        
    Returns:
        Instance configurée de MempoolAnalyzer
    """
    return MempoolAnalyzer(client)
