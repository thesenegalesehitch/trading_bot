"""
Web3 Intelligence Engine - Moteur principal.

Ce module orchestre tous les analyseurs Web3 et fournit
une interface unifiée pour le système de trading.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

from web3_innovation.settings import web3_config, Chain
from web3_innovation.clients.quicknode_client import QuickNodeClient, get_quicknode_client
from web3_innovation.clients.websocket_manager import WebSocketManager, get_websocket_manager
from web3_innovation.analyzers.mempool_analyzer import MempoolAnalyzer
from web3_innovation.analyzers.cross_chain_oracle import CrossChainOracle
from web3_innovation.analyzers.onchain_sentiment import OnChainSentimentAnalyzer
from web3_innovation.hooks.signal_dispatcher import (
    SignalDispatcher,
    IntegrationMode,
    Web3Signal,
    create_signal_from_whale_alert,
    create_signal_from_correlation,
    create_signal_from_sentiment,
)
from web3_innovation.hooks.event_bus import EventBus, Event, EventPriority, get_event_bus
from web3_innovation.models.mempool_types import WhaleAlert, MempoolSignal
from web3_innovation.models.correlation_types import CrossChainIndex, CorrelationEvent
from web3_innovation.models.sentiment_types import StakingSentiment, SellPressureScore

logger = logging.getLogger(__name__)


@dataclass
class Web3Status:
    """Statut global du système Web3."""
    is_running: bool
    mode: IntegrationMode
    
    # Statut des composants
    mempool_active: bool = False
    cross_chain_active: bool = False
    sentiment_active: bool = False
    
    # Métriques agrégées
    total_signals_generated: int = 0
    whale_alerts_24h: int = 0
    current_cci: float = 0.0
    eth_sentiment_score: Optional[float] = None
    
    # Santé
    connection_health: Dict[str, str] = field(default_factory=dict)
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'is_running': self.is_running,
            'mode': self.mode.value,
            'mempool_active': self.mempool_active,
            'cross_chain_active': self.cross_chain_active,
            'sentiment_active': self.sentiment_active,
            'total_signals_generated': self.total_signals_generated,
            'whale_alerts_24h': self.whale_alerts_24h,
            'current_cci': self.current_cci,
            'eth_sentiment_score': self.eth_sentiment_score,
            'connection_health': self.connection_health,
            'timestamp': self.timestamp.isoformat(),
        }


class Web3IntelligenceEngine:
    """
    Moteur principal d'intelligence Web3.
    
    DESCRIPTION:
    ============
    Orchestre tous les composants du module web3_innovation
    et fournit une interface simple pour le système de trading.
    
    ARCHITECTURE:
    =============
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                   Web3IntelligenceEngine                     │
    │                                                             │
    │  ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐  │
    │  │MempoolAnalyzer│ │CrossChainOracle│ │SentimentAnalyzer│  │
    │  └──────┬───────┘ └──────┬───────┘ └────────┬───────────┘  │
    │         │                │                  │               │
    │         └────────────────┼──────────────────┘               │
    │                          ▼                                  │
    │                   ┌─────────────┐                           │
    │                   │ EventBus    │                           │
    │                   └──────┬──────┘                           │
    │                          ▼                                  │
    │                 ┌──────────────────┐                        │
    │                 │ SignalDispatcher │                        │
    │                 └────────┬─────────┘                        │
    │                          │                                  │
    │                          ▼                                  │
    │              [Callbacks vers système principal]             │
    └─────────────────────────────────────────────────────────────┘
    ```
    
    INNOVATION:
    ===========
    Ce moteur unifie trois sources d'intelligence on-chain
    en un système cohérent pouvant enrichir n'importe quel
    système de trading existant.
    
    FONCTIONNALITÉS:
    ================
    - Démarrage/arrêt orchestré de tous les composants
    - Agrégation des signaux de tous les analyseurs
    - Interface simple pour le système de trading
    - Métriques et monitoring intégrés
    
    USAGE:
    ======
    ```python
    from web3_innovation import Web3IntelligenceEngine
    
    # Initialisation
    engine = Web3IntelligenceEngine(mode=IntegrationMode.ENRICHMENT)
    
    # Callback pour recevoir les signaux
    def on_signal(signal: Web3Signal):
        print(f"Signal: {signal.signal_type}")
    
    engine.register_callback(on_signal)
    
    # Démarrage
    await engine.start()
    
    # Récupérer le statut
    status = engine.get_status()
    
    # Récupérer l'analyse Web3 pour enrichir un signal
    web3_data = engine.get_current_analysis()
    
    # Arrêt
    await engine.stop()
    ```
    
    RISQUE:
    =======
    - Nécessite des endpoints QuickNode fonctionnels
    - Consomme des ressources réseau et mémoire
    
    MITIGATION:
    ===========
    - Mode dégradé si un composant échoue
    - Health checks réguliers
    - Logs détaillés pour debugging
    """
    
    def __init__(
        self,
        mode: IntegrationMode = IntegrationMode.ALERT_ONLY,
        enable_mempool: bool = True,
        enable_cross_chain: bool = True,
        enable_sentiment: bool = True,
    ):
        """
        Initialise le moteur Web3.
        
        DESCRIPTION:
        ============
        Configure les composants selon les paramètres fournis.
        
        Args:
            mode: Mode d'intégration avec le système de trading
            enable_mempool: Activer l'analyse mempool
            enable_cross_chain: Activer le cross-chain oracle
            enable_sentiment: Activer l'analyse de sentiment
        """
        self.mode = mode
        self.config = web3_config
        
        # Feature flags
        self._enable_mempool = enable_mempool and self.config.ENABLE_MEMPOOL_ANALYSIS
        self._enable_cross_chain = enable_cross_chain and self.config.ENABLE_CROSS_CHAIN_ORACLE
        self._enable_sentiment = enable_sentiment and self.config.ENABLE_SENTIMENT_ANALYSIS
        
        # Composants
        self._client: Optional[QuickNodeClient] = None
        self._ws_manager: Optional[WebSocketManager] = None
        self._event_bus: Optional[EventBus] = None
        self._dispatcher: Optional[SignalDispatcher] = None
        
        # Analyseurs
        self._mempool_analyzer: Optional[MempoolAnalyzer] = None
        self._cross_chain_oracle: Optional[CrossChainOracle] = None
        self._sentiment_analyzer: Optional[OnChainSentimentAnalyzer] = None
        
        # État
        self._running = False
        self._started_at: Optional[datetime] = None
        
        # Callbacks utilisateur
        self._user_callbacks: List[Callable[[Web3Signal], None]] = []
        
        # Métriques
        self._metrics = {
            'signals_generated': 0,
            'whale_alerts': 0,
            'correlation_events': 0,
            'sentiment_updates': 0,
        }
        
        logger.info(f"Web3IntelligenceEngine initialisé (mode: {mode.value})")
    
    async def start(self) -> bool:
        """
        Démarre le moteur Web3.
        
        DESCRIPTION:
        ============
        Initialise tous les composants activés et démarre
        les boucles de traitement.
        
        Returns:
            True si au moins un composant a démarré
            
        RISQUE:
        =======
        Peut échouer si les endpoints ne sont pas configurés.
        """
        if self._running:
            logger.warning("Engine déjà en cours d'exécution")
            return True
        
        logger.info("🚀 Démarrage du Web3IntelligenceEngine...")
        
        try:
            # Initialiser les composants core
            self._client = get_quicknode_client()
            self._ws_manager = get_websocket_manager()
            self._event_bus = get_event_bus()
            self._dispatcher = SignalDispatcher(self.mode)
            
            # Démarrer le bus événementiel et le dispatcher
            await self._event_bus.start()
            await self._dispatcher.start()
            
            # Démarrer le WS manager
            await self._ws_manager.start()
            
            # Initialiser et démarrer les analyseurs
            components_started = 0
            
            if self._enable_mempool:
                if await self._start_mempool_analyzer():
                    components_started += 1
            
            if self._enable_cross_chain:
                if await self._start_cross_chain_oracle():
                    components_started += 1
            
            if self._enable_sentiment:
                if await self._start_sentiment_analyzer():
                    components_started += 1
            
            if components_started == 0:
                logger.error("❌ Aucun composant n'a pu démarrer")
                return False
            
            self._running = True
            self._started_at = datetime.utcnow()
            
            # Enregistrer le callback interne pour le dispatcher
            for callback in self._user_callbacks:
                self._dispatcher.register_callback(callback)
            
            logger.info(
                f"✅ Web3IntelligenceEngine démarré "
                f"({components_started} composants actifs)"
            )
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur démarrage engine: {e}")
            return False
    
    async def _start_mempool_analyzer(self) -> bool:
        """Démarre l'analyseur mempool."""
        try:
            self._mempool_analyzer = MempoolAnalyzer(self._client)
            
            # Enregistrer les callbacks
            self._mempool_analyzer.register_whale_callback(self._on_whale_alert)
            self._mempool_analyzer.register_signal_callback(self._on_mempool_signal)
            
            success = await self._mempool_analyzer.start(Chain.ETHEREUM)
            
            if success:
                logger.info("  ✓ MempoolAnalyzer actif")
            
            return success
            
        except Exception as e:
            logger.error(f"  ✗ MempoolAnalyzer échoué: {e}")
            return False
    
    async def _start_cross_chain_oracle(self) -> bool:
        """Démarre le cross-chain oracle."""
        try:
            self._cross_chain_oracle = CrossChainOracle(self._client, self._client)
            
            # Enregistrer les callbacks
            self._cross_chain_oracle.register_event_callback(self._on_correlation_event)
            self._cross_chain_oracle.register_index_callback(self._on_index_update)
            
            success = await self._cross_chain_oracle.start()
            
            if success:
                logger.info("  ✓ CrossChainOracle actif")
            
            return success
            
        except Exception as e:
            logger.error(f"  ✗ CrossChainOracle échoué: {e}")
            return False
    
    async def _start_sentiment_analyzer(self) -> bool:
        """Démarre l'analyseur de sentiment."""
        try:
            self._sentiment_analyzer = OnChainSentimentAnalyzer(self._client)
            
            # Enregistrer les callbacks
            self._sentiment_analyzer.register_sentiment_callback(self._on_sentiment_update)
            self._sentiment_analyzer.register_pressure_callback(self._on_sell_pressure)
            
            success = await self._sentiment_analyzer.start(['ETH', 'SOL'])
            
            if success:
                logger.info("  ✓ SentimentAnalyzer actif")
            
            return success
            
        except Exception as e:
            logger.error(f"  ✗ SentimentAnalyzer échoué: {e}")
            return False
    
    async def stop(self):
        """
        Arrête le moteur Web3.
        
        DESCRIPTION:
        ============
        Arrête proprement tous les composants.
        """
        if not self._running:
            return
        
        logger.info("Arrêt du Web3IntelligenceEngine...")
        
        # Arrêter les analyseurs
        if self._mempool_analyzer:
            await self._mempool_analyzer.stop()
        
        if self._cross_chain_oracle:
            await self._cross_chain_oracle.stop()
        
        if self._sentiment_analyzer:
            await self._sentiment_analyzer.stop()
        
        # Arrêter les composants core
        if self._dispatcher:
            await self._dispatcher.stop()
        
        if self._event_bus:
            await self._event_bus.stop()
        
        if self._ws_manager:
            await self._ws_manager.stop()
        
        if self._client:
            await self._client.disconnect()
        
        self._running = False
        logger.info("✅ Web3IntelligenceEngine arrêté")
    
    def register_callback(self, callback: Callable[[Web3Signal], None]):
        """
        Enregistre un callback pour les signaux Web3.
        
        DESCRIPTION:
        ============
        Le callback sera appelé pour chaque nouveau signal
        généré par les analyseurs.
        
        Args:
            callback: Fonction(Web3Signal) -> None
        """
        self._user_callbacks.append(callback)
        
        # Si déjà en cours, ajouter au dispatcher
        if self._dispatcher:
            self._dispatcher.register_callback(callback)
    
    async def _on_whale_alert(self, alert: WhaleAlert):
        """Callback interne pour les alertes whale."""
        self._metrics['whale_alerts'] += 1
        
        # Créer un signal unifié
        signal = create_signal_from_whale_alert(alert)
        self._metrics['signals_generated'] += 1
        
        # Dispatcher le signal
        await self._dispatcher.dispatch_signal(signal)
        
        # Publier sur le bus
        await self._event_bus.publish(Event(
            event_type='whale_alert',
            data=alert.to_dict(),
            source='mempool',
            priority=EventPriority.HIGH,
        ))
    
    async def _on_mempool_signal(self, signal: MempoolSignal):
        """Callback pour les signaux mempool agrégés."""
        await self._event_bus.publish(Event(
            event_type='mempool_signal',
            data=signal.to_dict(),
            source='mempool',
            priority=EventPriority.NORMAL,
        ))
    
    async def _on_correlation_event(self, event: CorrelationEvent):
        """Callback pour les événements de corrélation."""
        self._metrics['correlation_events'] += 1
        
        signal = create_signal_from_correlation(event)
        self._metrics['signals_generated'] += 1
        
        await self._dispatcher.dispatch_signal(signal)
        
        await self._event_bus.publish(Event(
            event_type='correlation_event',
            data=event.to_dict(),
            source='cross_chain',
            priority=EventPriority.HIGH,
        ))
    
    async def _on_index_update(self, index: CrossChainIndex):
        """Callback pour les mises à jour de l'indice CCI."""
        await self._event_bus.publish(Event(
            event_type='cci_update',
            data=index.to_dict(),
            source='cross_chain',
            priority=EventPriority.NORMAL,
        ))
    
    async def _on_sentiment_update(self, sentiment: StakingSentiment):
        """Callback pour les mises à jour de sentiment."""
        self._metrics['sentiment_updates'] += 1
        
        signal = create_signal_from_sentiment(sentiment)
        self._metrics['signals_generated'] += 1
        
        await self._dispatcher.dispatch_signal(signal)
        
        await self._event_bus.publish(Event(
            event_type='sentiment_update',
            data=sentiment.to_dict(),
            source='sentiment',
            priority=EventPriority.NORMAL,
        ))
    
    async def _on_sell_pressure(self, pressure: SellPressureScore):
        """Callback pour les alertes de pression vendeuse."""
        await self._event_bus.publish(Event(
            event_type='sell_pressure_alert',
            data=pressure.to_dict(),
            source='sentiment',
            priority=EventPriority.HIGH if pressure.is_critical() else EventPriority.NORMAL,
        ))
    
    def get_status(self) -> Web3Status:
        """
        Retourne le statut du moteur.
        
        Returns:
            Web3Status avec l'état de tous les composants
        """
        cci = 0.0
        if self._cross_chain_oracle:
            cci = self._cross_chain_oracle.get_current_correlation()
        
        eth_sentiment = None
        if self._sentiment_analyzer:
            sentiment = self._sentiment_analyzer.get_sentiment('ETH')
            if sentiment:
                eth_sentiment = sentiment.score
        
        return Web3Status(
            is_running=self._running,
            mode=self.mode,
            mempool_active=self._mempool_analyzer is not None and self._running,
            cross_chain_active=self._cross_chain_oracle is not None and self._running,
            sentiment_active=self._sentiment_analyzer is not None and self._running,
            total_signals_generated=self._metrics['signals_generated'],
            whale_alerts_24h=self._metrics['whale_alerts'],
            current_cci=cci,
            eth_sentiment_score=eth_sentiment,
            connection_health=self._get_connection_health(),
        )
    
    def _get_connection_health(self) -> Dict[str, str]:
        """Retourne l'état de santé des connexions."""
        if not self._ws_manager:
            return {}
        
        health = self._ws_manager.get_health_status()
        return {
            chain: h.status.value
            for chain, h in health.items()
        }
    
    def get_current_analysis(self) -> Dict[str, Any]:
        """
        Retourne l'analyse Web3 courante.
        
        DESCRIPTION:
        ============
        Agrège les données de tous les analyseurs en un
        dictionnaire utilisable par le système de trading.
        
        Returns:
            Dictionnaire avec les analyses Web3
            
        USAGE:
        ======
        ```python
        web3_data = engine.get_current_analysis()
        
        # Utiliser dans votre logique de trading
        if web3_data['mempool_pressure'] > 50:
            # Forte pression acheteuse détectée
            pass
        ```
        """
        analysis = {
            'timestamp': datetime.utcnow().isoformat(),
            'is_active': self._running,
        }
        
        # Mempool
        if self._mempool_analyzer:
            alerts = self._mempool_analyzer.get_active_alerts()
            buy_alerts = [a for a in alerts if a.action.value == 'BUY']
            sell_alerts = [a for a in alerts if a.action.value == 'SELL']
            
            analysis['mempool'] = {
                'active_alerts': len(alerts),
                'buy_alerts': len(buy_alerts),
                'sell_alerts': len(sell_alerts),
                'pressure_score': (len(buy_alerts) - len(sell_alerts)) * 10,
                'total_volume_usd': sum(a.amount_usd for a in alerts),
            }
        
        # Cross-chain
        if self._cross_chain_oracle:
            index = self._cross_chain_oracle.get_current_index()
            analysis['cross_chain'] = {
                'cci_value': index.index_value if index else 0,
                'is_bullish': index.is_bullish() if index else False,
                'is_bearish': index.is_bearish() if index else False,
                'confidence': index.confidence if index else 0,
            }
        
        # Sentiment
        if self._sentiment_analyzer:
            eth_sentiment = self._sentiment_analyzer.get_sentiment('ETH')
            sol_sentiment = self._sentiment_analyzer.get_sentiment('SOL')
            
            analysis['sentiment'] = {
                'eth': {
                    'score': eth_sentiment.score if eth_sentiment else None,
                    'label': eth_sentiment.score_label if eth_sentiment else None,
                    'sell_pressure': eth_sentiment.sell_pressure_probability if eth_sentiment else None,
                },
                'sol': {
                    'score': sol_sentiment.score if sol_sentiment else None,
                    'label': sol_sentiment.score_label if sol_sentiment else None,
                    'sell_pressure': sol_sentiment.sell_pressure_probability if sol_sentiment else None,
                },
            }
        
        return analysis
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques complètes du moteur.
        
        Returns:
            Dictionnaire des métriques
        """
        metrics = {
            **self._metrics,
            'uptime_seconds': (
                (datetime.utcnow() - self._started_at).total_seconds()
                if self._started_at else 0
            ),
        }
        
        if self._mempool_analyzer:
            metrics['mempool'] = self._mempool_analyzer.get_metrics()
        
        if self._cross_chain_oracle:
            metrics['cross_chain'] = self._cross_chain_oracle.get_metrics()
        
        if self._sentiment_analyzer:
            metrics['sentiment'] = self._sentiment_analyzer.get_metrics()
        
        if self._dispatcher:
            metrics['dispatcher'] = self._dispatcher.get_metrics()
        
        return metrics


# Factory function
def create_web3_engine(
    mode: IntegrationMode = IntegrationMode.ALERT_ONLY,
    **kwargs
) -> Web3IntelligenceEngine:
    """
    Crée une instance du moteur Web3.
    
    Args:
        mode: Mode d'intégration
        **kwargs: Arguments additionnels
        
    Returns:
        Instance configurée du moteur
    """
    return Web3IntelligenceEngine(mode=mode, **kwargs)
