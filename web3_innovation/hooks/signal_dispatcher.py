"""
Signal Dispatcher - Système de dispatch des signaux Web3.

Ce module fait le pont entre les analyseurs Web3 et le système
de trading existant via un système de hooks non-intrusif.
"""

import asyncio
import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

from web3_innovation.models.mempool_types import WhaleAlert, MempoolSignal
from web3_innovation.models.correlation_types import CrossChainIndex, CorrelationEvent
from web3_innovation.models.sentiment_types import StakingSentiment, SellPressureScore

logger = logging.getLogger(__name__)


class IntegrationMode(Enum):
    """
    Modes d'intégration avec le système principal.
    
    DESCRIPTION:
    ============
    Définit comment les signaux Web3 interagissent avec
    les fonctions de trading existantes.
    """
    
    ALERT_ONLY = "alert_only"
    """
    Mode sûr: génère uniquement des alertes/logs.
    Les signaux sont enregistrés mais n'affectent pas
    les décisions de trading existantes.
    Recommandé pour la phase d'observation initiale.
    """
    
    ENRICHMENT = "enrichment"
    """
    Mode intermédiaire: enrichit les signaux existants.
    Les métadonnées Web3 sont ajoutées aux analyses
    sans modifier les décisions finales.
    """
    
    OVERRIDE = "override"
    """
    Mode avancé: peut modifier les décisions.
    Les signaux Web3 peuvent invalider ou renforcer
    les signaux de trading existants.
    À utiliser après validation extensive.
    """


class SignalPriority(Enum):
    """Priorité des signaux."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Web3Signal:
    """
    Signal Web3 unifié pour dispatch.
    
    DESCRIPTION:
    ============
    Structure commune pour tous les types de signaux Web3
    (mempool, cross-chain, sentiment) permettant un
    traitement uniforme par le dispatcher.
    """
    signal_id: str
    signal_type: str
    source: str  # 'mempool', 'cross_chain', 'sentiment'
    
    # Données du signal
    data: Dict[str, Any]
    
    # Métadonnées
    priority: SignalPriority = SignalPriority.MEDIUM
    confidence: float = 0.0
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    # État
    processed: bool = False
    dispatched_to: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Vérifie si le signal est encore valide."""
        if self.expires_at is None:
            return True
        return datetime.utcnow() < self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'signal_id': self.signal_id,
            'signal_type': self.signal_type,
            'source': self.source,
            'data': self.data,
            'priority': self.priority.name,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'processed': self.processed,
            'is_valid': self.is_valid(),
        }


@dataclass
class EnrichmentResult:
    """Résultat d'un enrichissement de signal."""
    original_signal: Dict[str, Any]
    web3_data: Dict[str, Any]
    enriched_signal: Dict[str, Any]
    modifications: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class SignalDispatcher:
    """
    Système de dispatch des signaux Web3 vers le système principal.
    
    DESCRIPTION:
    ============
    Ce dispatcher agit comme un pont entre les analyseurs Web3
    et les fonctions existantes du QuantumTradingSystem.
    Il permet une intégration progressive et non-intrusive.
    
    ARCHITECTURE:
    =============
    ```
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │ MempoolAnalyzer │     │ CrossChainOracle│     │ SentimentAnalyzer│
    └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     ▼
                          ┌─────────────────────┐
                          │   SignalDispatcher  │
                          │                     │
                          │ • Queue de signaux  │
                          │ • Filtrage          │
                          │ • Enrichissement    │
                          │ • Dispatch          │
                          └──────────┬──────────┘
                                     │
             ┌───────────────────────┼───────────────────────┐
             │                       │                       │
             ▼                       ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │ Alert Callbacks │     │Enrichment Handlers│   │Override Handlers│
    │  (Mode ALERT)   │     │  (Mode ENRICH)   │     │  (Mode OVERRIDE)│
    └─────────────────┘     └─────────────────┘     └─────────────────┘
    ```
    
    MODES D'INTÉGRATION:
    ====================
    
    1. **ALERT_ONLY** (Mode Safe):
       - Génère uniquement des logs et notifications
       - Aucun impact sur les décisions de trading
       - Recommandé pour commencer
    
    2. **ENRICHMENT** (Mode Intermédiaire):
       - Ajoute des métadonnées aux signaux existants
       - Les handlers peuvent lire les données Web3
       - Le système principal décide comment les utiliser
    
    3. **OVERRIDE** (Mode Avancé):
       - Les signaux Web3 peuvent modifier les décisions
       - Nécessite une validation extensive
       - À utiliser après plusieurs mois de backtesting
    
    USAGE:
    ======
    ```python
    # Dans main.py ou votre code existant
    from web3_innovation import SignalDispatcher, IntegrationMode
    
    # 1. Initialisation
    dispatcher = SignalDispatcher(mode=IntegrationMode.ENRICHMENT)
    
    # 2. Enregistrer des callbacks
    def on_whale_alert(signal: Web3Signal):
        print(f"🐋 Whale détectée: {signal.data}")
    
    dispatcher.register_callback(on_whale_alert)
    
    # 3. Optionnel: Handler d'enrichissement
    def enrich_trading_signal(original: dict, web3: Web3Signal) -> dict:
        original['web3_data'] = {
            'mempool_pressure': web3.data.get('pressure_score'),
            'confidence': web3.confidence
        }
        return original
    
    dispatcher.register_enrichment_handler(enrich_trading_signal)
    
    # 4. Les signaux sont dispatchés automatiquement par les analyseurs
    ```
    
    INNOVATION:
    ===========
    Ce système permet une adoption progressive de l'intelligence
    Web3 sans risquer de casser le système de trading existant.
    
    RISQUE:
    =======
    - En mode OVERRIDE, des signaux incorrects peuvent affecter les trades
    - Latence additionnelle pour le processing des signaux
    
    MITIGATION:
    ===========
    - Commencer toujours en mode ALERT_ONLY
    - Valider pendant au moins 30 jours avant de passer en ENRICHMENT
    - Ne jamais utiliser OVERRIDE sans backtesting extensif
    """
    
    # Taille maximale de la queue
    MAX_QUEUE_SIZE = 1000
    
    def __init__(self, mode: IntegrationMode = IntegrationMode.ALERT_ONLY):
        """
        Initialise le dispatcher de signaux.
        
        DESCRIPTION:
        ============
        Configure le mode d'intégration et initialise
        les queues et callbacks.
        
        Args:
            mode: Mode d'intégration (défaut: ALERT_ONLY)
        """
        self.mode = mode
        
        # Queue de signaux
        self._signal_queue: asyncio.Queue = asyncio.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self._pending_signals: Dict[str, Web3Signal] = {}
        
        # Callbacks par type
        self._callbacks: List[Callable[[Web3Signal], None]] = []
        self._typed_callbacks: Dict[str, List[Callable]] = {}
        
        # Handlers d'enrichissement
        self._enrichment_handlers: List[Callable[[Dict, Web3Signal], Dict]] = []
        
        # Handlers d'override
        self._override_handlers: List[Callable[[Dict, Web3Signal], Optional[Dict]]] = []
        
        # Historique
        self._signal_history: List[Web3Signal] = []
        self._max_history_size = 500
        
        # Métriques
        self._metrics = {
            'signals_received': 0,
            'signals_dispatched': 0,
            'signals_filtered': 0,
            'enrichments_applied': 0,
            'overrides_applied': 0,
        }
        
        # État
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        
        logger.info(f"SignalDispatcher initialisé en mode {mode.value}")
    
    async def start(self):
        """
        Démarre le processor de signaux.
        
        DESCRIPTION:
        ============
        Lance la boucle de traitement des signaux en arrière-plan.
        """
        if self._running:
            return
        
        self._running = True
        self._processor_task = asyncio.create_task(self._process_loop())
        logger.info("SignalDispatcher démarré")
    
    async def stop(self):
        """Arrête le processor."""
        self._running = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("SignalDispatcher arrêté")
    
    def register_callback(
        self,
        callback: Callable[[Web3Signal], None],
        signal_types: Optional[List[str]] = None
    ):
        """
        Enregistre un callback pour les signaux.
        
        DESCRIPTION:
        ============
        Le callback sera appelé pour chaque signal reçu
        (ou uniquement pour les types spécifiés).
        
        Args:
            callback: Fonction(Web3Signal) -> None
            signal_types: Types de signaux à filtrer (optionnel)
        """
        if signal_types:
            for signal_type in signal_types:
                if signal_type not in self._typed_callbacks:
                    self._typed_callbacks[signal_type] = []
                self._typed_callbacks[signal_type].append(callback)
        else:
            self._callbacks.append(callback)
        
        logger.debug(f"Callback enregistré pour types: {signal_types or 'tous'}")
    
    def register_enrichment_handler(
        self,
        handler: Callable[[Dict, Web3Signal], Dict]
    ):
        """
        Enregistre un handler d'enrichissement.
        
        DESCRIPTION:
        ============
        Le handler reçoit le signal de trading original et les
        données Web3, et doit retourner le signal enrichi.
        
        Ne fonctionne qu'en mode ENRICHMENT ou OVERRIDE.
        
        Args:
            handler: Fonction(original_signal, web3_signal) -> enriched_signal
        """
        if self.mode == IntegrationMode.ALERT_ONLY:
            logger.warning(
                "Enrichment handler enregistré mais mode est ALERT_ONLY. "
                "Handler sera ignoré."
            )
        self._enrichment_handlers.append(handler)
    
    def register_override_handler(
        self,
        handler: Callable[[Dict, Web3Signal], Optional[Dict]]
    ):
        """
        Enregistre un handler d'override.
        
        DESCRIPTION:
        ============
        Le handler peut modifier ou annuler un signal de trading.
        Retourner None annule le signal.
        
        Ne fonctionne qu'en mode OVERRIDE.
        
        Args:
            handler: Fonction(signal, web3) -> modified_signal ou None
            
        RISQUE:
        =======
        Ces handlers peuvent affecter directement les trades.
        À utiliser avec précaution.
        """
        if self.mode != IntegrationMode.OVERRIDE:
            logger.warning(
                "Override handler enregistré mais mode n'est pas OVERRIDE. "
                "Handler sera ignoré."
            )
        self._override_handlers.append(handler)
    
    async def dispatch_signal(self, signal: Web3Signal):
        """
        Dispatch un signal Web3.
        
        DESCRIPTION:
        ============
        Ajoute le signal à la queue de traitement.
        Le signal sera traité de manière asynchrone.
        
        Args:
            signal: Signal à dispatcher
        """
        try:
            self._metrics['signals_received'] += 1
            
            # Vérifier la validité
            if not signal.is_valid():
                self._metrics['signals_filtered'] += 1
                return
            
            # Ajouter à la queue
            await asyncio.wait_for(
                self._signal_queue.put(signal),
                timeout=1.0
            )
            
            self._pending_signals[signal.signal_id] = signal
            
        except asyncio.TimeoutError:
            logger.warning("Queue de signaux pleine, signal abandonné")
        except Exception as e:
            logger.error(f"Erreur dispatch signal: {e}")
    
    def dispatch_signal_sync(self, signal: Web3Signal):
        """
        Dispatch un signal de manière synchrone.
        
        Args:
            signal: Signal à dispatcher
        """
        try:
            self._signal_queue.put_nowait(signal)
            self._metrics['signals_received'] += 1
            self._pending_signals[signal.signal_id] = signal
        except asyncio.QueueFull:
            logger.warning("Queue de signaux pleine")
    
    async def _process_loop(self):
        """Boucle de traitement des signaux."""
        while self._running:
            try:
                # Attendre un signal
                signal = await asyncio.wait_for(
                    self._signal_queue.get(),
                    timeout=1.0
                )
                
                await self._process_signal(signal)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur processing signal: {e}")
    
    async def _process_signal(self, signal: Web3Signal):
        """
        Traite un signal individuel.
        
        Args:
            signal: Signal à traiter
        """
        try:
            # Vérifier la validité
            if not signal.is_valid():
                self._metrics['signals_filtered'] += 1
                return
            
            # Logger le signal
            self._log_signal(signal)
            
            # Appeler les callbacks généraux
            for callback in self._callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(signal)
                    else:
                        callback(signal)
                    signal.dispatched_to.append(callback.__name__)
                except Exception as e:
                    logger.error(f"Erreur callback {callback.__name__}: {e}")
            
            # Appeler les callbacks typés
            if signal.signal_type in self._typed_callbacks:
                for callback in self._typed_callbacks[signal.signal_type]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(signal)
                        else:
                            callback(signal)
                        signal.dispatched_to.append(callback.__name__)
                    except Exception as e:
                        logger.error(f"Erreur callback typé: {e}")
            
            # Marquer comme traité
            signal.processed = True
            self._metrics['signals_dispatched'] += 1
            
            # Ajouter à l'historique
            self._add_to_history(signal)
            
            # Retirer des pending
            if signal.signal_id in self._pending_signals:
                del self._pending_signals[signal.signal_id]
                
        except Exception as e:
            logger.error(f"Erreur traitement signal: {e}")
    
    def _log_signal(self, signal: Web3Signal):
        """Log un signal selon sa priorité."""
        message = (
            f"[{signal.source.upper()}] {signal.signal_type} - "
            f"Confidence: {signal.confidence:.2f}"
        )
        
        if signal.priority == SignalPriority.CRITICAL:
            logger.critical(f"🚨 {message}")
        elif signal.priority == SignalPriority.HIGH:
            logger.warning(f"⚠️ {message}")
        elif signal.priority == SignalPriority.MEDIUM:
            logger.info(f"📊 {message}")
        else:
            logger.debug(f"📝 {message}")
    
    def _add_to_history(self, signal: Web3Signal):
        """Ajoute un signal à l'historique."""
        self._signal_history.append(signal)
        
        # Limiter la taille
        if len(self._signal_history) > self._max_history_size:
            self._signal_history = self._signal_history[-self._max_history_size:]
    
    def enrich_signal(
        self,
        original_signal: Dict[str, Any],
        web3_signal: Web3Signal
    ) -> EnrichmentResult:
        """
        Enrichit un signal de trading avec les données Web3.
        
        DESCRIPTION:
        ============
        Applique les handlers d'enrichissement enregistrés
        au signal de trading original.
        
        Args:
            original_signal: Signal de trading existant
            web3_signal: Signal Web3 à utiliser pour l'enrichissement
            
        Returns:
            EnrichmentResult avec le signal enrichi
        """
        if self.mode == IntegrationMode.ALERT_ONLY:
            return EnrichmentResult(
                original_signal=original_signal,
                web3_data={},
                enriched_signal=original_signal,
                modifications=[],
            )
        
        enriched = original_signal.copy()
        modifications = []
        
        for handler in self._enrichment_handlers:
            try:
                before = enriched.copy()
                enriched = handler(enriched, web3_signal)
                
                # Tracker les modifications
                for key in enriched:
                    if key not in before or enriched[key] != before.get(key):
                        modifications.append(f"Modified: {key}")
                        
                self._metrics['enrichments_applied'] += 1
                
            except Exception as e:
                logger.error(f"Erreur enrichissement: {e}")
        
        return EnrichmentResult(
            original_signal=original_signal,
            web3_data=web3_signal.to_dict(),
            enriched_signal=enriched,
            modifications=modifications,
        )
    
    def apply_override(
        self,
        original_signal: Dict[str, Any],
        web3_signal: Web3Signal
    ) -> Optional[Dict[str, Any]]:
        """
        Applique les overrides basés sur les données Web3.
        
        DESCRIPTION:
        ============
        Permet aux handlers d'override de modifier ou annuler
        un signal de trading.
        
        Args:
            original_signal: Signal de trading existant
            web3_signal: Signal Web3
            
        Returns:
            Signal modifié, ou None si annulé
            
        RISQUE:
        =======
        Cette méthode peut affecter directement les décisions
        de trading. À utiliser avec précaution.
        """
        if self.mode != IntegrationMode.OVERRIDE:
            return original_signal
        
        result = original_signal
        
        for handler in self._override_handlers:
            try:
                result = handler(result, web3_signal)
                
                if result is None:
                    logger.warning(
                        f"Signal annulé par override handler: {handler.__name__}"
                    )
                    self._metrics['overrides_applied'] += 1
                    return None
                    
            except Exception as e:
                logger.error(f"Erreur override: {e}")
        
        return result
    
    def get_pending_signals(self) -> List[Web3Signal]:
        """
        Retourne les signaux en attente de traitement.
        
        Returns:
            Liste des signaux pending
        """
        return [
            s for s in self._pending_signals.values()
            if s.is_valid()
        ]
    
    def get_recent_signals(
        self,
        limit: int = 50,
        source: Optional[str] = None,
        signal_type: Optional[str] = None
    ) -> List[Web3Signal]:
        """
        Retourne les signaux récents de l'historique.
        
        Args:
            limit: Nombre maximum de signaux
            source: Filtrer par source
            signal_type: Filtrer par type
            
        Returns:
            Liste des signaux récents
        """
        signals = self._signal_history[-limit:]
        
        if source:
            signals = [s for s in signals if s.source == source]
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]
        
        return signals
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques du dispatcher.
        
        Returns:
            Dictionnaire des métriques
        """
        return {
            **self._metrics,
            'mode': self.mode.value,
            'pending_count': len(self._pending_signals),
            'history_size': len(self._signal_history),
            'callbacks_registered': len(self._callbacks),
            'typed_callbacks': {k: len(v) for k, v in self._typed_callbacks.items()},
            'enrichment_handlers': len(self._enrichment_handlers),
            'override_handlers': len(self._override_handlers),
            'running': self._running,
        }
    
    def set_mode(self, mode: IntegrationMode):
        """
        Change le mode d'intégration.
        
        Args:
            mode: Nouveau mode
        """
        old_mode = self.mode
        self.mode = mode
        logger.info(f"Mode changé: {old_mode.value} -> {mode.value}")


# Factory functions
def create_signal_from_whale_alert(alert: WhaleAlert) -> Web3Signal:
    """Crée un Web3Signal à partir d'une WhaleAlert."""
    return Web3Signal(
        signal_id=str(uuid.uuid4())[:8],
        signal_type=f"MEMPOOL_WHALE_{alert.action.value}",
        source="mempool",
        data=alert.to_dict(),
        priority=SignalPriority.HIGH if alert.amount_usd > 1000000 else SignalPriority.MEDIUM,
        confidence=alert.confidence,
        timestamp=alert.timestamp,
        expires_at=alert.expires_at,
    )


def create_signal_from_correlation(event: CorrelationEvent) -> Web3Signal:
    """Crée un Web3Signal à partir d'un CorrelationEvent."""
    priority = SignalPriority.HIGH if abs(event.cross_chain_index.index_value) > 0.7 else SignalPriority.MEDIUM
    
    return Web3Signal(
        signal_id=event.event_id,
        signal_type=event.signal_type,
        source="cross_chain",
        data=event.to_dict(),
        priority=priority,
        confidence=event.cross_chain_index.confidence,
        timestamp=event.timestamp,
        expires_at=event.expires_at,
    )


def create_signal_from_sentiment(sentiment: StakingSentiment) -> Web3Signal:
    """Crée un Web3Signal à partir d'un StakingSentiment."""
    signal_type = (
        "STAKING_SENTIMENT_BULLISH" if sentiment.is_bullish()
        else "STAKING_SENTIMENT_BEARISH" if sentiment.is_bearish()
        else "STAKING_SENTIMENT_NEUTRAL"
    )
    
    priority = (
        SignalPriority.HIGH if sentiment.sell_pressure_probability > 0.7
        else SignalPriority.MEDIUM
    )
    
    return Web3Signal(
        signal_id=str(uuid.uuid4())[:8],
        signal_type=signal_type,
        source="sentiment",
        data=sentiment.to_dict(),
        priority=priority,
        confidence=sentiment.confidence,
        timestamp=sentiment.timestamp,
    )


# Instance globale
_signal_dispatcher: Optional[SignalDispatcher] = None


def get_signal_dispatcher(mode: IntegrationMode = IntegrationMode.ALERT_ONLY) -> SignalDispatcher:
    """Retourne l'instance globale du dispatcher."""
    global _signal_dispatcher
    if _signal_dispatcher is None:
        _signal_dispatcher = SignalDispatcher(mode)
    return _signal_dispatcher
