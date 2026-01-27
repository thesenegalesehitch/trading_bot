"""
Gestionnaire de connexions WebSocket persistantes.

Ce module fournit une couche d'abstraction pour gérer plusieurs
connexions WebSocket avec monitoring de santé et métriques.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import time

from web3_innovation.settings import web3_config, Chain

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Statut de santé d'une connexion."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ConnectionMetrics:
    """
    Métriques de performance d'une connexion.
    
    DESCRIPTION:
    ============
    Suit les métriques clés pour évaluer la qualité
    de la connexion WebSocket.
    """
    chain: Chain
    
    # Compteurs
    messages_received: int = 0
    messages_sent: int = 0
    errors_count: int = 0
    reconnections_count: int = 0
    
    # Latence
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    latency_samples: List[float] = field(default_factory=list)
    
    # Timing
    connected_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None
    last_error_at: Optional[datetime] = None
    
    # Uptime
    total_uptime_seconds: float = 0.0
    current_session_start: Optional[datetime] = None
    
    def record_latency(self, latency_ms: float):
        """Enregistre une mesure de latence."""
        self.latency_samples.append(latency_ms)
        # Garder les 1000 derniers échantillons
        if len(self.latency_samples) > 1000:
            self.latency_samples = self.latency_samples[-1000:]
        
        # Recalculer les stats
        if self.latency_samples:
            self.avg_latency_ms = sum(self.latency_samples) / len(self.latency_samples)
            sorted_samples = sorted(self.latency_samples)
            p99_index = int(len(sorted_samples) * 0.99)
            self.p99_latency_ms = sorted_samples[min(p99_index, len(sorted_samples) - 1)]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'chain': self.chain.value,
            'messages_received': self.messages_received,
            'messages_sent': self.messages_sent,
            'errors_count': self.errors_count,
            'reconnections_count': self.reconnections_count,
            'avg_latency_ms': round(self.avg_latency_ms, 2),
            'p99_latency_ms': round(self.p99_latency_ms, 2),
            'connected_at': self.connected_at.isoformat() if self.connected_at else None,
            'last_message_at': self.last_message_at.isoformat() if self.last_message_at else None,
            'total_uptime_seconds': round(self.total_uptime_seconds, 1),
        }


@dataclass
class HealthCheck:
    """
    Résultat d'un health check.
    
    DESCRIPTION:
    ============
    Évalue l'état de santé global de la connexion
    basé sur plusieurs critères.
    """
    status: HealthStatus
    chain: Chain
    
    # Détails
    is_connected: bool = False
    last_message_age_seconds: Optional[float] = None
    error_rate_percent: float = 0.0
    latency_ok: bool = True
    
    # Seuils dépassés
    warnings: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'status': self.status.value,
            'chain': self.chain.value,
            'is_connected': self.is_connected,
            'last_message_age_seconds': self.last_message_age_seconds,
            'error_rate_percent': round(self.error_rate_percent, 2),
            'latency_ok': self.latency_ok,
            'warnings': self.warnings,
            'timestamp': self.timestamp.isoformat(),
        }


class WebSocketManager:
    """
    Gestionnaire de connexions WebSocket avec monitoring.
    
    DESCRIPTION:
    ============
    Couche de gestion au-dessus de QuickNodeClient qui ajoute:
    - Health checking périodique
    - Métriques de performance
    - Alertes sur dégradation
    - Logging structuré
    
    INNOVATION:
    ===========
    Monitoring proactif de la qualité des connexions pour
    détecter les dégradations avant qu'elles n'impactent
    les signaux de trading.
    
    USAGE:
    ======
    ```python
    manager = WebSocketManager()
    await manager.start()
    
    # Check health
    health = manager.get_health_status()
    ```
    
    RISQUE:
    =======
    Le monitoring ajoute un overhead minimal mais peut
    masquer des problèmes si les seuils sont mal configurés.
    """
    
    # Seuils de health check
    MAX_MESSAGE_AGE_SECONDS = 60  # Alerte si pas de message depuis 60s
    MAX_ERROR_RATE_PERCENT = 5.0  # Alerte si > 5% d'erreurs
    MAX_LATENCY_MS = 500  # Alerte si latence > 500ms
    
    def __init__(self):
        """
        Initialise le gestionnaire WebSocket.
        
        DESCRIPTION:
        ============
        Configure le monitoring et les métriques pour
        toutes les chaînes supportées.
        """
        self.config = web3_config
        
        # Métriques par chaîne
        self._metrics: Dict[Chain, ConnectionMetrics] = {
            chain: ConnectionMetrics(chain=chain) for chain in Chain
        }
        
        # Health check results
        self._health_cache: Dict[Chain, HealthCheck] = {}
        
        # Callbacks d'alerte
        self._alert_callbacks: List[Callable[[HealthCheck], None]] = []
        
        # Tâches de monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("WebSocketManager initialisé")
    
    async def start(self, health_check_interval: int = 30):
        """
        Démarre le monitoring des connexions.
        
        Args:
            health_check_interval: Intervalle entre les health checks (secondes)
        """
        if self._running:
            return
        
        self._running = True
        
        # Démarrer le health check périodique
        self._health_check_task = asyncio.create_task(
            self._health_check_loop(health_check_interval)
        )
        
        logger.info(f"WebSocketManager démarré (health check: {health_check_interval}s)")
    
    async def stop(self):
        """Arrête le monitoring."""
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("WebSocketManager arrêté")
    
    def record_message_received(self, chain: Chain):
        """Enregistre un message reçu."""
        metrics = self._metrics[chain]
        metrics.messages_received += 1
        metrics.last_message_at = datetime.utcnow()
    
    def record_message_sent(self, chain: Chain):
        """Enregistre un message envoyé."""
        metrics = self._metrics[chain]
        metrics.messages_sent += 1
    
    def record_error(self, chain: Chain):
        """Enregistre une erreur."""
        metrics = self._metrics[chain]
        metrics.errors_count += 1
        metrics.last_error_at = datetime.utcnow()
    
    def record_reconnection(self, chain: Chain):
        """Enregistre une reconnexion."""
        metrics = self._metrics[chain]
        metrics.reconnections_count += 1
    
    def record_latency(self, chain: Chain, latency_ms: float):
        """Enregistre une mesure de latence."""
        self._metrics[chain].record_latency(latency_ms)
    
    def record_connected(self, chain: Chain):
        """Enregistre une connexion établie."""
        metrics = self._metrics[chain]
        now = datetime.utcnow()
        metrics.connected_at = now
        metrics.current_session_start = now
    
    def record_disconnected(self, chain: Chain):
        """Enregistre une déconnexion."""
        metrics = self._metrics[chain]
        if metrics.current_session_start:
            session_duration = (datetime.utcnow() - metrics.current_session_start).total_seconds()
            metrics.total_uptime_seconds += session_duration
            metrics.current_session_start = None
    
    async def _health_check_loop(self, interval: int):
        """Boucle de health check périodique."""
        while self._running:
            try:
                await asyncio.sleep(interval)
                
                for chain in Chain:
                    health = self._perform_health_check(chain)
                    self._health_cache[chain] = health
                    
                    # Alerter si dégradé ou unhealthy
                    if health.status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]:
                        await self._trigger_alert(health)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur health check: {e}")
    
    def _perform_health_check(self, chain: Chain) -> HealthCheck:
        """
        Effectue un health check pour une chaîne.
        
        Args:
            chain: Chaîne à vérifier
            
        Returns:
            Résultat du health check
        """
        metrics = self._metrics[chain]
        warnings = []
        
        # Vérifier l'âge du dernier message
        last_message_age = None
        if metrics.last_message_at:
            last_message_age = (datetime.utcnow() - metrics.last_message_at).total_seconds()
            if last_message_age > self.MAX_MESSAGE_AGE_SECONDS:
                warnings.append(f"Pas de message depuis {last_message_age:.0f}s")
        
        # Calculer le taux d'erreur
        total_messages = metrics.messages_received + metrics.messages_sent
        error_rate = 0.0
        if total_messages > 0:
            error_rate = (metrics.errors_count / total_messages) * 100
            if error_rate > self.MAX_ERROR_RATE_PERCENT:
                warnings.append(f"Taux d'erreur élevé: {error_rate:.1f}%")
        
        # Vérifier la latence
        latency_ok = metrics.avg_latency_ms < self.MAX_LATENCY_MS
        if not latency_ok:
            warnings.append(f"Latence élevée: {metrics.avg_latency_ms:.0f}ms")
        
        # Déterminer le statut
        is_connected = metrics.connected_at is not None and metrics.current_session_start is not None
        
        if not is_connected:
            status = HealthStatus.UNHEALTHY
        elif len(warnings) >= 2:
            status = HealthStatus.UNHEALTHY
        elif len(warnings) == 1:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        return HealthCheck(
            status=status,
            chain=chain,
            is_connected=is_connected,
            last_message_age_seconds=last_message_age,
            error_rate_percent=error_rate,
            latency_ok=latency_ok,
            warnings=warnings,
        )
    
    async def _trigger_alert(self, health: HealthCheck):
        """Déclenche les alertes pour un health check."""
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(health)
                else:
                    callback(health)
            except Exception as e:
                logger.error(f"Erreur callback alerte: {e}")
    
    def register_alert_callback(self, callback: Callable[[HealthCheck], None]):
        """
        Enregistre un callback pour les alertes de santé.
        
        Args:
            callback: Fonction appelée quand une connexion est dégradée
        """
        self._alert_callbacks.append(callback)
    
    def get_health_status(self, chain: Optional[Chain] = None) -> Dict[str, HealthCheck]:
        """
        Retourne le statut de santé des connexions.
        
        Args:
            chain: Chaîne spécifique ou None pour toutes
            
        Returns:
            Dictionnaire des health checks
        """
        if chain:
            # Effectuer un check immédiat
            return {chain.value: self._perform_health_check(chain)}
        
        return {c.value: self._perform_health_check(c) for c in Chain}
    
    def get_metrics(self, chain: Optional[Chain] = None) -> Dict[str, Dict]:
        """
        Retourne les métriques des connexions.
        
        Args:
            chain: Chaîne spécifique ou None pour toutes
            
        Returns:
            Dictionnaire des métriques
        """
        if chain:
            return {chain.value: self._metrics[chain].to_dict()}
        
        return {c.value: self._metrics[c].to_dict() for c in Chain}
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé global du statut.
        
        Returns:
            Résumé avec métriques agrégées
        """
        health_checks = self.get_health_status()
        metrics = self.get_metrics()
        
        # Compter les statuts
        statuses = [h.status for h in health_checks.values()]
        
        return {
            'overall_status': (
                HealthStatus.HEALTHY.value if all(s == HealthStatus.HEALTHY for s in statuses)
                else HealthStatus.DEGRADED.value if any(s == HealthStatus.HEALTHY for s in statuses)
                else HealthStatus.UNHEALTHY.value
            ),
            'healthy_count': sum(1 for s in statuses if s == HealthStatus.HEALTHY),
            'degraded_count': sum(1 for s in statuses if s == HealthStatus.DEGRADED),
            'unhealthy_count': sum(1 for s in statuses if s == HealthStatus.UNHEALTHY),
            'total_messages_received': sum(m['messages_received'] for m in metrics.values()),
            'total_errors': sum(m['errors_count'] for m in metrics.values()),
            'chains': {
                chain: {
                    'health': health_checks[chain].to_dict(),
                    'metrics': metrics[chain],
                }
                for chain in metrics.keys()
            },
            'timestamp': datetime.utcnow().isoformat(),
        }


# Instance globale
_websocket_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """Retourne l'instance globale du gestionnaire WebSocket."""
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager()
    return _websocket_manager
