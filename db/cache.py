"""
Redis caching layer for Quantum Trading System.

Provides intelligent caching with auto-invalidation for market data, signals, and correlations.
"""

import redis
import json
import pickle
from typing import Any, Optional, Dict
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


class CacheManager:
    """
    Redis cache manager with intelligent invalidation.

    Caches:
    - Market data with TTL based on interval
    - Signals with short TTL
    - Correlations with rolling window TTL
    - Risk metrics with portfolio-based invalidation
    """

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, password: str = None):
        if not REDIS_AVAILABLE:
            logger.warning("Redis non disponible - cache désactivé")
            self.client = None
            return

        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False  # Pour les données binaires
            )
            self.client.ping()  # Test de connexion
            logger.info("Connexion Redis établie")
        except Exception as e:
            logger.error(f"Erreur connexion Redis: {e}")
            self.client = None

    def _make_key(self, prefix: str, *args) -> str:
        """Génère une clé Redis."""
        return f"{prefix}:{':'.join(str(arg) for arg in args)}"

    def set_market_data(self, symbol: str, interval: str, data: Any, ttl_hours: int = 24) -> bool:
        """Cache les données de marché."""
        if not self.client:
            return False

        key = self._make_key('market_data', symbol, interval)
        try:
            serialized = pickle.dumps(data)
            return self.client.setex(key, timedelta(hours=ttl_hours), serialized)
        except Exception as e:
            logger.error(f"Erreur cache market_data: {e}")
            return False

    def get_market_data(self, symbol: str, interval: str) -> Optional[Any]:
        """Récupère les données de marché du cache."""
        if not self.client:
            return None

        key = self._make_key('market_data', symbol, interval)
        try:
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Erreur récupération market_data: {e}")
        return None

    def set_signals(self, symbol: str, signals: Dict, ttl_minutes: int = 5) -> bool:
        """Cache les signaux avec TTL court."""
        if not self.client:
            return False

        key = self._make_key('signals', symbol)
        try:
            serialized = json.dumps(signals).encode('utf-8')
            return self.client.setex(key, timedelta(minutes=ttl_minutes), serialized)
        except Exception as e:
            logger.error(f"Erreur cache signals: {e}")
            return False

    def get_signals(self, symbol: str) -> Optional[Dict]:
        """Récupère les signaux du cache."""
        if not self.client:
            return None

        key = self._make_key('signals', symbol)
        try:
            data = self.client.get(key)
            if data:
                return json.loads(data.decode('utf-8'))
        except Exception as e:
            logger.error(f"Erreur récupération signals: {e}")
        return None

    def set_correlations(self, correlations: Any, ttl_hours: int = 1) -> bool:
        """Cache la matrice de corrélations."""
        if not self.client:
            return False

        key = 'correlations'
        try:
            serialized = pickle.dumps(correlations)
            return self.client.setex(key, timedelta(hours=ttl_hours), serialized)
        except Exception as e:
            logger.error(f"Erreur cache correlations: {e}")
            return False

    def get_correlations(self) -> Optional[Any]:
        """Récupère la matrice de corrélations."""
        if not self.client:
            return None

        try:
            data = self.client.get('correlations')
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Erreur récupération correlations: {e}")
        return None

    def set_risk_metrics(self, portfolio_id: str, metrics: Dict, ttl_minutes: int = 10) -> bool:
        """Cache les métriques de risque."""
        if not self.client:
            return False

        key = self._make_key('risk', portfolio_id)
        try:
            serialized = json.dumps(metrics).encode('utf-8')
            return self.client.setex(key, timedelta(minutes=ttl_minutes), serialized)
        except Exception as e:
            logger.error(f"Erreur cache risk_metrics: {e}")
            return False

    def get_risk_metrics(self, portfolio_id: str) -> Optional[Dict]:
        """Récupère les métriques de risque."""
        if not self.client:
            return None

        key = self._make_key('risk', portfolio_id)
        try:
            data = self.client.get(key)
            if data:
                return json.loads(data.decode('utf-8'))
        except Exception as e:
            logger.error(f"Erreur récupération risk_metrics: {e}")
        return None

    def invalidate_symbol(self, symbol: str):
        """Invalide tout le cache pour un symbole."""
        if not self.client:
            return

        try:
            # Pattern pour toutes les clés du symbole
            pattern = f"*:*{symbol}*"
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
                logger.info(f"Cache invalidé pour {symbol}: {len(keys)} clés")
        except Exception as e:
            logger.error(f"Erreur invalidation cache {symbol}: {e}")

    def invalidate_correlations(self):
        """Invalide le cache des corrélations."""
        if self.client:
            try:
                self.client.delete('correlations')
                logger.info("Cache des corrélations invalidé")
            except Exception as e:
                logger.error(f"Erreur invalidation correlations: {e}")

    def clear_all(self):
        """Vide tout le cache."""
        if self.client:
            try:
                self.client.flushdb()
                logger.info("Cache vidé complètement")
            except Exception as e:
                logger.error(f"Erreur vidage cache: {e}")

    def get_stats(self) -> Dict:
        """Retourne les statistiques du cache."""
        if not self.client:
            return {'status': 'disabled'}

        try:
            info = self.client.info()
            return {
                'status': 'connected',
                'keys': self.client.dbsize(),
                'memory_used': info.get('used_memory_human', 'N/A'),
                'connections': info.get('connected_clients', 0)
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


# Instance globale
cache_manager = None

def get_cache_manager():
    """Retourne l'instance globale du cache manager."""
    global cache_manager
    if cache_manager is None:
        from config.settings import config
        cache_manager = CacheManager(
            host=config.database.REDIS_HOST,
            port=config.database.REDIS_PORT,
            db=config.database.REDIS_DB,
            password=config.database.REDIS_PASSWORD or None
        )
    return cache_manager