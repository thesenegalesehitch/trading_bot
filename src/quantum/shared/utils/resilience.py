"""
Resilience Utilities - Mécanismes d'auto-guérison et de robustesse.
Fournit des décorateurs pour la gestion des fautes réseau intermittentes.
"""

import asyncio
import logging
from functools import wraps
from typing import Callable, Any, Type, Tuple

logger = logging.getLogger(__name__)

def retry_async(
    max_retries: int = 3, 
    delay: float = 1.0, 
    backoff: float = 2.0, 
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Décorateur asynchrone avec backoff exponentiel pour les appels volatiles (RPC/API).
    """
    def decorator(func: Callable[..., Any]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal delay
            last_exception = None
            
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(f"❌ Échec définitif après {max_retries} tentatives: {func.__name__} - {e}")
                        raise
                    
                    wait_time = delay * (backoff ** (attempt - 1))
                    logger.warning(f"⚠️ Tentative {attempt}/{max_retries} échouée pour {func.__name__}. Attente {wait_time:.1f}s... Erreur: {e}")
                    await asyncio.sleep(wait_time)
            
            # Ne devrait pas être atteint
            raise last_exception
            
        return wrapper
    return decorator

class GlobalLock:
    """Verrou asynchrone pour garantir l'atomicité des opérations sensibles (Trading Orders)."""
    _locks = {}

    @classmethod
    async def acquire(cls, name: str):
        if name not in cls._locks:
            cls._locks[name] = asyncio.Lock()
        await cls._locks[name].acquire()

    @classmethod
    def release(cls, name: str):
        if name in cls._locks:
            cls._locks[name].release()
