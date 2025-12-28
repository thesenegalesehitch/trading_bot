"""
Module de logging structuré pour le système de trading.
Gestion avancée des logs avec rotation, niveaux et formatage JSON.

Fonctionnalités:
- Logs structurés JSON
- Rotation automatique des fichiers
- Niveaux de log configurables
- Contexte enrichi (symbole, timeframe, etc.)
"""

import logging
import json
import os
import sys
from datetime import datetime
from typing import Dict, Optional, Any
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from functools import wraps
import time


class JSONFormatter(logging.Formatter):
    """Formateur JSON pour les logs structurés."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Ajouter les extras
        if hasattr(record, 'extra_data'):
            log_data['data'] = record.extra_data
        
        # Ajouter l'exception si présente
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """Formateur coloré pour la console."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Vert
        'WARNING': '\033[33m',   # Jaune
        'ERROR': '\033[31m',     # Rouge
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        
        # Format basique avec couleur
        timestamp = datetime.now().strftime('%H:%M:%S')
        message = f"{color}[{timestamp}] [{record.levelname:8}]{self.RESET} {record.getMessage()}"
        
        # Ajouter les extras si présents
        if hasattr(record, 'extra_data') and record.extra_data:
            extras = ' | '.join(f"{k}={v}" for k, v in record.extra_data.items())
            message += f" {color}({extras}){self.RESET}"
        
        return message


class TradingLogger:
    """
    Logger centralisé pour le système de trading.
    
    Usage:
        logger = TradingLogger('signal_generator')
        logger.info('Signal généré', symbol='EURUSD', direction='BUY')
        logger.warning('Volatilité élevée', atr=0.015)
        logger.error('Échec connexion API', api='Alpha Vantage')
    """
    
    def __init__(
        self,
        name: str = 'quantum_trading',
        log_level: str = 'INFO',
        log_dir: str = 'logs',
        console_output: bool = True,
        file_output: bool = True,
        json_format: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5
    ):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.propagate = False
        
        # Ne pas réinitialiser si déjà configuré
        if self.logger.handlers:
            return
        
        # Créer le répertoire de logs
        if file_output and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Handler console
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ConsoleFormatter())
            self.logger.addHandler(console_handler)
        
        # Handler fichier
        if file_output:
            log_file = os.path.join(log_dir, f'{name}.log')
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            
            if json_format:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
                ))
            
            self.logger.addHandler(file_handler)
    
    def _log(self, level: int, message: str, **kwargs):
        """Log interne avec extras."""
        record = self.logger.makeRecord(
            self.name, level, '', 0, message, (), None
        )
        if kwargs:
            record.extra_data = kwargs
        self.logger.handle(record)
    
    def debug(self, message: str, **kwargs):
        """Log niveau DEBUG."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log niveau INFO."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log niveau WARNING."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log niveau ERROR."""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log niveau CRITICAL."""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def trade(
        self,
        action: str,
        symbol: str,
        price: float,
        **kwargs
    ):
        """Log spécial pour les trades."""
        self.info(
            f"Trade: {action} {symbol} @ {price}",
            action=action,
            symbol=symbol,
            price=price,
            **kwargs
        )
    
    def signal(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        source: str = 'system',
        **kwargs
    ):
        """Log spécial pour les signaux."""
        self.info(
            f"Signal: {direction} {symbol} ({confidence:.1f}%)",
            signal_type='trading_signal',
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            source=source,
            **kwargs
        )
    
    def risk_event(
        self,
        event_type: str,
        severity: str,
        **kwargs
    ):
        """Log spécial pour les événements de risque."""
        level = logging.WARNING if severity == 'warning' else logging.CRITICAL
        self._log(
            level,
            f"Risk Event: {event_type}",
            event_type=event_type,
            severity=severity,
            **kwargs
        )


def log_execution_time(logger: TradingLogger = None):
    """Décorateur pour logger le temps d'exécution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                
                if logger:
                    logger.debug(
                        f"Exécution: {func.__name__}",
                        function=func.__name__,
                        duration_ms=round(elapsed * 1000, 2)
                    )
                return result
                
            except Exception as e:
                elapsed = time.time() - start
                if logger:
                    logger.error(
                        f"Erreur dans {func.__name__}: {str(e)}",
                        function=func.__name__,
                        duration_ms=round(elapsed * 1000, 2),
                        error=str(e)
                    )
                raise
        return wrapper
    return decorator


class LogContext:
    """Contexte de logging réutilisable."""
    
    def __init__(self, logger: TradingLogger, **context):
        self.logger = logger
        self.context = context
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, **{**self.context, **kwargs})
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, **{**self.context, **kwargs})
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, **{**self.context, **kwargs})


# Logger global par défaut
_default_logger = None


def get_logger(name: str = 'quantum_trading') -> TradingLogger:
    """Obtient un logger configuré."""
    global _default_logger
    
    if _default_logger is None or _default_logger.name != name:
        _default_logger = TradingLogger(name)
    
    return _default_logger


if __name__ == "__main__":
    print("=" * 60)
    print("TEST TRADING LOGGER")
    print("=" * 60)
    
    # Créer un logger
    logger = TradingLogger('test_logger', log_dir='logs')
    
    # Tests basiques
    logger.debug("Message de debug", extra_info="test")
    logger.info("Système démarré", version="1.0")
    logger.warning("Volatilité élevée", atr=0.025, symbol="EURUSD")
    logger.error("Connexion échouée", api="Alpha Vantage", attempts=3)
    
    # Log de signal
    logger.signal(
        symbol="EURUSD",
        direction="BUY",
        confidence=87.5,
        source="ml_ensemble",
        price=1.0850
    )
    
    # Log de risque
    logger.risk_event(
        event_type="max_drawdown_approaching",
        severity="warning",
        current_dd=0.045,
        limit=0.05
    )
    
    # Avec contexte
    ctx = LogContext(logger, symbol="EURUSD", timeframe="1h")
    ctx.info("Analyse démarrée")
    ctx.info("Signal trouvé", confidence=75)
    
    print("\n✅ Logs générés avec succès")
    print("📁 Vérifiez le dossier 'logs' pour les fichiers JSON")
