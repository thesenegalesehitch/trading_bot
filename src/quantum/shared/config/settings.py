"""
Quantum Trading System - Configuration centralisée
==================================================
Version épurée : uniquement les paramètres utilisés.

Author: Alexandre Albert Ndour
Copyright (c) 2026 Alexandre Albert Ndour. All Rights Reserved.
"""

from dataclasses import dataclass, field
from typing import List, Dict
from datetime import timedelta
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class SymbolConfig:
    """Configuration des symboles tradés."""
    
    # Forex Majeurs
    EURUSD: str = "EURUSD=X"
    GBPUSD: str = "GBPUSD=X"
    USDJPY: str = "USDJPY=X"
    USDCHF: str = "USDCHF=X"
    AUDUSD: str = "AUDUSD=X"
    USDCAD: str = "USDCAD=X"
    NZDUSD: str = "NZDUSD=X"
    
    # Forex Mineurs
    EURGBP: str = "EURGBP=X"
    EURJPY: str = "EURJPY=X"
    GBPJPY: str = "GBPJPY=X"
    
    # Métaux
    GOLD: str = "GC=F"
    SILVER: str = "SI=F"
    
    # Crypto
    BTCUSDT: str = "BTC-USD"
    ETHUSDT: str = "ETH-USD"
    SOLUSDT: str = "SOL-USD"
    
    # Indices
    SP500: str = "^GSPC"
    NASDAQ: str = "^IXIC"
    
    # Symboles actifs par défaut
    ACTIVE_SYMBOLS: List[str] = field(default_factory=lambda: [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", 
        "AUDUSD=X", "USDCAD=X", "NZDUSD=X",
        "GC=F", "SI=F",
        "BTC-USD", "ETH-USD"
    ])
    
    FOREX_PAIRS: List[str] = field(default_factory=lambda: [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", 
        "AUDUSD=X", "USDCAD=X", "NZDUSD=X",
        "EURGBP=X", "EURJPY=X", "GBPJPY=X"
    ])
    
    CRYPTO_PAIRS: List[str] = field(default_factory=lambda: [
        "BTC-USD", "ETH-USD", "SOL-USD"
    ])
    
    DISPLAY_NAMES: Dict[str, str] = field(default_factory=lambda: {
        "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD",
        "USDJPY=X": "USD/JPY", "USDCHF=X": "USD/CHF",
        "AUDUSD=X": "AUD/USD", "USDCAD=X": "USD/CAD",
        "NZDUSD=X": "NZD/USD", "EURGBP=X": "EUR/GBP",
        "EURJPY=X": "EUR/JPY", "GBPJPY=X": "GBP/JPY",
        "GC=F": "XAU/USD (Or)", "SI=F": "XAG/USD (Argent)",
        "BTC-USD": "BTC/USD", "ETH-USD": "ETH/USD",
        "SOL-USD": "SOL/USD",
        "^GSPC": "S&P 500", "^IXIC": "NASDAQ",
    })
    
    TRADING_SESSIONS: Dict[str, Dict] = field(default_factory=lambda: {
        "SYDNEY": {"open": "22:00", "close": "07:00", "timezone": "UTC"},
        "TOKYO": {"open": "00:00", "close": "09:00", "timezone": "UTC"},
        "LONDON": {"open": "08:00", "close": "17:00", "timezone": "UTC"},
        "NEW_YORK": {"open": "13:00", "close": "22:00", "timezone": "UTC"},
    })
    
    PAIR_PROPERTIES: Dict[str, Dict] = field(default_factory=lambda: {
        "EURUSD=X": {"pip_size": 0.0001, "pip_value": 10, "min_lot": 0.01},
        "GBPUSD=X": {"pip_size": 0.0001, "pip_value": 10, "min_lot": 0.01},
        "USDJPY=X": {"pip_size": 0.01, "pip_value": 9.1, "min_lot": 0.01},
        "USDCHF=X": {"pip_size": 0.0001, "pip_value": 10.3, "min_lot": 0.01},
        "GC=F": {"pip_size": 0.1, "pip_value": 10, "min_lot": 0.01},
        "BTC-USD": {"pip_size": 1, "pip_value": 1, "min_lot": 0.001},
        "ETH-USD": {"pip_size": 0.1, "pip_value": 1, "min_lot": 0.01},
    })


@dataclass
class TimeframeConfig:
    """Configuration des unités de temps."""
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["15m", "1h", "4h", "1d"])
    PRIMARY_TIMEFRAME: str = "1h"
    TIMEFRAME_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "15m": 0.15, "1h": 0.30, "4h": 0.35, "1d": 0.20
    })


@dataclass
class DataConfig:
    """Configuration du moteur de données."""
    HISTORICAL_YEARS: int = 2
    PRIMARY_DATA_SOURCE: str = "yfinance"
    
    # Feature engineering
    VOLATILITY_WINDOW: int = 14
    VOLUME_NORMALIZATION_WINDOW: int = 20
    EXTRACT_HOUR: bool = True
    EXTRACT_DAY_OF_WEEK: bool = True
    EXTRACT_MONTH: bool = True


@dataclass 
class TechnicalConfig:
    """Configuration de l'analyse technique."""
    REQUIRED_TF_CONFIRMATION: int = 3
    ORDER_BLOCK_LOOKBACK: int = 50
    FVG_MIN_GAP_PERCENT: float = 0.1
    ICHIMOKU_TENKAN: int = 9
    ICHIMOKU_KIJUN: int = 26
    ICHIMOKU_SENKOU_B: int = 52
    ICHIMOKU_DISPLACEMENT: int = 26
    DIVERGENCE_LOOKBACK: int = 50
    DIVERGENCE_MIN_STRENGTH: float = 0.3
    WYCKOFF_LOOKBACK: int = 100
    WYCKOFF_RANGE_THRESHOLD: float = 0.03


@dataclass
class RiskConfig:
    """Configuration de la gestion du risque."""
    RISK_PER_TRADE: float = 0.01
    MAX_DRAWDOWN: float = 0.05
    MAX_CONSECUTIVE_LOSSES: int = 3
    MAX_DAILY_LOSS: float = 0.02
    ATR_PERIOD: int = 14
    ATR_MULTIPLIER: float = 1.5
    TP_LEVELS: List[Dict] = field(default_factory=lambda: [
        {"ratio": 1.0, "size_percent": 50},
        {"ratio": 2.0, "size_percent": 30},
        {"ratio": 3.0, "size_percent": 20},
    ])
    INITIAL_CAPITAL: float = 10000.0
    USE_KELLY: bool = True
    KELLY_FRACTION: float = 0.5
    VAR_CONFIDENCE: float = 0.95
    VAR_METHOD: str = "monte_carlo"


@dataclass
class DatabaseConfig:
    """Configuration base de données SQLite."""
    DATABASE_URL: str = field(default_factory=lambda: os.getenv(
        'DATABASE_URL', 'sqlite+aiosqlite:///./quantum.db'
    ))
    DATABASE_URL_SYNC: str = field(default_factory=lambda: os.getenv(
        'DATABASE_URL_SYNC', 'sqlite:///./quantum.db'
    ))
    ECHO_SQL: bool = False


@dataclass
class AuthConfig:
    """Configuration authentification JWT."""
    SECRET_KEY: str = field(default_factory=lambda: os.getenv(
        'JWT_SECRET_KEY', 'DEV_INSECURE_SECRET_KEY_FOR_LOCAL_ONLY'
    ))
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 11520  # 8 jours
    DEMO_ACCOUNT_INITIAL_BALANCE: float = 1_000_000.0

    def __post_init__(self):
        if self.SECRET_KEY == 'DEV_INSECURE_SECRET_KEY_FOR_LOCAL_ONLY' and os.getenv('MODE') == 'production':
            raise ValueError("ERREUR: Définir JWT_SECRET_KEY dans .env pour la production.")


@dataclass
class SystemConfig:
    """Configuration système globale."""
    MODE: str = field(default_factory=lambda: os.getenv('MODE', 'dev'))
    LOG_LEVEL: str = "INFO"
    DATA_DIR: str = "data/cache"
    LOGS_DIR: str = "logs"


# Instance globale
class Config:
    """Conteneur global pour toutes les configurations."""
    symbols = SymbolConfig()
    timeframes = TimeframeConfig()
    data = DataConfig()
    technical = TechnicalConfig()
    risk = RiskConfig()
    database = DatabaseConfig()
    auth = AuthConfig()
    system = SystemConfig()


config = Config()
