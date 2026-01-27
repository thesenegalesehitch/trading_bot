"""
Configuration pour le module Web3 Innovation.
Centralise tous les paramètres QuickNode, seuils de détection et endpoints blockchain.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class Chain(Enum):
    """Blockchains supportées."""
    ETHEREUM = "ethereum"
    SOLANA = "solana"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    BASE = "base"


class SignalType(Enum):
    """Types de signaux Web3."""
    MEMPOOL_WHALE_BUY = "mempool_whale_buy"
    MEMPOOL_WHALE_SELL = "mempool_whale_sell"
    MEMPOOL_SANDWICH_DETECTED = "mempool_sandwich_detected"
    CROSS_CHAIN_CASCADE_BULLISH = "cross_chain_cascade_bullish"
    CROSS_CHAIN_CASCADE_BEARISH = "cross_chain_cascade_bearish"
    BRIDGE_FLOW_ALERT = "bridge_flow_alert"
    STAKING_SENTIMENT_BULLISH = "staking_sentiment_bullish"
    STAKING_SENTIMENT_BEARISH = "staking_sentiment_bearish"
    WHALE_UNSTAKE_ALERT = "whale_unstake_alert"


@dataclass
class QuickNodeConfig:
    """Configuration des endpoints QuickNode."""
    
    # Endpoints WebSocket (depuis variables d'environnement)
    ETH_WSS_ENDPOINT: str = field(
        default_factory=lambda: os.getenv(
            'QUICKNODE_ETH_ENDPOINT',
            'wss://eth-mainnet.g.alchemy.com/v2/demo'  # Fallback demo
        )
    )
    SOL_WSS_ENDPOINT: str = field(
        default_factory=lambda: os.getenv(
            'QUICKNODE_SOL_ENDPOINT',
            'wss://api.mainnet-beta.solana.com'  # Public fallback
        )
    )
    POLYGON_WSS_ENDPOINT: str = field(
        default_factory=lambda: os.getenv(
            'QUICKNODE_POLYGON_ENDPOINT',
            ''
        )
    )
    
    # Endpoints HTTP pour les appels RPC
    ETH_HTTP_ENDPOINT: str = field(
        default_factory=lambda: os.getenv(
            'QUICKNODE_ETH_HTTP',
            'https://eth-mainnet.g.alchemy.com/v2/demo'
        )
    )
    SOL_HTTP_ENDPOINT: str = field(
        default_factory=lambda: os.getenv(
            'QUICKNODE_SOL_HTTP',
            'https://api.mainnet-beta.solana.com'
        )
    )
    
    # Timeouts et reconnexion
    CONNECTION_TIMEOUT_SECONDS: int = 30
    RECONNECT_DELAY_SECONDS: int = 5
    MAX_RECONNECT_ATTEMPTS: int = 10
    HEARTBEAT_INTERVAL_SECONDS: int = 30


@dataclass
class WhaleThresholdConfig:
    """
    Seuils de détection des baleines.
    
    INNOVATION:
    ===========
    Seuils dynamiques basés sur la volatilité actuelle du marché.
    Un marché volatile abaisse les seuils pour capturer plus de signaux.
    """
    
    # Seuils statiques (valeurs par défaut)
    ETH_WHALE_THRESHOLD: float = field(
        default_factory=lambda: float(os.getenv('WEB3_ETH_WHALE_THRESHOLD', '100'))
    )
    SOL_WHALE_THRESHOLD: float = field(
        default_factory=lambda: float(os.getenv('WEB3_SOL_WHALE_THRESHOLD', '10000'))
    )
    USD_WHALE_THRESHOLD: float = field(
        default_factory=lambda: float(os.getenv('WEB3_USD_WHALE_THRESHOLD', '500000'))
    )
    
    # Multiplicateurs par volatilité
    VOLATILITY_MULTIPLIERS: Dict[str, float] = field(default_factory=lambda: {
        'low': 1.5,      # Marché calme: seuils plus élevés
        'medium': 1.0,   # Normal
        'high': 0.7,     # Volatile: seuils plus bas
        'extreme': 0.5   # Très volatile: capturer plus
    })
    
    # Tokens spécifiques avec seuils personnalisés
    TOKEN_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        'WETH': 100,
        'WBTC': 10,
        'USDC': 1000000,
        'USDT': 1000000,
        'SOL': 10000,
        'MATIC': 500000,
    })


@dataclass
class MempoolConfig:
    """Configuration de l'analyseur de mempool."""
    
    # Filtres de transactions
    MIN_GAS_PRICE_GWEI: float = 20  # Ignorer les TX à gas très bas
    MAX_PENDING_AGE_SECONDS: int = 120  # TX pendantes trop vieilles
    
    # DEX à monitorer
    MONITORED_DEX_ROUTERS: Dict[str, str] = field(default_factory=lambda: {
        # Ethereum
        'uniswap_v2': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
        'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
        'sushiswap': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
        'curve': '0x99a58482BD75cbab83b27EC03CA68fF489b5788f',
        '1inch': '0x1111111254fb6c44bAC0beD2854e76F90643097d',
        # Solana (program IDs)
        'jupiter': 'JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB',
        'raydium': '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8',
    })
    
    # Contrats de bridge
    BRIDGE_CONTRACTS: Dict[str, str] = field(default_factory=lambda: {
        'wormhole_eth': '0x3ee18B2214AFF97000D974cf647E7C347E8fa585',
        'portal_bridge': '0x3014ca10b91cb3D0AD85fEf7A3Cb95BCDAc90B3E',
        'layerzero': '0x66A71Dcef29A0fFBDBE3c6a460a3B5BC225Cd675',
    })
    
    # Timeout des signaux (après ce délai, signal invalide)
    SIGNAL_VALIDITY_SECONDS: int = 60
    
    # Score de confiance minimum
    MIN_CONFIDENCE_THRESHOLD: float = 0.6


@dataclass
class CrossChainConfig:
    """Configuration du Cross-Chain Correlation Oracle."""
    
    # Fenêtre de corrélation (en secondes)
    CORRELATION_WINDOW_SECONDS: int = 180  # 3 minutes
    
    # Seuil de corrélation pour émettre un signal
    CORRELATION_THRESHOLD: float = 0.7
    
    # Facteur de décroissance temporelle (lambda)
    TIME_DECAY_LAMBDA: float = 0.5  # Par minute
    
    # Poids des baleines par taille
    WHALE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'small': 1.0,      # 100-500 ETH
        'medium': 1.5,     # 500-2000 ETH
        'large': 2.0,      # 2000-10000 ETH
        'mega': 3.0,       # >10000 ETH
    })
    
    # Volume minimum pour considérer un mouvement significatif
    MIN_VOLUME_DELTA_PERCENT: float = 5.0
    
    # Lag maximum acceptable pour corrélation (secondes)
    MAX_ACCEPTABLE_LAG_SECONDS: int = 300  # 5 minutes


@dataclass
class SentimentConfig:
    """Configuration de l'analyseur de sentiment on-chain."""
    
    # Contrats de staking à monitorer
    STAKING_CONTRACTS: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        'ethereum': {
            'beacon_deposit': '0x00000000219ab540356cBB839Cbe05303d7705Fa',
            'lido': '0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84',
            'rocketpool': '0xDD3f50F8A6CafbE9b31a427582963f465E745AF8',
        },
        'solana': {
            'native_staking': 'Stake11111111111111111111111111111111111111',
            'marinade': 'MarBmsSgKXdrN1egZf5sqe1TMai9K1rChYNDJgjq7aD',
        },
        'polygon': {
            'matic_staking': '0x5e3Ef299fDDf15eAa0432E6e66473ace8c13D908',
        }
    })
    
    # Fenêtre d'analyse pour le ratio staking
    STAKING_RATIO_WINDOW_HOURS: int = 24
    
    # Seuils de sentiment
    SENTIMENT_THRESHOLDS: Dict[str, tuple] = field(default_factory=lambda: {
        'bearish': (0, 30),
        'neutral': (30, 50),
        'bullish': (50, 70),
        'extremely_bullish': (70, 100),
    })
    
    # Prédiction de pression vendeuse
    SELL_PRESSURE_HORIZON_HOURS: int = 24
    UNSTAKE_TO_SELL_PROBABILITY: float = 0.65  # 65% des unstakes mènent à une vente


@dataclass
class Web3Config:
    """Configuration globale du module Web3 Innovation."""
    
    quicknode: QuickNodeConfig = field(default_factory=QuickNodeConfig)
    whale_thresholds: WhaleThresholdConfig = field(default_factory=WhaleThresholdConfig)
    mempool: MempoolConfig = field(default_factory=MempoolConfig)
    cross_chain: CrossChainConfig = field(default_factory=CrossChainConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    
    # Mode d'intégration par défaut
    DEFAULT_INTEGRATION_MODE: str = "ALERT_ONLY"
    
    # Logging
    LOG_LEVEL: str = field(
        default_factory=lambda: os.getenv('WEB3_LOG_LEVEL', 'INFO')
    )
    LOG_TO_FILE: bool = True
    LOG_FILE_PATH: str = "logs/web3_innovation.log"
    
    # Feature flags
    ENABLE_MEMPOOL_ANALYSIS: bool = True
    ENABLE_CROSS_CHAIN_ORACLE: bool = True
    ENABLE_SENTIMENT_ANALYSIS: bool = True
    
    # Backtesting mode (désactive les connexions réelles)
    BACKTEST_MODE: bool = field(
        default_factory=lambda: os.getenv('WEB3_BACKTEST_MODE', 'false').lower() == 'true'
    )


# Instance globale
web3_config = Web3Config()


def get_web3_config() -> Web3Config:
    """Retourne l'instance globale de configuration Web3."""
    return web3_config
