"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    WEB3 INNOVATION - PREDICTIVE ON-CHAIN INTELLIGENCE        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Module de trading intelligence basé sur l'analyse blockchain en temps réel  ║
║                                                                              ║
║  🎯 Mempool Analysis: Détection des whale transactions avant validation      ║
║  🔗 Cross-Chain Oracle: Corrélation ETH ↔ SOL en temps réel                 ║
║  📊 On-Chain Sentiment: Prédiction via patterns de staking                   ║
║                                                                              ║
║  Conception: Alexandre Albert Ndour                                          ║
║  Date: Janvier 2026                                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Signature: 416c6578616e647265_416c62657274_4e646f7572

from web3_innovation.settings import Web3Config, web3_config

# Clients
from web3_innovation.clients.quicknode_client import QuickNodeClient
from web3_innovation.clients.websocket_manager import WebSocketManager

# Analyzers
from web3_innovation.analyzers.mempool_analyzer import MempoolAnalyzer
from web3_innovation.analyzers.cross_chain_oracle import CrossChainOracle
from web3_innovation.analyzers.onchain_sentiment import OnChainSentimentAnalyzer

# Hooks
from web3_innovation.hooks.signal_dispatcher import SignalDispatcher, IntegrationMode
from web3_innovation.hooks.event_bus import EventBus

# Models
from web3_innovation.models.mempool_types import WhaleAlert, TransactionIntent, MempoolSignal
from web3_innovation.models.correlation_types import CrossChainIndex, BridgeFlow
from web3_innovation.models.sentiment_types import StakingSentiment, SellPressureScore

# Main Engine
from web3_innovation.engine import Web3IntelligenceEngine

__version__ = "1.0.0"
__author__ = "Alexandre Albert Ndour"

__all__ = [
    # Config
    "Web3Config",
    "web3_config",
    # Clients
    "QuickNodeClient",
    "WebSocketManager",
    # Analyzers
    "MempoolAnalyzer",
    "CrossChainOracle",
    "OnChainSentimentAnalyzer",
    # Hooks
    "SignalDispatcher",
    "IntegrationMode",
    "EventBus",
    # Models
    "WhaleAlert",
    "TransactionIntent",
    "MempoolSignal",
    "CrossChainIndex",
    "BridgeFlow",
    "StakingSentiment",
    "SellPressureScore",
    # Engine
    "Web3IntelligenceEngine",
]
