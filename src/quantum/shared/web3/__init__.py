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

from quantum.shared.web3.settings import Web3Config, web3_config

# Clients
from quantum.shared.web3.clients.quicknode_client import QuickNodeClient
from quantum.shared.web3.clients.websocket_manager import WebSocketManager

# Analyzers
from quantum.shared.web3.analyzers.mempool_analyzer import MempoolAnalyzer
from quantum.shared.web3.analyzers.cross_chain_oracle import CrossChainOracle
from quantum.shared.web3.analyzers.onchain_sentiment import OnChainSentimentAnalyzer

# Hooks
from quantum.shared.web3.hooks.signal_dispatcher import SignalDispatcher, IntegrationMode
from quantum.shared.web3.hooks.event_bus import EventBus

# Models
from quantum.shared.web3.models.mempool_types import WhaleAlert, TransactionIntent, MempoolSignal
from quantum.shared.web3.models.correlation_types import CrossChainIndex, BridgeFlow
from quantum.shared.web3.models.sentiment_types import StakingSentiment, SellPressureScore

# Main Engine
from quantum.shared.web3.engine import Web3IntelligenceEngine

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
