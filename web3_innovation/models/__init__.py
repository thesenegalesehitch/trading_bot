"""
Data classes et types pour le module Web3 Innovation.
"""

from web3_innovation.models.mempool_types import (
    WhaleAlert,
    TransactionIntent,
    MempoolSignal,
    PendingTransaction,
)
from web3_innovation.models.correlation_types import (
    CrossChainIndex,
    BridgeFlow,
    ChainVolume,
    CorrelationEvent,
)
from web3_innovation.models.sentiment_types import (
    StakingSentiment,
    SellPressureScore,
    WhaleStakingAction,
    StakingMetrics,
)

__all__ = [
    # Mempool
    "WhaleAlert",
    "TransactionIntent",
    "MempoolSignal",
    "PendingTransaction",
    # Correlation
    "CrossChainIndex",
    "BridgeFlow",
    "ChainVolume",
    "CorrelationEvent",
    # Sentiment
    "StakingSentiment",
    "SellPressureScore",
    "WhaleStakingAction",
    "StakingMetrics",
]
