"""
Data classes et types pour le module Web3 Innovation.
"""

from quantum.shared.web3.models.mempool_types import (
    WhaleAlert,
    TransactionIntent,
    MempoolSignal,
    PendingTransaction,
)
from quantum.shared.web3.models.correlation_types import (
    CrossChainIndex,
    BridgeFlow,
    ChainVolume,
    CorrelationEvent,
)
from quantum.shared.web3.models.sentiment_types import (
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
