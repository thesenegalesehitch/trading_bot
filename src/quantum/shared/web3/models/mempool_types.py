"""
Types de données pour l'analyseur de Mempool.

Ce module définit les structures de données utilisées pour représenter
les transactions pendantes, les alertes baleines et les signaux générés.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class TransactionIntentType(Enum):
    """
    Types d'intentions de transaction détectées.
    
    INNOVATION:
    ===========
    Classification automatique des intentions basée sur l'analyse
    des calldata et des patterns de gas pricing.
    """
    SWAP = "swap"
    TRANSFER = "transfer"
    STAKE = "stake"
    UNSTAKE = "unstake"
    BRIDGE = "bridge"
    APPROVE = "approve"
    LIQUIDITY_ADD = "liquidity_add"
    LIQUIDITY_REMOVE = "liquidity_remove"
    NFT_MINT = "nft_mint"
    NFT_TRANSFER = "nft_transfer"
    UNKNOWN = "unknown"


class MempoolAction(Enum):
    """Actions détectées dans la mempool."""
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


@dataclass
class PendingTransaction:
    """
    Représente une transaction pendante dans la mempool.
    
    DESCRIPTION:
    ============
    Structure complète d'une transaction en attente de validation,
    enrichie avec des métadonnées d'analyse.
    
    INNOVATION:
    ===========
    Inclut un score de priorité basé sur le gas price et l'âge,
    permettant de prédire l'ordre de validation.
    
    RISQUE:
    =======
    Les transactions peuvent être remplacées (RBF) ou annulées
    à tout moment avant leur inclusion dans un bloc.
    """
    tx_hash: str
    chain: str
    from_address: str
    to_address: str
    value_wei: int
    value_eth: float
    gas_price_gwei: float
    gas_limit: int
    nonce: int
    input_data: str
    timestamp: datetime
    
    # Métadonnées enrichies
    value_usd: Optional[float] = None
    decoded_function: Optional[str] = None
    decoded_args: Optional[Dict[str, Any]] = None
    priority_score: float = 0.0  # Score de priorité de validation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'tx_hash': self.tx_hash,
            'chain': self.chain,
            'from_address': self.from_address,
            'to_address': self.to_address,
            'value_wei': self.value_wei,
            'value_eth': self.value_eth,
            'value_usd': self.value_usd,
            'gas_price_gwei': self.gas_price_gwei,
            'gas_limit': self.gas_limit,
            'nonce': self.nonce,
            'decoded_function': self.decoded_function,
            'priority_score': self.priority_score,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class TransactionIntent:
    """
    Représente l'intention décodée d'une transaction.
    
    DESCRIPTION:
    ============
    Analyse sémantique de ce que la transaction cherche à accomplir,
    au-delà des données brutes on-chain.
    
    INNOVATION:
    ===========
    Détection des patterns d'arbitrage et MEV pour identifier
    les bots et les distinguer des véritables mouvements de marché.
    
    RISQUE:
    =======
    Le décodage peut échouer sur des contrats non-vérifiés ou 
    des protocoles non-supportés.
    """
    intent_type: TransactionIntentType
    target_protocol: str
    tokens_involved: List[str]
    
    # Analyse avancée
    is_arbitrage: bool = False
    is_mev_bot: bool = False
    is_sandwich_attack: bool = False
    estimated_profit_usd: Optional[float] = None
    
    # Tokens spécifiques
    token_in: Optional[str] = None
    token_out: Optional[str] = None
    amount_in: Optional[float] = None
    amount_out: Optional[float] = None
    
    confidence: float = 0.0  # Confiance dans le décodage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'intent_type': self.intent_type.value,
            'target_protocol': self.target_protocol,
            'tokens_involved': self.tokens_involved,
            'is_arbitrage': self.is_arbitrage,
            'is_mev_bot': self.is_mev_bot,
            'is_sandwich_attack': self.is_sandwich_attack,
            'token_in': self.token_in,
            'token_out': self.token_out,
            'amount_in': self.amount_in,
            'amount_out': self.amount_out,
            'confidence': self.confidence,
        }


@dataclass
class WhaleAlert:
    """
    Alerte de transaction baleine détectée dans la mempool.
    
    DESCRIPTION:
    ============
    Signal généré quand une transaction significative (whale) est
    détectée dans la mempool AVANT sa validation dans un bloc.
    
    INNOVATION MARCHÉ:
    ==================
    Ce type d'alerte n'existe pas dans les outils grand public.
    Les plateformes comme Whale Alert ne signalent que les 
    transactions CONFIRMÉES. Notre système détecte les transactions
    10-30 secondes AVANT qu'elles ne soient publiques.
    
    AVANTAGE COMPÉTITIF:
    ====================
    - Anticipation de 10-30 secondes sur les CEX
    - Détection des patterns sandwich avant exécution
    - Identification des smart money moves en temps réel
    
    RISQUE ASSOCIÉ:
    ===============
    - La transaction peut être annulée (probability: ~5%)
    - Replaced-by-fee peut modifier le montant
    - Haute congestion réseau = délais imprévisibles
    - Les bots MEV peuvent front-runner notre signal
    
    MITIGATION:
    ===========
    - Le champ `confidence` quantifie la fiabilité (0-1)
    - Le champ `expires_at` invalide les signaux trop vieux
    - Mode ALERT_ONLY recommandé initialement
    """
    tx_hash: str
    chain: str
    whale_address: str
    action: MempoolAction
    
    # Token et montant
    token: str
    amount: float
    amount_usd: float
    
    # Impact estimé
    estimated_price_impact_percent: float
    
    # Fiabilité
    confidence: float  # 0-1
    
    # Timestamps
    timestamp: datetime
    expires_at: datetime  # Après ce timestamp, signal invalide
    
    # Métadonnées additionnelles
    target_dex: Optional[str] = None
    gas_price_gwei: Optional[float] = None
    is_smart_money: bool = False
    historical_success_rate: Optional[float] = None  # Si whale connue
    
    def is_valid(self) -> bool:
        """Vérifie si l'alerte est encore valide."""
        return datetime.utcnow() < self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'tx_hash': self.tx_hash,
            'chain': self.chain,
            'whale_address': self.whale_address,
            'action': self.action.value,
            'token': self.token,
            'amount': self.amount,
            'amount_usd': self.amount_usd,
            'estimated_price_impact_percent': self.estimated_price_impact_percent,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'is_valid': self.is_valid(),
            'target_dex': self.target_dex,
            'is_smart_money': self.is_smart_money,
        }


@dataclass
class MempoolSignal:
    """
    Signal agrégé basé sur l'analyse de la mempool.
    
    DESCRIPTION:
    ============
    Signal de haut niveau combinant plusieurs alertes individuelles
    pour générer une vue consolidée de la pression mémopool.
    
    INNOVATION:
    ===========
    Agrégation intelligente qui distingue le "bruit" (petites TX)
    du "signal" (mouvements coordonnés de baleines).
    
    USAGE:
    ======
    Ce signal peut être utilisé pour enrichir les décisions de trading
    existantes en ajoutant une couche de "mempool pressure".
    
    RISQUE:
    =======
    En cas de congestion extrême, la pression peut être surestimée.
    """
    signal_type: str  # 'BULLISH_PRESSURE', 'BEARISH_PRESSURE', 'NEUTRAL'
    
    # Métriques agrégées
    total_pending_volume_usd: float
    net_buy_pressure_usd: float  # Buy - Sell volume
    whale_count: int
    
    # Score final
    pressure_score: float  # -100 (bearish) to +100 (bullish)
    confidence: float
    
    # Signaux individuels (default factory)
    whale_alerts: List[WhaleAlert] = field(default_factory=list)
    
    # Timestamps
    timestamp: datetime = field(default_factory=datetime.utcnow)
    window_seconds: int = 60  # Fenêtre d'agrégation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'signal_type': self.signal_type,
            'total_pending_volume_usd': self.total_pending_volume_usd,
            'net_buy_pressure_usd': self.net_buy_pressure_usd,
            'whale_count': self.whale_count,
            'pressure_score': self.pressure_score,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'window_seconds': self.window_seconds,
            'whale_alerts': [alert.to_dict() for alert in self.whale_alerts],
        }
