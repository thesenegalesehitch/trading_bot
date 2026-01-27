"""
Classificateur d'adresses blockchain.

Ce module classifie les adresses en différentes catégories
(whale, smart money, bot, exchange, etc.) pour enrichir l'analyse.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AddressType(Enum):
    """Types d'adresses blockchain."""
    
    UNKNOWN = "unknown"
    """Adresse non classifiée."""
    
    WHALE = "whale"
    """Baleine - détenteur de grandes quantités."""
    
    SMART_MONEY = "smart_money"
    """Smart Money - historique de trades profitables."""
    
    MEV_BOT = "mev_bot"
    """Bot MEV - arbitrage/sandwich."""
    
    MARKET_MAKER = "market_maker"
    """Market Maker - haute fréquence."""
    
    EXCHANGE = "exchange"
    """Exchange centralisé (hot wallet)."""
    
    DEFI_PROTOCOL = "defi_protocol"
    """Contrat de protocole DeFi."""
    
    BRIDGE = "bridge"
    """Contrat de bridge cross-chain."""
    
    NFT_TRADER = "nft_trader"
    """Trader NFT actif."""
    
    DORMANT_WHALE = "dormant_whale"
    """Baleine inactive depuis longtemps."""


@dataclass
class AddressProfile:
    """
    Profil d'une adresse blockchain.
    
    DESCRIPTION:
    ============
    Agrège les informations connues sur une adresse
    pour contextualiser ses transactions.
    """
    address: str
    address_type: AddressType
    
    # Labels
    label: Optional[str] = None  # "Binance", "Wintermute", etc.
    tags: List[str] = field(default_factory=list)
    
    # Métriques de patrimoine
    balance_eth: float = 0.0
    balance_usd: float = 0.0
    
    # Métriques d'activité
    total_transactions: int = 0
    transactions_30d: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    
    # Métriques de profitabilité (pour smart money)
    win_rate: Optional[float] = None  # % de trades profitables
    total_pnl_usd: Optional[float] = None
    avg_trade_size_usd: Optional[float] = None
    
    # Métriques MEV (pour bots)
    sandwich_count: int = 0
    arbitrage_count: int = 0
    liquidation_count: int = 0
    
    # Score de confiance dans la classification
    confidence: float = 0.0
    
    # Timestamps
    classified_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'address': self.address,
            'address_type': self.address_type.value,
            'label': self.label,
            'tags': self.tags,
            'balance_eth': self.balance_eth,
            'balance_usd': self.balance_usd,
            'total_transactions': self.total_transactions,
            'win_rate': self.win_rate,
            'total_pnl_usd': self.total_pnl_usd,
            'confidence': self.confidence,
        }


class AddressClassifier:
    """
    Classificateur d'adresses blockchain.
    
    DESCRIPTION:
    ============
    Classifie les adresses en catégories pour enrichir
    l'analyse des transactions et signaux.
    
    INNOVATION:
    ===========
    Combine plusieurs signaux (balance, activité, patterns)
    pour déterminer le type d'une adresse avec confiance.
    
    SOURCES DE DONNÉES:
    ===================
    - Base de données intégrée d'adresses connues
    - Analyse heuristique des patterns on-chain
    - Labels communautaires (Etherscan, Arkham)
    
    USAGE:
    ======
    ```python
    classifier = AddressClassifier()
    
    # Classifier une adresse
    profile = classifier.classify("0x...")
    print(f"Type: {profile.address_type}, Label: {profile.label}")
    
    # Vérifier si whale
    if classifier.is_whale("0x..."):
        print("C'est une baleine!")
    ```
    
    RISQUE:
    =======
    Les classifications peuvent être incorrectes pour
    les adresses nouvelles ou peu actives.
    """
    
    # Seuils de classification
    WHALE_THRESHOLD_ETH = 1000  # >= 1000 ETH
    MEGA_WHALE_THRESHOLD_ETH = 10000  # >= 10000 ETH
    
    def __init__(self):
        """
        Initialise le classificateur.
        
        Charge la base de données d'adresses connues.
        """
        # Base de données d'adresses connues
        self._known_addresses: Dict[str, AddressProfile] = {}
        
        # Cache des classifications
        self._classification_cache: Dict[str, AddressProfile] = {}
        self._cache_ttl_seconds = 3600  # 1 heure
        
        # Charger les adresses connues
        self._load_known_addresses()
        
        logger.info("AddressClassifier initialisé")
    
    def _load_known_addresses(self):
        """Charge les adresses connues."""
        
        # Exchanges centralisés (exemples)
        exchanges = {
            '0x28c6c06298d514db089934071355e5743bf21d60': ('Binance', AddressType.EXCHANGE),
            '0x21a31ee1afc51d94c2efccaa2092ad1028285549': ('Binance', AddressType.EXCHANGE),
            '0xdfd5293d8e347dfe59e90efd55b2956a1343963d': ('Binance', AddressType.EXCHANGE),
            '0x56eddb7aa87536c09ccc2793473599fd21a8b17f': ('Coinbase', AddressType.EXCHANGE),
            '0x503828976d22510aad0201ac7ec88293211d23da': ('Coinbase', AddressType.EXCHANGE),
            '0x71660c4005ba85c37ccec55d0c4493e66fe775d3': ('Coinbase', AddressType.EXCHANGE),
            '0x2faf487a4414fe77e2327f0bf4ae2a264a776ad2': ('FTX', AddressType.EXCHANGE),
            '0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0': ('Kraken', AddressType.EXCHANGE),
        }
        
        # Market makers connus
        market_makers = {
            '0x0000000000007f150bd6f54c40a34d7c3d5e9f56': ('Wintermute', AddressType.MARKET_MAKER),
            '0x00000000ae347930bd1e7b0f35588b92280f9e75': ('Wintermute', AddressType.MARKET_MAKER),
            '0x5041ed759dd4afc3a72b8192c143f72f4724081a': ('Jump Trading', AddressType.MARKET_MAKER),
        }
        
        # Bridges
        bridges = {
            '0x3ee18b2214aff97000d974cf647e7c347e8fa585': ('Wormhole', AddressType.BRIDGE),
            '0x99c9fc46f92e8a1c0dec1b1747d010903e884be1': ('Optimism Bridge', AddressType.BRIDGE),
            '0x4dbd4fc535ac27206064b68ffcf827b0a60bab3f': ('Arbitrum Bridge', AddressType.BRIDGE),
        }
        
        # MEV bots connus
        mev_bots = {
            '0x000000000000006f6502b7f2bbac8c30a3f67e9a': ('Flashbots', AddressType.MEV_BOT),
            '0x0000000000000000000000000000000000000001': ('MEV Bot', AddressType.MEV_BOT),
        }
        
        # Créer les profils
        for addr, (label, addr_type) in {**exchanges, **market_makers, **bridges, **mev_bots}.items():
            self._known_addresses[addr.lower()] = AddressProfile(
                address=addr.lower(),
                address_type=addr_type,
                label=label,
                confidence=1.0,
            )
    
    def classify(self, address: str, force: bool = False) -> AddressProfile:
        """
        Classifie une adresse.
        
        Args:
            address: Adresse à classifier
            force: Forcer la reclassification même si en cache
            
        Returns:
            Profil de l'adresse
        """
        address = address.lower()
        
        # Vérifier le cache
        if not force and address in self._classification_cache:
            cached = self._classification_cache[address]
            age = (datetime.utcnow() - cached.updated_at).total_seconds()
            if age < self._cache_ttl_seconds:
                return cached
        
        # Vérifier les adresses connues
        if address in self._known_addresses:
            profile = self._known_addresses[address]
            self._classification_cache[address] = profile
            return profile
        
        # Classification heuristique
        profile = self._heuristic_classification(address)
        self._classification_cache[address] = profile
        
        return profile
    
    def _heuristic_classification(self, address: str) -> AddressProfile:
        """
        Classification heuristique basée sur les patterns.
        
        Args:
            address: Adresse à classifier
            
        Returns:
            Profil estimé
        """
        # Pour le prototype, retourner UNKNOWN
        # En production, on analyserait:
        # - L'historique des transactions
        # - Le balance actuel
        # - Les patterns de timing
        # - Les protocoles interagis
        
        return AddressProfile(
            address=address,
            address_type=AddressType.UNKNOWN,
            confidence=0.0,
        )
    
    def is_whale(self, address: str) -> bool:
        """
        Vérifie si une adresse est une baleine.
        
        Args:
            address: Adresse à vérifier
            
        Returns:
            True si c'est une baleine
        """
        profile = self.classify(address)
        return profile.address_type in [
            AddressType.WHALE,
            AddressType.DORMANT_WHALE,
        ]
    
    def is_smart_money(self, address: str) -> bool:
        """
        Vérifie si une adresse est considérée "smart money".
        
        Args:
            address: Adresse à vérifier
            
        Returns:
            True si smart money
        """
        profile = self.classify(address)
        return profile.address_type == AddressType.SMART_MONEY
    
    def is_bot(self, address: str) -> bool:
        """
        Vérifie si une adresse est un bot.
        
        Args:
            address: Adresse à vérifier
            
        Returns:
            True si c'est un bot
        """
        profile = self.classify(address)
        return profile.address_type in [
            AddressType.MEV_BOT,
            AddressType.MARKET_MAKER,
        ]
    
    def is_exchange(self, address: str) -> bool:
        """
        Vérifie si une adresse appartient à un exchange.
        
        Args:
            address: Adresse à vérifier
            
        Returns:
            True si c'est un exchange
        """
        profile = self.classify(address)
        return profile.address_type == AddressType.EXCHANGE
    
    def add_known_address(
        self,
        address: str,
        address_type: AddressType,
        label: Optional[str] = None,
        **kwargs
    ):
        """
        Ajoute une adresse connue à la base.
        
        Args:
            address: Adresse à ajouter
            address_type: Type de l'adresse
            label: Label optionnel
            **kwargs: Attributs additionnels
        """
        address = address.lower()
        
        profile = AddressProfile(
            address=address,
            address_type=address_type,
            label=label,
            confidence=kwargs.get('confidence', 1.0),
            tags=kwargs.get('tags', []),
        )
        
        self._known_addresses[address] = profile
        
        # Invalider le cache
        if address in self._classification_cache:
            del self._classification_cache[address]
        
        logger.info(f"Adresse ajoutée: {address} ({address_type.value})")
    
    def get_known_whales(self) -> List[AddressProfile]:
        """
        Retourne toutes les baleines connues.
        
        Returns:
            Liste des profils de baleines
        """
        return [
            p for p in self._known_addresses.values()
            if p.address_type in [AddressType.WHALE, AddressType.DORMANT_WHALE, AddressType.SMART_MONEY]
        ]
    
    def get_known_bots(self) -> List[AddressProfile]:
        """
        Retourne tous les bots connus.
        
        Returns:
            Liste des profils de bots
        """
        return [
            p for p in self._known_addresses.values()
            if p.address_type == AddressType.MEV_BOT
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du classificateur.
        
        Returns:
            Dictionnaire des stats
        """
        type_counts = {}
        for profile in self._known_addresses.values():
            addr_type = profile.address_type.value
            type_counts[addr_type] = type_counts.get(addr_type, 0) + 1
        
        return {
            'known_addresses_count': len(self._known_addresses),
            'cache_size': len(self._classification_cache),
            'addresses_by_type': type_counts,
        }


# Instance globale
_address_classifier: Optional[AddressClassifier] = None


def get_address_classifier() -> AddressClassifier:
    """Retourne l'instance globale du classificateur."""
    global _address_classifier
    if _address_classifier is None:
        _address_classifier = AddressClassifier()
    return _address_classifier
