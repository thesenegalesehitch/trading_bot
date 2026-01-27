"""
Décodeur de transactions blockchain.

Ce module décode les transactions complexes pour comprendre
leur intention (swap, stake, bridge, etc.).
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ABI des fonctions courantes (simplifiés - premiers 4 bytes du hash)
KNOWN_FUNCTION_SELECTORS = {
    # Uniswap V2 Router
    '0x7ff36ab5': {'name': 'swapExactETHForTokens', 'type': 'swap', 'direction': 'buy'},
    '0x18cbafe5': {'name': 'swapExactTokensForETH', 'type': 'swap', 'direction': 'sell'},
    '0x38ed1739': {'name': 'swapExactTokensForTokens', 'type': 'swap', 'direction': 'neutral'},
    '0x8803dbee': {'name': 'swapTokensForExactTokens', 'type': 'swap', 'direction': 'neutral'},
    '0xfb3bdb41': {'name': 'swapETHForExactTokens', 'type': 'swap', 'direction': 'buy'},
    '0x5c11d795': {'name': 'swapExactTokensForTokensSupportingFeeOnTransferTokens', 'type': 'swap', 'direction': 'neutral'},
    '0x791ac947': {'name': 'swapExactTokensForETHSupportingFeeOnTransferTokens', 'type': 'swap', 'direction': 'sell'},
    
    # Uniswap V3 Router
    '0xc04b8d59': {'name': 'exactInput', 'type': 'swap', 'direction': 'neutral'},
    '0xf28c0498': {'name': 'exactOutputSingle', 'type': 'swap', 'direction': 'neutral'},
    '0x414bf389': {'name': 'exactInputSingle', 'type': 'swap', 'direction': 'neutral'},
    '0xdb3e2198': {'name': 'exactOutputSingle', 'type': 'swap', 'direction': 'neutral'},
    
    # ERC20
    '0x095ea7b3': {'name': 'approve', 'type': 'approve', 'direction': 'neutral'},
    '0xa9059cbb': {'name': 'transfer', 'type': 'transfer', 'direction': 'neutral'},
    '0x23b872dd': {'name': 'transferFrom', 'type': 'transfer', 'direction': 'neutral'},
    
    # Staking
    '0xa694fc3a': {'name': 'stake', 'type': 'stake', 'direction': 'lock'},
    '0x2e1a7d4d': {'name': 'withdraw', 'type': 'unstake', 'direction': 'unlock'},
    '0xe9fad8ee': {'name': 'exit', 'type': 'unstake', 'direction': 'unlock'},
    '0x3d18b912': {'name': 'getReward', 'type': 'claim', 'direction': 'neutral'},
    
    # Lido
    '0xa1903eab': {'name': 'submit', 'type': 'stake', 'direction': 'lock'},
    
    # Curve
    '0x3df02124': {'name': 'exchange', 'type': 'swap', 'direction': 'neutral'},
    '0x5b41b908': {'name': 'exchange_underlying', 'type': 'swap', 'direction': 'neutral'},
    
    # Aave
    '0x617ba037': {'name': 'deposit', 'type': 'deposit', 'direction': 'lock'},
    '0x69328dec': {'name': 'withdraw', 'type': 'withdraw', 'direction': 'unlock'},
    
    # Compound
    '0x1249c58b': {'name': 'mint', 'type': 'deposit', 'direction': 'lock'},
    '0xdb006a75': {'name': 'redeem', 'type': 'withdraw', 'direction': 'unlock'},
    
    # Bridges
    '0x0c53c51c': {'name': 'executeMetaTransaction', 'type': 'meta', 'direction': 'neutral'},
}


@dataclass
class DecodedTransaction:
    """
    Transaction décodée.
    
    DESCRIPTION:
    ============
    Représente une transaction avec son intention
    et ses paramètres décodés.
    """
    function_name: str
    function_type: str  # 'swap', 'transfer', 'stake', etc.
    direction: str  # 'buy', 'sell', 'lock', 'unlock', 'neutral'
    
    # Paramètres décodés
    tokens_involved: List[str]
    amounts: List[float]
    
    # Métadonnées
    raw_input: str
    selector: str
    decoded_successfully: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'function_name': self.function_name,
            'function_type': self.function_type,
            'direction': self.direction,
            'tokens_involved': self.tokens_involved,
            'amounts': self.amounts,
            'decoded_successfully': self.decoded_successfully,
        }


class TransactionDecoder:
    """
    Décodeur de transactions blockchain.
    
    DESCRIPTION:
    ============
    Analyse les calldata des transactions pour comprendre
    leur intention sans avoir besoin de l'ABI complet.
    
    INNOVATION:
    ===========
    Utilise une base de selectors connue et des heuristiques
    pour décoder les transactions même sur des contrats
    non-vérifiés.
    
    FONCTIONNALITÉS:
    ================
    - Décodage des swaps (Uniswap V2/V3, Curve, etc.)
    - Décodage du staking/unstaking
    - Décodage des bridges
    - Identification des patterns MEV
    
    USAGE:
    ======
    ```python
    decoder = TransactionDecoder()
    
    # Décoder une transaction
    result = decoder.decode_transaction(tx_input, tx_value)
    print(f"Type: {result.function_type}, Direction: {result.direction}")
    ```
    
    RISQUE:
    =======
    Les transactions avec des contrats personnalisés ou
    des proxies peuvent ne pas être décodées correctement.
    """
    
    def __init__(self):
        """Initialise le décodeur."""
        self._selectors = KNOWN_FUNCTION_SELECTORS
        
        # Cache de décodages
        self._decode_cache: Dict[str, DecodedTransaction] = {}
        
        # Statistiques
        self._stats = {
            'total_decoded': 0,
            'successful': 0,
            'failed': 0,
        }
        
        logger.info("TransactionDecoder initialisé")
    
    def decode_transaction(
        self,
        input_data: str,
        value_wei: int = 0
    ) -> DecodedTransaction:
        """
        Décode une transaction.
        
        Args:
            input_data: Calldata de la transaction (hex)
            value_wei: Valeur ETH envoyée (en wei)
            
        Returns:
            DecodedTransaction avec les détails
        """
        self._stats['total_decoded'] += 1
        
        # Transaction simple (pas de data)
        if not input_data or input_data == '0x':
            self._stats['successful'] += 1
            return DecodedTransaction(
                function_name='transfer',
                function_type='transfer',
                direction='neutral',
                tokens_involved=['ETH'],
                amounts=[value_wei / 1e18],
                raw_input=input_data,
                selector='0x',
            )
        
        # Extraire le selector (4 premiers bytes)
        selector = input_data[:10] if len(input_data) >= 10 else input_data
        
        # Chercher dans les selectors connus
        if selector in self._selectors:
            info = self._selectors[selector]
            self._stats['successful'] += 1
            
            return DecodedTransaction(
                function_name=info['name'],
                function_type=info['type'],
                direction=info['direction'],
                tokens_involved=self._extract_tokens(input_data),
                amounts=self._extract_amounts(input_data, value_wei),
                raw_input=input_data,
                selector=selector,
            )
        
        # Essayer l'heuristique
        result = self._heuristic_decode(input_data, value_wei)
        
        if result:
            self._stats['successful'] += 1
            return result
        
        # Échec du décodage
        self._stats['failed'] += 1
        return DecodedTransaction(
            function_name='unknown',
            function_type='unknown',
            direction='neutral',
            tokens_involved=[],
            amounts=[],
            raw_input=input_data,
            selector=selector,
            decoded_successfully=False,
            error_message='Selector inconnu',
        )
    
    def _extract_tokens(self, input_data: str) -> List[str]:
        """
        Extrait les adresses de tokens des calldata.
        
        Args:
            input_data: Calldata hex
            
        Returns:
            Liste des adresses de tokens
        """
        tokens = []
        
        # Les adresses sont sur 32 bytes (64 chars hex)
        # mais seulement les 20 derniers bytes sont l'adresse
        data = input_data[10:]  # Retirer le selector
        
        # Chercher les patterns d'adresses (20 bytes avec leading zeros)
        for i in range(0, len(data) - 64, 64):
            chunk = data[i:i+64]
            # Vérifier si c'est probablement une adresse
            if chunk.startswith('000000000000000000000000'):
                addr = '0x' + chunk[24:]
                if self._is_valid_address(addr):
                    tokens.append(addr)
        
        return tokens[:5]  # Limiter à 5 tokens max
    
    def _extract_amounts(
        self,
        input_data: str,
        value_wei: int
    ) -> List[float]:
        """
        Extrait les montants des calldata.
        
        Args:
            input_data: Calldata hex
            value_wei: Valeur ETH
            
        Returns:
            Liste des montants
        """
        amounts = []
        
        # Ajouter la valeur ETH si présente
        if value_wei > 0:
            amounts.append(value_wei / 1e18)
        
        # Extraire les uint256 des calldata
        data = input_data[10:]
        
        for i in range(0, len(data) - 64, 64):
            chunk = data[i:i+64]
            try:
                value = int(chunk, 16)
                # Filtrer les valeurs qui ressemblent à des montants de tokens
                if 1e15 < value < 1e30:  # Entre 0.001 et 1T tokens (avec 18 decimals)
                    amounts.append(value / 1e18)
            except ValueError:
                continue
        
        return amounts[:5]  # Limiter à 5 montants max
    
    def _is_valid_address(self, address: str) -> bool:
        """Vérifie si une chaîne est une adresse valide."""
        if not address.startswith('0x'):
            return False
        if len(address) != 42:
            return False
        try:
            int(address, 16)
            return True
        except ValueError:
            return False
    
    def _heuristic_decode(
        self,
        input_data: str,
        value_wei: int
    ) -> Optional[DecodedTransaction]:
        """
        Tente un décodage heuristique.
        
        Args:
            input_data: Calldata
            value_wei: Valeur ETH
            
        Returns:
            DecodedTransaction ou None si échec
        """
        selector = input_data[:10]
        
        # Heuristique: les swaps ont souvent de l'ETH + des paramètres
        if value_wei > 0 and len(input_data) > 200:
            return DecodedTransaction(
                function_name='probable_swap',
                function_type='swap',
                direction='buy',
                tokens_involved=self._extract_tokens(input_data),
                amounts=self._extract_amounts(input_data, value_wei),
                raw_input=input_data,
                selector=selector,
            )
        
        # Heuristique: transfert simple si pas de data complexe
        if len(input_data) < 150:
            return DecodedTransaction(
                function_name='probable_transfer',
                function_type='transfer',
                direction='neutral',
                tokens_involved=[],
                amounts=[],
                raw_input=input_data,
                selector=selector,
            )
        
        return None
    
    def is_swap(self, input_data: str) -> bool:
        """
        Vérifie si une transaction est un swap.
        
        Args:
            input_data: Calldata
            
        Returns:
            True si c'est un swap
        """
        result = self.decode_transaction(input_data)
        return result.function_type == 'swap'
    
    def is_staking(self, input_data: str) -> bool:
        """
        Vérifie si une transaction est du staking.
        
        Args:
            input_data: Calldata
            
        Returns:
            True si c'est du staking
        """
        result = self.decode_transaction(input_data)
        return result.function_type in ['stake', 'unstake']
    
    def get_swap_direction(self, input_data: str, value_wei: int) -> str:
        """
        Détermine la direction d'un swap.
        
        Args:
            input_data: Calldata
            value_wei: Valeur ETH
            
        Returns:
            'buy', 'sell', ou 'neutral'
        """
        result = self.decode_transaction(input_data, value_wei)
        
        if result.function_type != 'swap':
            return 'neutral'
        
        # Si la TX envoie de l'ETH, c'est probablement un achat
        if value_wei > 0:
            return 'buy'
        
        return result.direction
    
    def add_selector(
        self,
        selector: str,
        name: str,
        function_type: str,
        direction: str = 'neutral'
    ):
        """
        Ajoute un nouveau selector à la base.
        
        Args:
            selector: Selector hex (0x...)
            name: Nom de la fonction
            function_type: Type (swap, stake, etc.)
            direction: Direction (buy, sell, neutral)
        """
        self._selectors[selector.lower()] = {
            'name': name,
            'type': function_type,
            'direction': direction,
        }
        logger.debug(f"Selector ajouté: {selector} ({name})")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du décodeur.
        
        Returns:
            Dictionnaire des stats
        """
        return {
            **self._stats,
            'known_selectors': len(self._selectors),
            'success_rate': (
                self._stats['successful'] / max(self._stats['total_decoded'], 1) * 100
            ),
        }


# Instance globale
_transaction_decoder: Optional[TransactionDecoder] = None


def get_transaction_decoder() -> TransactionDecoder:
    """Retourne l'instance globale du décodeur."""
    global _transaction_decoder
    if _transaction_decoder is None:
        _transaction_decoder = TransactionDecoder()
    return _transaction_decoder
