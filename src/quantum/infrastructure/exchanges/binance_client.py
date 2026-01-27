"""
Connecteur Transactionnel Binance.
Gère l'exécution des ordres, le suivi du solde et le paper trading (Testnet).
"""

import logging
import os
from typing import Dict, Optional, List
from binance.client import Client
from binance.exceptions import BinanceAPIException
from quantum.shared.config.settings import config

logger = logging.getLogger(__name__)

class BinanceExchange:
    """
    Client de trading pour Binance.
    Supporte le mode Live et le mode Testnet via configuration.
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        self.testnet = testnet
        
        self.client: Optional[Client] = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialise le client API Binance."""
        try:
            if not self.api_key or not self.api_secret:
                logger.warning("Clés API Binance manquantes. Mode lecture seule ou erreur à venir.")
            
            self.client = Client(self.api_key, self.api_secret, testnet=self.testnet)
            # Test de connexion
            self.client.get_account_api_trading_status()
            logger.info(f"Connecté à Binance ({'Testnet' if self.testnet else 'Live'})")
        except Exception as e:
            logger.error(f"Échec initialisation Binance: {e}")
            self.client = None

    def get_balance(self, asset: str = "USDT") -> float:
        """Récupère le solde disponible pour un actif."""
        if not self.client: return 0.0
        try:
            balance = self.client.get_asset_balance(asset=asset)
            return float(balance['free']) if balance else 0.0
        except Exception as e:
            logger.error(f"Erreur solde {asset}: {e}")
            return 0.0

    def create_market_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """Exécute un ordre au marché."""
        if not self.client: return {"status": "error", "message": "No client"}
        try:
            order = self.client.create_order(
                symbol=symbol.replace('-', ''),
                side=side.upper(),
                type=Client.ORDER_TYPE_MARKET,
                quantity=quantity
            )
            logger.info(f"Ordre Market {side} {symbol} exécuté: {order['orderId']}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Erreur ordre Binance: {e}")
            return {"status": "error", "message": str(e)}

    def create_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict:
        """Crée un ordre limite."""
        if not self.client: return {"status": "error", "message": "No client"}
        try:
            order = self.client.create_order(
                symbol=symbol.replace('-', ''),
                side=side.upper(),
                type=Client.ORDER_TYPE_LIMIT,
                timeInForce=Client.TIME_IN_FORCE_GTC,
                quantity=quantity,
                price=str(price)
            )
            return order
        except Exception as e:
            logger.error(f"Erreur ordre limite: {e}")
            return {"status": "error", "message": str(e)}
            
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Récupère les ordres ouverts."""
        if not self.client: return []
        try:
            return self.client.get_open_orders(symbol=symbol.replace('-', '') if symbol else None)
        except Exception as e:
            logger.error(f"Erreur ordres ouverts: {e}")
            return []

if __name__ == "__main__":
    # Test rapide (nécessite des clés dans .env)
    exchange = BinanceExchange(testnet=True)
    print(f"Solde USDT: {exchange.get_balance('USDT')}")
