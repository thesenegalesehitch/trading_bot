"""
Connecteur Transactionnel Interactive Brokers (IBKR).
Utilise ib_insync pour une interface asynchrone simplifiée avec TWS/Gateway.
"""

import logging
import asyncio
from typing import Optional, List, Dict
from ib_insync import IB, Forex, MarketOrder, LimitOrder, Contract
from quantum.shared.config.settings import config

logger = logging.getLogger(__name__)

class IBKRExchange:
    """
    Client de trading pour Interactive Brokers.
    Nécessite TWS ou IB Gateway ouvert sur la machine ou le réseau.
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id
        
        self.ib = IB()
        self._connected = False

    async def connect(self) -> bool:
        """Établit la connexion asynchrone avec IBKR."""
        try:
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            self._connected = True
            logger.info(f"Connecté à IBKR ({self.host}:{self.port})")
            return True
        except Exception as e:
            logger.error(f"Échec connexion IBKR: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Ferme la connexion."""
        if self.ib.isConnected():
            self.ib.disconnect()
            self._connected = False

    async def get_balance(self) -> float:
        """Récupère la valeur nette de liquidation du compte (Total Cash)."""
        if not self._connected: return 0.0
        try:
            summary = await self.ib.accountSummaryAsync()
            for item in summary:
                if item.tag == 'NetLiquidation':
                    return float(item.value)
            return 0.0
        except Exception as e:
            logger.error(f"Erreur solde IBKR: {e}")
            return 0.0

    async def execute_forex_order(self, symbol: str, side: str, quantity: float, order_type: str = "MKT", price: float = None) -> Dict:
        """
        Exécute un ordre sur le Forex.
        Ex: symbol="EURUSD", side="BUY", quantity=20000
        """
        if not self._connected: return {"status": "error", "message": "Not connected"}
        
        try:
            # Créer le contrat
            curr1, curr2 = symbol[:3], symbol[3:]
            contract = Forex(f"{curr1}{curr2}")
            await self.ib.qualifyContractsAsync(contract)
            
            # Créer l'ordre
            if order_type.upper() == "MKT":
                order = MarketOrder(side.upper(), quantity)
            else:
                order = LimitOrder(side.upper(), quantity, price)
                
            trade = self.ib.placeOrder(contract, order)
            
            # Attendre un peu pour confirmation de transmission
            await asyncio.sleep(1)
            
            logger.info(f"Ordre IBKR {side} {symbol} soumis: {trade.order.orderId}")
            return {
                "status": "submitted",
                "order_id": trade.order.orderId,
                "contract": str(contract),
                "trade_state": trade.orderStatus.status
            }
        except Exception as e:
            logger.error(f"Erreur ordre IBKR: {e}")
            return {"status": "error", "message": str(e)}

    def get_positions(self) -> List[Dict]:
        """Récupère les positions actuelles du portefeuille."""
        if not self._connected: return []
        positions = []
        for pos in self.ib.positions():
            positions.append({
                'contract': str(pos.contract),
                'amount': pos.position,
                'avg_cost': pos.avgCost
            })
        return positions

async def test_ibkr():
    exchange = IBKRExchange()
    if await exchange.connect():
        print(f"Solde Net: {await exchange.get_balance()}$")
        exchange.disconnect()

if __name__ == "__main__":
    asyncio.run(test_ibkr())
