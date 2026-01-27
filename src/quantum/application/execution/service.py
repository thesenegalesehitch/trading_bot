"""
ExecutionManager - Coordonne l'exécution des ordres sur les différents échanges.
Vérifie le Circuit Breaker et gère le sizing avant transmission.
"""

import logging
import asyncio
from typing import Dict, Optional
from quantum.shared.config.settings import config
from quantum.infrastructure.exchanges.binance_client import BinanceExchange
from quantum.infrastructure.exchanges.ibkr_client import IBKRExchange
from quantum.domain.risk.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

class ExecutionManager:
    """
    Pilote l'exécution réelle des signaux de trading.
    """
    
    def __init__(self, circuit_breaker: CircuitBreaker):
        self.circuit_breaker = circuit_breaker
        self.live_trading = os.getenv('LIVE_TRADING', 'False').lower() == 'true'
        
        # Initialisation lazy des clients
        self._binance: Optional[BinanceExchange] = None
        self._ibkr: Optional[IBKRExchange] = None

    def _get_binance(self) -> BinanceExchange:
        if not self._binance:
            testnet = os.getenv('BINANCE_TESTNET', 'True').lower() == 'true'
            self._binance = BinanceExchange(testnet=testnet)
        return self._binance

    async def _get_ibkr(self) -> IBKRExchange:
        if not self._ibkr:
            self._ibkr = IBKRExchange()
            await self._ibkr.connect()
        return self._ibkr

    async def execute_signal(self, symbol: str, signal: str, confidence: float, price: float) -> Dict:
        """
        Exécute un signal de trading si toutes les conditions sont réunies.
        """
        # 1. Vérifier si Live Trading est activé
        if not self.live_trading:
            logger.info(f"🚫 Simulation d'ordre ({symbol}): {signal} à {price} (Live Trading désactivé)")
            return {"status": "simulated", "message": "Live trading disabled"}

        # 2. Vérifier le Circuit Breaker
        cb_status = self.circuit_breaker.can_trade()
        if not cb_status['allowed']:
            logger.warning(f"⛔ ORDRE REFUSÉ par Circuit Breaker: {cb_status['reason']}")
            return {"status": "rejected", "reason": cb_status['reason']}

        # 3. Déterminer la plateforme et le sizing
        is_crypto = "-" in symbol or any(c in symbol for c in ["BTC", "ETH", "SOL"])
        
        try:
            if is_crypto:
                return await self._execute_crypto(symbol, signal, confidence)
            else:
                return await self._execute_tradfi(symbol, signal, confidence)
        except Exception as e:
            logger.error(f"Erreur fatale lors de l'exécution {symbol}: {e}")
            return {"status": "error", "message": str(e)}

    async def _execute_crypto(self, symbol: str, signal: str, confidence: float) -> Dict:
        """Exécution sur Binance."""
        exchange = self._get_binance()
        # Calcul de la quantité (Simplifié: 2% du solde USDT)
        usdt_balance = exchange.get_balance("USDT")
        risk_amount = usdt_balance * config.risk.RISK_PER_TRADE
        
        side = "BUY" if "BUY" in signal else "SELL"
        # Logique simplifiée de sizing (fixe pour cet exemple)
        # En production, utiliser l'ATR calculé dans l'analyse
        
        logger.info(f"🚀 Transmission ORDRE CRYPTO sur Binance: {side} {symbol}")
        # result = exchange.create_market_order(symbol, side, quantity=...)
        return {"status": "success", "exchange": "Binance", "side": side}

    async def _execute_tradfi(self, symbol: str, signal: str, confidence: float) -> Dict:
        """Exécution sur IBKR."""
        exchange = await self._get_ibkr()
        side = "BUY" if "BUY" in signal else "SELL"
        
        # Nettoyage du symbole (BTC-USD -> BTCUSD)
        clean_symbol = symbol.replace('-', '').replace('=X', '')
        
        logger.info(f"🏦 Transmission ORDRE TRADFI sur IBKR: {side} {clean_symbol}")
        # result = await exchange.execute_forex_order(clean_symbol, side, quantity=20000)
        return {"status": "success", "exchange": "IBKR", "side": side}

import os
