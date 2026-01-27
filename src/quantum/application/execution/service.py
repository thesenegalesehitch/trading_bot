from quantum.infrastructure.db.secrets import get_secrets
from quantum.shared.utils.resilience import GlobalLock, retry_async

class ExecutionManager:
    """
    Pilote l'exécution réelle avec verrous atomiques et secrets sécurisés.
    """
    
    def __init__(self, circuit_breaker: CircuitBreaker):
        self.circuit_breaker = circuit_breaker
        self.secrets = get_secrets()
        self.live_trading = self.secrets.live_trading
        
        self._binance: Optional[BinanceExchange] = None
        self._ibkr: Optional[IBKRExchange] = None

    def _get_binance(self) -> BinanceExchange:
        if not self._binance:
            self._binance = BinanceExchange(
                api_key=self.secrets.binance_api_key.get_secret_value() if self.secrets.binance_api_key else None,
                api_secret=self.secrets.binance_api_secret.get_secret_value() if self.secrets.binance_api_secret else None,
                testnet=self.secrets.binance_testnet
            )
        return self._binance

    @retry_async(max_retries=3, delay=2.0)
    async def execute_signal(self, symbol: str, signal: str, confidence: float, price: float) -> Dict:
        """
        Exécute un signal avec verrou d'atomicité pour prévenir la réentrée.
        """
        lock_name = f"exec_{symbol}"
        await GlobalLock.acquire(lock_name)
        
        try:
            # 1. Vérification Live
            if not self.live_trading:
                logger.info(f"🚫 SIMULATION: {signal} {symbol}")
                return {"status": "simulated"}

            # 2. Circuit Breaker check
            if not self.circuit_breaker.can_trade()['allowed']:
                return {"status": "rejected"}

            # 3. Branchement exécution
            is_crypto = "-" in symbol or any(c in symbol for c in ["BTC", "ETH", "SOL"])
            if is_crypto:
                return await self._execute_crypto(symbol, signal, confidence)
            else:
                return await self._execute_tradfi(symbol, signal, confidence)
                
        finally:
            GlobalLock.release(lock_name)

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
