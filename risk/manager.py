"""
Gestionnaire de risque pour le calcul des positions et stops.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import config


@dataclass
class TradeSetup:
    """Configuration complète d'un trade."""
    symbol: str
    direction: str  # "BUY" ou "SELL"
    entry_price: float
    stop_loss: float
    take_profits: List[Dict]  # [{price, size_percent}]
    position_size: float
    risk_amount: float
    risk_reward_ratio: float


class RiskManager:
    """
    Gère le risque par trade et le sizing des positions.
    
    Règles:
    - Maximum 1% du capital par trade
    - Stop-Loss basé sur ATR
    - 3 niveaux de Take-Profit
    """
    
    def __init__(
        self,
        capital: float = None,
        risk_per_trade: float = None,
        atr_multiplier: float = None
    ):
        self.capital = capital or config.risk.INITIAL_CAPITAL
        self.risk_per_trade = risk_per_trade or config.risk.RISK_PER_TRADE
        self.atr_multiplier = atr_multiplier or config.risk.ATR_MULTIPLIER
        self.atr_period = config.risk.ATR_PERIOD
        self.tp_levels = config.risk.TP_LEVELS
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        capital: float = None
    ) -> float:
        """
        Calcule la taille de position basée sur le risque.
        
        Formula: Position = (Capital * Risk%) / |Entry - SL|
        """
        capital = capital or self.capital
        risk_amount = capital * self.risk_per_trade
        
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return 0
        
        position_size = risk_amount / stop_distance
        
        return position_size
    
    def calculate_stop_loss(
        self,
        df: pd.DataFrame,
        entry_price: float,
        direction: str
    ) -> float:
        """
        Calcule le Stop-Loss basé sur l'ATR.
        
        Args:
            df: DataFrame OHLCV
            entry_price: Prix d'entrée
            direction: "BUY" ou "SELL"
        
        Returns:
            Niveau de Stop-Loss
        """
        atr = self._calculate_atr(df)
        
        if atr is None or atr == 0:
            # Fallback: 1% du prix
            atr = entry_price * 0.01
        
        stop_distance = atr * self.atr_multiplier
        
        if direction == "BUY":
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        return round(stop_loss, 5)
    
    def calculate_take_profits(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str
    ) -> List[Dict]:
        """
        Calcule les niveaux de Take-Profit.
        
        Utilise des ratios Risk:Reward configurés.
        """
        risk = abs(entry_price - stop_loss)
        take_profits = []
        
        for tp_config in self.tp_levels:
            ratio = tp_config['ratio']
            size_pct = tp_config['size_percent']
            
            if direction == "BUY":
                tp_price = entry_price + (risk * ratio)
            else:
                tp_price = entry_price - (risk * ratio)
            
            take_profits.append({
                'price': round(tp_price, 5),
                'size_percent': size_pct,
                'ratio': f"1:{ratio}"
            })
        
        return take_profits
    
    def create_trade_setup(
        self,
        df: pd.DataFrame,
        symbol: str,
        direction: str,
        entry_price: float = None
    ) -> TradeSetup:
        """
        Crée un setup de trade complet.
        
        Args:
            df: DataFrame OHLCV
            symbol: Symbole tradé
            direction: "BUY" ou "SELL"
            entry_price: Prix d'entrée (défaut: dernier close)
        
        Returns:
            TradeSetup avec tous les paramètres
        """
        entry_price = entry_price or df['Close'].iloc[-1]
        
        # Stop-Loss
        stop_loss = self.calculate_stop_loss(df, entry_price, direction)
        
        # Take-Profits
        take_profits = self.calculate_take_profits(entry_price, stop_loss, direction)
        
        # Position size
        position_size = self.calculate_position_size(entry_price, stop_loss)
        
        # Risk amount
        risk_amount = self.capital * self.risk_per_trade
        
        # Risk/Reward (basé sur TP1)
        if take_profits:
            reward = abs(take_profits[0]['price'] - entry_price)
            risk = abs(entry_price - stop_loss)
            rr_ratio = reward / risk if risk > 0 else 0
        else:
            rr_ratio = 0
        
        return TradeSetup(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profits=take_profits,
            position_size=position_size,
            risk_amount=risk_amount,
            risk_reward_ratio=rr_ratio
        )
    
    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calcule l'ATR."""
        if len(df) < self.atr_period:
            return None
        
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean().iloc[-1]
        
        return atr
    
    def update_capital(self, pnl: float):
        """Met à jour le capital après un trade."""
        self.capital += pnl
    
    def get_max_position_value(self) -> float:
        """Valeur maximale d'une position."""
        return self.capital * 0.1  # Max 10% du capital


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    n = 50
    
    df = pd.DataFrame({
        'Open': 1.08 + np.random.randn(n) * 0.001,
        'High': 1.08 + abs(np.random.randn(n)) * 0.002,
        'Low': 1.08 - abs(np.random.randn(n)) * 0.002,
        'Close': 1.08 + np.random.randn(n) * 0.001,
        'Volume': np.random.randint(1000, 10000, n)
    })
    
    rm = RiskManager(capital=10000)
    setup = rm.create_trade_setup(df, "EUR/USD", "BUY")
    
    print("=== Trade Setup ===")
    print(f"Symbol: {setup.symbol}")
    print(f"Direction: {setup.direction}")
    print(f"Entry: {setup.entry_price:.5f}")
    print(f"Stop-Loss: {setup.stop_loss:.5f}")
    print(f"Position Size: {setup.position_size:.2f}")
    print(f"Risk Amount: ${setup.risk_amount:.2f}")
    print(f"\nTake-Profits:")
    for tp in setup.take_profits:
        print(f"  {tp['ratio']}: {tp['price']:.5f} ({tp['size_percent']}%)")
