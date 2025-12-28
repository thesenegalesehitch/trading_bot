"""
Module de simulation des coûts de trading réalistes.
Inclut slippage dynamique, spread, commissions et impact de marché.

Les coûts réalistes sont critiques pour un backtesting fiable.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class OrderSide(Enum):
    """Côté de l'ordre."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class TradingCosts:
    """Résultat du calcul des coûts."""
    spread_cost: float          # Coût du spread
    slippage_cost: float        # Coût du slippage
    commission_cost: float      # Commission broker
    market_impact: float        # Impact de marché (gros ordres)
    total_cost: float          # Coût total
    effective_price: float     # Prix effectif après tous les coûts
    cost_bps: float            # Coût en basis points


class TradingCostSimulator:
    """
    Simulateur de coûts de trading réalistes.
    
    Prend en compte:
    1. Spread bid/ask (variable selon liquidité)
    2. Slippage (variable selon volatilité)
    3. Commission broker
    4. Impact de marché (pour gros volumes)
    """
    
    # Spreads typiques par paire (en pips)
    DEFAULT_SPREADS = {
        # Forex Majeurs (très liquides)
        'EURUSD': 0.8, 'GBPUSD': 1.0, 'USDJPY': 0.9, 'USDCHF': 1.0,
        'AUDUSD': 1.2, 'USDCAD': 1.2, 'NZDUSD': 1.5,
        # Forex Mineurs
        'EURGBP': 1.2, 'EURJPY': 1.5, 'GBPJPY': 2.0, 'AUDJPY': 1.8,
        # Crypto (très variable)
        'BTCUSD': 50, 'ETHUSD': 5, 'XRPUSD': 0.01,
        # Métaux
        'XAUUSD': 30, 'XAGUSD': 5,
        # Default
        'DEFAULT': 2.0
    }
    
    # Valeur d'un pip par paire
    PIP_VALUES = {
        'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.01, 'USDCHF': 0.0001,
        'AUDUSD': 0.0001, 'USDCAD': 0.0001, 'NZDUSD': 0.0001,
        'BTCUSD': 1, 'ETHUSD': 0.1, 'XAUUSD': 0.1, 'XAGUSD': 0.01,
        'DEFAULT': 0.0001
    }
    
    def __init__(
        self,
        commission_per_lot: float = 7.0,  # USD par lot standard
        base_slippage_pct: float = 0.01,  # 0.01% de base
        spread_multiplier: float = 1.0,   # Multiplicateur du spread
        market_impact_threshold: float = 100000  # USD notionnel
    ):
        self.commission_per_lot = commission_per_lot
        self.base_slippage_pct = base_slippage_pct
        self.spread_multiplier = spread_multiplier
        self.market_impact_threshold = market_impact_threshold
    
    def calculate_costs(
        self,
        symbol: str,
        side: OrderSide,
        price: float,
        quantity: float,  # En unités (lot size dépend du symbole)
        volatility: Optional[float] = None,  # ATR ou vol récente
        avg_volume: Optional[float] = None,  # Volume moyen
        order_volume: Optional[float] = None  # Volume de l'ordre en notionnel
    ) -> TradingCosts:
        """
        Calcule tous les coûts de trading.
        
        Args:
            symbol: Symbole à trader
            side: BUY ou SELL
            price: Prix actuel
            quantity: Quantité à trader
            volatility: Volatilité récente (optionnel, pour slippage)
            avg_volume: Volume moyen (optionnel, pour market impact)
            order_volume: Volume de l'ordre en valeur notionnelle
        
        Returns:
            TradingCosts avec tous les coûts détaillés
        """
        notional_value = price * quantity
        
        # 1. Spread
        spread_cost = self._calculate_spread_cost(symbol, price, quantity)
        
        # 2. Slippage
        slippage_cost = self._calculate_slippage(
            symbol, price, quantity, volatility, side
        )
        
        # 3. Commission
        commission_cost = self._calculate_commission(symbol, notional_value)
        
        # 4. Market Impact
        market_impact = self._calculate_market_impact(
            notional_value, avg_volume
        )
        
        # Total
        total_cost = spread_cost + slippage_cost + commission_cost + market_impact
        
        # Prix effectif
        if side == OrderSide.BUY:
            effective_price = price + (total_cost / quantity)
        else:
            effective_price = price - (total_cost / quantity)
        
        # Coût en basis points
        cost_bps = (total_cost / notional_value) * 10000
        
        return TradingCosts(
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            commission_cost=commission_cost,
            market_impact=market_impact,
            total_cost=total_cost,
            effective_price=effective_price,
            cost_bps=cost_bps
        )
    
    def _calculate_spread_cost(
        self,
        symbol: str,
        price: float,
        quantity: float
    ) -> float:
        """Calcule le coût du spread."""
        # Obtenir le spread en pips
        spread_pips = self.DEFAULT_SPREADS.get(
            symbol.upper().replace('=X', '').replace('-USD', 'USD'),
            self.DEFAULT_SPREADS['DEFAULT']
        )
        spread_pips *= self.spread_multiplier
        
        # Valeur du pip
        pip_value = self.PIP_VALUES.get(
            symbol.upper().replace('=X', '').replace('-USD', 'USD'),
            self.PIP_VALUES['DEFAULT']
        )
        
        # Coût = spread * valeur pip * quantité
        spread_in_price = spread_pips * pip_value
        cost = spread_in_price * quantity / 2  # On paie la moitié du spread
        
        return cost
    
    def _calculate_slippage(
        self,
        symbol: str,
        price: float,
        quantity: float,
        volatility: Optional[float],
        side: OrderSide
    ) -> float:
        """
        Calcule le slippage basé sur la volatilité.
        
        Le slippage augmente avec:
        - La volatilité du marché
        - La taille de l'ordre
        """
        base_slippage = self.base_slippage_pct / 100
        
        # Ajustement volatilité
        if volatility:
            # Plus de volatilité = plus de slippage
            vol_factor = min(3.0, max(0.5, volatility / 0.01))  # Normalize autour de 1%
            base_slippage *= vol_factor
        
        # Slippage aléatoire (simulation)
        random_factor = 1 + np.random.uniform(-0.3, 0.3)
        slippage_pct = base_slippage * random_factor
        
        cost = price * quantity * slippage_pct
        
        return cost
    
    def _calculate_commission(
        self,
        symbol: str,
        notional_value: float
    ) -> float:
        """Calcule la commission broker."""
        # Standard lot = 100,000 pour forex
        lot_size = 100000
        
        # Nombre de lots
        lots = notional_value / lot_size
        
        # Commission
        commission = lots * self.commission_per_lot
        
        return commission
    
    def _calculate_market_impact(
        self,
        notional_value: float,
        avg_volume: Optional[float]
    ) -> float:
        """
        Calcule l'impact de marché pour les gros ordres.
        
        Les ordres représentant une part significative du volume
        moyen ont un impact sur le prix.
        """
        if notional_value <= self.market_impact_threshold:
            return 0.0
        
        # Impact quadratique (modèle simplifié)
        excess = notional_value - self.market_impact_threshold
        
        if avg_volume and avg_volume > 0:
            participation_rate = notional_value / avg_volume
            # Impact = racine carrée de la participation
            impact_pct = min(0.01, np.sqrt(participation_rate) * 0.001)
        else:
            # Impact par défaut basé sur le notionnel
            impact_pct = (excess / self.market_impact_threshold) * 0.001
        
        return notional_value * impact_pct


class DynamicSpreadModel:
    """
    Modèle de spread dynamique basé sur les conditions de marché.
    
    Le spread s'élargit pendant:
    - Les événements économiques
    - La haute volatilité
    - Les heures de faible liquidité
    """
    
    def __init__(self, base_spreads: Dict[str, float] = None):
        self.base_spreads = base_spreads or TradingCostSimulator.DEFAULT_SPREADS
    
    def get_dynamic_spread(
        self,
        symbol: str,
        hour_utc: int,
        volatility_percentile: float = 50,
        is_news_time: bool = False
    ) -> float:
        """
        Calcule le spread dynamique.
        
        Args:
            symbol: Symbole
            hour_utc: Heure UTC (0-23)
            volatility_percentile: Percentile de vol (0-100)
            is_news_time: Événement économique imminent
        
        Returns:
            Spread en pips
        """
        base = self.base_spreads.get(symbol, self.base_spreads['DEFAULT'])
        
        # Facteur session
        session_factor = self._get_session_factor(hour_utc)
        
        # Facteur volatilité
        vol_factor = 1 + (volatility_percentile - 50) / 100
        
        # Facteur news
        news_factor = 3.0 if is_news_time else 1.0
        
        return base * session_factor * vol_factor * news_factor
    
    def _get_session_factor(self, hour_utc: int) -> float:
        """Retourne le facteur de spread selon l'heure."""
        # Chevauchement Londres/NY: meilleure liquidité
        if 13 <= hour_utc <= 17:
            return 0.8
        # Session Londres
        elif 8 <= hour_utc <= 17:
            return 0.9
        # Session NY
        elif 13 <= hour_utc <= 22:
            return 0.95
        # Session Tokyo
        elif 0 <= hour_utc <= 9:
            return 1.1
        # Faible liquidité (fin NY avant Tokyo)
        else:
            return 1.5


class TradeExecutionSimulator:
    """
    Simulateur d'exécution de trades avec coûts réalistes.
    
    Combine tous les modèles de coûts pour une simulation complète.
    """
    
    def __init__(
        self,
        cost_simulator: TradingCostSimulator = None,
        spread_model: DynamicSpreadModel = None
    ):
        self.cost_simulator = cost_simulator or TradingCostSimulator()
        self.spread_model = spread_model or DynamicSpreadModel()
    
    def simulate_execution(
        self,
        symbol: str,
        side: OrderSide,
        price: float,
        quantity: float,
        df: Optional[pd.DataFrame] = None,
        timestamp: Optional[pd.Timestamp] = None
    ) -> Tuple[float, TradingCosts]:
        """
        Simule l'exécution d'un trade avec tous les coûts.
        
        Args:
            symbol: Symbole
            side: BUY ou SELL
            price: Prix demandé
            quantity: Quantité
            df: DataFrame OHLCV pour context (volatilité, volume)
            timestamp: Heure du trade
        
        Returns:
            (prix_effectif, détails_coûts)
        """
        # Extraire contexte du DataFrame si disponible
        volatility = None
        avg_volume = None
        
        if df is not None and len(df) >= 14:
            # ATR comme proxy de volatilité
            high = df['High'].values[-14:]
            low = df['Low'].values[-14:]
            close = df['Close'].values[-14:]
            
            tr = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1))
                )
            )[1:]
            volatility = np.mean(tr) / price
            
            # Volume moyen
            if 'Volume' in df.columns:
                avg_volume = df['Volume'].iloc[-20:].mean() * price
        
        # Calculer les coûts
        costs = self.cost_simulator.calculate_costs(
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            volatility=volatility,
            avg_volume=avg_volume
        )
        
        return costs.effective_price, costs
    
    def apply_to_backtest(
        self,
        trades: pd.DataFrame,
        df: pd.DataFrame,
        entry_col: str = 'entry_price',
        exit_col: str = 'exit_price',
        side_col: str = 'side',
        quantity_col: str = 'quantity'
    ) -> pd.DataFrame:
        """
        Applique les coûts à un DataFrame de trades de backtest.
        
        Returns:
            DataFrame avec colonnes de coûts ajoutées
        """
        trades = trades.copy()
        
        # Colonnes de coûts
        trades['entry_costs'] = 0.0
        trades['exit_costs'] = 0.0
        trades['total_costs'] = 0.0
        trades['effective_entry'] = trades[entry_col]
        trades['effective_exit'] = trades[exit_col]
        
        for idx, trade in trades.iterrows():
            # Coûts d'entrée
            entry_side = OrderSide.BUY if trade[side_col] == 'long' else OrderSide.SELL
            entry_costs = self.cost_simulator.calculate_costs(
                symbol=trade.get('symbol', 'DEFAULT'),
                side=entry_side,
                price=trade[entry_col],
                quantity=trade[quantity_col]
            )
            
            # Coûts de sortie (opposé)
            exit_side = OrderSide.SELL if trade[side_col] == 'long' else OrderSide.BUY
            exit_costs = self.cost_simulator.calculate_costs(
                symbol=trade.get('symbol', 'DEFAULT'),
                side=exit_side,
                price=trade[exit_col],
                quantity=trade[quantity_col]
            )
            
            trades.loc[idx, 'entry_costs'] = entry_costs.total_cost
            trades.loc[idx, 'exit_costs'] = exit_costs.total_cost
            trades.loc[idx, 'total_costs'] = entry_costs.total_cost + exit_costs.total_cost
            trades.loc[idx, 'effective_entry'] = entry_costs.effective_price
            trades.loc[idx, 'effective_exit'] = exit_costs.effective_price
        
        # Recalculer PnL avec coûts
        if 'pnl' in trades.columns:
            trades['pnl_after_costs'] = trades['pnl'] - trades['total_costs']
        
        return trades


if __name__ == "__main__":
    print("=" * 60)
    print("TEST TRADING COSTS SIMULATOR")
    print("=" * 60)
    
    simulator = TradingCostSimulator()
    
    # Test EUR/USD
    costs = simulator.calculate_costs(
        symbol='EURUSD',
        side=OrderSide.BUY,
        price=1.0850,
        quantity=100000,  # 1 lot standard
        volatility=0.0015
    )
    
    print("\n--- EUR/USD (1 lot) ---")
    print(f"Prix demandé: 1.0850")
    print(f"Spread: ${costs.spread_cost:.2f}")
    print(f"Slippage: ${costs.slippage_cost:.2f}")
    print(f"Commission: ${costs.commission_cost:.2f}")
    print(f"Market Impact: ${costs.market_impact:.2f}")
    print(f"TOTAL: ${costs.total_cost:.2f} ({costs.cost_bps:.1f} bps)")
    print(f"Prix effectif: {costs.effective_price:.5f}")
    
    # Test BTC/USD
    costs_btc = simulator.calculate_costs(
        symbol='BTCUSD',
        side=OrderSide.BUY,
        price=42000,
        quantity=1,  # 1 BTC
        volatility=0.03
    )
    
    print("\n--- BTC/USD (1 BTC) ---")
    print(f"Prix demandé: $42,000")
    print(f"TOTAL: ${costs_btc.total_cost:.2f} ({costs_btc.cost_bps:.1f} bps)")
    print(f"Prix effectif: ${costs_btc.effective_price:.2f}")
    
    # Test spread dynamique
    print("\n--- Spread Dynamique ---")
    spread_model = DynamicSpreadModel()
    
    for hour in [3, 10, 15, 21]:
        spread = spread_model.get_dynamic_spread('EURUSD', hour)
        print(f"  {hour:02d}:00 UTC: {spread:.2f} pips")
    
    print("\n  Pendant les news: ", end="")
    spread_news = spread_model.get_dynamic_spread('EURUSD', 15, is_news_time=True)
    print(f"{spread_news:.2f} pips")
