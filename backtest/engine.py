"""
Moteur de backtesting avec VectorBT.
Simule les performances historiques de la stratégie.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import config

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    print("⚠️ VectorBT non disponible, backtesting simplifié")


class BacktestEngine:
    """
    Moteur de backtesting pour évaluer les stratégies.
    
    Utilise VectorBT si disponible, sinon fallback sur une
    implémentation simplifiée.
    """
    
    def __init__(
        self,
        initial_capital: float = None,
        commission: float = 0.0001,  # 0.01% = 1 pip
        slippage: float = 0.0001
    ):
        self.initial_capital = initial_capital or config.risk.INITIAL_CAPITAL
        self.commission = commission
        self.slippage = slippage
        self.results: Optional[Dict] = None
    
    def run(
        self,
        df: pd.DataFrame,
        entries: pd.Series,
        exits: pd.Series,
        short_entries: pd.Series = None,
        short_exits: pd.Series = None
    ) -> Dict:
        """
        Exécute le backtest.
        
        Args:
            df: DataFrame avec prix
            entries: Série booléenne d'entrées long
            exits: Série booléenne de sorties long
            short_entries: Série booléenne d'entrées short
            short_exits: Série booléenne de sorties short
        
        Returns:
            Résultats du backtest
        """
        close = df['Close']
        
        if VBT_AVAILABLE:
            return self._run_vectorbt(close, entries, exits, short_entries, short_exits)
        else:
            return self._run_simple(close, entries, exits)
    
    def _run_vectorbt(
        self,
        close: pd.Series,
        entries: pd.Series,
        exits: pd.Series,
        short_entries: pd.Series = None,
        short_exits: pd.Series = None
    ) -> Dict:
        """Backtest avec VectorBT."""
        # Portfolio long only
        pf = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            fees=self.commission,
            slippage=self.slippage
        )
        
        self.results = {
            "total_return": pf.total_return() * 100,
            "sharpe_ratio": pf.sharpe_ratio(),
            "sortino_ratio": pf.sortino_ratio(),
            "max_drawdown": pf.max_drawdown() * 100,
            "win_rate": pf.trades.win_rate() * 100 if len(pf.trades.records) > 0 else 0,
            "total_trades": len(pf.trades.records),
            "profit_factor": pf.trades.profit_factor() if len(pf.trades.records) > 0 else 0,
            "final_value": pf.final_value(),
            "calmar_ratio": pf.calmar_ratio() if pf.max_drawdown() > 0 else 0,
            "trades": self._extract_trades(pf) if hasattr(pf, 'trades') else []
        }
        
        return self.results
    
    def _run_simple(
        self,
        close: pd.Series,
        entries: pd.Series,
        exits: pd.Series
    ) -> Dict:
        """Backtest simplifié sans VectorBT."""
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = [capital]
        peak = capital
        max_dd = 0
        
        for i in range(len(close)):
            if position == 0 and entries.iloc[i]:
                # Entrer en position
                position = capital / close.iloc[i]
                entry_price = close.iloc[i]
                capital = 0
            
            elif position > 0 and exits.iloc[i]:
                # Sortir de position
                exit_price = close.iloc[i]
                pnl = position * (exit_price - entry_price)
                pnl -= abs(position * exit_price * self.commission * 2)  # Aller-retour
                
                capital = position * exit_price
                is_win = pnl > 0
                
                trades.append({
                    "entry": entry_price,
                    "exit": exit_price,
                    "pnl": pnl,
                    "is_win": is_win
                })
                
                position = 0
                entry_price = 0
            
            # Equity
            if position > 0:
                current_equity = position * close.iloc[i]
            else:
                current_equity = capital
            
            equity_curve.append(current_equity)
            
            # Drawdown
            if current_equity > peak:
                peak = current_equity
            dd = (peak - current_equity) / peak
            if dd > max_dd:
                max_dd = dd
        
        # Clôturer position ouverte
        if position > 0:
            capital = position * close.iloc[-1]
        
        # Calcul des métriques
        total_return = (capital - self.initial_capital) / self.initial_capital * 100
        win_trades = [t for t in trades if t['is_win']]
        win_rate = len(win_trades) / len(trades) * 100 if trades else 0
        
        # Sharpe simplifié
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24) if returns.std() > 0 else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        self.results = {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sharpe * 0.8,  # Approximation
            "max_drawdown": max_dd * 100,
            "win_rate": win_rate,
            "total_trades": len(trades),
            "profit_factor": profit_factor,
            "final_value": capital,
            "trades": trades[:50]  # Premiers 50 trades
        }
        
        return self.results
    
    def _extract_trades(self, pf) -> List[Dict]:
        """Extrait les trades du portfolio VectorBT."""
        if not hasattr(pf, 'trades') or len(pf.trades.records) == 0:
            return []
        
        trades = []
        for record in pf.trades.records[:50]:  # Premiers 50
            trades.append({
                "entry_idx": int(record['entry_idx']),
                "exit_idx": int(record['exit_idx']),
                "pnl": float(record['pnl']),
                "return": float(record['return']) * 100
            })
        return trades
    
    def run_with_strategy(
        self,
        df: pd.DataFrame,
        strategy_func: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> Dict:
        """
        Exécute le backtest avec une fonction de stratégie.
        
        Args:
            df: DataFrame avec données OHLCV
            strategy_func: Fonction qui prend df et retourne df avec colonnes 'entry' et 'exit'
        
        Returns:
            Résultats du backtest
        """
        # Appliquer la stratégie
        signals_df = strategy_func(df)
        
        entries = signals_df['entry'] if 'entry' in signals_df else pd.Series(False, index=df.index)
        exits = signals_df['exit'] if 'exit' in signals_df else pd.Series(False, index=df.index)
        
        return self.run(df, entries, exits)
    
    def print_report(self) -> str:
        """Affiche un rapport formaté des résultats."""
        if self.results is None:
            return "Aucun backtest exécuté"
        
        r = self.results
        report = []
        report.append("═" * 50)
        report.append("         RAPPORT DE BACKTEST")
        report.append("═" * 50)
        report.append(f"Capital initial   : ${self.initial_capital:,.2f}")
        report.append(f"Valeur finale     : ${r['final_value']:,.2f}")
        report.append(f"Rendement total   : {r['total_return']:+.2f}%")
        report.append("─" * 50)
        report.append(f"Sharpe Ratio      : {r['sharpe_ratio']:.2f}")
        report.append(f"Sortino Ratio     : {r['sortino_ratio']:.2f}")
        report.append(f"Max Drawdown      : {r['max_drawdown']:.2f}%")
        report.append("─" * 50)
        report.append(f"Nombre de trades  : {r['total_trades']}")
        report.append(f"Win Rate          : {r['win_rate']:.1f}%")
        report.append(f"Profit Factor     : {r['profit_factor']:.2f}")
        report.append("═" * 50)
        
        final_report = "\n".join(report)
        print(final_report)
        return final_report
    
    def is_strategy_profitable(self, min_sharpe: float = 1.0, min_win_rate: float = 50) -> bool:
        """Vérifie si la stratégie est rentable."""
        if self.results is None:
            return False
        
        return (
            self.results['sharpe_ratio'] >= min_sharpe and
            self.results['win_rate'] >= min_win_rate and
            self.results['total_return'] > 0
        )


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    n = 1000
    
    # Données de test
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    dates = pd.date_range(start='2023-01-01', periods=n, freq='1H')
    
    df = pd.DataFrame({
        'Open': price + np.random.randn(n) * 0.1,
        'High': price + abs(np.random.randn(n)) * 0.2,
        'Low': price - abs(np.random.randn(n)) * 0.2,
        'Close': price,
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    # Signaux simples (basés sur MA crossover)
    sma_fast = df['Close'].rolling(10).mean()
    sma_slow = df['Close'].rolling(30).mean()
    
    entries = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
    exits = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))
    
    # Backtest
    engine = BacktestEngine(initial_capital=10000)
    results = engine.run(df, entries.fillna(False), exits.fillna(False))
    
    # Rapport
    engine.print_report()
    
    print(f"\nStratégie rentable: {engine.is_strategy_profitable()}")
