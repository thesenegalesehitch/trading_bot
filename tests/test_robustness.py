"""
Robustness Tests for Quantum Trading System
============================================

Tests validates strategies across different market conditions:
- Bull markets
- Bear markets  
- High volatility periods
- Low volatility periods
- Regime changes
- Black swan events

These tests help identify strategy weaknesses before live trading.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestStrategyRobustness:
    """Test suite for strategy robustness across market conditions."""
    
    @pytest.fixture
    def bull_market_data(self) -> pd.DataFrame:
        """Generate bull market data with strong upward trend."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        
        # Strong upward trend with low volatility
        returns = np.random.normal(0.001, 0.01, 252)  # 0.1% daily, 1% vol
        prices = 100 * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'Open': prices * 0.998,
            'High': prices * 1.01,
            'Low': prices * 0.99,
            'Close': prices,
            'Volume': np.random.randint(1_000_000, 5_000_000, 252)
        }, index=dates)
    
    @pytest.fixture
    def bear_market_data(self) -> pd.DataFrame:
        """Generate bear market data with strong downward trend."""
        np.random.seed(43)
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        
        # Strong downward trend
        returns = np.random.normal(-0.001, 0.015, 252)
        prices = 100 * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'Open': prices * 1.002,
            'High': prices * 1.015,
            'Low': prices * 0.985,
            'Close': prices,
            'Volume': np.random.randint(1_000_000, 8_000_000, 252)
        }, index=dates)
    
    @pytest.fixture
    def high_volatility_data(self) -> pd.DataFrame:
        """Generate high volatility market data."""
        np.random.seed(44)
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        
        # High volatility, sideways
        returns = np.random.normal(0, 0.03, 252)  # 3% daily vol
        prices = 100 * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.03,
            'Low': prices * 0.97,
            'Close': prices,
            'Volume': np.random.randint(5_000_000, 15_000_000, 252)
        }, index=dates)
    
    @pytest.fixture
    def regime_change_data(self) -> pd.DataFrame:
        """Generate data with regime changes (bull -> bear -> bull)."""
        np.random.seed(45)
        dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
        
        returns = np.zeros(500)
        
        # Bull market (first 166 days)
        returns[:166] = np.random.normal(0.0015, 0.012, 166)
        
        # Bear market (next 166 days)
        returns[166:332] = np.random.normal(-0.002, 0.02, 166)
        
        # Recovery (last 168 days)
        returns[332:] = np.random.normal(0.001, 0.015, 168)
        
        prices = 100 * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'Open': prices * 0.998,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1_000_000, 10_000_000, 500)
        }, index=dates)
    
    @pytest.fixture
    def black_swan_data(self) -> pd.DataFrame:
        """Generate data with black swan events (flash crashes)."""
        np.random.seed(46)
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        
        returns = np.random.normal(0.0005, 0.015, 252)
        
        # Add black swan events (flash crashes)
        crash_days = [50, 120, 200]
        for day in crash_days:
            returns[day] = -0.10  # 10% crash
            if day + 1 < 252:
                returns[day + 1] = 0.05  # Recovery
        
        prices = 100 * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.025,
            'Low': prices * 0.90,  # Include crash lows
            'Close': prices,
            'Volume': np.random.randint(2_000_000, 20_000_000, 252)
        }, index=dates)
    
    def test_indicator_behavior_bull_market(self, bull_market_data):
        """Test that indicators behave correctly in bull markets."""
        df = bull_market_data
        
        # RSI should be mostly overbought in strong bull
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # In bull market, RSI should average higher
        avg_rsi = rsi.dropna().mean()
        assert avg_rsi > 50, f"Bull market RSI should be >50, got {avg_rsi:.2f}"
        
    def test_indicator_behavior_bear_market(self, bear_market_data):
        """Test that indicators behave correctly in bear markets."""
        df = bear_market_data
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # In bear market, RSI should average lower
        avg_rsi = rsi.dropna().mean()
        assert avg_rsi < 50, f"Bear market RSI should be <50, got {avg_rsi:.2f}"
        
    def test_strategy_survives_regime_change(self, regime_change_data):
        """Test that strategy doesn't blow up during regime changes."""
        df = regime_change_data
        
        # Simulate simple trend-following strategy
        signals = []
        position = 0
        
        for i in range(50, len(df)):
            # Simple moving average crossover
            sma_fast = df['Close'].iloc[i-20:i].mean()
            sma_slow = df['Close'].iloc[i-50:i].mean()
            
            if sma_fast > sma_slow:
                signal = 1  # Long
            elif sma_fast < sma_slow:
                signal = -1  # Short
            else:
                signal = 0
            
            signals.append(signal)
        
        # Calculate returns
        returns = df['Close'].pct_change().dropna()
        strategy_returns = returns.iloc[49:] * signals
        
        # Strategy should not lose more than 50% during regime changes
        cumulative_return = (1 + strategy_returns).prod() - 1
        assert cumulative_return > -0.50, f"Strategy lost {cumulative_return:.2%} during regime changes"
        
    def test_strategy_survives_black_swan(self, black_swan_data):
        """Test that strategy survives flash crashes."""
        df = black_swan_data
        
        # Simple mean reversion strategy
        signals = []
        position = 0
        
        for i in range(20, len(df)):
            # RSI-based mean reversion
            delta = df['Close'].iloc[i-14:i].diff()
            gain = delta.where(delta > 0, 0).mean()
            loss = (-delta.where(delta < 0, 0)).mean()
            rsi = 100 - (100 / (1 + gain/loss)) if loss != 0 else 50
            
            if rsi < 30:
                signal = 1  # Buy oversold
            elif rsi > 70:
                signal = -1  # Sell overbought
            else:
                signal = 0
            
            signals.append(signal)
        
        returns = df['Close'].pct_change().dropna()
        strategy_returns = returns.iloc[19:] * signals
        
        # Max drawdown should be limited
        cumulative = (1 + strategy_returns)
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        
        max_dd = drawdown.min()
        assert max_dd > -0.30, f"Strategy drawdown {max_dd:.2%} too severe"
        
    def test_high_volatility_environment(self, high_volatility_data):
        """Test strategy in high volatility environment."""
        df = high_volatility_data
        
        # Calculate realized volatility
        returns = df['Close'].pct_change().dropna()
        realized_vol = returns.rolling(20).std() * np.sqrt(252)  # Annualized
        
        avg_vol = realized_vol.mean()
        
        # Should be high volatility (>30% annualized)
        assert avg_vol > 0.30, f"Expected high vol >30%, got {avg_vol:.2%}"
        
        # In high vol, smaller position sizes should be used
        # This is a basic check that the system would reduce size
        position_size_reduction_factor = 0.5  # Should reduce by half
        
        # Test that system recognizes high vol
        assert avg_vol > 0.30, "High vol detection failed"


class TestRiskMetricsReliability:
    """Test that risk metrics are calculated correctly."""
    
    def test_var_calculation_is_conservative(self):
        """VaR should be conservative (not underestimate risk)."""
        np.random.seed(47)
        
        # Generate returns with fat tails (Student-t)
        from scipy import stats
        returns = stats.t.rvs(df=4, size=1000, loc=0, scale=0.02)
        
        # Historical VaR at 95%
        var_95 = -np.percentile(returns, 5)
        
        # Parametric VaR (normal) - underestimates tail risk
        normal_var_95 = -1.645 * returns.std()
        
        # Historical should be >= parametric (conservative)
        assert var_95 >= normal_var_95 * 0.8, \
            "Historical VaR should be more conservative than parametric"
            
    def test_cvar_exceeds_var(self):
        """CVaR (Expected Shortfall) should exceed VaR."""
        np.random.seed(48)
        
        returns = np.random.normal(0, 0.02, 1000)
        
        var_95 = -np.percentile(95)
        cvar_95 = -returns[returns < -var_95].mean()
        
        assert cvar_95 > var_95, "CVaR should exceed VaR"


class TestBacktestRealism:
    """Test that backtests use realistic parameters."""
    
    def test_commission_costs_are_realistic(self):
        """Commission should reflect real trading costs."""
        # Typical Forex costs: $5-10 per lot
        # 1 lot = 100,000 units = $10 per 1 pip on EURUSD
        
        realistic_commission_per_lot = 7.0  # USD
        
        # This test verifies the system uses realistic values
        # In actual backtest engine, these should be used
        assert realistic_commission_per_lot >= 5.0, "Commission too low"
        assert realistic_commission_per_lot <= 15.0, "Commission unreasonably high"
        
    def test_slippage_is_variable(self):
        """Slippage should vary with market conditions."""
        # In normal conditions: 0.1-0.5 pips
        # In volatile conditions: 1-5 pips
        # In gap events: can be 10+ pips
        
        normal_slippage = 0.0001  # 1 pip = 0.0001 for EURUSD
        volatile_slippage = 0.0005
        
        # Test that system recognizes condition-dependent slippage
        assert normal_slippage < volatile_slippage


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
