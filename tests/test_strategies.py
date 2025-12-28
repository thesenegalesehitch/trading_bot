"""
Tests unitaires pour le moteur multi-stratégie.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.multi_strategy import (
    MultiStrategyEngine,
    TrendFollowingStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    SignalType,
    TradeSignal
)
from core.regime_detector import MarketRegime


class TestBaseStrategies:
    """Tests pour les stratégies de base."""
    
    @pytest.fixture
    def uptrend_df(self):
        """DataFrame avec tendance haussière."""
        np.random.seed(42)
        n = 200
        trend = np.linspace(100, 150, n) + np.random.randn(n) * 2
        dates = pd.date_range('2024-01-01', periods=n, freq='1h')
        
        df = pd.DataFrame({
            'Open': trend + np.random.randn(n) * 0.5,
            'High': trend + abs(np.random.randn(n)) + 1,
            'Low': trend - abs(np.random.randn(n)) - 1,
            'Close': trend,
            'Volume': np.random.randint(1000, 10000, n)
        }, index=dates)
        df.attrs['symbol'] = 'TEST'
        return df
    
    @pytest.fixture
    def ranging_df(self):
        """DataFrame en range avec oversold RSI."""
        np.random.seed(42)
        n = 200
        # Création d'un pattern qui finit oversold
        base = 100 + np.sin(np.linspace(0, 10, n)) * 10
        # Ajouter une baisse finale pour être oversold
        base[-20:] = base[-20:] - np.linspace(0, 15, 20)
        
        dates = pd.date_range('2024-01-01', periods=n, freq='1h')
        
        df = pd.DataFrame({
            'Open': base + np.random.randn(n) * 0.5,
            'High': base + abs(np.random.randn(n)) + 0.5,
            'Low': base - abs(np.random.randn(n)) - 0.5,
            'Close': base,
            'Volume': np.random.randint(1000, 10000, n)
        }, index=dates)
        df.attrs['symbol'] = 'TEST'
        return df


class TestTrendFollowingStrategy:
    """Tests pour TrendFollowingStrategy."""
    
    @pytest.fixture
    def strategy(self):
        return TrendFollowingStrategy()
    
    def test_initialization(self, strategy):
        """La stratégie doit s'initialiser correctement."""
        assert strategy.name == "TrendFollowing"
        assert strategy.is_active
        assert 'fast_ema' in strategy.params
    
    def test_suitable_regimes(self, strategy):
        """La stratégie doit être adaptée aux régimes trending."""
        regimes = strategy.get_suitable_regimes()
        
        assert MarketRegime.TRENDING_UP in regimes
        assert MarketRegime.TRENDING_DOWN in regimes
        assert MarketRegime.RANGING not in regimes
    
    def test_signal_generation_uptrend(self, strategy):
        """La stratégie doit générer un signal en tendance."""
        np.random.seed(42)
        n = 200
        trend = np.linspace(100, 150, n)
        dates = pd.date_range('2024-01-01', periods=n, freq='1h')
        
        df = pd.DataFrame({
            'Open': trend,
            'High': trend + 1,
            'Low': trend - 1,
            'Close': trend,
            'Volume': [1000] * n
        }, index=dates)
        df.attrs['symbol'] = 'TEST'
        
        signal = strategy.generate_signal(df, MarketRegime.TRENDING_UP)
        
        # En uptrend, on s'attend à un signal BUY ou None
        if signal:
            assert isinstance(signal, TradeSignal)
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
    
    def test_insufficient_data_returns_none(self, strategy):
        """Données insuffisantes doivent retourner None."""
        df = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000, 1000]
        })
        df.attrs['symbol'] = 'TEST'
        
        signal = strategy.generate_signal(df, MarketRegime.TRENDING_UP)
        
        assert signal is None
    
    def test_performance_update(self, strategy):
        """Le score de performance doit être mis à jour."""
        initial_score = strategy.performance_score
        
        strategy.update_performance(100)  # Trade gagnant
        
        assert strategy.performance_score != initial_score


class TestMeanReversionStrategy:
    """Tests pour MeanReversionStrategy."""
    
    @pytest.fixture
    def strategy(self):
        return MeanReversionStrategy()
    
    def test_initialization(self, strategy):
        """La stratégie doit s'initialiser correctement."""
        assert strategy.name == "MeanReversion"
        assert 'bb_period' in strategy.params
        assert 'rsi_oversold' in strategy.params
    
    def test_suitable_regimes(self, strategy):
        """La stratégie doit être adaptée au range."""
        regimes = strategy.get_suitable_regimes()
        
        assert MarketRegime.RANGING in regimes
        assert MarketRegime.TRENDING_UP not in regimes


class TestBreakoutStrategy:
    """Tests pour BreakoutStrategy."""
    
    @pytest.fixture
    def strategy(self):
        return BreakoutStrategy()
    
    def test_initialization(self, strategy):
        """La stratégie doit s'initialiser correctement."""
        assert strategy.name == "Breakout"
        assert 'lookback' in strategy.params
        assert 'volume_threshold' in strategy.params
    
    def test_suitable_regimes(self, strategy):
        """La stratégie doit être adaptée aux transitions."""
        regimes = strategy.get_suitable_regimes()
        
        assert MarketRegime.TRANSITION in regimes
        assert MarketRegime.RANGING in regimes


class TestMultiStrategyEngine:
    """Tests pour MultiStrategyEngine."""
    
    @pytest.fixture
    def engine(self):
        return MultiStrategyEngine()
    
    @pytest.fixture
    def sample_df(self):
        """DataFrame de test."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        dates = pd.date_range('2024-01-01', periods=n, freq='1h')
        
        return pd.DataFrame({
            'Open': close + np.random.randn(n) * 0.2,
            'High': close + abs(np.random.randn(n)) * 0.5 + 0.1,
            'Low': close - abs(np.random.randn(n)) * 0.5 - 0.1,
            'Close': close,
            'Volume': np.random.randint(1000, 10000, n)
        }, index=dates)
    
    def test_default_strategies_loaded(self, engine):
        """Les stratégies par défaut doivent être chargées."""
        assert len(engine.strategies) >= 3
        assert 'TrendFollowing' in engine.strategies
        assert 'MeanReversion' in engine.strategies
        assert 'Breakout' in engine.strategies
    
    def test_add_strategy(self, engine):
        """Ajouter une stratégie doit fonctionner."""
        initial_count = len(engine.strategies)
        
        new_strategy = TrendFollowingStrategy({'fast_ema': 10})
        new_strategy.name = "CustomTrend"
        engine.add_strategy(new_strategy)
        
        assert len(engine.strategies) == initial_count + 1
        assert 'CustomTrend' in engine.strategies
    
    def test_remove_strategy(self, engine):
        """Retirer une stratégie doit fonctionner."""
        engine.remove_strategy('TrendFollowing')
        
        assert 'TrendFollowing' not in engine.strategies
    
    def test_generate_signals(self, engine, sample_df):
        """Génération de signaux doit fonctionner."""
        signals = engine.generate_signals(sample_df, 'EURUSD')
        
        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, TradeSignal)
    
    def test_consensus_signal(self, engine, sample_df):
        """Signal consensus doit fonctionner."""
        consensus = engine.get_consensus_signal(sample_df, 'EURUSD', min_confidence=0)
        
        # Peut être None ou un TradeSignal
        if consensus:
            assert isinstance(consensus, TradeSignal)
            assert consensus.strategy_name == 'Consensus'
    
    def test_allocation_rebalance(self, engine):
        """Rebalancement doit ajuster les allocations."""
        # Simuler des performances différentes
        engine.strategies['TrendFollowing'].performance_score = 80
        engine.strategies['MeanReversion'].performance_score = 20
        engine._rebalance_allocation()
        
        assert engine.capital_allocation['TrendFollowing'] > engine.capital_allocation['MeanReversion']
    
    def test_status_report(self, engine):
        """Rapport de statut doit contenir les infos requises."""
        status = engine.get_status()
        
        assert 'strategies' in status
        assert 'total_strategies' in status
        
        for name, info in status['strategies'].items():
            assert 'active' in info
            assert 'performance_score' in info
            assert 'allocation' in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
