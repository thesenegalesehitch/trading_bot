"""
Tests unitaires pour le détecteur de régimes de marché.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os



from quantum.domain.core.regime_detector import (
    RegimeDetector,
    MarketRegime,
    RegimeAnalysis
)


class TestRegimeDetector:
    """Tests pour RegimeDetector."""
    
    @pytest.fixture
    def detector(self):
        return RegimeDetector()
    
    @pytest.fixture
    def uptrend_df(self):
        """DataFrame avec tendance haussière claire."""
        np.random.seed(42)
        n = 200
        trend = np.linspace(100, 150, n) + np.random.randn(n) * 2
        dates = pd.date_range('2024-01-01', periods=n, freq='1h')
        
        return pd.DataFrame({
            'Open': trend + np.random.randn(n) * 0.5,
            'High': trend + abs(np.random.randn(n)) + 1,
            'Low': trend - abs(np.random.randn(n)) - 1,
            'Close': trend,
            'Volume': np.random.randint(1000, 10000, n)
        }, index=dates)
    
    @pytest.fixture
    def downtrend_df(self):
        """DataFrame avec tendance baissière claire."""
        np.random.seed(42)
        n = 200
        trend = np.linspace(150, 100, n) + np.random.randn(n) * 2
        dates = pd.date_range('2024-01-01', periods=n, freq='1h')
        
        return pd.DataFrame({
            'Open': trend + np.random.randn(n) * 0.5,
            'High': trend + abs(np.random.randn(n)) + 1,
            'Low': trend - abs(np.random.randn(n)) - 1,
            'Close': trend,
            'Volume': np.random.randint(1000, 10000, n)
        }, index=dates)
    
    @pytest.fixture
    def ranging_df(self):
        """DataFrame en range (consolidation)."""
        np.random.seed(42)
        n = 200
        # Oscillation autour de 100
        ranging = 100 + np.sin(np.linspace(0, 20, n)) * 2 + np.random.randn(n) * 0.5
        dates = pd.date_range('2024-01-01', periods=n, freq='1h')
        
        return pd.DataFrame({
            'Open': ranging + np.random.randn(n) * 0.2,
            'High': ranging + abs(np.random.randn(n)) * 0.3 + 0.1,
            'Low': ranging - abs(np.random.randn(n)) * 0.3 - 0.1,
            'Close': ranging,
            'Volume': np.random.randint(1000, 10000, n)
        }, index=dates)
    
    def test_detect_returns_analysis(self, detector, uptrend_df):
        """detect() doit retourner un RegimeAnalysis."""
        result = detector.detect(uptrend_df)
        
        assert isinstance(result, RegimeAnalysis)
        assert isinstance(result.current_regime, MarketRegime)
        assert 0 <= result.confidence <= 100
    
    def test_uptrend_detected(self, detector, uptrend_df):
        """Tendance haussière doit être détectée."""
        result = detector.detect(uptrend_df)
        
        assert result.current_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRANSITION]
        assert result.trend_strength > 0 or result.current_regime == MarketRegime.TRANSITION
    
    def test_downtrend_detected(self, detector, downtrend_df):
        """Tendance baissière doit être détectée."""
        result = detector.detect(downtrend_df)
        
        assert result.current_regime in [MarketRegime.TRENDING_DOWN, MarketRegime.TRANSITION]
        assert result.trend_strength < 0 or result.current_regime == MarketRegime.TRANSITION
    
    def test_ranging_detected(self, detector, ranging_df):
        """Range doit être détecté."""
        result = detector.detect(ranging_df)
        
        # Le range peut être détecté comme RANGING ou TRANSITION
        assert result.current_regime in [MarketRegime.RANGING, MarketRegime.TRANSITION, MarketRegime.UNKNOWN]
    
    def test_insufficient_data(self, detector):
        """Données insuffisantes doivent retourner UNKNOWN."""
        short_df = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000, 1000]
        })
        
        result = detector.detect(short_df)
        
        assert result.current_regime == MarketRegime.UNKNOWN
        assert result.confidence == 0
    
    def test_volatility_percentile_calculated(self, detector, uptrend_df):
        """Percentile de volatilité doit être calculé."""
        result = detector.detect(uptrend_df)
        
        assert 0 <= result.volatility_percentile <= 100
    
    def test_regime_duration_tracked(self, detector, uptrend_df):
        """Durée du régime doit être suivie."""
        # Premier detect
        result1 = detector.detect(uptrend_df)
        
        # Deuxième detect
        result2 = detector.detect(uptrend_df)
        
        # La durée devrait augmenter si le régime est le même
        if result1.current_regime == result2.current_regime:
            assert result2.regime_duration >= result1.regime_duration
    
    def test_recommendation_generated(self, detector, uptrend_df):
        """Recommandation doit être générée."""
        result = detector.detect(uptrend_df)
        
        assert len(result.recommendation) > 0
    
    def test_strategy_params_for_regime(self, detector, uptrend_df):
        """Paramètres de stratégie doivent être retournés."""
        result = detector.detect(uptrend_df)
        params = detector.get_regime_for_strategy(result.current_regime)
        
        assert 'strategy_type' in params
        assert 'direction_bias' in params
        assert 'position_size_factor' in params


class TestMarketRegimeEnum:
    """Tests pour l'enum MarketRegime."""
    
    def test_all_regimes_defined(self):
        """Tous les régimes doivent être définis."""
        expected_regimes = [
            'TRENDING_UP', 'TRENDING_DOWN', 'RANGING',
            'HIGH_VOLATILITY', 'TRANSITION', 'UNKNOWN'
        ]
        
        actual_regimes = [r.name for r in MarketRegime]
        
        for regime in expected_regimes:
            assert regime in actual_regimes
    
    def test_regime_values_are_strings(self):
        """Les valeurs des régimes doivent être des strings."""
        for regime in MarketRegime:
            assert isinstance(regime.value, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
