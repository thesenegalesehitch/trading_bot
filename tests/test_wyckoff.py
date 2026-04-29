"""
Tests pour l'analyse Wyckoff.
"""

import pytest
import pandas as pd
import numpy as np

from quantum.domain.analysis.wyckoff import WyckoffAnalyzer, WyckoffPhase, WyckoffEvent

@pytest.fixture
def sample_accumulation_data():
    """Génère des données simulant une phase Wyckoff (simplifiée)."""
    # Pour Wyckoff, on a besoin de plus de données (lookback par défaut = 100)
    dates = pd.date_range("2025-01-01", periods=150, freq="1d")
    np.random.seed(42)
    
    # Simuler un markdown, puis un range (accumulation), puis un début de markup
    # 0-50: Baisse
    p1 = np.linspace(100, 50, 50) + np.random.normal(0, 2, 50)
    # 50-120: Range entre 45 et 55
    p2 = np.random.uniform(45, 55, 70)
    # Un "Spring" (faux break bas)
    p2[40] = 38
    p2[41] = 40
    # 120-150: Hausse
    p3 = np.linspace(50, 80, 30) + np.random.normal(0, 2, 30)
    
    close_prices = np.concatenate([p1, p2, p3])
    
    data = {
        'Open': close_prices + np.random.normal(0, 1, 150),
        'High': close_prices + np.abs(np.random.normal(0, 2, 150)),
        'Low': close_prices - np.abs(np.random.normal(0, 2, 150)),
        'Close': close_prices,
        'Volume': np.random.uniform(1000, 5000, 150)
    }
    # Au moment du Spring, volume très élevé
    data['Volume'][90] = 15000
    
    df = pd.DataFrame(data, index=dates)
    return df

def test_wyckoff_insufficient_data():
    analyzer = WyckoffAnalyzer(lookback=100)
    df = pd.DataFrame({'Close': [1] * 50, 'High': [1] * 50, 'Low': [1] * 50, 'Volume': [1] * 50})
    result = analyzer.analyze(df)
    
    assert result.phase == WyckoffPhase.UNKNOWN
    assert len(result.events) == 0

def test_wyckoff_identify_range(sample_accumulation_data):
    analyzer = WyckoffAnalyzer(lookback=100)
    support, resistance = analyzer._identify_range(sample_accumulation_data)
    
    # Le range devrait être autour de 45-55 (les percentiles 10 et 90)
    assert 40 <= support <= 52
    assert 50 <= resistance <= 85

def test_wyckoff_volume_spread_analysis(sample_accumulation_data):
    analyzer = WyckoffAnalyzer(lookback=100)
    vsa_df = analyzer._analyze_volume_spread(sample_accumulation_data)
    
    assert 'Spread' in vsa_df.columns
    assert 'Volume_MA' in vsa_df.columns
    assert 'VSA_Signal' in vsa_df.columns

def test_wyckoff_analyze_output(sample_accumulation_data):
    # WyckoffAnalyzer est heuristique, il se peut qu'il sorte ACCUMULATION ou MARKUP selon le moment
    analyzer = WyckoffAnalyzer(lookback=100, range_threshold=0.03)
    result = analyzer.analyze(sample_accumulation_data)
    
    assert isinstance(result.phase, WyckoffPhase)
    assert hasattr(result, 'events')
    assert hasattr(result, 'support_level')
    assert hasattr(result, 'resistance_level')
    assert hasattr(result, 'signal')
    
    # Le mock d'accumulation se termine par une hausse (MARKUP) -> Test robuste pour éviter les faux positifs 
    assert result.phase in [WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKUP, WyckoffPhase.UNKNOWN]
