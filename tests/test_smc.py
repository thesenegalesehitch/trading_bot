"""
Tests pour l'analyse Smart Money Concepts (SMC).
"""

import pytest
import pandas as pd
import numpy as np

from quantum.domain.analysis.smc import SmartMoneyConceptsAnalyzer, OrderBlock, FairValueGap

@pytest.fixture
def sample_bullish_data():
    """Données pour tester un Order Block Bullish et un FVG Bullish."""
    dates = pd.date_range("2026-01-01", periods=10, freq="1h")
    # Structure: 
    # Open High Low Close
    # On simule un mouvement fort à la hausse pour créer OB et FVG
    data = [
        [100, 105,  95, 102], # 0
        [102, 108,  98, 106], # 1
        [106, 107, 105, 105], # 2: Petite bougie rouge ou indecision (OB potentiel)
        [105, 120, 102, 118], # 3: Forte bougie verte (crée FVG Bullish et valide OB en 2)
        [118, 122, 115, 121], # 4
        [121, 125, 120, 124], # 5
        [124, 128, 122, 127], # 6
        [127, 130, 125, 129], # 7
        [129, 135, 128, 134], # 8
        [134, 140, 132, 138]  # 9
    ]
    df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close'], index=dates)
    return df

@pytest.fixture
def sample_bearish_data():
    """Données pour tester un Order Block Bearish et un FVG Bearish."""
    dates = pd.date_range("2026-01-01", periods=10, freq="1h")
    data = [
        [140, 145, 135, 138],
        [138, 142, 136, 140],
        [140, 142, 138, 141], # Petite bougie d'indécision
        [141, 142, 120, 122], # Forte bougie rouge
        [122, 125, 118, 120],
        [120, 122, 115, 118],
        [118, 120, 112, 115],
        [115, 118, 110, 112],
        [112, 115, 105, 108],
        [108, 110, 100, 105]
    ]
    df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close'], index=dates)
    return df

def test_detect_order_blocks_bullish(sample_bullish_data):
    analyzer = SmartMoneyConceptsAnalyzer(lookback=10, min_fvg_percent=0.1)
    analyzer._detect_order_blocks(sample_bullish_data)
    
    assert len(analyzer.order_blocks) > 0
    bullish_obs = [ob for ob in analyzer.order_blocks if ob.type == "BULLISH" and ob.is_valid]
    assert len(bullish_obs) > 0

def test_detect_order_blocks_bearish(sample_bearish_data):
    analyzer = SmartMoneyConceptsAnalyzer(lookback=10, min_fvg_percent=0.1)
    analyzer._detect_order_blocks(sample_bearish_data)
    
    assert len(analyzer.order_blocks) > 0
    bearish_obs = [ob for ob in analyzer.order_blocks if ob.type == "BEARISH" and ob.is_valid]
    assert len(bearish_obs) > 0

def test_detect_fair_value_gaps_bullish(sample_bullish_data):
    analyzer = SmartMoneyConceptsAnalyzer(lookback=10, min_fvg_percent=0.5)
    analyzer._detect_fair_value_gaps(sample_bullish_data)
    
    fvgs = [f for f in analyzer.fvg_list if f.type == "BULLISH"]
    assert len(fvgs) > 0
    # Le Low de la bougie 4 (115) est plus haut que le High de la bougie 2 (107)
    assert fvgs[0].high == 115.0 or fvgs[0].low == 107.0

def test_analyze_full_output(sample_bullish_data):
    analyzer = SmartMoneyConceptsAnalyzer(lookback=10, min_fvg_percent=0.1)
    result = analyzer.analyze(sample_bullish_data)
    
    assert "market_structure" in result
    assert "order_blocks" in result
    assert "fair_value_gaps" in result
    assert "current_analysis" in result
    
    assert result["market_structure"]["trend"] in ["BULLISH", "BEARISH", "CONSOLIDATION", "UNDEFINED"]

def test_get_ob_proximity_score(sample_bullish_data):
    analyzer = SmartMoneyConceptsAnalyzer(lookback=10, min_fvg_percent=0.1)
    analyzer.analyze(sample_bullish_data)
    score = analyzer.get_ob_proximity_score(sample_bullish_data)
    
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0
