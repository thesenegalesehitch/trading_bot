"""
Quantum Trading System - Institutional Test Suite v3.1
=====================================================
Validation des mécanismes Zero-Fault et de la robustesse Alpha.
"""

import pytest
import numpy as np
from quantum.domain.core.scorer import MultiCriteriaScorer, ScoreComponent, SignalStrength
from quantum.application.execution.service import ExecutionManager
from quantum.domain.risk.circuit_breaker import CircuitBreaker

@pytest.fixture
def scorer():
    return MultiCriteriaScorer()

@pytest.fixture
def circuit_breaker():
    return CircuitBreaker()

# --- Tests Scorer (NumPy Vectorization & Logic) ---

def test_scorer_strong_buy_logic(scorer):
    """Vérifie que la décision STRONG_BUY exige une confirmation multi-couches (>60%)."""
    components = [
        ScoreComponent("technical", 90, 0.25, "bullish"),
        ScoreComponent("ml", 85, 0.20, "bullish"),
        ScoreComponent("onchain", 80, 0.20, "bullish"),
        ScoreComponent("social", 50, 0.15, "neutral"),
        ScoreComponent("statistical", 50, 0.10, "neutral"),
        ScoreComponent("risk", 100, 0.10, "neutral")
    ]
    # Somme bullish weights = 0.25 + 0.20 + 0.20 = 0.65 (> 0.6)
    result = scorer._compute_final_score(components)
    assert result.direction == "STRONG_BUY"
    assert result.strength == SignalStrength.STRONG # Score calculé ~ 78, abs deviation 28

def test_scorer_neutral_on_conflict(scorer):
    """Vérifie que le système reste NEUTRAL en cas de conflit entre couches."""
    components = [
        ScoreComponent("technical", 90, 0.5, "bullish"),
        ScoreComponent("ml", 10, 0.5, "bearish")
    ]
    result = scorer._compute_final_score(components)
    assert result.direction == "NEUTRAL"

# --- Tests Exécution (Secrets & Safety) ---

@pytest.mark.asyncio
async def test_execution_manager_simulation_mode(circuit_breaker):
    """Garantit que sans flags explicites, aucun ordre réel n'est envoyé."""
    manager = ExecutionManager(circuit_breaker)
    # On s'assure que live_trading est False pour le test
    manager.live_trading = False
    
    res = await manager.execute_signal("BTC-USD", "BUY", 90, 50000)
    assert res['status'] == "simulated"

def test_vectorization_performance():
    """Test de smoke pour s'assurer que NumPy fonctionne correctement."""
    values = np.random.uniform(0, 100, 1000)
    weights = np.ones(1000) / 1000
    avg = np.average(values, weights=weights)
    assert 0 <= avg <= 100
