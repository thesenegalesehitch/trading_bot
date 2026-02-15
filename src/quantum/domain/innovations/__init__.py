"""
Innovations Module - The 5 AI Innovations for Quantum Trading System v2
Phase 4: Innovations - Trade Advisor & Coach

Ce module contient les 5 innovations:
1. What If Simulator - Permet de replay n'importe quel scénario
2. Auto-Post-Mortem - Génère analyse automatique après chaque trade
3. Mistake Predictor - Prédit quand l'utilisateur va faire une erreur
4. Confusion Resolver - Résout les contradictions d'indicateurs (FLAGSHP)
5. Reverse Engineering - Analyse les trades winners pour en extraire des leçons
"""

from .whatif_simulator import WhatIfSimulator, SimulationResult, SimulationType
from .postmortem import AutoPostMortem, PostMortemAnalysis
from .mistake_predictor import MistakePredictor, MistakePrediction, MistakePattern
from .confusion_resolver import ConfusionResolver, ConfusionResolution, IndicatorAnalysis, IndicatorSignal

__all__ = [
    # What If Simulator
    "WhatIfSimulator",
    "SimulationResult",
    "SimulationType",
    
    # Auto-Post-Mortem
    "AutoPostMortem",
    "PostMortemAnalysis",
    
    # Mistake Predictor
    "MistakePredictor",
    "MistakePrediction",
    "MistakePattern",
    
    # Confusion Resolver (FLAGSHP)
    "ConfusionResolver",
    "ConfusionResolution",
    "IndicatorAnalysis",
    "IndicatorSignal"
]
