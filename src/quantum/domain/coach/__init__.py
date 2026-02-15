"""
Coach Module - Trade Advisor & Coach Features
Phase 3: Coach Features for Quantum Trading System v2

Ce module fournit les fonctionnalités de coaching et de validation:
- Trade Validator: Valide les trades saisis par l'utilisateur
- LLM Explainer: Génère des explications naturelles
- Trade History: Stockage et analyse de l'historique des trades
"""

from .validator import TradeValidator, TradeValidation, ValidationIssue, ValidationSeverity
from .explainer import LLMExplainer, ExplanationContext, SignalDirection
from .history import TradeHistory, Trade, TradeOutcome, TradeReason

__all__ = [
    "TradeValidator",
    "TradeValidation", 
    "ValidationIssue",
    "ValidationSeverity",
    "LLMExplainer",
    "ExplanationContext",
    "SignalDirection",
    "TradeHistory",
    "Trade",
    "TradeOutcome",
    "TradeReason"
]
