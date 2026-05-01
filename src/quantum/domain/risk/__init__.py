"""
Module de gestion du risque.
Contient le gestionnaire de risque, VaR et Kelly Criterion.
"""

from .manager import RiskManager
from .var_calculator import VaRCalculator, KellyCriterion

__all__ = ["RiskManager", "VaRCalculator", "KellyCriterion"]
