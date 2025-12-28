"""
Module mathématique et statistique du système de trading.
Contient les analyses de co-intégration, Hurst et Z-Score.
"""

from .cointegration import CointegrationAnalyzer
from .hurst import HurstExponent
from .zscore import BollingerZScore

__all__ = ["CointegrationAnalyzer", "HurstExponent", "BollingerZScore"]
