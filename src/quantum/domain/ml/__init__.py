"""
Module Machine Learning pour la validation des signaux.
Contient le modèle XGBoost et le pipeline d'entraînement.
"""

from .model import SignalClassifier
from .trainer import ModelTrainer
from .features import MLFeaturesPreparer

__all__ = ["SignalClassifier", "ModelTrainer", "MLFeaturesPreparer"]
