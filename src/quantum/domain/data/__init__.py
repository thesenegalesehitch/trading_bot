"""
Moteur de données pour le système de trading.
Téléchargement et préparation des données historiques via yFinance.
"""

from .downloader import DataDownloader
from .feature_engine import FeatureEngine

__all__ = ["DataDownloader", "FeatureEngine"]
