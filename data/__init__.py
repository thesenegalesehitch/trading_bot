"""
Moteur de données pour le système de trading quantitatif.
Téléchargement, nettoyage et préparation des données historiques.
"""

from .downloader import DataDownloader
from .kalman_filter import KalmanFilter
from .feature_engine import FeatureEngine

__all__ = ["DataDownloader", "KalmanFilter", "FeatureEngine"]
