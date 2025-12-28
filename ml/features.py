"""
Préparation des features pour le Machine Learning.
Transforme les indicateurs bruts en features normalisées.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import config


class MLFeaturesPreparer:
    """
    Prépare les features pour l'entraînement et l'inférence ML.
    
    Responsabilités:
    1. Sélection des features pertinentes
    2. Normalisation/Standardisation
    3. Gestion des valeurs manquantes
    4. Création de la variable cible
    """
    
    def __init__(self, feature_columns: List[str] = None):
        self.feature_columns = feature_columns or config.ml.FEATURE_COLUMNS
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def create_target(
        self,
        df: pd.DataFrame,
        forward_periods: int = 5,
        min_return: float = 0.001
    ) -> pd.Series:
        """
        Crée la variable cible (succès du trade).
        
        Un trade est considéré réussi si le prix évolue favorablement
        dans les N prochaines bougies.
        
        Args:
            df: DataFrame avec prix
            forward_periods: Horizon de prédiction
            min_return: Rendement minimum pour succès
        
        Returns:
            Série binaire (1=succès, 0=échec)
        """
        close = df['Close']
        
        # Rendement futur
        future_return = close.shift(-forward_periods) / close - 1
        
        # Cible binaire
        target = (future_return > min_return).astype(int)
        
        return target
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        fit: bool = False
    ) -> pd.DataFrame:
        """
        Prépare les features pour le ML.
        
        Args:
            df: DataFrame avec toutes les colonnes
            fit: Si True, fit le scaler
        
        Returns:
            DataFrame avec features normalisées
        """
        # Sélectionner colonnes disponibles
        available = [c for c in self.feature_columns if c in df.columns]
        
        if not available:
            raise ValueError("Aucune feature configurée trouvée dans les données")
        
        features = df[available].copy()
        
        # Remplacer inf par NaN
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill puis backward fill
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # Normalisation
        if fit:
            self.scaler.fit(features)
            self.is_fitted = True
        
        if self.is_fitted:
            scaled = self.scaler.transform(features)
            features = pd.DataFrame(scaled, index=features.index, columns=available)
        
        return features
    
    def prepare_train_data(
        self,
        df: pd.DataFrame,
        forward_periods: int = 5,
        min_return: float = 0.001
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prépare X et y pour l'entraînement.
        
        Returns:
            Tuple (features, target)
        """
        # Créer la cible
        target = self.create_target(df, forward_periods, min_return)
        
        # Préparer features
        features = self.prepare_features(df, fit=True)
        
        # Aligner et supprimer NaN
        valid_idx = target.notna() & ~features.isna().any(axis=1)
        
        return features[valid_idx], target[valid_idx]
    
    def get_feature_importance_columns(self) -> List[str]:
        """Retourne les colonnes de features utilisées."""
        return [c for c in self.feature_columns if self.is_fitted]


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        'Close': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'zscore': np.random.randn(n),
        'hurst': np.random.uniform(0.3, 0.7, n),
        'rsi': np.random.uniform(20, 80, n),
        'atr_normalized': np.random.uniform(0.01, 0.03, n)
    })
    
    preparer = MLFeaturesPreparer(['zscore', 'hurst', 'rsi', 'atr_normalized'])
    X, y = preparer.prepare_train_data(df)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
