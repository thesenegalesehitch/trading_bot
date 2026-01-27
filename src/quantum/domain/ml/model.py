"""
Modèle XGBoost pour la classification des signaux de trading.
Prédit la probabilité de succès d'un trade.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import pickle
import os
import sys


from quantum.shared.config.settings import config

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier


class SignalClassifier:
    """
    Classificateur de signaux basé sur XGBoost.
    
    Prédit la probabilité qu'un signal soit profitable.
    Un signal n'est émis que si la probabilité dépasse le seuil configuré.
    """
    
    def __init__(self, model_params: Dict = None):
        self.params = model_params or config.ml.XGBOOST_PARAMS
        self.min_threshold = config.ml.MIN_PROBABILITY_THRESHOLD
        self.strong_threshold = config.ml.STRONG_SIGNAL_THRESHOLD
        
        if XGBOOST_AVAILABLE:
            self.model = XGBClassifier(**self.params)
        else:
            print("⚠️ XGBoost non disponible, utilisation de RandomForest")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        
        self.is_trained = False
        self.feature_names = []
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Entraîne le modèle.
        
        Args:
            X: Features
            y: Target
        
        Returns:
            Métriques d'entraînement
        """
        self.feature_names = list(X.columns)
        
        # Séparer en train/validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Entraîner
        if XGBOOST_AVAILABLE:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Évaluer
        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)
        
        # Prédictions probabilistes
        val_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calcul AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_val, val_proba)
        except:
            auc = 0.5
        
        return {
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "auc": auc,
            "train_size": len(X_train),
            "val_size": len(X_val)
        }
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prédit la probabilité de succès.
        
        Returns:
            Array de probabilités (0-1)
        """
        if not self.is_trained:
            raise ValueError("Modèle non entraîné")
        
        return self.model.predict_proba(X)[:, 1]
    
    def predict_signal(self, X: pd.DataFrame) -> Dict:
        """
        Génère un signal avec niveau de confiance.
        
        Args:
            X: Features (une seule ligne pour temps réel)
        
        Returns:
            Signal avec probabilité et recommandation
        """
        proba = self.predict_proba(X)
        
        if len(proba) == 1:
            proba = proba[0]
        else:
            proba = proba[-1]  # Dernière valeur
        
        if proba >= self.strong_threshold:
            signal = "STRONG_BUY"
            action = "Entrer en position avec confiance élevée"
        elif proba >= self.min_threshold:
            signal = "BUY"
            action = "Signal valide, confirmer avec d'autres indicateurs"
        elif proba <= 1 - self.strong_threshold:
            signal = "STRONG_AVOID"
            action = "Éviter ce trade, probabilité d'échec élevée"
        elif proba <= 1 - self.min_threshold:
            signal = "AVOID"
            action = "Probabilité de succès insuffisante"
        else:
            signal = "WAIT"
            action = "Zone d'incertitude, attendre un meilleur setup"
        
        return {
            "signal": signal,
            "probability": proba * 100,
            "action": action,
            "threshold_met": proba >= self.min_threshold,
            "is_strong": proba >= self.strong_threshold
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Retourne l'importance des features."""
        if not self.is_trained:
            return pd.DataFrame()
        
        importance = self.model.feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save(self, path: str):
        """Sauvegarde le modèle."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }, f)
    
    def load(self, path: str):
        """Charge le modèle."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.is_trained = data['is_trained']


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    n = 500
    
    # Données synthétiques
    X = pd.DataFrame({
        'zscore': np.random.randn(n),
        'hurst': np.random.uniform(0.3, 0.7, n),
        'rsi': np.random.uniform(20, 80, n),
        'macd': np.random.randn(n) * 0.1
    })
    
    # Target corrélée aux features
    y = ((X['zscore'] < -1) | (X['rsi'] < 30)).astype(int)
    
    # Entraîner
    classifier = SignalClassifier()
    metrics = classifier.train(X, y)
    
    print("=== Métriques d'entraînement ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print("\n=== Importance des features ===")
    print(classifier.get_feature_importance())
    
    print("\n=== Prédiction signal ===")
    signal = classifier.predict_signal(X.tail(1))
    for k, v in signal.items():
        print(f"  {k}: {v}")
