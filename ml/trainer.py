"""
Pipeline d'entraînement du modèle ML.
Gère l'entraînement, la validation croisée et l'optimisation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import TimeSeriesSplit
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import config
from .features import MLFeaturesPreparer
from .model import SignalClassifier


class ModelTrainer:
    """
    Gère l'entraînement complet du modèle.
    
    Étapes:
    1. Préparation des features
    2. Validation croisée temporelle
    3. Entraînement final
    4. Évaluation
    """
    
    def __init__(self):
        self.preparer = MLFeaturesPreparer()
        self.classifier = SignalClassifier()
        self.cv_results = []
    
    def train_with_cross_validation(
        self,
        df: pd.DataFrame,
        n_splits: int = None
    ) -> Dict:
        """
        Entraîne avec validation croisée temporelle.
        
        Args:
            df: DataFrame complet avec features
            n_splits: Nombre de splits CV
        
        Returns:
            Résultats de la validation croisée
        """
        n_splits = n_splits or config.ml.CV_SPLITS
        
        # Préparer données
        X, y = self.preparer.prepare_train_data(df)
        
        if len(X) < n_splits * 50:
            print("⚠️ Données insuffisantes pour CV complète")
            n_splits = max(2, len(X) // 100)
        
        # Time Series Split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Nouveau classificateur pour ce fold
            fold_clf = SignalClassifier()
            
            # Entraîner
            temp_metrics = self._train_fold(fold_clf, X_train, y_train, X_val, y_val)
            temp_metrics['fold'] = fold
            fold_results.append(temp_metrics)
        
        self.cv_results = fold_results
        
        # Moyennes
        avg_metrics = {
            'mean_accuracy': np.mean([r['accuracy'] for r in fold_results]),
            'std_accuracy': np.std([r['accuracy'] for r in fold_results]),
            'mean_auc': np.mean([r.get('auc', 0.5) for r in fold_results]),
            'n_folds': n_splits
        }
        
        # Entraîner modèle final sur toutes les données
        final_metrics = self.classifier.train(X, y)
        
        return {
            'cv_results': fold_results,
            'cv_summary': avg_metrics,
            'final_model': final_metrics
        }
    
    def _train_fold(
        self,
        clf: SignalClassifier,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict:
        """Entraîne un fold de CV."""
        # Fit temporaire
        clf.model.fit(X_train, y_train)
        clf.is_trained = True
        
        # Évaluer
        train_acc = clf.model.score(X_train, y_train)
        val_acc = clf.model.score(X_val, y_val)
        
        # AUC
        from sklearn.metrics import roc_auc_score
        try:
            val_proba = clf.model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_proba)
        except:
            auc = 0.5
        
        return {
            'accuracy': val_acc,
            'train_accuracy': train_acc,
            'auc': auc,
            'train_size': len(X_train),
            'val_size': len(X_val)
        }
    
    def evaluate_on_test(
        self,
        test_df: pd.DataFrame
    ) -> Dict:
        """
        Évalue sur données de test.
        
        Args:
            test_df: DataFrame de test
        
        Returns:
            Métriques de test
        """
        if not self.classifier.is_trained:
            raise ValueError("Modèle non entraîné")
        
        X_test, y_test = self.preparer.prepare_train_data(test_df)
        
        # Prédictions
        y_pred = self.classifier.model.predict(X_test)
        y_proba = self.classifier.predict_proba(X_test)
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'n_samples': len(y_test)
        }
    
    def get_trading_statistics(
        self,
        df: pd.DataFrame,
        min_proba: float = None
    ) -> Dict:
        """
        Calcule les statistiques de trading basées sur les prédictions.
        
        Simule les trades qui auraient été pris avec le seuil de probabilité.
        """
        min_proba = min_proba or config.ml.MIN_PROBABILITY_THRESHOLD
        
        X, y = self.preparer.prepare_train_data(df)
        
        if not self.classifier.is_trained:
            return {"error": "Modèle non entraîné"}
        
        probas = self.classifier.predict_proba(X)
        
        # Trades au-dessus du seuil
        above_threshold = probas >= min_proba
        trades_taken = above_threshold.sum()
        
        if trades_taken == 0:
            return {
                "trades_taken": 0,
                "message": "Aucun trade au-dessus du seuil"
            }
        
        # Win rate sur les trades pris
        wins = (y[above_threshold] == 1).sum()
        win_rate = wins / trades_taken
        
        # Comparaison avec random
        overall_win_rate = y.mean()
        
        return {
            "trades_taken": int(trades_taken),
            "total_opportunities": len(y),
            "selectivity": trades_taken / len(y) * 100,
            "wins": int(wins),
            "losses": int(trades_taken - wins),
            "win_rate": win_rate * 100,
            "baseline_win_rate": overall_win_rate * 100,
            "edge": (win_rate - overall_win_rate) * 100,
            "threshold_used": min_proba * 100
        }
    
    def save_model(self, path: str):
        """Sauvegarde modèle et preparer."""
        self.classifier.save(path)
    
    def load_model(self, path: str):
        """Charge modèle."""
        self.classifier.load(path)


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    n = 1000
    
    # Données synthétiques
    df = pd.DataFrame({
        'Close': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'zscore': np.random.randn(n),
        'hurst': np.random.uniform(0.3, 0.7, n),
        'rsi': np.random.uniform(20, 80, n),
        'macd_signal': np.random.randn(n) * 0.1,
        'atr_normalized': np.random.uniform(0.01, 0.03, n)
    })
    
    # Entraîner
    trainer = ModelTrainer()
    results = trainer.train_with_cross_validation(df, n_splits=3)
    
    print("=== Résultats CV ===")
    for fold_res in results['cv_results']:
        print(f"  Fold {fold_res['fold']}: acc={fold_res['accuracy']:.3f}, auc={fold_res['auc']:.3f}")
    
    print("\n=== Résumé CV ===")
    summary = results['cv_summary']
    print(f"  Accuracy: {summary['mean_accuracy']:.3f} ± {summary['std_accuracy']:.3f}")
    print(f"  AUC: {summary['mean_auc']:.3f}")
    
    print("\n=== Statistiques Trading ===")
    stats = trainer.get_trading_statistics(df)
    for k, v in stats.items():
        print(f"  {k}: {v}")
