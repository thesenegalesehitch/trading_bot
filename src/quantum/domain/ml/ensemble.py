"""
Module d'ensemble de modèles ML pour la classification des signaux.
Combine XGBoost, LightGBM, CatBoost et Random Forest avec voting/stacking.

Avantages de l'ensemble:
- Réduit le surapprentissage
- Améliore la robustesse
- Calibration des probabilités
- Feature importance agrégée
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')


from quantum.shared.config.settings import config

# Import des modèles avec gestion des erreurs
MODELS_AVAILABLE = {}

try:
    from xgboost import XGBClassifier
    MODELS_AVAILABLE['xgboost'] = True
except ImportError:
    MODELS_AVAILABLE['xgboost'] = False

try:
    import lightgbm as lgb
    MODELS_AVAILABLE['lightgbm'] = True
except ImportError:
    MODELS_AVAILABLE['lightgbm'] = False

try:
    from catboost import CatBoostClassifier
    MODELS_AVAILABLE['catboost'] = True
except ImportError:
    MODELS_AVAILABLE['catboost'] = False

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score


@dataclass
class EnsembleConfig:
    """Configuration de l'ensemble."""
    use_xgboost: bool = True
    use_lightgbm: bool = True
    use_catboost: bool = True
    use_random_forest: bool = True
    voting_method: str = 'soft'  # 'hard' ou 'soft'
    use_stacking: bool = False
    calibrate_probabilities: bool = True
    random_state: int = 42


class ModelFactory:
    """Factory pour créer les modèles individuels."""
    
    @staticmethod
    def create_xgboost(random_state: int = 42) -> Optional[Any]:
        """Crée un classificateur XGBoost."""
        if not MODELS_AVAILABLE.get('xgboost', False):
            return None
        
        return XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=random_state,
            verbosity=0
        )
    
    @staticmethod
    def create_lightgbm(random_state: int = 42) -> Optional[Any]:
        """Crée un classificateur LightGBM."""
        if not MODELS_AVAILABLE.get('lightgbm', False):
            return None
        
        return lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary',
            random_state=random_state,
            verbosity=-1
        )
    
    @staticmethod
    def create_catboost(random_state: int = 42) -> Optional[Any]:
        """Crée un classificateur CatBoost."""
        if not MODELS_AVAILABLE.get('catboost', False):
            return None
        
        return CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            loss_function='Logloss',
            random_state=random_state,
            verbose=False
        )
    
    @staticmethod
    def create_random_forest(random_state: int = 42) -> RandomForestClassifier:
        """Crée un classificateur Random Forest."""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )


class EnsembleClassifier:
    """
    Classificateur ensemble combinant plusieurs modèles.
    
    Supporte:
    - Voting (hard/soft)
    - Stacking avec meta-learner
    - Calibration des probabilités
    """
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.models: Dict[str, Any] = {}
        self.ensemble = None
        self.is_trained = False
        self.feature_names: List[str] = []
        self.feature_importance: Optional[pd.DataFrame] = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialise les modèles individuels."""
        factory = ModelFactory()
        
        if self.config.use_xgboost:
            model = factory.create_xgboost(self.config.random_state)
            if model:
                self.models['xgboost'] = model
                print("✅ XGBoost activé")
            else:
                print("⚠️ XGBoost non disponible")
        
        if self.config.use_lightgbm:
            model = factory.create_lightgbm(self.config.random_state)
            if model:
                self.models['lightgbm'] = model
                print("✅ LightGBM activé")
            else:
                print("⚠️ LightGBM non disponible")
        
        if self.config.use_catboost:
            model = factory.create_catboost(self.config.random_state)
            if model:
                self.models['catboost'] = model
                print("✅ CatBoost activé")
            else:
                print("⚠️ CatBoost non disponible")
        
        if self.config.use_random_forest:
            self.models['random_forest'] = factory.create_random_forest(self.config.random_state)
            print("✅ Random Forest activé")
        
        print(f"📊 Ensemble configuré avec {len(self.models)} modèles")
    
    def _create_ensemble(self):
        """Crée l'ensemble de modèles."""
        if len(self.models) == 0:
            raise ValueError("Aucun modèle disponible pour l'ensemble")
        
        estimators = [(name, model) for name, model in self.models.items()]
        
        if self.config.use_stacking:
            # Stacking avec Logistic Regression comme meta-learner
            self.ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=5,
                passthrough=False,
                n_jobs=-1
            )
        else:
            # Voting classifier
            self.ensemble = VotingClassifier(
                estimators=estimators,
                voting=self.config.voting_method,
                n_jobs=-1
            )
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Entraîne l'ensemble de modèles.
        
        Args:
            X: Features
            y: Target
            validation_split: Ratio de validation
        
        Returns:
            Métriques d'entraînement
        """
        self.feature_names = list(X.columns)
        
        # Split train/validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Créer l'ensemble
        self._create_ensemble()
        
        # Entraîner
        print("🔄 Entraînement de l'ensemble...")
        self.ensemble.fit(X_train, y_train)
        
        # Calibration des probabilités si demandé
        if self.config.calibrate_probabilities:
            print("🔄 Calibration des probabilités...")
            self.ensemble = CalibratedClassifierCV(
                self.ensemble,
                method='isotonic',
                cv='prefit'
            )
            self.ensemble.fit(X_val, y_val)
        
        self.is_trained = True
        
        # Calculer les métriques
        metrics = self._calculate_metrics(X_train, y_train, X_val, y_val)
        
        # Calculer l'importance des features
        self._calculate_feature_importance(X_train, y_train)
        
        return metrics
    
    def _calculate_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict:
        """Calcule les métriques d'évaluation."""
        # Prédictions
        train_pred = self.ensemble.predict(X_train)
        val_pred = self.ensemble.predict(X_val)
        
        train_proba = self.predict_proba(X_train)
        val_proba = self.predict_proba(X_val)
        
        metrics = {
            'train': {
                'accuracy': accuracy_score(y_train, train_pred),
                'precision': precision_score(y_train, train_pred, zero_division=0),
                'recall': recall_score(y_train, train_pred, zero_division=0),
                'f1': f1_score(y_train, train_pred, zero_division=0),
                'auc': roc_auc_score(y_train, train_proba) if len(np.unique(y_train)) > 1 else 0.5,
                'samples': len(y_train)
            },
            'validation': {
                'accuracy': accuracy_score(y_val, val_pred),
                'precision': precision_score(y_val, val_pred, zero_division=0),
                'recall': recall_score(y_val, val_pred, zero_division=0),
                'f1': f1_score(y_val, val_pred, zero_division=0),
                'auc': roc_auc_score(y_val, val_proba) if len(np.unique(y_val)) > 1 else 0.5,
                'samples': len(y_val)
            },
            'models_used': list(self.models.keys()),
            'ensemble_type': 'stacking' if self.config.use_stacking else 'voting',
            'calibrated': self.config.calibrate_probabilities
        }
        
        return metrics
    
    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """Calcule l'importance des features agrégée."""
        importances = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # Entraîner le modèle individuellement pour l'importance
                model.fit(X, y)
                imp = model.feature_importances_
                importances[name] = imp
        
        if importances:
            # Moyenne des importances
            avg_importance = np.mean(list(importances.values()), axis=0)
            
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prédit les classes."""
        if not self.is_trained:
            raise ValueError("Modèle non entraîné")
        return self.ensemble.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Prédit les probabilités."""
        if not self.is_trained:
            raise ValueError("Modèle non entraîné")
        
        proba = self.ensemble.predict_proba(X)
        
        # Retourner seulement la probabilité de la classe positive
        if proba.ndim == 2:
            return proba[:, 1]
        return proba
    
    def predict_signal(
        self,
        X: pd.DataFrame,
        min_threshold: float = 0.6,
        strong_threshold: float = 0.75
    ) -> Dict:
        """
        Génère un signal de trading avec niveau de confiance.
        
        Args:
            X: Features (dernière ligne pour temps réel)
            min_threshold: Seuil minimum pour signal
            strong_threshold: Seuil pour signal fort
        
        Returns:
            Dict avec signal, probabilité et recommandation
        """
        proba = self.predict_proba(X)
        
        if len(proba) > 1:
            proba = proba[-1]  # Dernière valeur
        else:
            proba = proba[0]
        
        if proba >= strong_threshold:
            signal = "STRONG_BUY"
            action = "Signal fort - Entrer en position avec confiance"
        elif proba >= min_threshold:
            signal = "BUY"
            action = "Signal valide - Confirmer avec d'autres indicateurs"
        elif proba <= 1 - strong_threshold:
            signal = "STRONG_AVOID"
            action = "Éviter ce trade - Risque élevé"
        elif proba <= 1 - min_threshold:
            signal = "AVOID"
            action = "Probabilité insuffisante"
        else:
            signal = "NEUTRAL"
            action = "Zone d'incertitude - Attendre"
        
        # Consensus des modèles individuels
        individual_preds = {}
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                try:
                    ind_proba = model.predict_proba(X)
                    if ind_proba.ndim == 2:
                        ind_proba = ind_proba[-1, 1] if len(ind_proba) > 1 else ind_proba[0, 1]
                    individual_preds[name] = float(ind_proba)
                except:
                    pass
        
        consensus = sum(1 for p in individual_preds.values() if p > 0.5) / len(individual_preds) if individual_preds else 0.5
        
        return {
            "signal": signal,
            "probability": round(proba * 100, 2),
            "action": action,
            "threshold_met": proba >= min_threshold,
            "is_strong": proba >= strong_threshold,
            "model_consensus": round(consensus * 100, 2),
            "individual_predictions": individual_preds
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Retourne l'importance des features."""
        if self.feature_importance is None:
            return pd.DataFrame()
        return self.feature_importance
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> Dict:
        """
        Validation croisée temporelle.
        
        Args:
            X: Features
            y: Target
            n_splits: Nombre de splits
        
        Returns:
            Résultats de la CV
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Créer un nouvel ensemble pour ce fold
            self._create_ensemble()
            self.ensemble.fit(X_train, y_train)
            
            val_pred = self.ensemble.predict(X_val)
            val_proba = self.ensemble.predict_proba(X_val)
            if val_proba.ndim == 2:
                val_proba = val_proba[:, 1]
            
            fold_results.append({
                'fold': fold,
                'accuracy': accuracy_score(y_val, val_pred),
                'auc': roc_auc_score(y_val, val_proba) if len(np.unique(y_val)) > 1 else 0.5,
                'f1': f1_score(y_val, val_pred, zero_division=0),
                'train_size': len(y_train),
                'val_size': len(y_val)
            })
        
        return {
            'folds': fold_results,
            'mean_accuracy': np.mean([r['accuracy'] for r in fold_results]),
            'std_accuracy': np.std([r['accuracy'] for r in fold_results]),
            'mean_auc': np.mean([r['auc'] for r in fold_results]),
            'mean_f1': np.mean([r['f1'] for r in fold_results])
        }
    
    def save(self, path: str):
        """Sauvegarde l'ensemble."""
        with open(path, 'wb') as f:
            pickle.dump({
                'ensemble': self.ensemble,
                'models': self.models,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'config': self.config,
                'is_trained': self.is_trained
            }, f)
    
    def load(self, path: str):
        """Charge l'ensemble."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.ensemble = data['ensemble']
            self.models = data['models']
            self.feature_names = data['feature_names']
            self.feature_importance = data['feature_importance']
            self.config = data['config']
            self.is_trained = data['is_trained']


if __name__ == "__main__":
    print("=" * 60)
    print("TEST ENSEMBLE CLASSIFIER")
    print("=" * 60)
    
    print(f"\nModèles disponibles: {MODELS_AVAILABLE}")
    
    # Données de test
    np.random.seed(42)
    n = 1000
    
    X = pd.DataFrame({
        'zscore': np.random.randn(n),
        'hurst': np.random.uniform(0.3, 0.7, n),
        'rsi': np.random.uniform(20, 80, n),
        'macd': np.random.randn(n) * 0.1,
        'atr': np.random.uniform(0.01, 0.03, n),
        'volume_ratio': np.random.uniform(0.5, 2, n)
    })
    
    # Target corrélée aux features
    y = ((X['zscore'] < -1) | (X['rsi'] < 30)).astype(int)
    
    # Créer et entraîner l'ensemble
    config = EnsembleConfig(
        use_stacking=False,
        calibrate_probabilities=True
    )
    ensemble = EnsembleClassifier(config)
    
    print("\n--- Entraînement ---")
    metrics = ensemble.train(X, y)
    
    print(f"\nTrain Accuracy: {metrics['train']['accuracy']:.3f}")
    print(f"Val Accuracy: {metrics['validation']['accuracy']:.3f}")
    print(f"Val AUC: {metrics['validation']['auc']:.3f}")
    
    print("\n--- Feature Importance ---")
    importance = ensemble.get_feature_importance()
    print(importance)
    
    print("\n--- Signal Prediction ---")
    signal = ensemble.predict_signal(X.tail(1))
    print(f"Signal: {signal['signal']}")
    print(f"Probabilité: {signal['probability']}%")
    print(f"Consensus: {signal['model_consensus']}%")
    
    print("\n--- Cross Validation ---")
    cv_results = ensemble.cross_validate(X, y, n_splits=3)
    print(f"Mean Accuracy: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
    print(f"Mean AUC: {cv_results['mean_auc']:.3f}")
