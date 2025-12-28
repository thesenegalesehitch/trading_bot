"""
Module d'optimisation bayésienne des hyperparamètres.
Utilise Optuna pour une recherche efficace.

Avantages:
- Plus efficace que grid search
- Pruning early des mauvais essais
- Visualisation des résultats
- Suggestion intelligente des hyperparamètres
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional, Any, Tuple
from dataclasses import dataclass
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import config

# Import Optuna avec fallback
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna non disponible, optimisation basique utilisée")

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# Import des modèles
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


@dataclass
class OptimizationResult:
    """Résultat de l'optimisation."""
    best_params: Dict
    best_score: float
    n_trials: int
    optimization_history: List[Dict]
    feature_importance: Optional[pd.DataFrame] = None


class HyperparameterOptimizer:
    """
    Optimiseur bayésien des hyperparamètres.
    
    Supporte:
    - XGBoost
    - LightGBM
    - Random Forest
    - Custom objectives
    """
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        n_trials: int = 50,
        cv_splits: int = 5,
        scoring: str = 'roc_auc',
        random_state: int = 42
    ):
        self.model_type = model_type
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.scoring = scoring
        self.random_state = random_state
        
        self.study: Optional[Any] = None
        self.best_model: Optional[Any] = None
        self.optimization_history: List[Dict] = []
    
    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict = None
    ) -> OptimizationResult:
        """
        Optimise les hyperparamètres.
        
        Args:
            X: Features
            y: Target
            param_space: Espace de paramètres personnalisé
        
        Returns:
            OptimizationResult avec les meilleurs paramètres
        """
        if OPTUNA_AVAILABLE:
            return self._optimize_optuna(X, y, param_space)
        else:
            return self._optimize_basic(X, y, param_space)
    
    def _optimize_optuna(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict = None
    ) -> OptimizationResult:
        """Optimisation avec Optuna."""
        
        # Créer le sampler et pruner
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        
        # Créer l'étude
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        # Définir l'objective
        objective = self._create_objective(X, y, param_space)
        
        # Optimiser
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Récupérer les résultats
        best_params = self.study.best_params
        best_score = self.study.best_value
        
        # Créer le modèle optimal
        self.best_model = self._create_model(best_params)
        self.best_model.fit(X, y)
        
        # Historique
        history = [
            {
                'trial': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state)
            }
            for t in self.study.trials
        ]
        
        # Feature importance si disponible
        feature_importance = None
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=len(self.study.trials),
            optimization_history=history,
            feature_importance=feature_importance
        )
    
    def _create_objective(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict = None
    ) -> Callable:
        """Crée la fonction objective pour Optuna."""
        
        def objective(trial: 'optuna.Trial') -> float:
            # Suggérer les hyperparamètres selon le type de modèle
            if self.model_type == 'xgboost':
                params = self._suggest_xgboost_params(trial, param_space)
            elif self.model_type == 'lightgbm':
                params = self._suggest_lightgbm_params(trial, param_space)
            elif self.model_type == 'random_forest':
                params = self._suggest_rf_params(trial, param_space)
            else:
                params = param_space or {}
            
            # Créer le modèle
            model = self._create_model(params)
            
            # Validation croisée temporelle
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                
                if self.scoring == 'roc_auc':
                    y_proba = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) > 1 else 0.5
                else:
                    y_pred = model.predict(X_val)
                    score = accuracy_score(y_val, y_pred)
                
                scores.append(score)
                
                # Pruning - arrêter si les résultats sont mauvais
                trial.report(np.mean(scores), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(scores)
        
        return objective
    
    def _suggest_xgboost_params(
        self,
        trial: 'optuna.Trial',
        custom_space: Dict = None
    ) -> Dict:
        """Suggère les hyperparamètres XGBoost."""
        if custom_space:
            return {k: trial.suggest_categorical(k, [v]) for k, v in custom_space.items()}
        
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0, 5),
        }
    
    def _suggest_lightgbm_params(
        self,
        trial: 'optuna.Trial',
        custom_space: Dict = None
    ) -> Dict:
        """Suggère les hyperparamètres LightGBM."""
        if custom_space:
            return custom_space
        
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
    
    def _suggest_rf_params(
        self,
        trial: 'optuna.Trial',
        custom_space: Dict = None
    ) -> Dict:
        """Suggère les hyperparamètres Random Forest."""
        if custom_space:
            return custom_space
        
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        }
    
    def _create_model(self, params: Dict) -> Any:
        """Crée un modèle avec les paramètres donnés."""
        params = params.copy()
        params['random_state'] = self.random_state
        
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            params['use_label_encoder'] = False
            params['eval_metric'] = 'logloss'
            params['verbosity'] = 0
            return XGBClassifier(**params)
        
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            params['verbosity'] = -1
            return lgb.LGBMClassifier(**params)
        
        elif self.model_type == 'random_forest':
            params['n_jobs'] = -1
            return RandomForestClassifier(**params)
        
        else:
            # Fallback sur Random Forest
            return RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
    
    def _optimize_basic(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict = None
    ) -> OptimizationResult:
        """Optimisation basique sans Optuna (random search)."""
        from sklearn.model_selection import RandomizedSearchCV
        
        # Définir l'espace de paramètres par défaut
        if param_space is None:
            if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
                param_space = {
                    'n_estimators': [50, 100, 150, 200],
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                }
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
            elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                param_space = {
                    'n_estimators': [50, 100, 150, 200],
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                }
                model = lgb.LGBMClassifier(verbosity=-1)
            else:
                param_space = {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                }
                model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        else:
            model = self._create_model({})
        
        # Random search avec CV temporelle
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        
        search = RandomizedSearchCV(
            model,
            param_space,
            n_iter=min(self.n_trials, 20),
            cv=tscv,
            scoring='roc_auc',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        search.fit(X, y)
        
        self.best_model = search.best_estimator_
        
        # Feature importance
        feature_importance = None
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return OptimizationResult(
            best_params=search.best_params_,
            best_score=search.best_score_,
            n_trials=len(search.cv_results_['mean_test_score']),
            optimization_history=[{
                'params': p,
                'score': s
            } for p, s in zip(search.cv_results_['params'], search.cv_results_['mean_test_score'])],
            feature_importance=feature_importance
        )
    
    def get_best_model(self) -> Any:
        """Retourne le meilleur modèle entraîné."""
        return self.best_model
    
    def get_optimization_importance(self) -> Optional[pd.DataFrame]:
        """Analyse l'importance des hyperparamètres."""
        if not OPTUNA_AVAILABLE or self.study is None:
            return None
        
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return pd.DataFrame({
                'parameter': list(importance.keys()),
                'importance': list(importance.values())
            }).sort_values('importance', ascending=False)
        except:
            return None


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization pour éviter le surapprentissage.
    
    Divise les données en plusieurs fenêtres:
    - In-sample: Optimisation des paramètres
    - Out-of-sample: Validation
    """
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        n_splits: int = 5,
        train_ratio: float = 0.7,
        n_trials_per_window: int = 30
    ):
        self.model_type = model_type
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.n_trials_per_window = n_trials_per_window
    
    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict:
        """
        Exécute l'optimisation walk-forward.
        
        Args:
            X: Features
            y: Target
        
        Returns:
            Résultats par fenêtre et métriques globales
        """
        results = []
        window_size = len(X) // self.n_splits
        
        for i in range(self.n_splits - 1):
            # Définir les fenêtres
            train_end = (i + 1) * window_size
            test_start = train_end
            test_end = min(test_start + window_size, len(X))
            
            # In-sample (optimisation)
            in_sample_end = int(train_end * self.train_ratio)
            X_train = X.iloc[:in_sample_end]
            y_train = y.iloc[:in_sample_end]
            
            # Validation in-sample
            X_val = X.iloc[in_sample_end:train_end]
            y_val = y.iloc[in_sample_end:train_end]
            
            # Out-of-sample
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            if len(X_train) < 100 or len(X_test) < 20:
                continue
            
            # Optimiser sur in-sample
            optimizer = HyperparameterOptimizer(
                model_type=self.model_type,
                n_trials=self.n_trials_per_window,
                cv_splits=3
            )
            
            opt_result = optimizer.optimize(
                pd.concat([X_train, X_val]),
                pd.concat([y_train, y_val])
            )
            
            # Évaluer sur out-of-sample
            model = optimizer.get_best_model()
            
            test_proba = model.predict_proba(X_test)[:, 1]
            test_pred = model.predict(X_test)
            
            oos_auc = roc_auc_score(y_test, test_proba) if len(np.unique(y_test)) > 1 else 0.5
            oos_accuracy = accuracy_score(y_test, test_pred)
            
            results.append({
                'window': i,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'is_score': opt_result.best_score,
                'oos_auc': oos_auc,
                'oos_accuracy': oos_accuracy,
                'best_params': opt_result.best_params,
                'robustness_ratio': oos_auc / opt_result.best_score if opt_result.best_score > 0 else 0
            })
        
        # Statistiques globales
        if results:
            avg_is = np.mean([r['is_score'] for r in results])
            avg_oos = np.mean([r['oos_auc'] for r in results])
            avg_robustness = np.mean([r['robustness_ratio'] for r in results])
        else:
            avg_is = avg_oos = avg_robustness = 0
        
        return {
            'windows': results,
            'avg_in_sample_score': avg_is,
            'avg_out_sample_auc': avg_oos,
            'avg_robustness_ratio': avg_robustness,
            'is_robust': avg_robustness > 0.8  # Le modèle est robuste si ratio > 80%
        }


if __name__ == "__main__":
    print("=" * 60)
    print("TEST OPTIMISATION BAYÉSIENNE")
    print("=" * 60)
    
    print(f"\nOptuna disponible: {OPTUNA_AVAILABLE}")
    
    # Données de test
    np.random.seed(42)
    n = 1000
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.uniform(0, 1, n),
        'feature3': np.random.randn(n),
        'feature4': np.random.uniform(-1, 1, n),
    })
    
    y = ((X['feature1'] > 0) & (X['feature2'] > 0.5)).astype(int)
    
    # Optimiser
    print("\n--- Optimisation XGBoost ---")
    optimizer = HyperparameterOptimizer(
        model_type='xgboost' if XGBOOST_AVAILABLE else 'random_forest',
        n_trials=20,
        cv_splits=3
    )
    
    result = optimizer.optimize(X, y)
    
    print(f"\nMeilleurs paramètres: {result.best_params}")
    print(f"Meilleur score: {result.best_score:.4f}")
    print(f"Nombre d'essais: {result.n_trials}")
    
    if result.feature_importance is not None:
        print(f"\nImportance des features:")
        print(result.feature_importance)
    
    # Walk-Forward
    print("\n--- Walk-Forward Optimization ---")
    wfo = WalkForwardOptimizer(
        model_type='random_forest',
        n_splits=4,
        n_trials_per_window=10
    )
    
    wfo_results = wfo.optimize(X, y)
    
    print(f"\nScore in-sample moyen: {wfo_results['avg_in_sample_score']:.4f}")
    print(f"Score out-of-sample moyen: {wfo_results['avg_out_sample_auc']:.4f}")
    print(f"Ratio de robustesse: {wfo_results['avg_robustness_ratio']:.4f}")
    print(f"Modèle robuste: {wfo_results['is_robust']}")
