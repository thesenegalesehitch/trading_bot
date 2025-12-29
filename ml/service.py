"""
MLService - Service de Machine Learning pour la classification des signaux de trading.

Ce service fournit une interface unifiée pour :
- Chargement des modèles d'ensemble
- Prédiction des signaux avec confiance
- Mise à jour automatique des modèles
- Calibration des scores de confiance
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import os
import sys
import logging
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import config
from ml.ensemble import EnsembleClassifier, EnsembleConfig

logger = logging.getLogger(__name__)


class MLService:
    """
    Service ML pour la prédiction de signaux de trading.

    Gère le cycle de vie des modèles d'ensemble et fournit
    des prédictions avec scores de confiance calibrés.
    """

    def __init__(self, model_path: str = None):
        """
        Initialise le service ML.

        Args:
            model_path: Chemin vers le modèle sauvegardé (optionnel)
        """
        self.model_path = model_path or config.ml.MODEL_PATH
        self.models: Dict[str, EnsembleClassifier] = {}
        self.last_update = None
        self.min_confidence = config.ml.MIN_PROBABILITY_THRESHOLD
        self.strong_confidence = config.ml.STRONG_SIGNAL_THRESHOLD

        # Charger les modèles existants
        self._load_models()

    def _load_models(self):
        """Charge les modèles d'ensemble sauvegardés."""
        try:
            if os.path.exists(self.model_path):
                # Charger le modèle principal
                ensemble = EnsembleClassifier()
                ensemble.load(self.model_path)
                self.models['main'] = ensemble
                self.last_update = datetime.now()
                logger.info(f"Modèle chargé depuis {self.model_path}")
            else:
                logger.warning(f"Aucun modèle trouvé à {self.model_path}")
                # Créer un modèle par défaut si nécessaire
                self._create_default_model()

        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            self._create_default_model()

    def _create_default_model(self):
        """Crée un modèle par défaut si aucun n'existe."""
        config_ensemble = EnsembleConfig(
            use_xgboost=True,
            use_lightgbm=True,
            use_catboost=True,
            calibrate_probabilities=True
        )
        self.models['main'] = EnsembleClassifier(config_ensemble)
        logger.info("Modèle par défaut créé")

    def predict_signal(self, features: Dict) -> Dict:
        """
        Prédit un signal de trading avec score de confiance.

        Args:
            features: Dictionnaire des features d'entrée

        Returns:
            Dictionnaire contenant le signal, probabilité et recommandations
        """
        try:
            # Convertir en DataFrame
            X = pd.DataFrame([features])

            # Utiliser le modèle principal
            model = self.models.get('main')
            if not model or not model.is_trained:
                return self._fallback_prediction(features)

            # Prédiction
            signal_data = model.predict_signal(
                X,
                min_threshold=self.min_confidence,
                strong_threshold=self.strong_confidence
            )

            # Enrichir avec métadonnées
            signal_data.update({
                'timestamp': datetime.now().isoformat(),
                'model_version': 'ensemble_v1',
                'features_used': list(features.keys()),
                'confidence_level': self._calculate_confidence_level(signal_data['probability'] / 100)
            })

            return signal_data

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            return self._fallback_prediction(features)

    def _fallback_prediction(self, features: Dict) -> Dict:
        """Prédiction de secours en cas d'erreur."""
        return {
            'signal': 'UNKNOWN',
            'probability': 50.0,
            'action': 'Erreur de prédiction - Vérifier manuellement',
            'threshold_met': False,
            'is_strong': False,
            'model_consensus': 50.0,
            'individual_predictions': {},
            'timestamp': datetime.now().isoformat(),
            'model_version': 'fallback',
            'features_used': list(features.keys()),
            'confidence_level': 'low',
            'error': True
        }

    def _calculate_confidence_level(self, probability: float) -> str:
        """Calcule le niveau de confiance basé sur la probabilité."""
        if probability >= self.strong_confidence:
            return 'high'
        elif probability >= self.min_confidence:
            return 'medium'
        else:
            return 'low'

    def update_models(self, new_data: pd.DataFrame):
        """
        Met à jour les modèles avec de nouvelles données.

        Args:
            new_data: Nouvelles données d'entraînement
        """
        try:
            if new_data.empty:
                logger.warning("Aucune nouvelle donnée fournie")
                return

            # Réentraîner le modèle principal
            model = self.models.get('main')
            if model:
                # Préparer les features et target
                # Note: Cette logique devrait être adaptée selon vos données
                X = new_data.drop(columns=['target'], errors='ignore')
                y = new_data.get('target', pd.Series())

                if not y.empty:
                    logger.info("Réentraînement du modèle avec nouvelles données")
                    metrics = model.train(X, y)

                    # Sauvegarder le modèle mis à jour
                    model.save(self.model_path)
                    self.last_update = datetime.now()

                    logger.info(f"Modèle mis à jour - Accuracy: {metrics['validation']['accuracy']:.3f}")
                else:
                    logger.warning("Aucune target trouvée dans les nouvelles données")

        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du modèle: {e}")

    def get_model_info(self) -> Dict:
        """Retourne les informations sur les modèles chargés."""
        info = {
            'models_loaded': list(self.models.keys()),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'model_path': self.model_path,
            'min_confidence': self.min_confidence,
            'strong_confidence': self.strong_confidence
        }

        # Informations détaillées sur le modèle principal
        main_model = self.models.get('main')
        if main_model and main_model.is_trained:
            info['main_model'] = {
                'trained': True,
                'feature_count': len(main_model.feature_names),
                'features': main_model.feature_names,
                'models_used': list(main_model.models.keys()),
                'ensemble_type': 'stacking' if main_model.config.use_stacking else 'voting'
            }
        else:
            info['main_model'] = {'trained': False}

        return info

    def get_feature_importance(self) -> pd.DataFrame:
        """Retourne l'importance des features."""
        model = self.models.get('main')
        if model and model.is_trained:
            return model.get_feature_importance()
        return pd.DataFrame()

    def validate_features(self, features: Dict) -> Dict:
        """
        Valide que les features fournies sont compatibles.

        Args:
            features: Features à valider

        Returns:
            Résultat de validation
        """
        model = self.models.get('main')
        if not model or not model.is_trained:
            return {'valid': False, 'reason': 'Modèle non entraîné'}

        expected_features = set(model.feature_names)
        provided_features = set(features.keys())

        missing = expected_features - provided_features
        extra = provided_features - expected_features

        if missing:
            return {
                'valid': False,
                'reason': f'Features manquantes: {list(missing)}',
                'missing_features': list(missing)
            }

        result = {'valid': True, 'expected_features': list(expected_features)}

        if extra:
            result['extra_features'] = list(extra)
            result['warning'] = 'Features supplémentaires ignorées'

        return result


# Instance globale du service
ml_service = MLService()


if __name__ == "__main__":
    # Test du service
    print("=== Test MLService ===")

    # Features de test
    test_features = {
        'zscore': -1.5,
        'hurst': 0.45,
        'rsi': 25.0,
        'macd': -0.02,
        'atr': 0.015,
        'volume_ratio': 1.2
    }

    # Prédiction
    result = ml_service.predict_signal(test_features)

    print(f"Signal: {result['signal']}")
    print(f"Probabilité: {result['probability']}%")
    print(f"Action: {result['action']}")
    print(f"Confiance: {result['confidence_level']}")

    # Informations sur le modèle
    info = ml_service.get_model_info()
    print(f"\nModèles chargés: {info['models_loaded']}")
    print(f"Dernière mise à jour: {info['last_update']}")

    # Validation des features
    validation = ml_service.validate_features(test_features)
    print(f"Features valides: {validation['valid']}")