"""
Implémentation du filtre de Kalman pour le lissage des prix.
Élimine le bruit de marché tout en préservant le signal réel.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import sys
import os


from quantum.shared.config.settings import config


class KalmanFilter:
    """
    Filtre de Kalman univarié pour le lissage des séries temporelles financières.
    
    Le filtre de Kalman est optimal pour :
    - Éliminer le bruit de mesure
    - Estimer l'état vrai du système
    - Prédire les valeurs futures
    
    Modèle utilisé:
    - État: x(t) = A * x(t-1) + w (processus)
    - Mesure: z(t) = H * x(t) + v (observation)
    
    où w ~ N(0, Q) et v ~ N(0, R)
    """
    
    def __init__(
        self,
        process_noise: float = None,
        measurement_noise: float = None,
        initial_estimate: float = None,
        initial_error: float = 1.0
    ):
        """
        Initialise le filtre de Kalman.
        
        Args:
            process_noise: Variance du bruit de processus (Q)
            measurement_noise: Variance du bruit de mesure (R)
            initial_estimate: Estimation initiale de l'état
            initial_error: Erreur initiale d'estimation
        """
        self.Q = process_noise or config.data.KALMAN_PROCESS_NOISE
        self.R = measurement_noise or config.data.KALMAN_MEASUREMENT_NOISE
        self.initial_estimate = initial_estimate
        self.initial_error = initial_error
        
        # État du filtre
        self.x_hat = None  # Estimation de l'état
        self.P = None      # Covariance de l'erreur
        
        # Historique
        self.estimates = []
        self.gains = []
    
    def reset(self, initial_value: float = None):
        """Réinitialise le filtre."""
        self.x_hat = initial_value or self.initial_estimate
        self.P = self.initial_error
        self.estimates = []
        self.gains = []
    
    def update(self, measurement: float) -> Tuple[float, float]:
        """
        Met à jour le filtre avec une nouvelle mesure.
        
        Args:
            measurement: Nouvelle observation
        
        Returns:
            Tuple (estimation, gain de Kalman)
        """
        # Initialisation si nécessaire
        if self.x_hat is None:
            self.x_hat = measurement
            self.P = self.initial_error
            return self.x_hat, 0.0
        
        # === ÉTAPE DE PRÉDICTION ===
        # Prédiction de l'état (modèle à marche aléatoire: A = 1)
        x_hat_prior = self.x_hat
        P_prior = self.P + self.Q
        
        # === ÉTAPE DE MISE À JOUR ===
        # Calcul du gain de Kalman (H = 1)
        K = P_prior / (P_prior + self.R)
        
        # Mise à jour de l'estimation
        innovation = measurement - x_hat_prior
        self.x_hat = x_hat_prior + K * innovation
        
        # Mise à jour de la covariance de l'erreur
        self.P = (1 - K) * P_prior
        
        # Sauvegarde
        self.estimates.append(self.x_hat)
        self.gains.append(K)
        
        return self.x_hat, K
    
    def filter_series(self, series: pd.Series) -> pd.Series:
        """
        Applique le filtre de Kalman à une série de prix.
        
        Args:
            series: Série pandas de prix
        
        Returns:
            Série filtrée (même index)
        """
        self.reset(series.iloc[0])
        
        filtered_values = []
        for value in series:
            if pd.isna(value):
                filtered_values.append(np.nan)
            else:
                estimate, _ = self.update(value)
                filtered_values.append(estimate)
        
        return pd.Series(filtered_values, index=series.index, name=f"{series.name}_kalman")
    
    def filter_dataframe(
        self,
        df: pd.DataFrame,
        columns: list = None
    ) -> pd.DataFrame:
        """
        Applique le filtre de Kalman aux colonnes spécifiées du DataFrame.
        
        Args:
            df: DataFrame avec les données OHLCV
            columns: Colonnes à filtrer (défaut: ['Close'])
        
        Returns:
            DataFrame avec colonnes filtrées ajoutées
        """
        columns = columns or ['Close']
        result = df.copy()
        
        for col in columns:
            if col in df.columns:
                self.reset()
                result[f"{col}_Kalman"] = self.filter_series(df[col])
        
        return result
    
    def get_signal_to_noise_ratio(self, original: pd.Series, filtered: pd.Series) -> float:
        """
        Calcule le ratio signal/bruit.
        
        Args:
            original: Série originale
            filtered: Série filtrée
        
        Returns:
            SNR en dB
        """
        noise = original - filtered
        signal_power = np.var(filtered)
        noise_power = np.var(noise)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr


class AdaptiveKalmanFilter(KalmanFilter):
    """
    Filtre de Kalman adaptatif qui ajuste automatiquement les paramètres
    en fonction de la volatilité du marché.
    """
    
    def __init__(
        self,
        base_process_noise: float = None,
        base_measurement_noise: float = None,
        adaptation_factor: float = 0.1,
        volatility_window: int = 20
    ):
        """
        Initialise le filtre adaptatif.
        
        Args:
            base_process_noise: Bruit de processus de base
            base_measurement_noise: Bruit de mesure de base
            adaptation_factor: Facteur d'adaptation (0-1)
            volatility_window: Fenêtre pour calculer la volatilité
        """
        super().__init__(base_process_noise, base_measurement_noise)
        
        self.base_Q = self.Q
        self.base_R = self.R
        self.adaptation_factor = adaptation_factor
        self.volatility_window = volatility_window
        self.recent_innovations = []
    
    def update(self, measurement: float) -> Tuple[float, float]:
        """
        Met à jour avec adaptation des paramètres.
        """
        # Calculer l'innovation avant la mise à jour
        if self.x_hat is not None:
            innovation = measurement - self.x_hat
            self.recent_innovations.append(innovation)
            
            # Garder seulement la fenêtre récente
            if len(self.recent_innovations) > self.volatility_window:
                self.recent_innovations.pop(0)
            
            # Adapter les paramètres basé sur la volatilité des innovations
            if len(self.recent_innovations) >= 5:
                innovation_var = np.var(self.recent_innovations)
                
                # Ajuster R (bruit de mesure) selon la volatilité
                self.R = self.base_R * (1 + self.adaptation_factor * innovation_var)
        
        return super().update(measurement)


class MultiVariateKalmanFilter:
    """
    Filtre de Kalman multivarié pour traiter plusieurs variables corrélées.
    Utile pour filtrer OHLC ensemble en préservant leurs relations.
    """
    
    def __init__(
        self,
        n_variables: int = 4,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1
    ):
        """
        Initialise le filtre multivarié.
        
        Args:
            n_variables: Nombre de variables (OHLC = 4)
            process_noise: Bruit de processus
            measurement_noise: Bruit de mesure
        """
        self.n = n_variables
        
        # Matrices du filtre
        self.A = np.eye(n_variables)  # Matrice de transition
        self.H = np.eye(n_variables)  # Matrice d'observation
        self.Q = np.eye(n_variables) * process_noise  # Covariance processus
        self.R = np.eye(n_variables) * measurement_noise  # Covariance mesure
        
        # État
        self.x = None
        self.P = None
    
    def reset(self, initial_state: np.ndarray = None):
        """Réinitialise le filtre."""
        self.x = initial_state
        self.P = np.eye(self.n)
    
    def update(self, measurements: np.ndarray) -> np.ndarray:
        """
        Met à jour avec un vecteur de mesures.
        
        Args:
            measurements: Vecteur de mesures [O, H, L, C]
        
        Returns:
            Vecteur d'état estimé
        """
        if self.x is None:
            self.x = measurements.copy()
            self.P = np.eye(self.n)
            return self.x
        
        # Prédiction
        x_prior = self.A @ self.x
        P_prior = self.A @ self.P @ self.A.T + self.Q
        
        # Mise à jour
        S = self.H @ P_prior @ self.H.T + self.R
        K = P_prior @ self.H.T @ np.linalg.inv(S)
        
        innovation = measurements - self.H @ x_prior
        self.x = x_prior + K @ innovation
        self.P = (np.eye(self.n) - K @ self.H) @ P_prior
        
        return self.x
    
    def filter_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtre les colonnes OHLC d'un DataFrame.
        
        Args:
            df: DataFrame avec colonnes Open, High, Low, Close
        
        Returns:
            DataFrame avec colonnes filtrées
        """
        self.reset()
        
        ohlc_cols = ['Open', 'High', 'Low', 'Close']
        filtered_data = []
        
        for idx, row in df.iterrows():
            measurements = np.array([row[col] for col in ohlc_cols])
            filtered = self.update(measurements)
            filtered_data.append(filtered)
        
        result = df.copy()
        filtered_df = pd.DataFrame(
            filtered_data,
            index=df.index,
            columns=[f"{col}_Kalman" for col in ohlc_cols]
        )
        
        return pd.concat([result, filtered_df], axis=1)


if __name__ == "__main__":
    # Test du filtre de Kalman
    import matplotlib.pyplot as plt
    
    # Générer des données de test avec bruit
    np.random.seed(42)
    n_points = 200
    
    # Signal réel (tendance)
    true_signal = np.cumsum(np.random.randn(n_points) * 0.1) + 100
    
    # Signal bruité (ce qu'on observe)
    noise = np.random.randn(n_points) * 0.5
    noisy_signal = true_signal + noise
    
    # Appliquer le filtre
    kf = KalmanFilter(process_noise=0.01, measurement_noise=0.25)
    filtered_signal = []
    
    for value in noisy_signal:
        estimate, _ = kf.update(value)
        filtered_signal.append(estimate)
    
    # Calculer le SNR
    snr = kf.get_signal_to_noise_ratio(
        pd.Series(noisy_signal),
        pd.Series(filtered_signal)
    )
    print(f"SNR: {snr:.2f} dB")
    
    # Visualisation
    plt.figure(figsize=(12, 6))
    plt.plot(noisy_signal, alpha=0.5, label='Signal bruité')
    plt.plot(true_signal, label='Signal réel', linewidth=2)
    plt.plot(filtered_signal, label='Filtre de Kalman', linewidth=2)
    plt.legend()
    plt.title('Filtre de Kalman - Réduction du bruit')
    plt.savefig('kalman_test.png')
    print("Graphique sauvegardé: kalman_test.png")
