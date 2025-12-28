"""
Z-Score de Bollinger pour la détection des extrêmes statistiques.
Identifie les points de retournement avec haute probabilité.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import config


class BollingerZScore:
    """
    Calcule le Z-Score de Bollinger pour identifier les extrêmes statistiques.
    
    Le Z-Score mesure combien d'écarts-types le prix est éloigné de sa moyenne.
    À 3 écarts-types, la probabilité de retournement est >99% (théoriquement).
    
    Utilise les bandes de Bollinger comme base, mais ajoute:
    - Calcul de probabilités exactes
    - Détection multi-niveau (2σ, 3σ, 4σ)
    - Score de confiance ajusté par la volatilité
    """
    
    def __init__(
        self,
        window: int = None,
        signal_threshold: float = None,
        extreme_threshold: float = None
    ):
        """
        Initialise le calculateur de Z-Score.
        
        Args:
            window: Fenêtre pour le calcul de la moyenne mobile
            signal_threshold: Seuil pour signaux normaux (défaut: 2σ)
            extreme_threshold: Seuil pour signaux extrêmes (défaut: 3σ)
        """
        self.window = window or config.statistical.ZSCORE_WINDOW
        self.signal_threshold = signal_threshold or config.statistical.ZSCORE_SIGNAL_THRESHOLD
        self.extreme_threshold = extreme_threshold or config.statistical.ZSCORE_EXTREME_THRESHOLD
    
    def calculate(self, series: pd.Series) -> pd.DataFrame:
        """
        Calcule le Z-Score et les bandes de Bollinger.
        
        Args:
            series: Série de prix (typiquement Close)
        
        Returns:
            DataFrame avec zscore, bandes et signaux
        """
        # Moyenne mobile
        sma = series.rolling(window=self.window).mean()
        
        # Écart-type rolling
        std = series.rolling(window=self.window).std()
        
        # Z-Score
        zscore = (series - sma) / (std + 1e-10)
        
        # Bandes de Bollinger
        bb_upper_2 = sma + 2 * std
        bb_lower_2 = sma - 2 * std
        bb_upper_3 = sma + 3 * std
        bb_lower_3 = sma - 3 * std
        
        # Probabilité de retour à la moyenne (basée sur distribution normale)
        probability = 2 * (1 - stats.norm.cdf(np.abs(zscore)))
        reversal_probability = 1 - probability  # Probabilité de retournement
        
        result = pd.DataFrame({
            'price': series,
            'sma': sma,
            'std': std,
            'zscore': zscore,
            'bb_upper_2': bb_upper_2,
            'bb_lower_2': bb_lower_2,
            'bb_upper_3': bb_upper_3,
            'bb_lower_3': bb_lower_3,
            'reversal_probability': reversal_probability
        }, index=series.index)
        
        # Ajouter les signaux
        result['signal'] = self._generate_signals(zscore)
        
        return result
    
    def _generate_signals(self, zscore: pd.Series) -> pd.Series:
        """
        Génère les signaux basés sur le Z-Score.
        
        Signaux:
        - 0: Neutre
        - 1: BUY signal (z < -threshold)
        - -1: SELL signal (z > threshold)
        - 2: STRONG BUY (z < -extreme)
        - -2: STRONG SELL (z > extreme)
        """
        signals = pd.Series(0, index=zscore.index)
        
        # Signaux normaux
        signals[zscore < -self.signal_threshold] = 1
        signals[zscore > self.signal_threshold] = -1
        
        # Signaux extrêmes
        signals[zscore < -self.extreme_threshold] = 2
        signals[zscore > self.extreme_threshold] = -2
        
        return signals
    
    def get_current_status(self, series: pd.Series) -> Dict:
        """
        Analyse le statut actuel du Z-Score.
        
        Args:
            series: Série de prix
        
        Returns:
            Dictionnaire avec analyse détaillée
        """
        result = self.calculate(series)
        
        current = {
            'price': result['price'].iloc[-1],
            'zscore': result['zscore'].iloc[-1],
            'sma': result['sma'].iloc[-1],
            'std': result['std'].iloc[-1],
            'signal': result['signal'].iloc[-1],
            'reversal_probability': result['reversal_probability'].iloc[-1]
        }
        
        # Interprétation
        z = current['zscore']
        
        if np.isnan(z):
            current['interpretation'] = "Données insuffisantes"
            current['confidence'] = 0.0
        elif abs(z) >= 3:
            direction = "suracheté" if z > 0 else "survendu"
            current['interpretation'] = f"Extrême {direction} - Retournement très probable"
            current['confidence'] = min(95, 80 + abs(z) * 5)
        elif abs(z) >= 2:
            direction = "surachat" if z > 0 else "survente"
            current['interpretation'] = f"Zone de {direction} - Signal potentiel"
            current['confidence'] = min(85, 60 + abs(z) * 10)
        elif abs(z) >= 1:
            direction = "haut" if z > 0 else "bas"
            current['interpretation'] = f"Légèrement {direction} de la moyenne"
            current['confidence'] = 40
        else:
            current['interpretation'] = "Dans la zone normale"
            current['confidence'] = 20
        
        # Distances aux bandes
        current['distance_to_upper_2'] = result['bb_upper_2'].iloc[-1] - current['price']
        current['distance_to_lower_2'] = current['price'] - result['bb_lower_2'].iloc[-1]
        
        return current
    
    def detect_divergence(
        self,
        price: pd.Series,
        indicator: pd.Series,
        lookback: int = 14
    ) -> Dict:
        """
        Détecte les divergences entre le prix et un indicateur.
        
        Une divergence se produit quand le prix fait un nouveau high/low
        mais l'indicateur ne confirme pas.
        
        Args:
            price: Série de prix
            indicator: Série d'indicateur (RSI, MACD, etc.)
            lookback: Période de lookback pour chercher les extrêmes
        
        Returns:
            Dictionnaire avec type de divergence détectée
        """
        if len(price) < lookback:
            return {"divergence": None, "type": "Données insuffisantes"}
        
        # Fenêtres pour comparaison
        recent_price = price.iloc[-lookback:]
        older_price = price.iloc[-2*lookback:-lookback]
        recent_indicator = indicator.iloc[-lookback:]
        older_indicator = indicator.iloc[-2*lookback:-lookback]
        
        # Extrêmes récents vs anciens
        recent_high = recent_price.max()
        older_high = older_price.max()
        recent_low = recent_price.min()
        older_low = older_price.min()
        
        recent_ind_high = recent_indicator.max()
        older_ind_high = older_indicator.max()
        recent_ind_low = recent_indicator.min()
        older_ind_low = older_indicator.min()
        
        # Divergence bearish: prix fait higher high, indicateur fait lower high
        if recent_high > older_high and recent_ind_high < older_ind_high:
            return {
                "divergence": "BEARISH",
                "type": "Regular bearish divergence",
                "description": "Prix en nouveaux sommets mais indicateur en baisse",
                "signal": "SELL",
                "strength": abs(recent_ind_high - older_ind_high)
            }
        
        # Divergence bullish: prix fait lower low, indicateur fait higher low
        if recent_low < older_low and recent_ind_low > older_ind_low:
            return {
                "divergence": "BULLISH",
                "type": "Regular bullish divergence",
                "description": "Prix en nouveaux creux mais indicateur en hausse",
                "signal": "BUY",
                "strength": abs(recent_ind_low - older_ind_low)
            }
        
        # Hidden divergence bearish
        if recent_high < older_high and recent_ind_high > older_ind_high:
            return {
                "divergence": "HIDDEN_BEARISH",
                "type": "Hidden bearish divergence",
                "description": "Continuation baissière probable",
                "signal": "SELL",
                "strength": abs(recent_ind_high - older_ind_high) * 0.7
            }
        
        # Hidden divergence bullish
        if recent_low > older_low and recent_ind_low < older_ind_low:
            return {
                "divergence": "HIDDEN_BULLISH",
                "type": "Hidden bullish divergence",
                "description": "Continuation haussière probable",
                "signal": "BUY",
                "strength": abs(recent_ind_low - older_ind_low) * 0.7
            }
        
        return {"divergence": None, "type": "Aucune divergence détectée"}
    
    def calculate_probability_of_reversal(
        self,
        zscore: float,
        lookback_data: pd.Series = None
    ) -> Dict:
        """
        Calcule la probabilité précise de retournement.
        
        Args:
            zscore: Z-Score actuel
            lookback_data: Données historiques pour calibration
        
        Returns:
            Dictionnaire avec probabilités détaillées
        """
        # Probabilité théorique (distribution normale)
        prob_beyond = 2 * (1 - stats.norm.cdf(abs(zscore)))
        theoretical_reversal = 1 - prob_beyond
        
        # Intervalles de confiance
        result = {
            'zscore': zscore,
            'theoretical_reversal_prob': theoretical_reversal * 100,
            'prob_beyond_current': prob_beyond * 100,
        }
        
        # Probabilités par niveau
        levels = [1, 2, 2.5, 3, 3.5, 4]
        for level in levels:
            prob = 2 * (1 - stats.norm.cdf(level))
            result[f'prob_beyond_{level}sigma'] = prob * 100
        
        # Calibration empirique si données fournies
        if lookback_data is not None and len(lookback_data) > 100:
            historical_zscore = (lookback_data - lookback_data.rolling(self.window).mean()) / \
                               (lookback_data.rolling(self.window).std() + 1e-10)
            
            # Compter les occurrences au-delà du z-score actuel
            exceedances = (abs(historical_zscore) > abs(zscore)).sum()
            empirical_prob = exceedances / len(historical_zscore.dropna())
            
            result['empirical_reversal_prob'] = (1 - empirical_prob) * 100
            result['historical_exceedances'] = exceedances
        
        # Score de confiance final
        if abs(zscore) >= 3:
            result['confidence_level'] = "TRÈS ÉLEVÉ (>90%)"
            result['recommendation'] = "Signal fort de retournement"
        elif abs(zscore) >= 2:
            result['confidence_level'] = "ÉLEVÉ (75-90%)"
            result['recommendation'] = "Signal modéré, attendre confirmation"
        elif abs(zscore) >= 1.5:
            result['confidence_level'] = "MOYEN (60-75%)"
            result['recommendation'] = "Zone d'intérêt, surveiller"
        else:
            result['confidence_level'] = "FAIBLE (<60%)"
            result['recommendation'] = "Prix dans zone normale"
        
        return result
    
    def find_extreme_levels(
        self,
        series: pd.Series,
        threshold: float = 3.0
    ) -> Dict:
        """
        Trouve les niveaux de prix correspondant aux extrêmes statistiques.
        
        Args:
            series: Série de prix
            threshold: Seuil de Z-Score (défaut: 3σ)
        
        Returns:
            Niveaux de prix pour les extrêmes haut et bas
        """
        result = self.calculate(series)
        
        current_sma = result['sma'].iloc[-1]
        current_std = result['std'].iloc[-1]
        current_price = result['price'].iloc[-1]
        
        levels = {
            'current_price': current_price,
            'sma': current_sma,
            'std': current_std,
        }
        
        # Calculer les niveaux d'extrême
        for sigma in [1, 2, 2.5, 3, 3.5]:
            levels[f'upper_{sigma}sigma'] = current_sma + sigma * current_std
            levels[f'lower_{sigma}sigma'] = current_sma - sigma * current_std
        
        # Distance actuelle aux niveaux
        levels['distance_to_upper_3'] = levels['upper_3sigma'] - current_price
        levels['distance_to_lower_3'] = current_price - levels['lower_3sigma']
        
        # En % du prix
        levels['pct_to_upper_3'] = (levels['distance_to_upper_3'] / current_price) * 100
        levels['pct_to_lower_3'] = (levels['distance_to_lower_3'] / current_price) * 100
        
        return levels


if __name__ == "__main__":
    # Test du module
    np.random.seed(42)
    n = 200
    
    # Simuler un prix avec mean-reversion
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    # Ajouter quelques pics extrêmes
    price[50:55] += 8  # Spike up
    price[150:155] -= 8  # Spike down
    
    dates = pd.date_range(start='2024-01-01', periods=n, freq='1H')
    series = pd.Series(price, index=dates, name="Price")
    
    # Calculer Z-Score
    zscore_calc = BollingerZScore()
    result = zscore_calc.calculate(series)
    
    print("=== Test du Z-Score de Bollinger ===\n")
    print("Dernières valeurs:")
    print(result.tail(10))
    
    # Statut actuel
    print("\n=== Statut Actuel ===")
    status = zscore_calc.get_current_status(series)
    for key, value in status.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Probabilité de retournement au pic
    print("\n=== Analyse au pic (index 52) ===")
    zscore_at_peak = result['zscore'].iloc[52]
    prob = zscore_calc.calculate_probability_of_reversal(zscore_at_peak, series)
    for key, value in prob.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Niveaux extrêmes
    print("\n=== Niveaux de Prix Extrêmes ===")
    levels = zscore_calc.find_extreme_levels(series)
    for key, value in levels.items():
        print(f"{key}: {value:.4f}")
