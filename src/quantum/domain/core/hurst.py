"""
Calcul de l'exposant de Hurst pour déterminer le régime de marché.
H > 0.5: Tendance persistante
H < 0.5: Mean-reversion
H ≈ 0.5: Random walk
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from scipy import stats
import sys
import os


from quantum.shared.config.settings import config


class HurstExponent:
    """
    Calcule l'exposant de Hurst pour caractériser le comportement d'une série.
    
    L'exposant de Hurst (H) est une mesure de la mémoire longue:
    - H > 0.5: Série persistante (tendance) - les mouvements sont suivis
    - H < 0.5: Série anti-persistante (mean-reversion) - les mouvements sont inversés
    - H = 0.5: Random walk - aucune mémoire
    
    Méthodes implémentées:
    1. R/S Analysis (Rescaled Range)
    2. DFA (Detrended Fluctuation Analysis)
    3. Variance Ratio
    """
    
    def __init__(self, window: int = None):
        """
        Initialise le calculateur d'exposant de Hurst.
        
        Args:
            window: Taille de la fenêtre pour les calculs rolling
        """
        self.window = window or config.statistical.HURST_WINDOW
        self.trend_threshold = config.statistical.HURST_TREND_THRESHOLD
        self.mean_revert_threshold = config.statistical.HURST_MEAN_REVERT_THRESHOLD
    
    def calculate_rs(self, series: pd.Series) -> float:
        """
        Calcule l'exposant de Hurst par analyse R/S (Rescaled Range).
        
        Méthode classique basée sur les travaux de Hurst sur le Nil.
        
        Args:
            series: Série temporelle de prix ou rendements
        
        Returns:
            Exposant de Hurst (0-1)
        """
        ts = series.dropna().values
        n = len(ts)
        
        if n < 20:
            return 0.5  # Pas assez de données
        
        # Différentes tailles de sous-séries
        max_k = min(n // 2, 100)
        sizes = list(range(10, max_k, max(1, (max_k - 10) // 20)))
        
        if len(sizes) < 3:
            return 0.5
        
        rs_values = []
        
        for size in sizes:
            # Nombre de sous-séries de cette taille
            n_subseries = n // size
            if n_subseries == 0:
                continue
            
            rs_list = []
            
            for i in range(n_subseries):
                subseries = ts[i * size:(i + 1) * size]
                
                # Calculer R/S pour cette sous-série
                mean = np.mean(subseries)
                deviations = subseries - mean
                cumulative_deviations = np.cumsum(deviations)
                
                R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                S = np.std(subseries, ddof=1)
                
                if S > 0:
                    rs_list.append(R / S)
            
            if rs_list:
                rs_values.append((np.log(size), np.log(np.mean(rs_list))))
        
        if len(rs_values) < 3:
            return 0.5
        
        # Régression linéaire log-log
        rs_array = np.array(rs_values)
        slope, _, _, _, _ = stats.linregress(rs_array[:, 0], rs_array[:, 1])
        
        # L'exposant de Hurst est la pente
        hurst = np.clip(slope, 0, 1)
        
        return hurst
    
    def calculate_dfa(self, series: pd.Series, order: int = 1) -> float:
        """
        Calcule l'exposant de Hurst par DFA (Detrended Fluctuation Analysis).
        
        Plus robuste que R/S pour les séries avec tendances.
        
        Args:
            series: Série temporelle
            order: Ordre du polynôme pour le detrending (1=linéaire)
        
        Returns:
            Exposant de Hurst
        """
        ts = series.dropna().values
        n = len(ts)
        
        if n < 50:
            return 0.5
        
        # Intégrer la série (profil cumulatif)
        profile = np.cumsum(ts - np.mean(ts))
        
        # Différentes tailles de fenêtre
        min_window = 10
        max_window = n // 4
        n_windows = min(20, max_window - min_window)
        
        if n_windows < 3:
            return 0.5
        
        window_sizes = np.unique(np.logspace(
            np.log10(min_window),
            np.log10(max_window),
            n_windows
        ).astype(int))
        
        fluctuations = []
        
        for window_size in window_sizes:
            n_segments = n // window_size
            if n_segments == 0:
                continue
            
            F2 = []
            
            for i in range(n_segments):
                segment = profile[i * window_size:(i + 1) * window_size]
                
                # Fit polynomial et calculer résidus
                x = np.arange(window_size)
                coeffs = np.polyfit(x, segment, order)
                trend = np.polyval(coeffs, x)
                
                residuals = segment - trend
                F2.append(np.mean(residuals ** 2))
            
            mean_F2 = np.mean(F2)
            if mean_F2 > 0:
                fluctuations.append((np.log(window_size), 0.5 * np.log(mean_F2)))
        
        if len(fluctuations) < 3:
            return 0.5
        
        # Régression log-log
        fl_array = np.array(fluctuations)
        slope, _, _, _, _ = stats.linregress(fl_array[:, 0], fl_array[:, 1])
        
        hurst = np.clip(slope, 0, 1)
        
        return hurst
    
    def calculate_variance_ratio(self, series: pd.Series, lag: int = 2) -> float:
        """
        Calcule l'exposant de Hurst via le ratio de variance.
        
        Méthode rapide et simple basée sur la théorie que pour un random walk,
        la variance des rendements sur n périodes = n × variance sur 1 période.
        
        Args:
            series: Série de prix
            lag: Décalage pour le calcul du ratio
        
        Returns:
            Exposant de Hurst approximatif
        """
        returns = series.pct_change().dropna()
        
        if len(returns) < lag * 2:
            return 0.5
        
        # Variance des rendements sur 1 période
        var_1 = returns.var()
        
        # Variance des rendements sur lag périodes
        returns_lag = series.pct_change(lag).dropna()
        var_lag = returns_lag.var()
        
        if var_1 == 0:
            return 0.5
        
        # Ratio de variance
        vr = var_lag / (lag * var_1)
        
        # Convertir VR en Hurst approximatif
        # VR = 1 pour random walk (H=0.5)
        # VR > 1 pour trending (H>0.5)
        # VR < 1 pour mean-reverting (H<0.5)
        
        if vr > 0:
            hurst = np.log(vr) / (2 * np.log(lag)) + 0.5
            hurst = np.clip(hurst, 0, 1)
        else:
            hurst = 0.5
        
        return hurst
    
    def calculate(
        self,
        series: pd.Series,
        method: str = "rs"
    ) -> float:
        """
        Calcule l'exposant de Hurst avec la méthode spécifiée.
        
        Args:
            series: Série temporelle
            method: Méthode ("rs", "dfa", "variance_ratio", "all")
        
        Returns:
            Exposant de Hurst
        """
        if method == "rs":
            return self.calculate_rs(series)
        elif method == "dfa":
            return self.calculate_dfa(series)
        elif method == "variance_ratio":
            return self.calculate_variance_ratio(series)
        elif method == "all":
            # Moyenne des 3 méthodes
            h_rs = self.calculate_rs(series)
            h_dfa = self.calculate_dfa(series)
            h_vr = self.calculate_variance_ratio(series)
            return np.mean([h_rs, h_dfa, h_vr])
        else:
            raise ValueError(f"Méthode inconnue: {method}")
    
    def calculate_rolling(
        self,
        series: pd.Series,
        window: int = None,
        method: str = "rs"
    ) -> pd.Series:
        """
        Calcule l'exposant de Hurst en rolling.
        
        Args:
            series: Série temporelle
            window: Taille de la fenêtre
            method: Méthode de calcul
        
        Returns:
            Série des valeurs de Hurst
        """
        window = window or self.window
        
        hurst_values = []
        
        for i in range(len(series)):
            if i < window:
                hurst_values.append(np.nan)
            else:
                subseries = series.iloc[i - window:i]
                h = self.calculate(subseries, method)
                hurst_values.append(h)
        
        return pd.Series(hurst_values, index=series.index, name="hurst")
    
    def get_regime(self, hurst: float) -> str:
        """
        Détermine le régime de marché basé sur l'exposant de Hurst.
        
        Args:
            hurst: Valeur de l'exposant de Hurst
        
        Returns:
            Régime ("TRENDING", "MEAN_REVERTING", "RANDOM")
        """
        if hurst > self.trend_threshold:
            return "TRENDING"
        elif hurst < self.mean_revert_threshold:
            return "MEAN_REVERTING"
        else:
            return "RANDOM"
    
    def get_strategy_recommendation(self, hurst: float) -> Dict:
        """
        Recommande une stratégie basée sur le régime.
        
        Args:
            hurst: Valeur de l'exposant de Hurst
        
        Returns:
            Dictionnaire avec recommandations
        """
        regime = self.get_regime(hurst)
        
        recommendations = {
            "TRENDING": {
                "regime": "TRENDING",
                "hurst": hurst,
                "strategy": "TREND_FOLLOWING",
                "description": "Marché en tendance - Suivre le mouvement",
                "indicators": ["Moving Averages", "MACD", "ADX"],
                "entry": "Breakouts, pullbacks dans la tendance",
                "exit": "Trailing stop, inversion de tendance"
            },
            "MEAN_REVERTING": {
                "regime": "MEAN_REVERTING",
                "hurst": hurst,
                "strategy": "MEAN_REVERSION",
                "description": "Marché oscillant - Acheter les creux, vendre les sommets",
                "indicators": ["RSI", "Bollinger Bands", "Stochastic"],
                "entry": "Extrêmes statistiques, rebonds sur supports/résistances",
                "exit": "Retour à la moyenne"
            },
            "RANDOM": {
                "regime": "RANDOM",
                "hurst": hurst,
                "strategy": "NEUTRAL",
                "description": "Marché aléatoire - Prudence recommandée",
                "indicators": ["Aucun particulier"],
                "entry": "Éviter ou réduire la taille des positions",
                "exit": "Stops serrés"
            }
        }
        
        return recommendations[regime]
    
    def analyze_series(self, series: pd.Series) -> Dict:
        """
        Analyse complète d'une série.
        
        Args:
            series: Série temporelle de prix
        
        Returns:
            Dictionnaire avec toutes les métriques
        """
        h_rs = self.calculate_rs(series)
        h_dfa = self.calculate_dfa(series)
        h_vr = self.calculate_variance_ratio(series)
        h_avg = np.mean([h_rs, h_dfa, h_vr])
        
        regime = self.get_regime(h_avg)
        recommendation = self.get_strategy_recommendation(h_avg)
        
        return {
            "hurst_rs": h_rs,
            "hurst_dfa": h_dfa,
            "hurst_variance_ratio": h_vr,
            "hurst_average": h_avg,
            "regime": regime,
            "recommendation": recommendation
        }


if __name__ == "__main__":
    # Tests du module
    np.random.seed(42)
    n = 500
    
    print("=== Test de l'exposant de Hurst ===\n")
    
    # 1. Random Walk (H ≈ 0.5)
    random_walk = np.cumsum(np.random.randn(n))
    random_series = pd.Series(random_walk, name="Random Walk")
    
    # 2. Trending (H > 0.5)
    trend = np.cumsum(np.random.randn(n) + 0.1)  # Biais positif
    trend_series = pd.Series(trend, name="Trending")
    
    # 3. Mean-reverting (H < 0.5)
    mean_rev = np.zeros(n)
    mean_rev[0] = 0
    for i in range(1, n):
        mean_rev[i] = 0.5 * mean_rev[i-1] + np.random.randn()  # AR(1) avec phi < 1
    mean_rev_series = pd.Series(mean_rev, name="Mean-Reverting")
    
    # Analyser chaque série
    hurst_calc = HurstExponent()
    
    for name, series in [
        ("Random Walk", random_series),
        ("Trending", trend_series),
        ("Mean-Reverting", mean_rev_series)
    ]:
        print(f"--- {name} ---")
        analysis = hurst_calc.analyze_series(series)
        print(f"  Hurst R/S: {analysis['hurst_rs']:.3f}")
        print(f"  Hurst DFA: {analysis['hurst_dfa']:.3f}")
        print(f"  Hurst VR:  {analysis['hurst_variance_ratio']:.3f}")
        print(f"  Moyenne:   {analysis['hurst_average']:.3f}")
        print(f"  Régime:    {analysis['regime']}")
        print(f"  Stratégie: {analysis['recommendation']['strategy']}")
        print()
