"""
Analyse de co-intégration entre EUR/USD et XAU/USD.
Détecte les opportunités d'arbitrage statistique.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from scipy import stats
import sys
import os


from quantum.shared.config.settings import config


class CointegrationAnalyzer:
    """
    Analyse la relation de co-intégration entre deux actifs.
    
    La co-intégration signifie que deux séries non-stationnaires
    ont une combinaison linéaire stationnaire. Quand cette relation
    dévie de sa moyenne, il y a une opportunité d'arbitrage.
    
    Tests utilisés:
    1. Test d'Engle-Granger (2 étapes)
    2. Test de Johansen (si statsmodels disponible)
    """
    
    def __init__(self, lookback: int = None):
        """
        Initialise l'analyseur de co-intégration.
        
        Args:
            lookback: Période de lookback pour l'analyse
        """
        self.lookback = lookback or config.statistical.COINTEGRATION_LOOKBACK
        self.pvalue_threshold = config.statistical.COINTEGRATION_PVALUE_THRESHOLD
        
        # Résultats stockés
        self.beta = None  # Coefficient de régression
        self.spread = None  # Série du spread
        self.spread_mean = None
        self.spread_std = None
        self.is_cointegrated = False
        self.pvalue = None
    
    def test_cointegration(
        self,
        series1: pd.Series,
        series2: pd.Series,
        method: str = "engle_granger"
    ) -> Dict:
        """
        Teste la co-intégration entre deux séries.
        
        Args:
            series1: Première série (ex: EUR/USD)
            series2: Deuxième série (ex: XAU/USD)
            method: Méthode de test ("engle_granger" ou "johansen")
        
        Returns:
            Dictionnaire avec résultats du test
        """
        # Aligner les séries
        aligned1, aligned2 = self._align_series(series1, series2)
        
        if len(aligned1) < self.lookback:
            return {
                "is_cointegrated": False,
                "error": "Données insuffisantes"
            }
        
        # Utiliser seulement la période de lookback
        s1 = aligned1.tail(self.lookback)
        s2 = aligned2.tail(self.lookback)
        
        if method == "engle_granger":
            return self._engle_granger_test(s1, s2)
        elif method == "johansen":
            return self._johansen_test(s1, s2)
        else:
            raise ValueError(f"Méthode inconnue: {method}")
    
    def _engle_granger_test(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> Dict:
        """
        Test de co-intégration d'Engle-Granger en 2 étapes.
        
        Étape 1: Régression OLS de series1 sur series2
        Étape 2: Test ADF sur les résidus
        """
        # Étape 1: Régression linéaire
        X = series2.values.reshape(-1, 1)
        y = series1.values
        
        # OLS: y = alpha + beta * X + epsilon
        X_with_const = np.column_stack([np.ones(len(X)), X])
        coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        alpha, beta = coeffs[0], coeffs[1]
        
        # Calculer les résidus (spread)
        residuals = y - (alpha + beta * X.flatten())
        
        # Étape 2: Test ADF sur les résidus
        adf_result = self._adf_test(residuals)
        
        # Stocker les résultats
        self.beta = beta
        self.spread = pd.Series(residuals, index=series1.index)
        self.spread_mean = np.mean(residuals)
        self.spread_std = np.std(residuals)
        self.pvalue = adf_result['pvalue']
        self.is_cointegrated = adf_result['pvalue'] < self.pvalue_threshold
        
        return {
            "is_cointegrated": self.is_cointegrated,
            "pvalue": self.pvalue,
            "beta": beta,
            "alpha": alpha,
            "adf_statistic": adf_result['statistic'],
            "critical_values": adf_result['critical_values'],
            "spread_mean": self.spread_mean,
            "spread_std": self.spread_std
        }
    
    def _adf_test(self, series: np.ndarray) -> Dict:
        """
        Test de Dickey-Fuller Augmenté simplifié.
        Teste si une série est stationnaire.
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series, autolag='AIC')
            return {
                'statistic': result[0],
                'pvalue': result[1],
                'critical_values': result[4]
            }
        except ImportError:
            # Implémentation simplifiée si statsmodels non disponible
            return self._simple_adf(series)
    
    def _simple_adf(self, series: np.ndarray) -> Dict:
        """
        ADF simplifié sans statsmodels.
        Régression: Δy(t) = γ*y(t-1) + ε(t)
        H0: γ = 0 (non-stationnaire)
        """
        y = series[1:]
        y_lag = series[:-1]
        dy = np.diff(series)
        
        # Régression de Δy sur y(t-1)
        X = y_lag.reshape(-1, 1)
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        try:
            coeffs = np.linalg.lstsq(X_with_const, dy, rcond=None)[0]
            gamma = coeffs[1]
            
            # Calcul de la statistique t
            residuals = dy - X_with_const @ coeffs
            mse = np.sum(residuals**2) / (len(residuals) - 2)
            
            XtX_inv = np.linalg.inv(X_with_const.T @ X_with_const)
            se_gamma = np.sqrt(mse * XtX_inv[1, 1])
            
            t_stat = gamma / se_gamma
            
            # Valeurs critiques approximatives pour ADF
            # (basées sur tables de MacKinnon pour n=250)
            critical_values = {
                '1%': -3.43,
                '5%': -2.86,
                '10%': -2.57
            }
            
            # p-value approximative
            if t_stat < -3.43:
                pvalue = 0.01
            elif t_stat < -2.86:
                pvalue = 0.05
            elif t_stat < -2.57:
                pvalue = 0.10
            else:
                pvalue = 0.5  # Non stationnaire
            
            return {
                'statistic': t_stat,
                'pvalue': pvalue,
                'critical_values': critical_values
            }
        except:
            return {
                'statistic': 0,
                'pvalue': 1.0,
                'critical_values': {}
            }
    
    def _johansen_test(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> Dict:
        """
        Test de co-intégration de Johansen.
        Plus puissant qu'Engle-Granger pour plusieurs séries.
        """
        try:
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
            
            data = np.column_stack([series1.values, series2.values])
            result = coint_johansen(data, det_order=0, k_ar_diff=1)
            
            # Vérifier les valeurs propres
            trace_stat = result.lr1[0]  # Statistique de trace
            critical_value = result.cvt[0, 1]  # Valeur critique 5%
            
            self.is_cointegrated = trace_stat > critical_value
            
            return {
                "is_cointegrated": self.is_cointegrated,
                "trace_statistic": trace_stat,
                "critical_value_5pct": critical_value,
                "eigenvalues": result.eig.tolist()
            }
        except ImportError:
            # Fallback sur Engle-Granger
            return self._engle_granger_test(series1, series2)
    
    def calculate_spread(
        self,
        series1: pd.Series,
        series2: pd.Series,
        beta: float = None
    ) -> pd.Series:
        """
        Calcule le spread entre deux séries.
        
        Args:
            series1: Première série
            series2: Deuxième série
            beta: Coefficient de régression (calculé si None)
        
        Returns:
            Série du spread
        """
        aligned1, aligned2 = self._align_series(series1, series2)
        
        if beta is None:
            if self.beta is None:
                # Calculer beta par régression
                coeffs = np.polyfit(aligned2.values, aligned1.values, 1)
                beta = coeffs[0]
            else:
                beta = self.beta
        
        spread = aligned1 - beta * aligned2
        return spread
    
    def get_zscore_of_spread(self, spread: pd.Series = None) -> pd.Series:
        """
        Calcule le Z-score du spread.
        
        Args:
            spread: Série du spread (utilise self.spread si None)
        
        Returns:
            Série du Z-score
        """
        spread = spread if spread is not None else self.spread
        
        if spread is None:
            raise ValueError("Aucun spread calculé. Exécutez test_cointegration d'abord.")
        
        mean = spread.rolling(window=self.lookback).mean()
        std = spread.rolling(window=self.lookback).std()
        
        zscore = (spread - mean) / (std + 1e-10)
        return zscore
    
    def detect_arbitrage_opportunity(
        self,
        series1: pd.Series,
        series2: pd.Series,
        zscore_threshold: float = 2.0
    ) -> Dict:
        """
        Détecte les opportunités d'arbitrage basées sur la co-intégration.
        
        Args:
            series1: Première série
            series2: Deuxième série
            zscore_threshold: Seuil de Z-score pour signal
        
        Returns:
            Dictionnaire avec signal et métriques
        """
        # Tester la co-intégration
        coint_result = self.test_cointegration(series1, series2)
        
        if not coint_result.get("is_cointegrated", False):
            return {
                "signal": "NONE",
                "reason": "Pas de co-intégration détectée",
                "coint_pvalue": coint_result.get("pvalue", 1.0)
            }
        
        # Calculer le Z-score du spread actuel
        zscore = self.get_zscore_of_spread()
        current_zscore = zscore.iloc[-1]
        
        # Déterminer le signal
        if current_zscore > zscore_threshold:
            signal = "SHORT_SPREAD"  # Spread trop haut, va revenir
            action = f"SHORT {series1.name}, LONG {series2.name}"
        elif current_zscore < -zscore_threshold:
            signal = "LONG_SPREAD"  # Spread trop bas, va revenir
            action = f"LONG {series1.name}, SHORT {series2.name}"
        else:
            signal = "NEUTRAL"
            action = "Attendre meilleure opportunité"
        
        return {
            "signal": signal,
            "action": action,
            "current_zscore": current_zscore,
            "zscore_threshold": zscore_threshold,
            "spread_beta": self.beta,
            "coint_pvalue": self.pvalue,
            "is_cointegrated": True
        }
    
    def _align_series(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Aligne deux séries sur leur index commun."""
        common_index = series1.index.intersection(series2.index)
        return series1.loc[common_index], series2.loc[common_index]
    
    def get_half_life(self) -> float:
        """
        Calcule la demi-vie du mean reversion du spread.
        Indique combien de temps le spread met pour revenir à moitié vers sa moyenne.
        """
        if self.spread is None:
            raise ValueError("Aucun spread calculé")
        
        spread = self.spread.values
        
        # Régression AR(1): spread(t) = phi * spread(t-1) + noise
        spread_lag = spread[:-1]
        spread_diff = spread[1:]
        
        # phi = 1 - lambda, où lambda est le taux de mean reversion
        coeffs = np.polyfit(spread_lag, spread_diff - spread_lag, 1)
        lambda_mr = -coeffs[0]
        
        if lambda_mr <= 0:
            return float('inf')  # Pas de mean reversion
        
        half_life = np.log(2) / lambda_mr
        return half_life


if __name__ == "__main__":
    # Test du module
    np.random.seed(42)
    
    # Créer deux séries co-intégrées synthétiques
    n = 500
    
    # Série commune (facteur commun)
    common = np.cumsum(np.random.randn(n) * 0.1)
    
    # Deux séries avec le même facteur commun + bruit
    series1 = common + np.random.randn(n) * 0.5 + 10
    series2 = 2 * common + np.random.randn(n) * 0.5 + 20
    
    dates = pd.date_range(start='2023-01-01', periods=n, freq='1H')
    s1 = pd.Series(series1, index=dates, name="EUR/USD")
    s2 = pd.Series(series2, index=dates, name="XAU/USD")
    
    # Tester la co-intégration
    analyzer = CointegrationAnalyzer(lookback=252)
    result = analyzer.test_cointegration(s1, s2)
    
    print("=== Résultats du test de co-intégration ===")
    for key, value in result.items():
        print(f"{key}: {value}")
    
    # Détecter opportunité d'arbitrage
    print("\n=== Détection d'arbitrage ===")
    arb = analyzer.detect_arbitrage_opportunity(s1, s2)
    for key, value in arb.items():
        print(f"{key}: {value}")
    
    # Demi-vie
    half_life = analyzer.get_half_life()
    print(f"\nDemi-vie du mean reversion: {half_life:.1f} périodes")
