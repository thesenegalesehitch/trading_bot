"""
Module de calcul du Value at Risk (VaR).
Mesure le risque de perte maximale sur un horizon donné.

Méthodes implémentées:
1. VaR Historique - Basé sur la distribution empirique
2. VaR Paramétrique - Gaussian (normal)
3. VaR Monte Carlo - Simulation de scénarios
4. CVaR (Conditional VaR) - Expected Shortfall
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import config


@dataclass
class VaRResult:
    """Résultat d'un calcul de VaR."""
    var_value: float           # Perte maximale (valeur absolue)
    var_percent: float         # En pourcentage
    confidence_level: float    # Niveau de confiance
    horizon_days: int          # Horizon en jours
    method: str                # Méthode utilisée
    cvar: Optional[float]      # Conditional VaR si calculé
    details: Optional[Dict]    # Détails supplémentaires


class VaRCalculator:
    """
    Calcule le Value at Risk avec plusieurs méthodes.
    
    Le VaR répond à la question:
    "Quelle est la perte maximale avec X% de confiance sur Y jours?"
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        horizon_days: int = 1
    ):
        self.confidence_level = confidence_level
        self.horizon_days = horizon_days
    
    def calculate_historical_var(
        self,
        returns: pd.Series,
        portfolio_value: float = 10000
    ) -> VaRResult:
        """
        VaR Historique - Utilise la distribution empirique des rendements.
        
        Simple et ne fait pas d'hypothèse sur la distribution.
        
        Args:
            returns: Série des rendements
            portfolio_value: Valeur du portefeuille
        
        Returns:
            VaRResult
        """
        returns = returns.dropna()
        
        if len(returns) < 30:
            return VaRResult(
                var_value=0,
                var_percent=0,
                confidence_level=self.confidence_level,
                horizon_days=self.horizon_days,
                method='historical',
                cvar=None,
                details={'error': 'Données insuffisantes'}
            )
        
        # Percentile correspondant au niveau de confiance
        var_percentile = (1 - self.confidence_level) * 100
        var_return = np.percentile(returns, var_percentile)
        
        # Ajuster pour l'horizon (racine du temps pour les rendements)
        var_return_adjusted = var_return * np.sqrt(self.horizon_days)
        
        # VaR en valeur absolue
        var_value = abs(var_return_adjusted * portfolio_value)
        
        # CVaR (Expected Shortfall) - Moyenne des pertes au-delà du VaR
        cvar_returns = returns[returns <= var_return]
        cvar_return = cvar_returns.mean() if len(cvar_returns) > 0 else var_return
        cvar_value = abs(cvar_return * np.sqrt(self.horizon_days) * portfolio_value)
        
        return VaRResult(
            var_value=round(var_value, 2),
            var_percent=round(abs(var_return_adjusted) * 100, 4),
            confidence_level=self.confidence_level,
            horizon_days=self.horizon_days,
            method='historical',
            cvar=round(cvar_value, 2),
            details={
                'n_observations': len(returns),
                'mean_return': float(returns.mean()),
                'std_return': float(returns.std()),
                'min_return': float(returns.min()),
                'max_return': float(returns.max())
            }
        )
    
    def calculate_parametric_var(
        self,
        returns: pd.Series,
        portfolio_value: float = 10000
    ) -> VaRResult:
        """
        VaR Paramétrique (Gaussien) - Assume une distribution normale.
        
        Plus simple mais moins précis pour les distributions à queues épaisses.
        
        Args:
            returns: Série des rendements
            portfolio_value: Valeur du portefeuille
        
        Returns:
            VaRResult
        """
        returns = returns.dropna()
        
        if len(returns) < 30:
            return VaRResult(
                var_value=0,
                var_percent=0,
                confidence_level=self.confidence_level,
                horizon_days=self.horizon_days,
                method='parametric',
                cvar=None,
                details={'error': 'Données insuffisantes'}
            )
        
        # Paramètres de la distribution
        mean = returns.mean()
        std = returns.std()
        
        # Z-score pour le niveau de confiance
        z_score = stats.norm.ppf(1 - self.confidence_level)
        
        # VaR
        var_return = mean + z_score * std * np.sqrt(self.horizon_days)
        var_value = abs(var_return * portfolio_value)
        
        # CVaR paramétrique
        # Pour une distribution normale: CVaR = μ - σ * φ(z) / (1-α)
        phi_z = stats.norm.pdf(z_score)
        cvar_return = mean - std * np.sqrt(self.horizon_days) * phi_z / (1 - self.confidence_level)
        cvar_value = abs(cvar_return * portfolio_value)
        
        return VaRResult(
            var_value=round(var_value, 2),
            var_percent=round(abs(var_return) * 100, 4),
            confidence_level=self.confidence_level,
            horizon_days=self.horizon_days,
            method='parametric',
            cvar=round(cvar_value, 2),
            details={
                'mean': float(mean),
                'std': float(std),
                'z_score': float(z_score),
                'skewness': float(stats.skew(returns)),
                'kurtosis': float(stats.kurtosis(returns))
            }
        )
    
    def calculate_monte_carlo_var(
        self,
        returns: pd.Series,
        portfolio_value: float = 10000,
        n_simulations: int = 10000
    ) -> VaRResult:
        """
        VaR Monte Carlo - Simule de nombreux scénarios possibles.
        
        Plus flexible, peut modéliser des distributions complexes.
        
        Args:
            returns: Série des rendements
            portfolio_value: Valeur du portefeuille
            n_simulations: Nombre de simulations
        
        Returns:
            VaRResult
        """
        returns = returns.dropna()
        
        if len(returns) < 30:
            return VaRResult(
                var_value=0,
                var_percent=0,
                confidence_level=self.confidence_level,
                horizon_days=self.horizon_days,
                method='monte_carlo',
                cvar=None,
                details={'error': 'Données insuffisantes'}
            )
        
        mean = returns.mean()
        std = returns.std()
        
        # Simuler les rendements
        np.random.seed(42)
        
        # Utiliser une distribution t de Student pour capturer les queues épaisses
        # Estimer les degrés de liberté
        df_estimate = 2 + 6 / (stats.kurtosis(returns) + 3) if stats.kurtosis(returns) > -3 else 30
        df_estimate = max(3, min(30, df_estimate))  # Borner entre 3 et 30
        
        # Générer les simulations
        simulated_returns = stats.t.rvs(
            df=df_estimate,
            loc=mean * self.horizon_days,
            scale=std * np.sqrt(self.horizon_days),
            size=n_simulations
        )
        
        # Calculer le VaR
        var_percentile = (1 - self.confidence_level) * 100
        var_return = np.percentile(simulated_returns, var_percentile)
        var_value = abs(var_return * portfolio_value)
        
        # CVaR
        cvar_returns = simulated_returns[simulated_returns <= var_return]
        cvar_return = cvar_returns.mean() if len(cvar_returns) > 0 else var_return
        cvar_value = abs(cvar_return * portfolio_value)
        
        return VaRResult(
            var_value=round(var_value, 2),
            var_percent=round(abs(var_return) * 100, 4),
            confidence_level=self.confidence_level,
            horizon_days=self.horizon_days,
            method='monte_carlo',
            cvar=round(cvar_value, 2),
            details={
                'n_simulations': n_simulations,
                'df_estimate': float(df_estimate),
                'simulated_mean': float(simulated_returns.mean()),
                'simulated_std': float(simulated_returns.std()),
                'worst_case': float(simulated_returns.min()) * portfolio_value
            }
        )
    
    def calculate_all_methods(
        self,
        returns: pd.Series,
        portfolio_value: float = 10000
    ) -> Dict[str, VaRResult]:
        """
        Calcule le VaR avec toutes les méthodes.
        
        Returns:
            Dict avec résultats par méthode
        """
        return {
            'historical': self.calculate_historical_var(returns, portfolio_value),
            'parametric': self.calculate_parametric_var(returns, portfolio_value),
            'monte_carlo': self.calculate_monte_carlo_var(returns, portfolio_value)
        }
    
    def get_risk_metrics(
        self,
        returns: pd.Series,
        portfolio_value: float = 10000
    ) -> Dict:
        """
        Calcule un ensemble complet de métriques de risque.
        
        Returns:
            Dict avec toutes les métriques
        """
        returns = returns.dropna()
        
        if len(returns) < 30:
            return {'error': 'Données insuffisantes'}
        
        # VaR avec différentes méthodes
        var_results = self.calculate_all_methods(returns, portfolio_value)
        
        # Métriques additionnelles
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        sortino = returns.mean() / returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_dd = drawdowns.min()
        
        # Calmar Ratio
        calmar = (returns.mean() * 252) / abs(max_dd) if max_dd != 0 else 0
        
        return {
            'var_95_1d': var_results['historical'].var_value,
            'cvar_95_1d': var_results['historical'].cvar,
            'var_methods': {
                'historical': var_results['historical'].var_percent,
                'parametric': var_results['parametric'].var_percent,
                'monte_carlo': var_results['monte_carlo'].var_percent
            },
            'sharpe_ratio': round(sharpe, 3),
            'sortino_ratio': round(sortino, 3),
            'max_drawdown': round(max_dd * 100, 2),
            'calmar_ratio': round(calmar, 3),
            'volatility_annual': round(returns.std() * np.sqrt(252) * 100, 2),
            'skewness': round(float(stats.skew(returns)), 3),
            'kurtosis': round(float(stats.kurtosis(returns)), 3),
            'portfolio_value': portfolio_value
        }


class KellyCriterion:
    """
    Calcul du Kelly Criterion pour le position sizing optimal.
    
    Le Kelly Criterion maximise la croissance du capital à long terme.
    
    Formule: f* = (bp - q) / b
    où:
    - f* = fraction du capital à risquer
    - b = odds (gain moyen / perte moyenne)
    - p = probabilité de gain
    - q = probabilité de perte (1-p)
    """
    
    def __init__(
        self,
        use_fractional: bool = True,
        fraction: float = 0.5  # Demi-Kelly par défaut
    ):
        self.use_fractional = use_fractional
        self.fraction = fraction  # 0.5 = demi-Kelly, 0.25 = quart-Kelly
    
    def calculate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> Dict:
        """
        Calcule le Kelly Criterion.
        
        Args:
            win_rate: Taux de réussite (0-1)
            avg_win: Gain moyen
            avg_loss: Perte moyenne (valeur positive)
        
        Returns:
            Dict avec Kelly et recommandations
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return {
                'kelly': 0,
                'fractional_kelly': 0,
                'recommendation': "Données invalides",
                'is_positive_expectancy': False
            }
        
        # Probabilités
        p = win_rate
        q = 1 - win_rate
        
        # Odds (risk/reward ratio inversé)
        b = avg_win / avg_loss
        
        # Full Kelly
        kelly = (b * p - q) / b
        
        # Fractional Kelly (plus conservateur)
        fractional_kelly = kelly * self.fraction if self.use_fractional else kelly
        
        # Limiter entre 0 et 1
        kelly = max(0, min(1, kelly))
        fractional_kelly = max(0, min(1, fractional_kelly))
        
        # Espérance mathématique
        expectancy = p * avg_win - q * avg_loss
        is_positive = expectancy > 0
        
        # Recommandation
        if kelly <= 0:
            recommendation = "NE PAS TRADER - Espérance négative"
        elif fractional_kelly < 0.01:
            recommendation = "Risque minime recommandé (<1%)"
        elif fractional_kelly < 0.05:
            recommendation = f"Risque conservateur: {fractional_kelly*100:.1f}% du capital"
        elif fractional_kelly < 0.15:
            recommendation = f"Risque modéré: {fractional_kelly*100:.1f}% du capital"
        else:
            recommendation = f"Risque élevé: {fractional_kelly*100:.1f}% du capital (prudence!)"
        
        return {
            'kelly': round(kelly * 100, 2),  # En pourcentage
            'fractional_kelly': round(fractional_kelly * 100, 2),
            'recommended_risk_percent': round(min(fractional_kelly * 100, 5), 2),  # Max 5%
            'expectancy': round(expectancy, 4),
            'profit_factor': round(b * p / q, 2) if q > 0 else 0,
            'is_positive_expectancy': is_positive,
            'recommendation': recommendation,
            'inputs': {
                'win_rate': round(win_rate * 100, 2),
                'avg_win': round(avg_win, 4),
                'avg_loss': round(avg_loss, 4),
                'risk_reward': round(b, 2)
            }
        }
    
    def calculate_from_trades(self, trades: pd.DataFrame) -> Dict:
        """
        Calcule le Kelly à partir d'un historique de trades.
        
        Args:
            trades: DataFrame avec colonne 'pnl' (profit/perte)
        
        Returns:
            Dict avec Kelly et statistiques
        """
        if 'pnl' not in trades.columns or len(trades) < 10:
            return {'error': 'Données insuffisantes'}
        
        pnl = trades['pnl']
        
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return {'error': 'Pas assez de wins ou losses'}
        
        win_rate = len(wins) / len(pnl)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        result = self.calculate(win_rate, avg_win, avg_loss)
        
        # Ajouter des statistiques
        result['trade_stats'] = {
            'total_trades': len(pnl),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'largest_win': float(wins.max()),
            'largest_loss': float(losses.min()),
            'consecutive_losses_max': self._max_consecutive(pnl < 0)
        }
        
        return result
    
    def _max_consecutive(self, series: pd.Series) -> int:
        """Calcule le max de valeurs True consécutives."""
        max_count = 0
        current_count = 0
        
        for val in series:
            if val:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def dynamic_kelly(
        self,
        base_kelly: float,
        current_drawdown: float,
        max_drawdown_limit: float = 0.20
    ) -> float:
        """
        Kelly dynamique ajusté selon le drawdown actuel.
        
        Réduit le risque quand le drawdown augmente.
        
        Args:
            base_kelly: Kelly de base (en décimal)
            current_drawdown: Drawdown actuel (en décimal, ex: 0.05 = 5%)
            max_drawdown_limit: Limite de drawdown maximale
        
        Returns:
            Kelly ajusté
        """
        if current_drawdown >= max_drawdown_limit:
            return 0  # Arrêter de trader
        
        # Réduction linéaire du Kelly basée sur le drawdown
        # À 0% DD -> 100% du Kelly
        # À max DD -> 0% du Kelly
        reduction_factor = 1 - (current_drawdown / max_drawdown_limit)
        
        return base_kelly * reduction_factor


if __name__ == "__main__":
    print("=" * 60)
    print("TEST VALUE AT RISK & KELLY CRITERION")
    print("=" * 60)
    
    # Générer des données de test
    np.random.seed(42)
    n = 500
    
    # Rendements simulés (légèrement positifs avec volatilité)
    returns = pd.Series(np.random.normal(0.0002, 0.015, n))
    
    # Test VaR
    print("\n--- Value at Risk ---")
    var_calc = VaRCalculator(confidence_level=0.95, horizon_days=1)
    
    var_hist = var_calc.calculate_historical_var(returns, portfolio_value=10000)
    print(f"VaR Historique (95%, 1 jour): ${var_hist.var_value} ({var_hist.var_percent}%)")
    print(f"CVaR (Expected Shortfall): ${var_hist.cvar}")
    
    var_mc = var_calc.calculate_monte_carlo_var(returns, portfolio_value=10000)
    print(f"VaR Monte Carlo: ${var_mc.var_value} ({var_mc.var_percent}%)")
    
    # Métriques complètes
    print("\n--- Métriques de Risque Complètes ---")
    metrics = var_calc.get_risk_metrics(returns, portfolio_value=10000)
    for k, v in metrics.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")
    
    # Test Kelly
    print("\n--- Kelly Criterion ---")
    kelly = KellyCriterion(use_fractional=True, fraction=0.5)
    
    result = kelly.calculate(
        win_rate=0.55,
        avg_win=0.02,
        avg_loss=0.015
    )
    
    print(f"Full Kelly: {result['kelly']}%")
    print(f"Demi-Kelly: {result['fractional_kelly']}%")
    print(f"Risque recommandé: {result['recommended_risk_percent']}%")
    print(f"Espérance: {result['expectancy']}")
    print(f"Recommandation: {result['recommendation']}")
    
    # Kelly dynamique
    print("\n--- Kelly Dynamique ---")
    base_kelly = result['fractional_kelly'] / 100
    for dd in [0, 0.05, 0.10, 0.15, 0.20]:
        adjusted = kelly.dynamic_kelly(base_kelly, dd) * 100
        print(f"  Drawdown {dd*100:.0f}% -> Kelly ajusté: {adjusted:.2f}%")
