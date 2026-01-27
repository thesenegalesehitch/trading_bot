"""
Module de simulation Monte Carlo pour le backtesting.
Génère des milliers de scénarios pour évaluer la robustesse d'une stratégie.

Fonctionnalités:
- Simulation de trajectoires de prix
- Bootstrap des rendements historiques  
- Intervalles de confiance sur les métriques
- Distribution des résultats possibles
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats
import sys
import os


from quantum.shared.config.settings import config


@dataclass
class MonteCarloResult:
    """Résultats de la simulation Monte Carlo."""
    n_simulations: int
    mean_return: float
    median_return: float
    std_return: float
    confidence_interval_95: Tuple[float, float]
    probability_profit: float
    probability_target: float
    max_drawdown_mean: float
    sharpe_mean: float
    var_95: float
    results_distribution: np.ndarray


class MonteCarloSimulator:
    """
    Simulateur Monte Carlo pour l'évaluation de stratégies.
    
    Méthodes:
    1. Simulation de prix par processus stochastique (GBM)
    2. Bootstrap des rendements historiques
    3. Permutation de trades
    """
    
    def __init__(
        self,
        n_simulations: int = 10000,
        random_seed: int = 42
    ):
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def simulate_returns_gbm(
        self,
        initial_value: float,
        n_periods: int,
        mu: float,
        sigma: float
    ) -> np.ndarray:
        """
        Simule des trajectoires de prix avec Geometric Brownian Motion.
        
        dS = μSdt + σSdW
        
        Args:
            initial_value: Valeur initiale
            n_periods: Nombre de périodes
            mu: Drift (rendement moyen)
            sigma: Volatilité
        
        Returns:
            Array de shape (n_simulations, n_periods)
        """
        dt = 1  # 1 période
        
        # Générer les chocs aléatoires
        Z = np.random.standard_normal((self.n_simulations, n_periods))
        
        # Calculer les rendements log
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        
        # Convertir en prix
        log_prices = np.zeros((self.n_simulations, n_periods + 1))
        log_prices[:, 0] = np.log(initial_value)
        log_prices[:, 1:] = log_prices[:, 0:1] + np.cumsum(log_returns, axis=1)
        
        return np.exp(log_prices)
    
    def bootstrap_returns(
        self,
        historical_returns: np.ndarray,
        n_periods: int,
        block_size: int = 5
    ) -> np.ndarray:
        """
        Bootstrap des rendements historiques (block bootstrap).
        
        Préserve une partie de l'autocorrélation.
        
        Args:
            historical_returns: Rendements historiques
            n_periods: Nombre de périodes à simuler
            block_size: Taille des blocs pour le bootstrap
        
        Returns:
            Array de shape (n_simulations, n_periods)
        """
        n_hist = len(historical_returns)
        n_blocks = int(np.ceil(n_periods / block_size))
        
        simulated = np.zeros((self.n_simulations, n_periods))
        
        for sim in range(self.n_simulations):
            blocks = []
            for _ in range(n_blocks):
                # Choisir un point de départ aléatoire
                start = np.random.randint(0, n_hist - block_size)
                blocks.append(historical_returns[start:start + block_size])
            
            # Concaténer et tronquer
            full_sim = np.concatenate(blocks)[:n_periods]
            simulated[sim, :] = full_sim
        
        return simulated
    
    def permute_trades(
        self,
        trades_pnl: np.ndarray
    ) -> np.ndarray:
        """
        Permute l'ordre des trades pour évaluer la dépendance à la séquence.
        
        Args:
            trades_pnl: Array des P&L des trades
        
        Returns:
            Array de shape (n_simulations, n_trades) avec equity curves
        """
        n_trades = len(trades_pnl)
        
        equity_curves = np.zeros((self.n_simulations, n_trades + 1))
        equity_curves[:, 0] = 1.0  # Capital initial normalisé
        
        for sim in range(self.n_simulations):
            # Permuter les trades
            shuffled = np.random.permutation(trades_pnl)
            
            # Calculer l'equity curve
            for i, pnl in enumerate(shuffled):
                equity_curves[sim, i + 1] = equity_curves[sim, i] * (1 + pnl)
        
        return equity_curves
    
    def run_strategy_monte_carlo(
        self,
        strategy_func: Callable[[pd.DataFrame], Tuple[pd.Series, pd.Series]],
        price_data: pd.DataFrame,
        initial_capital: float = 10000,
        target_return: float = 0.20  # 20% target
    ) -> MonteCarloResult:
        """
        Exécute une simulation Monte Carlo complète d'une stratégie.
        
        Args:
            strategy_func: Fonction qui prend les prix et retourne (entries, exits)
            price_data: DataFrame avec prix historiques
            initial_capital: Capital initial
            target_return: Rendement cible
        
        Returns:
            MonteCarloResult avec statistiques complètes
        """
        # Calculer les rendements historiques
        returns = price_data['Close'].pct_change().dropna().values
        
        n_periods = len(returns)
        mu = returns.mean()
        sigma = returns.std()
        
        # Simuler des trajectoires de prix
        simulated_prices = self.simulate_returns_gbm(
            price_data['Close'].iloc[0],
            n_periods,
            mu * 252,  # Annualiser
            sigma * np.sqrt(252)
        )
        
        # Évaluer la stratégie sur chaque trajectoire
        final_values = []
        max_drawdowns = []
        sharpe_ratios = []
        
        for sim_idx in range(min(self.n_simulations, 1000)):  # Limiter pour la performance
            # Créer un DataFrame pour cette simulation
            sim_df = pd.DataFrame({
                'Open': simulated_prices[sim_idx, :-1],
                'High': simulated_prices[sim_idx, 1:] * 1.005,
                'Low': simulated_prices[sim_idx, 1:] * 0.995,
                'Close': simulated_prices[sim_idx, 1:],
                'Volume': 1000
            })
            
            try:
                # Exécuter la stratégie
                entries, exits = strategy_func(sim_df)
                
                # Backtester simplifié
                result = self._simple_backtest(sim_df, entries, exits, initial_capital)
                
                final_values.append(result['final_value'])
                max_drawdowns.append(result['max_drawdown'])
                sharpe_ratios.append(result['sharpe'])
                
            except Exception:
                # En cas d'erreur, utiliser les valeurs par défaut
                final_values.append(initial_capital)
                max_drawdowns.append(0)
                sharpe_ratios.append(0)
        
        # Compléter avec bootstrap si nécessaire
        if len(final_values) < self.n_simulations:
            bootstrap_returns = self.bootstrap_returns(returns, n_periods)
            
            for sim_idx in range(len(final_values), self.n_simulations):
                # Calculer le rendement total de cette simulation
                total_return = np.prod(1 + bootstrap_returns[sim_idx - len(final_values)]) - 1
                final_values.append(initial_capital * (1 + total_return))
        
        final_values = np.array(final_values)
        max_drawdowns = np.array(max_drawdowns) if max_drawdowns else np.zeros(len(final_values))
        sharpe_ratios = np.array(sharpe_ratios) if sharpe_ratios else np.zeros(len(final_values))
        
        # Calculer les statistiques
        returns_pct = (final_values - initial_capital) / initial_capital * 100
        
        return MonteCarloResult(
            n_simulations=self.n_simulations,
            mean_return=round(returns_pct.mean(), 2),
            median_return=round(np.median(returns_pct), 2),
            std_return=round(returns_pct.std(), 2),
            confidence_interval_95=(
                round(np.percentile(returns_pct, 2.5), 2),
                round(np.percentile(returns_pct, 97.5), 2)
            ),
            probability_profit=round((returns_pct > 0).mean() * 100, 2),
            probability_target=round((returns_pct >= target_return * 100).mean() * 100, 2),
            max_drawdown_mean=round(max_drawdowns.mean() * 100, 2) if len(max_drawdowns) > 0 else 0,
            sharpe_mean=round(sharpe_ratios[sharpe_ratios != 0].mean(), 2) if len(sharpe_ratios[sharpe_ratios != 0]) > 0 else 0,
            var_95=round(np.percentile(returns_pct, 5), 2),
            results_distribution=returns_pct
        )
    
    def _simple_backtest(
        self,
        df: pd.DataFrame,
        entries: pd.Series,
        exits: pd.Series,
        initial_capital: float
    ) -> Dict:
        """Backtest simplifié pour Monte Carlo."""
        capital = initial_capital
        position = 0
        entry_price = 0
        peak = capital
        max_dd = 0
        returns_list = []
        
        close = df['Close'].values
        entries_val = entries.values if hasattr(entries, 'values') else entries
        exits_val = exits.values if hasattr(exits, 'values') else exits
        
        for i in range(len(df)):
            if position == 0 and entries_val[i]:
                position = capital / close[i]
                entry_price = close[i]
                capital = 0
            
            elif position > 0 and exits_val[i]:
                capital = position * close[i]
                ret = (close[i] - entry_price) / entry_price
                returns_list.append(ret)
                position = 0
            
            # Track drawdown
            current_value = capital + position * close[i] if position > 0 else capital
            if current_value > peak:
                peak = current_value
            dd = (peak - current_value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        # Fermer position ouverte
        if position > 0:
            capital = position * close[-1]
        
        # Sharpe
        if returns_list:
            returns_arr = np.array(returns_list)
            sharpe = returns_arr.mean() / returns_arr.std() * np.sqrt(len(returns_list)) if returns_arr.std() > 0 else 0
        else:
            sharpe = 0
        
        return {
            'final_value': capital,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'n_trades': len(returns_list)
        }
    
    def run_trade_permutation(
        self,
        trades_pnl: List[float],
        initial_capital: float = 10000
    ) -> Dict:
        """
        Analyse de permutation des trades.
        
        Évalue si la performance est due à l'ordre des trades ou aux trades eux-mêmes.
        
        Args:
            trades_pnl: Liste des P&L en pourcentage
            initial_capital: Capital initial
        
        Returns:
            Dict avec statistiques de permutation
        """
        trades_arr = np.array(trades_pnl)
        
        # Résultat réel (ordre original)
        real_equity = initial_capital
        for pnl in trades_arr:
            real_equity *= (1 + pnl)
        real_return = (real_equity - initial_capital) / initial_capital * 100
        
        # Permutations
        equity_curves = self.permute_trades(trades_arr)
        final_values = equity_curves[:, -1] * initial_capital
        returns_pct = (final_values - initial_capital) / initial_capital * 100
        
        # Calcul du percentile du résultat réel
        percentile = stats.percentileofscore(returns_pct, real_return)
        
        # Drawdowns par simulation
        drawdowns = []
        for curve in equity_curves:
            peak = np.maximum.accumulate(curve)
            dd = (peak - curve) / peak
            drawdowns.append(dd.max())
        drawdowns = np.array(drawdowns)
        
        return {
            'real_return': round(real_return, 2),
            'mean_permuted_return': round(returns_pct.mean(), 2),
            'std_permuted_return': round(returns_pct.std(), 2),
            'real_percentile': round(percentile, 1),
            'confidence_interval_95': (
                round(np.percentile(returns_pct, 2.5), 2),
                round(np.percentile(returns_pct, 97.5), 2)
            ),
            'is_sequence_dependent': percentile < 5 or percentile > 95,
            'probability_worse': round((returns_pct < real_return).mean() * 100, 2),
            'mean_max_drawdown': round(drawdowns.mean() * 100, 2),
            'worst_max_drawdown': round(drawdowns.max() * 100, 2)
        }
    
    def get_confidence_statistics(
        self,
        result: MonteCarloResult
    ) -> Dict:
        """
        Génère un rapport de confiance basé sur les résultats Monte Carlo.
        """
        dist = result.results_distribution
        
        return {
            'summary': {
                'simulations': result.n_simulations,
                'mean_return': f"{result.mean_return}%",
                'median_return': f"{result.median_return}%",
                'std_return': f"{result.std_return}%"
            },
            'confidence_intervals': {
                '50%': (round(np.percentile(dist, 25), 2), round(np.percentile(dist, 75), 2)),
                '90%': (round(np.percentile(dist, 5), 2), round(np.percentile(dist, 95), 2)),
                '95%': result.confidence_interval_95,
                '99%': (round(np.percentile(dist, 0.5), 2), round(np.percentile(dist, 99.5), 2))
            },
            'probabilities': {
                'profit': f"{result.probability_profit}%",
                'target_reached': f"{result.probability_target}%",
                'loss_>10%': f"{round((dist < -10).mean() * 100, 2)}%",
                'loss_>20%': f"{round((dist < -20).mean() * 100, 2)}%"
            },
            'risk_metrics': {
                'var_95': f"{result.var_95}%",
                'avg_max_drawdown': f"{result.max_drawdown_mean}%",
                'avg_sharpe': result.sharpe_mean
            },
            'recommendation': self._get_recommendation(result)
        }
    
    def _get_recommendation(self, result: MonteCarloResult) -> str:
        """Génère une recommandation basée sur les résultats."""
        if result.probability_profit < 50:
            return "⚠️ ATTENTION: Probabilité de profit < 50%. Stratégie risquée."
        elif result.mean_return < 0:
            return "⚠️ ATTENTION: Rendement moyen négatif. Revoir la stratégie."
        elif result.var_95 < -15:
            return "⚠️ ATTENTION: VaR 95% < -15%. Risque de pertes importantes."
        elif result.probability_profit > 70 and result.mean_return > 10:
            return "✅ Stratégie prometteuse. Considérer pour trading réel."
        elif result.probability_profit > 60:
            return "👍 Résultats acceptables. Tester sur plus de données."
        else:
            return "📊 Résultats mitigés. Optimisation recommandée."


if __name__ == "__main__":
    print("=" * 60)
    print("TEST MONTE CARLO SIMULATOR")
    print("=" * 60)
    
    # Créer des données de test
    np.random.seed(42)
    n = 500
    
    # Prix simulés
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        'Open': price + np.random.randn(n) * 0.2,
        'High': price + abs(np.random.randn(n)) * 0.5,
        'Low': price - abs(np.random.randn(n)) * 0.5,
        'Close': price,
        'Volume': np.random.randint(1000, 10000, n)
    })
    
    # Monte Carlo
    mc = MonteCarloSimulator(n_simulations=5000)
    
    # Stratégie simple pour test
    def simple_strategy(df):
        sma_fast = df['Close'].rolling(10).mean()
        sma_slow = df['Close'].rolling(30).mean()
        entries = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
        exits = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))
        return entries.fillna(False), exits.fillna(False)
    
    print("\n--- Simulation Monte Carlo ---")
    result = mc.run_strategy_monte_carlo(simple_strategy, df)
    
    print(f"Simulations: {result.n_simulations}")
    print(f"Rendement moyen: {result.mean_return}%")
    print(f"Écart-type: {result.std_return}%")
    print(f"IC 95%: {result.confidence_interval_95}")
    print(f"Probabilité de profit: {result.probability_profit}%")
    print(f"VaR 95%: {result.var_95}%")
    
    # Statistiques de confiance
    print("\n--- Statistiques de Confiance ---")
    stats_report = mc.get_confidence_statistics(result)
    print(f"Recommandation: {stats_report['recommendation']}")
    
    # Permutation de trades
    print("\n--- Analyse de Permutation des Trades ---")
    trades = [0.02, -0.01, 0.03, -0.015, 0.025, -0.02, 0.04, -0.01, 0.015, -0.005]
    perm_result = mc.run_trade_permutation(trades)
    
    print(f"Rendement réel: {perm_result['real_return']}%")
    print(f"Rendement moyen (permutations): {perm_result['mean_permuted_return']}%")
    print(f"Percentile du réel: {perm_result['real_percentile']}%")
    print(f"Dépendant de la séquence: {perm_result['is_sequence_dependent']}")
