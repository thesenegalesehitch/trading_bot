"""
Module de gestion de portefeuille multi-actifs.
Optimise l'allocation en fonction de la corrélation et du risque.

Fonctionnalités:
- Corrélation temps réel entre actifs
- Optimisation Mean-Variance (Markowitz)
- Maximum Sharpe Ratio portfolio
- Risk Parity allocation
- Rebalancing automatique
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
import sys
import os


from quantum.shared.config.settings import config


@dataclass
class PortfolioAllocation:
    """Allocation de portefeuille."""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float


class PortfolioManager:
    """
    Gère l'allocation et le risque d'un portefeuille multi-actifs.
    
    Objectifs:
    1. Maximiser le rendement ajusté au risque
    2. Minimiser la corrélation entre positions
    3. Contrôler le risque global
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,  # 2% annuel
        max_position_weight: float = 0.40,  # Max 40% par position
        min_position_weight: float = 0.05,  # Min 5% par position
        rebalance_threshold: float = 0.10  # Rebalancer si déviation > 10%
    ):
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_position_weight
        self.min_weight = min_position_weight
        self.rebalance_threshold = rebalance_threshold
        
        self.current_allocation: Optional[PortfolioAllocation] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None
    
    def calculate_correlation_matrix(
        self,
        returns_dict: Dict[str, pd.Series],
        window: int = 60
    ) -> pd.DataFrame:
        """
        Calcule la matrice de corrélation entre actifs.
        
        Args:
            returns_dict: Dict {symbol: returns_series}
            window: Fenêtre pour la corrélation rolling
        
        Returns:
            Matrice de corrélation
        """
        # Créer un DataFrame avec tous les rendements
        df = pd.DataFrame(returns_dict)
        
        # Aligner les données
        df = df.dropna()
        
        if len(df) < window:
            window = max(len(df) // 2, 10)
        
        # Corrélation rolling (dernière valeur)
        self.correlation_matrix = df.tail(window).corr()
        
        return self.correlation_matrix
    
    def get_correlation_risk(self) -> Dict:
        """
        Évalue le risque de corrélation du portefeuille.
        
        Returns:
            Dict avec métriques de corrélation
        """
        if self.correlation_matrix is None:
            return {'error': 'Matrice de corrélation non calculée'}
        
        corr = self.correlation_matrix.values
        
        # Corrélation moyenne (hors diagonale)
        n = len(corr)
        off_diagonal = corr[np.triu_indices(n, k=1)]
        avg_correlation = off_diagonal.mean() if len(off_diagonal) > 0 else 0
        
        # Corrélation maximale
        max_correlation = off_diagonal.max() if len(off_diagonal) > 0 else 0
        
        # Paires les plus corrélées
        symbols = list(self.correlation_matrix.columns)
        high_corr_pairs = []
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                if abs(corr[i, j]) > 0.7:
                    high_corr_pairs.append({
                        'pair': (symbols[i], symbols[j]),
                        'correlation': round(corr[i, j], 3)
                    })
        
        # Niveau de risque
        if avg_correlation > 0.7:
            risk_level = "HIGH"
            recommendation = "Portefeuille très corrélé - Diversifier!"
        elif avg_correlation > 0.4:
            risk_level = "MEDIUM"
            recommendation = "Corrélation modérée - Surveiller"
        else:
            risk_level = "LOW"
            recommendation = "Bonne diversification"
        
        return {
            'average_correlation': round(avg_correlation, 3),
            'max_correlation': round(max_correlation, 3),
            'high_corr_pairs': high_corr_pairs,
            'risk_level': risk_level,
            'recommendation': recommendation
        }
    
    def optimize_max_sharpe(
        self,
        returns_dict: Dict[str, pd.Series]
    ) -> PortfolioAllocation:
        """
        Optimise le portefeuille pour maximiser le ratio de Sharpe.
        
        Args:
            returns_dict: Dict {symbol: returns_series}
        
        Returns:
            PortfolioAllocation optimale
        """
        # Préparer les données
        df = pd.DataFrame(returns_dict).dropna()
        symbols = list(df.columns)
        n_assets = len(symbols)
        
        if n_assets < 2:
            # Portfolio mono-actif
            return PortfolioAllocation(
                weights={symbols[0]: 1.0} if symbols else {},
                expected_return=0,
                expected_volatility=0,
                sharpe_ratio=0,
                diversification_ratio=1
            )
        
        # Rendements et covariance
        mean_returns = df.mean() * 252  # Annualisé
        cov_matrix = df.cov() * 252
        
        # Fonction objective: Negative Sharpe Ratio
        def neg_sharpe(weights):
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
            return -sharpe
        
        # Contraintes
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Somme = 100%
        ]
        
        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Point de départ équipondéré
        init_weights = np.array([1/n_assets] * n_assets)
        
        # Optimisation
        result = minimize(
            neg_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        
        # Calculer les métriques du portefeuille optimal
        port_return = np.dot(optimal_weights, mean_returns)
        port_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        # Diversification ratio
        weighted_vol = np.dot(optimal_weights, np.sqrt(np.diag(cov_matrix)))
        div_ratio = weighted_vol / port_vol if port_vol > 0 else 1
        
        self.current_allocation = PortfolioAllocation(
            weights={s: round(w, 4) for s, w in zip(symbols, optimal_weights)},
            expected_return=round(port_return * 100, 2),
            expected_volatility=round(port_vol * 100, 2),
            sharpe_ratio=round(sharpe, 3),
            diversification_ratio=round(div_ratio, 3)
        )
        
        return self.current_allocation
    
    def optimize_min_variance(
        self,
        returns_dict: Dict[str, pd.Series]
    ) -> PortfolioAllocation:
        """
        Optimise le portefeuille pour minimiser la variance.
        
        Plus conservateur que max Sharpe.
        """
        df = pd.DataFrame(returns_dict).dropna()
        symbols = list(df.columns)
        n_assets = len(symbols)
        
        if n_assets < 2:
            return PortfolioAllocation(
                weights={symbols[0]: 1.0} if symbols else {},
                expected_return=0,
                expected_volatility=0,
                sharpe_ratio=0,
                diversification_ratio=1
            )
        
        mean_returns = df.mean() * 252
        cov_matrix = df.cov() * 252
        
        # Fonction objective: Variance
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        init_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            portfolio_variance,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        
        port_return = np.dot(optimal_weights, mean_returns)
        port_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        weighted_vol = np.dot(optimal_weights, np.sqrt(np.diag(cov_matrix)))
        div_ratio = weighted_vol / port_vol if port_vol > 0 else 1
        
        return PortfolioAllocation(
            weights={s: round(w, 4) for s, w in zip(symbols, optimal_weights)},
            expected_return=round(port_return * 100, 2),
            expected_volatility=round(port_vol * 100, 2),
            sharpe_ratio=round(sharpe, 3),
            diversification_ratio=round(div_ratio, 3)
        )
    
    def optimize_risk_parity(
        self,
        returns_dict: Dict[str, pd.Series]
    ) -> PortfolioAllocation:
        """
        Risk Parity - Chaque actif contribue également au risque total.
        
        Populaire pour les portefeuilles diversifiés.
        """
        df = pd.DataFrame(returns_dict).dropna()
        symbols = list(df.columns)
        n_assets = len(symbols)
        
        if n_assets < 2:
            return PortfolioAllocation(
                weights={symbols[0]: 1.0} if symbols else {},
                expected_return=0,
                expected_volatility=0,
                sharpe_ratio=0,
                diversification_ratio=1
            )
        
        mean_returns = df.mean() * 252
        cov_matrix = df.cov() * 252
        
        # Target: chaque actif contribue 1/n au risque
        target_risk = np.array([1/n_assets] * n_assets)
        
        def risk_contribution(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / port_vol if port_vol > 0 else np.zeros(n_assets)
            risk_contrib = risk_contrib / risk_contrib.sum() if risk_contrib.sum() > 0 else target_risk
            return np.sum((risk_contrib - target_risk) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0.01, 0.5) for _ in range(n_assets)]
        init_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            risk_contribution,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        
        port_return = np.dot(optimal_weights, mean_returns)
        port_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        weighted_vol = np.dot(optimal_weights, np.sqrt(np.diag(cov_matrix)))
        div_ratio = weighted_vol / port_vol if port_vol > 0 else 1
        
        return PortfolioAllocation(
            weights={s: round(w, 4) for s, w in zip(symbols, optimal_weights)},
            expected_return=round(port_return * 100, 2),
            expected_volatility=round(port_vol * 100, 2),
            sharpe_ratio=round(sharpe, 3),
            diversification_ratio=round(div_ratio, 3)
        )
    
    def optimize_black_litterman(
        self,
        returns_dict: Dict[str, pd.Series],
        views: Dict[str, float],
        view_confidences: Dict[str, float],
        tau: float = 0.05
    ) -> PortfolioAllocation:
        """
        Optimisation Black-Litterman.
        Combine l'équilibre du marché avec les vues (signaux) du système.
        
        Args:
            returns_dict: Données historiques
            views: Rendements attendus par actif (vues)
            view_confidences: Confiance dans chaque vue (0-1)
            tau: Scalaire d'incertitude du marché (défaut 0.05)
        """
        df = pd.DataFrame(returns_dict).dropna()
        symbols = list(df.columns)
        n = len(symbols)
        
        if n < 2:
            return self.optimize_max_sharpe(returns_dict)
            
        # 1. Préparer les inputs
        cov_matrix = df.cov() * 252
        delta = 2.5 # Aversion au risque moyenne
        
        # Poids d'équilibre (Market Prior) - Ici équipondéré par défaut
        w_eq = np.array([1/n] * n)
        
        # Pi: Rendements d'équilibre implicites
        pi = delta * np.dot(cov_matrix, w_eq)
        
        # 2. Construire les matrices de vues P et Q
        # Pour simplifier, chaque vue porte sur un seul actif
        active_symbols = [s for s in symbols if s in views]
        k = len(active_symbols)
        
        if k == 0:
            return self.optimize_max_sharpe(returns_dict)
            
        P = np.zeros((k, n))
        Q = np.zeros(k)
        Omega = np.zeros((k, k))
        
        for i, symbol in enumerate(active_symbols):
            idx = symbols.index(symbol)
            P[i, idx] = 1
            Q[i] = views[symbol]
            # Incertitude de la vue (basée sur la confiance)
            # Plus la confiance est haute, plus l'incertitude (Omega) est basse
            conf = view_confidences.get(symbol, 0.5)
            Omega[i, i] = cov_matrix.iloc[idx, idx] * (1 - conf) / (conf + 1e-6)
            
        # 3. Calculer les rendements combinés (E[R])
        # Formule de Black-Litterman
        tau_sigma_inv = np.linalg.inv(tau * cov_matrix)
        p_omega_inv_p = np.dot(P.T, np.dot(np.linalg.inv(Omega), P))
        
        first_term = np.linalg.inv(tau_sigma_inv + p_omega_inv_p)
        second_term = np.dot(tau_sigma_inv, pi) + np.dot(P.T, np.dot(np.linalg.inv(Omega), Q))
        
        er = np.dot(first_term, second_term)
        
        # 4. Optimiser les poids avec les nouveaux rendements attendus
        def objective(w):
            port_return = np.dot(w, er)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            return -(port_return / port_vol) if port_vol > 0 else 0

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
        
        result = minimize(objective, w_eq, method='SLSQP', bounds=bounds, constraints=constraints)
        optimal_weights = result.x
        
        # Métriques
        port_return = np.dot(optimal_weights, er)
        port_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        
        return PortfolioAllocation(
            weights={s: round(w, 4) for s, w in zip(symbols, optimal_weights)},
            expected_return=round(port_return * 100, 2),
            expected_volatility=round(port_vol * 100, 2),
            sharpe_ratio=round(port_return / port_vol if port_vol > 0 else 0, 3),
            diversification_ratio=1.0 # Placeholder
        )

    def check_rebalance_needed(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> Tuple[bool, Dict]:
        """
        Vérifie si un rebalancement est nécessaire.
        
        Args:
            current_weights: Poids actuels
            target_weights: Poids cibles
        
        Returns:
            Tuple (need_rebalance, adjustments)
        """
        adjustments = {}
        max_deviation = 0
        
        for asset in target_weights:
            current = current_weights.get(asset, 0)
            target = target_weights[asset]
            deviation = abs(current - target)
            
            if deviation > 0.001:  # Ignorer les très petites déviations
                adjustments[asset] = {
                    'current': round(current, 4),
                    'target': round(target, 4),
                    'change': round(target - current, 4),
                    'deviation_pct': round(deviation * 100, 2)
                }
            
            max_deviation = max(max_deviation, deviation)
        
        need_rebalance = max_deviation > self.rebalance_threshold
        
        return need_rebalance, {
            'need_rebalance': need_rebalance,
            'max_deviation': round(max_deviation * 100, 2),
            'threshold': round(self.rebalance_threshold * 100, 2),
            'adjustments': adjustments
        }
    
    def get_position_sizes(
        self,
        allocation: PortfolioAllocation,
        total_capital: float
    ) -> Dict[str, Dict]:
        """
        Calcule les tailles de position en valeur monétaire.
        
        Args:
            allocation: Allocation cible
            total_capital: Capital total disponible
        
        Returns:
            Dict avec tailles par actif
        """
        positions = {}
        
        for asset, weight in allocation.weights.items():
            value = total_capital * weight
            positions[asset] = {
                'weight_pct': round(weight * 100, 2),
                'value': round(value, 2),
                'max_risk': round(value * 0.02, 2)  # 2% de la position
            }
        
        return {
            'positions': positions,
            'total_capital': total_capital,
            'expected_return_annual': allocation.expected_return,
            'expected_volatility_annual': allocation.expected_volatility,
            'sharpe_ratio': allocation.sharpe_ratio
        }


if __name__ == "__main__":
    print("=" * 60)
    print("TEST PORTFOLIO MANAGER")
    print("=" * 60)
    
    # Générer des données de test
    np.random.seed(42)
    n = 252  # Un an de données journalières
    
    # Simuler 4 actifs avec différentes caractéristiques
    returns = {
        'EURUSD': pd.Series(np.random.normal(0.0001, 0.008, n)),
        'GOLD': pd.Series(np.random.normal(0.0002, 0.012, n)),
        'BTCUSD': pd.Series(np.random.normal(0.0005, 0.035, n)),
        'SP500': pd.Series(np.random.normal(0.0003, 0.015, n))
    }
    
    # Ajouter de la corrélation entre certains actifs
    returns['GOLD'] = returns['GOLD'] + returns['EURUSD'] * 0.3
    
    pm = PortfolioManager()
    
    # Corrélation
    print("\n--- Matrice de Corrélation ---")
    corr = pm.calculate_correlation_matrix(returns)
    print(corr.round(2))
    
    corr_risk = pm.get_correlation_risk()
    print(f"\nRisque de corrélation: {corr_risk['risk_level']}")
    print(f"Corrélation moyenne: {corr_risk['average_correlation']}")
    
    # Optimisation Max Sharpe
    print("\n--- Optimisation Max Sharpe ---")
    allocation_sharpe = pm.optimize_max_sharpe(returns)
    print(f"Poids: {allocation_sharpe.weights}")
    print(f"Rendement attendu: {allocation_sharpe.expected_return}%")
    print(f"Volatilité attendue: {allocation_sharpe.expected_volatility}%")
    print(f"Sharpe Ratio: {allocation_sharpe.sharpe_ratio}")
    
    # Optimisation Min Variance
    print("\n--- Optimisation Min Variance ---")
    allocation_var = pm.optimize_min_variance(returns)
    print(f"Poids: {allocation_var.weights}")
    print(f"Volatilité attendue: {allocation_var.expected_volatility}%")
    
    # Risk Parity
    print("\n--- Risk Parity ---")
    allocation_rp = pm.optimize_risk_parity(returns)
    print(f"Poids: {allocation_rp.weights}")
    
    # Tailles de position
    print("\n--- Tailles de Position (capital: 10,000$) ---")
    positions = pm.get_position_sizes(allocation_sharpe, 10000)
    for asset, pos in positions['positions'].items():
        print(f"  {asset}: {pos['weight_pct']}% = ${pos['value']}")
