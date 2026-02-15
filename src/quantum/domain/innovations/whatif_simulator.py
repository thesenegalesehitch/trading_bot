"""
What If Simulator - Permet de replay n'importe quel scénario de trading.
Phase 4: Innovations - Trade Advisor & Coach

Cet outil permet de simuler ce qui se serait passé si l'utilisateur
avait entré à un certain prix à une certaine date.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from quantum.domain.data.downloader import DataDownloader
from quantum.domain.data.feature_engine import TechnicalIndicators


class SimulationType(Enum):
    ENTRY_ONLY = "entry_only"  # Simule juste l'entrée
    WITH_SL_TP = "with_sl_tp"  # Simule avec Stop Loss et Take Profit
    TRAILING_STOP = "trailing_stop"  # Simule avec trailing stop


@dataclass
class SimulationResult:
    scenario: str
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float]
    exit_time: Optional[datetime]
    exit_reason: str  # "tp_hit", "sl_hit", "trailing_stop", "manual", "end_of_data"
    pnl_pips: float
    pnl_percent: float
    max_profit_pips: float
    max_loss_pips: float
    duration_hours: float
    details: Dict[str, Any]


class WhatIfSimulator:
    """
    Simule des scénarios de trading "What If".
    
    Exemples d'utilisation:
    - "What if j'avais bought à 1.0850 le 15 Mars?"
    - "What if j'avais mis mon SL 20 pips plus serré?"
    - "What if j'avais utilisé un trailing stop?"
    """
    
    def __init__(self):
        self.downloader = DataDownloader()
        self.indicators = TechnicalIndicators()
    
    def simulate(
        self,
        symbol: str,
        direction: str,  # "BUY" or "SELL"
        entry_price: float,
        entry_date: datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        simulation_type: SimulationType = SimulationType.WITH_SL_TP,
        trailing_stop_pct: float = 2.0,  # % de trailing stop
        exit_date: Optional[datetime] = None
    ) -> SimulationResult:
        """
        Simule un scénario de trading.
        
        Args:
            symbol: Symbole (ex: "EURUSD")
            direction: "BUY" ou "SELL"
            entry_price: Prix d'entrée souhaité
            entry_date: Date d'entrée souhaitée
            stop_loss: Stop Loss (optionnel)
            take_profit: Take Profit (optionnel)
            simulation_type: Type de simulation
            trailing_stop_pct: Pourcentage de trailing stop
            exit_date: Date de fin de simulation (optionnel)
        
        Returns:
            SimulationResult avec tous les détails de la simulation
        """
        # Télécharger les données depuis la date d'entrée
        start_date = entry_date - timedelta(days=7)  # 7 jours avant pour avoir le contexte
        end_date = exit_date or datetime.now() + timedelta(days=1)
        
        try:
            df = self.downloader.download_data(
                symbol=symbol,
                interval="1h",
                years=1
            )
            if df is None or len(df) == 0:
                return self._error_result("Données non disponibles pour ce symbole")
            
            # Filtrer les données à partir de la date d'entrée
            df = df[df.index >= entry_date].copy()
            if len(df) == 0:
                return self._error_result("Aucune donnée disponible après la date d'entrée")
            
        except Exception as e:
            return self._error_result(f"Erreur lors du téléchargement: {str(e)}")
        
        # Trouver l'index le plus proche de la date d'entrée
        df['time_diff'] = abs(df.index - entry_date)
        nearest_idx = df['time_diff'].idxmin()
        
        # Runner la simulation
        if simulation_type == SimulationType.ENTRY_ONLY:
            return self._simulate_entry_only(df, direction, entry_price, nearest_idx)
        elif simulation_type == SimulationType.TRAILING_STOP:
            return self._simulate_trailing_stop(df, direction, entry_price, nearest_idx, trailing_stop_pct)
        else:
            return self._simulate_sl_tp(
                df, direction, entry_price, nearest_idx, stop_loss, take_profit
            )
    
    def _simulate_entry_only(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
        entry_idx
    ) -> SimulationResult:
        """Simule simplement l'entrée sans SL/TP."""
        # Obtenir les prix après l'entrée
        prices_after = df.loc[entry_idx:, 'Close']
        
        if len(prices_after) == 0:
            return self._error_result("Aucune donnée après l'entrée")
        
        max_price = prices_after.max()
        min_price = prices_after.min()
        
        if direction == "BUY":
            pnl_pips = (prices_after.iloc[-1] - entry_price) * 10000
            max_profit = (max_price - entry_price) * 10000
            max_loss = (min_price - entry_price) * 10000
        else:
            pnl_pips = (entry_price - prices_after.iloc[-1]) * 10000
            max_profit = (entry_price - min_price) * 10000
            max_loss = (entry_price - max_price) * 10000
        
        duration = (df.index[-1] - entry_idx).total_seconds() / 3600
        
        return SimulationResult(
            scenario="Entry Only (sans SL/TP)",
            entry_price=entry_price,
            entry_time=entry_idx,
            exit_price=prices_after.iloc[-1],
            exit_time=df.index[-1],
            exit_reason="end_of_data",
            pnl_pips=round(pnl_pips, 1),
            pnl_percent=round(pnl_pips / entry_price * 100, 2),
            max_profit_pips=round(max_profit, 1),
            max_loss_pips=round(max_loss, 1),
            duration_hours=round(duration, 1),
            details={
                'highest_price': max_price,
                'lowest_price': min_price,
                'final_price': prices_after.iloc[-1]
            }
        )
    
    def _simulate_sl_tp(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
        entry_idx,
        stop_loss: Optional[float],
        take_profit: Optional[float]
    ) -> SimulationResult:
        """Simule avec Stop Loss et Take Profit."""
        prices = df.loc[entry_idx:, 'Close']
        
        if len(prices) == 0:
            return self._error_result("Aucune donnée après l'entrée")
        
        exit_price = None
        exit_reason = "end_of_data"
        exit_time = None
        
        max_profit = 0
        max_loss = 0
        
        for i, (timestamp, price) in enumerate(prices.items()):
            if i == 0:
                continue
            
            # Calculer le P&L actuel
            if direction == "BUY":
                current_pnl = (price - entry_price) * 10000
                profit = (price - entry_price) * 10000
                loss = (price - entry_price) * 10000
            else:
                current_pnl = (entry_price - price) * 10000
                profit = (entry_price - price) * 10000
                loss = (entry_price - price) * 10000
            
            max_profit = max(max_profit, profit)
            max_loss = min(max_loss, loss)
            
            # Vérifier le Stop Loss
            if stop_loss:
                if direction == "BUY" and price <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "sl_hit"
                    exit_time = timestamp
                    break
                elif direction == "SELL" and price >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "sl_hit"
                    exit_time = timestamp
                    break
            
            # Vérifier le Take Profit
            if take_profit:
                if direction == "BUY" and price >= take_profit:
                    exit_price = take_profit
                    exit_reason = "tp_hit"
                    exit_time = timestamp
                    break
                elif direction == "SELL" and price <= take_profit:
                    exit_price = take_profit
                    exit_reason = "tp_hit"
                    exit_time = timestamp
                    break
        
        # Si pas de sortie, utiliser le dernier prix
        if exit_price is None:
            exit_price = prices.iloc[-1]
            exit_time = df.index[-1]
            exit_reason = "end_of_data"
        
        # Calculer le P&L final
        if direction == "BUY":
            pnl_pips = (exit_price - entry_price) * 10000
        else:
            pnl_pips = (entry_price - exit_price) * 10000
        
        duration = (exit_time - entry_idx).total_seconds() / 3600
        
        return SimulationResult(
            scenario=f"Entry + SL ({stop_loss}) + TP ({take_profit})",
            entry_price=entry_price,
            entry_time=entry_idx,
            exit_price=exit_price,
            exit_time=exit_time,
            exit_reason=exit_reason,
            pnl_pips=round(pnl_pips, 1),
            pnl_percent=round(pnl_pips / entry_price * 100, 2),
            max_profit_pips=round(max_profit, 1),
            max_loss_pips=round(max_loss, 1),
            duration_hours=round(duration, 1),
            details={
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'sl_distance_pips': abs(entry_price - stop_loss) * 10000 if stop_loss else None,
                'tp_distance_pips': abs(take_profit - entry_price) * 10000 if take_profit else None
            }
        )
    
    def _simulate_trailing_stop(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
        entry_idx,
        trailing_stop_pct: float
    ) -> SimulationResult:
        """Simule avec trailing stop."""
        prices = df.loc[entry_idx:, 'Close']
        
        if len(prices) == 0:
            return self._error_result("Aucune donnée après l'entrée")
        
        # Initialiser le trailing stop
        trailing_stop = None
        exit_price = None
        exit_reason = "end_of_data"
        exit_time = None
        
        max_profit = 0
        peak_price = entry_price
        
        for timestamp, price in prices.items():
            # Mettre à jour le prix峰值
            if direction == "BUY":
                if price > peak_price:
                    peak_price = price
                    # Ajuster le trailing stop
                    trailing_stop = peak_price * (1 - trailing_stop_pct / 100)
            else:
                if price < peak_price:
                    peak_price = price
                    trailing_stop = peak_price * (1 + trailing_stop_pct / 100)
            
            # Calculer le profit actuel
            if direction == "BUY":
                profit = (price - entry_price) * 10000
                max_profit = max(max_profit, profit)
                # Vérifier le trailing stop
                if trailing_stop and price <= trailing_stop:
                    exit_price = price
                    exit_reason = "trailing_stop"
                    exit_time = timestamp
                    break
            else:
                profit = (entry_price - price) * 10000
                max_profit = max(max_profit, profit)
                if trailing_stop and price >= trailing_stop:
                    exit_price = price
                    exit_reason = "trailing_stop"
                    exit_time = timestamp
                    break
        
        if exit_price is None:
            exit_price = prices.iloc[-1]
            exit_time = df.index[-1]
        
        if direction == "BUY":
            pnl_pips = (exit_price - entry_price) * 10000
        else:
            pnl_pips = (entry_price - exit_price) * 10000
        
        duration = (exit_time - entry_idx).total_seconds() / 3600
        
        return SimulationResult(
            scenario=f"Trailing Stop {trailing_stop_pct}%",
            entry_price=entry_price,
            entry_time=entry_idx,
            exit_price=exit_price,
            exit_time=exit_time,
            exit_reason=exit_reason,
            pnl_pips=round(pnl_pips, 1),
            pnl_percent=round(pnl_pips / entry_price * 100, 2),
            max_profit_pips=round(max_profit, 1),
            max_loss_pips=round(min(0, pnl_pips), 1),
            duration_hours=round(duration, 1),
            details={
                'trailing_stop_pct': trailing_stop_pct,
                'peak_price': peak_price,
                'final_trailing_stop': trailing_stop
            }
        )
    
    def _error_result(self, error_message: str) -> SimulationResult:
        """Crée un résultat d'erreur."""
        return SimulationResult(
            scenario="ERROR",
            entry_price=0,
            entry_time=datetime.now(),
            exit_price=None,
            exit_time=None,
            exit_reason=error_message,
            pnl_pips=0,
            pnl_percent=0,
            max_profit_pips=0,
            max_loss_pips=0,
            duration_hours=0,
            details={'error': error_message}
        )
    
    def compare_scenarios(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        entry_date: datetime,
        scenarios: List[Dict]
    ) -> Dict[str, SimulationResult]:
        """
        Compare plusieurs scénarios.
        
        Args:
            scenarios: Liste de configurations à tester
                [
                    {'name': 'Scenario 1', 'stop_loss': 1.0800, 'take_profit': 1.0900},
                    {'name': 'Scenario 2', 'stop_loss': 1.0820, 'take_profit': 1.0920},
                    ...
                ]
        """
        results = {}
        
        for scenario in scenarios:
            name = scenario.get('name', 'Unnamed')
            result = self.simulate(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                entry_date=entry_date,
                stop_loss=scenario.get('stop_loss'),
                take_profit=scenario.get('take_profit'),
                simulation_type=SimulationType.WITH_SL_TP
            )
            results[name] = result
        
        return results


def whatif_example():
    """Exemple d'utilisation du What If Simulator."""
    simulator = WhatIfSimulator()
    
    # Simuler: "What if j'avais bought EURUSD à 1.0850 le 10 février?"
    result = simulator.simulate(
        symbol="EURUSD",
        direction="BUY",
        entry_price=1.0850,
        entry_date=datetime(2026, 2, 10),
        stop_loss=1.0820,
        take_profit=1.0910,
        simulation_type=SimulationType.WITH_SL_TP
    )
    
    print(f"\n{'='*60}")
    print(f"WHAT IF SIMULATOR - RÉSULTAT")
    print(f"{'='*60}")
    print(f"Scenario: {result.scenario}")
    print(f"Entry: {result.entry_price} à {result.entry_time}")
    print(f"Exit: {result.exit_price} à {result.exit_time}")
    print(f"Raison de sortie: {result.exit_reason}")
    print(f"P&L: {result.pnl_pips} pips ({result.pnl_percent}%)")
    print(f"Profit max: {result.max_profit_pips} pips")
    print(f"Perte max: {result.max_loss_pips} pips")
    print(f"Durée: {result.duration_hours} heures")


if __name__ == "__main__":
    whatif_example()
