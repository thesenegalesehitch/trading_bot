"""
Trade History - Stockage et analyse de l'historique des trades.
Phase 3: Coach Features - Trade Advisor & Coach
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd


class TradeOutcome(Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    OPEN = "OPEN"


class TradeReason(Enum):
    SIGNAL = "signal"  # Basé sur un signal
    MANUAL = "manual"  # Entré manuellement par l'utilisateur
    COPY = "copy"  # Copié d'un autre trader


@dataclass
class Trade:
    id: str
    symbol: str
    direction: str  # "BUY" or "SELL"
    entry_price: float
    exit_price: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    stop_loss: float
    take_profit: float
    quantity: float
    outcome: TradeOutcome
    pnl: float  # Profit/Perte en %
    pnl_pips: float
    reasoning: str
    notes: str
    setup_name: str  # Nom du setup ICT utilisé
    timeframe: str
    validation_score: Optional[int]  # Score de validation du trade
    tags: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['entry_time'] = self.entry_time.isoformat() if self.entry_time else None
        data['exit_time'] = self.exit_time.isoformat() if self.exit_time else None
        data['outcome'] = self.outcome.value if isinstance(self.outcome, TradeOutcome) else self.outcome
        return data


class TradeHistory:
    """
    Gère l'historique des trades et fournit des analytics.
    """
    
    def __init__(self, storage_path: str = "data/trades"):
        self.storage_path = storage_path
        self.trades: List[Trade] = []
        self._ensure_storage_dir()
        self._load_trades()
    
    def _ensure_storage_dir(self):
        """Crée le répertoire de stockage si nécessaire."""
        os.makedirs(self.storage_path, exist_ok=True)
    
    def _load_trades(self):
        """Charge les trades depuis le stockage."""
        trades_file = os.path.join(self.storage_path, "trades.json")
        if os.path.exists(trades_file):
            try:
                with open(trades_file, 'r') as f:
                    data = json.load(f)
                    for trade_data in data:
                        trade_data['entry_time'] = datetime.fromisoformat(trade_data['entry_time'])
                        if trade_data.get('exit_time'):
                            trade_data['exit_time'] = datetime.fromisoformat(trade_data['exit_time'])
                        if isinstance(trade_data.get('outcome'), str):
                            trade_data['outcome'] = TradeOutcome(trade_data['outcome'])
                        self.trades.append(Trade(**trade_data))
            except Exception as e:
                print(f"Erreur lors du chargement des trades: {e}")
    
    def _save_trades(self):
        """Sauvegarde les trades dans le stockage."""
        trades_file = os.path.join(self.storage_path, "trades.json")
        try:
            with open(trades_file, 'w') as f:
                json.dump([t.to_dict() for t in self.trades], f, indent=2)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des trades: {e}")
    
    def add_trade(self, trade: Trade):
        """Ajoute un nouveau trade."""
        self.trades.append(trade)
        self._save_trades()
    
    def close_trade(self, trade_id: str, exit_price: float, exit_time: datetime = None):
        """Ferme un trade existant."""
        for trade in self.trades:
            if trade.id == trade_id:
                trade.exit_price = exit_price
                trade.exit_time = exit_time or datetime.now()
                
                # Calculer le P&L
                if trade.direction == "BUY":
                    trade.pnl = ((exit_price - trade.entry_price) / trade.entry_price) * 100
                    trade.pnl_pips = (exit_price - trade.entry_price) * 10000
                else:
                    trade.pnl = ((trade.entry_price - exit_price) / trade.entry_price) * 100
                    trade.pnl_pips = (trade.entry_price - exit_price) * 10000
                
                # Déterminer l'outcome
                if trade.pnl > 0:
                    trade.outcome = TradeOutcome.WIN
                elif trade.pnl < 0:
                    trade.outcome = TradeOutcome.LOSS
                else:
                    trade.outcome = TradeOutcome.BREAKEVEN
                
                self._save_trades()
                return True
        return False
    
    def get_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Calcule les statistiques des trades sur une période donnée."""
        if not self.trades:
            return self._empty_stats()
        
        # Filtrer par période
        cutoff_date = datetime.now() - timedelta(days=days)
        period_trades = [t for t in self.trades if t.exit_time and t.exit_time >= cutoff_date]
        
        if not period_trades:
            return self._empty_stats()
        
        # Calculer les stats
        total_trades = len(period_trades)
        wins = sum(1 for t in period_trades if t.outcome == TradeOutcome.WIN)
        losses = sum(1 for t in period_trades if t.outcome == TradeOutcome.LOSS)
        breakeven = sum(1 for t in period_trades if t.outcome == TradeOutcome.BREAKEVEN)
        
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        loss_rate = (losses / total_trades * 100) if total_trades > 0 else 0
        
        # P&L
        total_pnl = sum(t.pnl for t in period_trades)
        avg_win = sum(t.pnl for t in period_trades if t.outcome == TradeOutcome.WIN) / wins if wins > 0 else 0
        avg_loss = sum(t.pnl for t in period_trades if t.outcome == TradeOutcome.LOSS) / losses if losses > 0 else 0
        
        # Plus grand win/loss
        biggest_win = max((t.pnl for t in period_trades if t.outcome == TradeOutcome.WIN), default=0)
        biggest_loss = min((t.pnl for t in period_trades if t.outcome == TradeOutcome.LOSS), default=0)
        
        # Duration moyenne
        durations = []
        for t in period_trades:
            if t.exit_time and t.entry_time:
                durations.append((t.exit_time - t.entry_time).total_seconds() / 3600)  # en heures
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            'period_days': days,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'breakeven': breakeven,
            'win_rate': round(win_rate, 2),
            'loss_rate': round(loss_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'biggest_win': round(biggest_win, 2),
            'biggest_loss': round(biggest_loss, 2),
            'avg_duration_hours': round(avg_duration, 2),
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
        }
    
    def _empty_stats(self) -> Dict[str, Any]:
        """Retourne des stats vides."""
        return {
            'period_days': 0,
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'breakeven': 0,
            'win_rate': 0,
            'loss_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'biggest_win': 0,
            'biggest_loss': 0,
            'avg_duration_hours': 0,
            'profit_factor': 0
        }
    
    def get_trades_dataframe(self, days: Optional[int] = None) -> pd.DataFrame:
        """Retourne les trades sous forme de DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = []
        for t in self.trades:
            if days:
                cutoff = datetime.now() - timedelta(days=days)
                if t.entry_time < cutoff:
                    continue
            
            trades_data.append({
                'id': t.id,
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'outcome': t.outcome.value if isinstance(t.outcome, TradeOutcome) else t.outcome,
                'pnl': t.pnl,
                'pnl_pips': t.pnl_pips,
                'setup': t.setup_name,
                'timeframe': t.timeframe,
                'duration_hours': (t.exit_time - t.entry_time).total_seconds() / 3600 if t.exit_time else None
            })
        
        return pd.DataFrame(trades_data)
    
    def get_open_trades(self) -> List[Trade]:
        """Retourne les trades ouverts."""
        return [t for t in self.trades if t.outcome == TradeOutcome.OPEN]
    
    def get_recent_trades(self, count: int = 10) -> List[Trade]:
        """Retourne les N derniers trades fermés."""
        closed_trades = [t for t in self.trades if t.outcome != TradeOutcome.OPEN]
        closed_trades.sort(key=lambda x: x.exit_time or datetime.min, reverse=True)
        return closed_trades[:count]
    
    def analyze_by_setup(self) -> Dict[str, Dict[str, Any]]:
        """Analyse les performances par setup de trading."""
        setups = {}
        
        for trade in self.trades:
            if trade.outcome == TradeOutcome.OPEN:
                continue
            
            setup = trade.setup_name or "Unknown"
            if setup not in setups:
                setups[setup] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0
                }
            
            setups[setup]['trades'] += 1
            if trade.outcome == TradeOutcome.WIN:
                setups[setup]['wins'] += 1
            else:
                setups[setup]['losses'] += 1
            setups[setup]['total_pnl'] += trade.pnl
        
        # Calculer les moyennes
        for setup in setups:
            if setups[setup]['trades'] > 0:
                setups[setup]['win_rate'] = setups[setup]['wins'] / setups[setup]['trades'] * 100
                setups[setup]['avg_pnl'] = setups[setup]['total_pnl'] / setups[setup]['trades']
        
        return setups
    
    def analyze_by_timeframe(self) -> Dict[str, Dict[str, Any]]:
        """Analyse les performances par timeframe."""
        timeframes = {}
        
        for trade in self.trades:
            if trade.outcome == TradeOutcome.OPEN:
                continue
            
            tf = trade.timeframe or "Unknown"
            if tf not in timeframes:
                timeframes[tf] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0
                }
            
            timeframes[tf]['trades'] += 1
            if trade.outcome == TradeOutcome.WIN:
                timeframes[tf]['wins'] += 1
            else:
                timeframes[tf]['losses'] += 1
            timeframes[tf]['total_pnl'] += trade.pnl
        
        return timeframes


def create_sample_trades():
    """Crée des trades示例 pour les tests."""
    history = TradeHistory()
    
    # Créer quelques trades exemple
    trades = [
        Trade(
            id="trade_001",
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.0850,
            exit_price=1.0875,
            entry_time=datetime.now() - timedelta(days=5),
            exit_time=datetime.now() - timedelta(days=4),
            stop_loss=1.0820,
            take_profit=1.0900,
            quantity=1.0,
            outcome=TradeOutcome.WIN,
            pnl=2.30,
            pnl_pips=25,
            reasoning="Signal BUY sur rebond support",
            notes="Trade bien exécuté",
            setup_name="Bull Flag",
            timeframe="1H",
            validation_score=85,
            tags=["trend_following", "ict"],
            metadata={}
        ),
        Trade(
            id="trade_002",
            symbol="GBPUSD",
            direction="SELL",
            entry_price=1.2650,
            exit_price=1.2620,
            entry_time=datetime.now() - timedelta(days=3),
            exit_time=datetime.now() - timedelta(days=2),
            stop_loss=1.2680,
            take_profit=1.2580,
            quantity=1.0,
            outcome=TradeOutcome.WIN,
            pnl=2.37,
            pnl_pips=30,
            reasoning="Signal SELL sur résistance",
            notes="Bon timing",
            setup_name="Bear Trap",
            timeframe="4H",
            validation_score=80,
            tags=["reversal", "ict"],
            metadata={}
        ),
        Trade(
            id="trade_003",
            symbol="USDJPY",
            direction="BUY",
            entry_price=149.50,
            exit_price=148.80,
            entry_time=datetime.now() - timedelta(days=1),
            exit_time=datetime.now(),
            stop_loss=149.00,
            take_profit=151.00,
            quantity=1.0,
            outcome=TradeOutcome.LOSS,
            pnl=-1.40,
            pnl_pips=-70,
            reasoning="Entrée trop précoce",
            notes="SL hit - manque de patience",
            setup_name="FVG Buy",
            timeframe="1H",
            validation_score=45,
            tags=["ict", "fvg"],
            metadata={}
        )
    ]
    
    for trade in trades:
        history.add_trade(trade)
    
    # Afficher les stats
    print("=== Statistiques des 30 derniers jours ===")
    stats = history.get_statistics(30)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== Analyse par Setup ===")
    setup_stats = history.analyze_by_setup()
    for setup, stats in setup_stats.items():
        print(f"{setup}: {stats}")
    
    return history


if __name__ == "__main__":
    history = create_sample_trades()
