"""
Circuit Breaker - Système d'arrêt d'urgence.
Arrête le trading si les conditions de risque sont dépassées.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import sys
import os


from quantum.shared.config.settings import config


@dataclass
class TradeRecord:
    """Enregistrement d'un trade."""
    timestamp: datetime
    pnl: float
    is_win: bool


class CircuitBreaker:
    """
    Système de protection contre les pertes excessives.
    
    Conditions d'arrêt:
    1. Drawdown max atteint (5%)
    2. Pertes consécutives max (3)
    3. Perte journalière max
    """
    
    def __init__(
        self,
        max_drawdown: float = None,
        max_consecutive_losses: int = None,
        initial_capital: float = None
    ):
        self.max_drawdown = max_drawdown or config.risk.MAX_DRAWDOWN
        self.max_consecutive_losses = max_consecutive_losses or config.risk.MAX_CONSECUTIVE_LOSSES
        self.initial_capital = initial_capital or config.risk.INITIAL_CAPITAL
        
        # État
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.trade_history: List[TradeRecord] = []
        self.is_active = True
        self.halt_reason: Optional[str] = None
        self.halt_until: Optional[datetime] = None
    
    def record_trade(self, pnl: float, timestamp: datetime = None):
        """
        Enregistre un trade et vérifie les conditions d'arrêt.
        
        Args:
            pnl: Profit/Perte du trade
            timestamp: Horodatage
        """
        timestamp = timestamp or datetime.now()
        
        self.current_capital += pnl
        is_win = pnl > 0
        
        self.trade_history.append(TradeRecord(timestamp, pnl, is_win))
        
        # Mettre à jour le peak
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Vérifier les conditions d'arrêt
        self._check_conditions()
    
    def _check_conditions(self):
        """Vérifie toutes les conditions d'arrêt."""
        # 1. Drawdown
        current_drawdown = self._calculate_drawdown()
        if current_drawdown >= self.max_drawdown:
            self._halt(f"Drawdown max atteint: {current_drawdown*100:.1f}%")
            return
        
        # 2. Pertes consécutives
        consecutive_losses = self._count_consecutive_losses()
        if consecutive_losses >= self.max_consecutive_losses:
            self._halt(f"Pertes consécutives: {consecutive_losses}")
            return
        
        # 3. Perte journalière (optionnel)
        daily_pnl = self._calculate_daily_pnl()
        if daily_pnl < -self.initial_capital * 0.02:  # -2% par jour
            self._halt(f"Perte journalière max: ${daily_pnl:.2f}")
    
    def _calculate_drawdown(self) -> float:
        """Calcule le drawdown actuel."""
        if self.peak_capital == 0:
            return 0
        return (self.peak_capital - self.current_capital) / self.peak_capital
    
    def _count_consecutive_losses(self) -> int:
        """Compte les pertes consécutives."""
        if not self.trade_history:
            return 0
        
        count = 0
        for trade in reversed(self.trade_history):
            if not trade.is_win:
                count += 1
            else:
                break
        return count
    
    def _calculate_daily_pnl(self) -> float:
        """Calcule le P&L du jour."""
        today = datetime.now().date()
        
        daily_pnl = sum(
            t.pnl for t in self.trade_history
            if t.timestamp.date() == today
        )
        return daily_pnl
    
    def _halt(self, reason: str, duration_hours: int = 24):
        """Arrête le trading."""
        self.is_active = False
        self.halt_reason = reason
        self.halt_until = datetime.now() + timedelta(hours=duration_hours)
        print(f"⛔ CIRCUIT BREAKER: {reason}")
        print(f"   Trading suspendu jusqu'à: {self.halt_until}")
    
    def can_trade(self) -> Dict:
        """
        Vérifie si le trading est autorisé.
        
        Returns:
            Dict avec status et raison
        """
        # Vérifier si le halt a expiré
        if not self.is_active and self.halt_until:
            if datetime.now() > self.halt_until:
                self.reset()
        
        if not self.is_active:
            return {
                "allowed": False,
                "reason": self.halt_reason,
                "resume_at": str(self.halt_until) if self.halt_until else None
            }
        
        # Vérifications préventives
        current_dd = self._calculate_drawdown()
        warning_dd = self.max_drawdown * 0.8
        
        if current_dd > warning_dd:
            return {
                "allowed": True,
                "warning": f"Drawdown proche du max: {current_dd*100:.1f}%",
                "reduce_size": True
            }
        
        return {"allowed": True}
    
    def reset(self):
        """Réinitialise le circuit breaker."""
        self.is_active = True
        self.halt_reason = None
        self.halt_until = None
        print("✅ Circuit Breaker réinitialisé")
    
    def get_status(self) -> Dict:
        """Retourne le statut complet."""
        return {
            "is_active": self.is_active,
            "current_capital": self.current_capital,
            "initial_capital": self.initial_capital,
            "peak_capital": self.peak_capital,
            "current_drawdown": self._calculate_drawdown() * 100,
            "max_drawdown": self.max_drawdown * 100,
            "consecutive_losses": self._count_consecutive_losses(),
            "max_consecutive_losses": self.max_consecutive_losses,
            "daily_pnl": self._calculate_daily_pnl(),
            "total_trades": len(self.trade_history),
            "halt_reason": self.halt_reason
        }


if __name__ == "__main__":
    # Test
    cb = CircuitBreaker(initial_capital=10000, max_consecutive_losses=3)
    
    print("=== Test Circuit Breaker ===\n")
    
    # Simuler des trades
    trades = [100, -50, -50, -60]  # 3 pertes consécutives
    
    for i, pnl in enumerate(trades):
        print(f"Trade {i+1}: PnL = ${pnl}")
        cb.record_trade(pnl)
        
        status = cb.can_trade()
        print(f"  Can trade: {status['allowed']}")
        if not status['allowed']:
            print(f"  Raison: {status['reason']}")
            break
    
    print("\n=== Statut Final ===")
    status = cb.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
