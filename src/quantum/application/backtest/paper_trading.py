"""
Simulateur de Paper Trading.
Permet de tester les stratégies en temps réel sans risquer de capital.

Fonctionnalités:
- Simulation d'exécution d'ordres
- Suivi des positions en temps réel
- Calcul du P&L
- Comparaison paper vs backtest
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import os
import sys




class OrderType(Enum):
    """Type d'ordre."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Statut d'un ordre."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(Enum):
    """Côté de la position."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Order:
    """Représente un ordre de trading."""
    id: str
    symbol: str
    side: PositionSide
    order_type: OrderType
    quantity: float
    price: Optional[float]  # Pour limit orders
    stop_price: Optional[float] = None  # Pour stop orders
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0
    filled_price: float = 0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    commission: float = 0
    slippage: float = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class Position:
    """Représente une position ouverte."""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    max_profit: float = 0
    max_loss: float = 0
    
    def update_pnl(self, current_price: float):
        """Met à jour le P&L non réalisé."""
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
        
        self.max_profit = max(self.max_profit, self.unrealized_pnl)
        self.max_loss = min(self.max_loss, self.unrealized_pnl)


@dataclass
class Trade:
    """Représente un trade terminé."""
    id: str
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    commission: float
    slippage: float
    max_profit: float
    max_loss: float
    duration: timedelta
    exit_reason: str


class PaperTradingSimulator:
    """
    Simulateur de paper trading.
    
    Simule l'exécution d'ordres avec slippage, commissions et gestion de positions.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        commission_pct: float = 0.001,  # 0.1%
        slippage_pct: float = 0.0005,   # 0.05%
        max_positions: int = 5,
        save_trades: bool = True,
        trades_file: str = "paper_trades.json"
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.max_positions = max_positions
        self.save_trades = save_trades
        self.trades_file = trades_file
        
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.order_counter = 0
        
        # Statistiques
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'peak_capital': initial_capital
        }
    
    def place_order(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: float = None,
        stop_price: float = None,
        stop_loss: float = None,
        take_profit: float = None,
        metadata: Dict = None
    ) -> Order:
        """
        Place un ordre de trading.
        
        Args:
            symbol: Symbole à trader
            side: LONG ou SHORT
            quantity: Quantité
            order_type: Type d'ordre
            price: Prix limite (pour limit orders)
            stop_price: Prix stop (pour stop orders)
            stop_loss: Stop loss
            take_profit: Take profit
            metadata: Métadonnées additionnelles
        
        Returns:
            Order créé
        """
        # Générer l'ID
        self.order_counter += 1
        order_id = f"ORD_{self.order_counter:06d}"
        
        # Créer l'ordre
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            metadata=metadata or {}
        )
        
        # Stocker SL/TP
        order.metadata['stop_loss'] = stop_loss
        order.metadata['take_profit'] = take_profit
        
        self.orders[order_id] = order
        
        return order
    
    def execute_order(
        self,
        order: Order,
        current_price: float,
        current_time: datetime = None
    ) -> bool:
        """
        Exécute un ordre au prix actuel.
        
        Args:
            order: Ordre à exécuter
            current_price: Prix de marché actuel
            current_time: Heure actuelle
        
        Returns:
            True si exécuté
        """
        if order.status != OrderStatus.PENDING:
            return False
        
        current_time = current_time or datetime.now()
        
        # Vérifier les conditions d'exécution
        if order.order_type == OrderType.LIMIT:
            if order.side == PositionSide.LONG and current_price > order.price:
                return False  # Prix trop haut pour limite d'achat
            if order.side == PositionSide.SHORT and current_price < order.price:
                return False  # Prix trop bas pour limite de vente
        
        if order.order_type == OrderType.STOP:
            if order.side == PositionSide.LONG and current_price < order.stop_price:
                return False  # Stop non atteint
            if order.side == PositionSide.SHORT and current_price > order.stop_price:
                return False
        
        # Vérifier le capital
        cost = current_price * order.quantity
        commission = cost * self.commission_pct
        
        if cost + commission > self.capital:
            order.status = OrderStatus.REJECTED
            order.metadata['rejection_reason'] = "Insufficient capital"
            return False
        
        # Vérifier le nombre de positions
        if len(self.positions) >= self.max_positions and order.symbol not in self.positions:
            order.status = OrderStatus.REJECTED
            order.metadata['rejection_reason'] = "Max positions reached"
            return False
        
        # Calculer le slippage
        slippage = current_price * self.slippage_pct
        if order.side == PositionSide.LONG:
            filled_price = current_price + slippage
        else:
            filled_price = current_price - slippage
        
        # Mettre à jour l'ordre
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = filled_price
        order.filled_at = current_time
        order.commission = commission
        order.slippage = slippage * order.quantity
        
        # Déduire le capital
        self.capital -= commission
        
        # Créer ou mettre à jour la position
        self._update_position(order)
        
        return True
    
    def _update_position(self, order: Order):
        """Met à jour la position après exécution."""
        symbol = order.symbol
        
        if symbol in self.positions:
            pos = self.positions[symbol]
            
            # Fermeture de position opposée
            if pos.side != order.side:
                self._close_position(symbol, order.filled_price, order.filled_at, "Signal opposé")
            else:
                # Ajout à la position existante
                total_cost = pos.entry_price * pos.quantity + order.filled_price * order.quantity
                pos.quantity += order.quantity
                pos.entry_price = total_cost / pos.quantity
        else:
            # Nouvelle position
            self.positions[symbol] = Position(
                symbol=symbol,
                side=order.side,
                quantity=order.quantity,
                entry_price=order.filled_price,
                entry_time=order.filled_at,
                stop_loss=order.metadata.get('stop_loss'),
                take_profit=order.metadata.get('take_profit')
            )
    
    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_time: datetime,
        reason: str
    ) -> Optional[Trade]:
        """Ferme une position et enregistre le trade."""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        
        # Calculer le P&L
        if pos.side == PositionSide.LONG:
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity
        
        pnl_percent = pnl / (pos.entry_price * pos.quantity) * 100
        
        # Commission de sortie
        commission = exit_price * pos.quantity * self.commission_pct
        pnl -= commission
        
        # Créer le trade
        trade = Trade(
            id=f"TRD_{len(self.trades) + 1:06d}",
            symbol=symbol,
            side=pos.side,
            quantity=pos.quantity,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            pnl=pnl,
            pnl_percent=pnl_percent,
            commission=commission * 2,  # Entrée + sortie
            slippage=0,  # Déjà inclus dans les prix
            max_profit=pos.max_profit,
            max_loss=pos.max_loss,
            duration=exit_time - pos.entry_time,
            exit_reason=reason
        )
        
        self.trades.append(trade)
        self._update_stats(trade)
        
        # Mettre à jour le capital
        self.capital += pos.entry_price * pos.quantity + pnl
        
        # Supprimer la position
        del self.positions[symbol]
        
        # Sauvegarder
        if self.save_trades:
            self._save_trades()
        
        return trade
    
    def update_positions(
        self,
        prices: Dict[str, float],
        current_time: datetime = None
    ) -> List[Trade]:
        """
        Met à jour toutes les positions avec les nouveaux prix.
        Vérifie SL/TP.
        
        Returns:
            Liste des trades fermés par SL/TP
        """
        current_time = current_time or datetime.now()
        closed_trades = []
        
        for symbol in list(self.positions.keys()):
            if symbol not in prices:
                continue
            
            pos = self.positions[symbol]
            price = prices[symbol]
            
            # Mettre à jour le P&L
            pos.update_pnl(price)
            
            # Vérifier Stop Loss
            if pos.stop_loss:
                if pos.side == PositionSide.LONG and price <= pos.stop_loss:
                    trade = self._close_position(symbol, pos.stop_loss, current_time, "Stop Loss")
                    if trade:
                        closed_trades.append(trade)
                    continue
                
                if pos.side == PositionSide.SHORT and price >= pos.stop_loss:
                    trade = self._close_position(symbol, pos.stop_loss, current_time, "Stop Loss")
                    if trade:
                        closed_trades.append(trade)
                    continue
            
            # Vérifier Take Profit
            if pos.take_profit:
                if pos.side == PositionSide.LONG and price >= pos.take_profit:
                    trade = self._close_position(symbol, pos.take_profit, current_time, "Take Profit")
                    if trade:
                        closed_trades.append(trade)
                    continue
                
                if pos.side == PositionSide.SHORT and price <= pos.take_profit:
                    trade = self._close_position(symbol, pos.take_profit, current_time, "Take Profit")
                    if trade:
                        closed_trades.append(trade)
                    continue
        
        return closed_trades
    
    def close_position(
        self,
        symbol: str,
        current_price: float,
        reason: str = "Manual close"
    ) -> Optional[Trade]:
        """Ferme manuellement une position."""
        return self._close_position(symbol, current_price, datetime.now(), reason)
    
    def close_all_positions(
        self,
        prices: Dict[str, float],
        reason: str = "Close all"
    ) -> List[Trade]:
        """Ferme toutes les positions."""
        trades = []
        for symbol in list(self.positions.keys()):
            if symbol in prices:
                trade = self._close_position(symbol, prices[symbol], datetime.now(), reason)
                if trade:
                    trades.append(trade)
        return trades
    
    def _update_stats(self, trade: Trade):
        """Met à jour les statistiques."""
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += trade.pnl
        
        if trade.pnl > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
        
        # Drawdown
        if self.capital > self.stats['peak_capital']:
            self.stats['peak_capital'] = self.capital
        
        drawdown = (self.stats['peak_capital'] - self.capital) / self.stats['peak_capital']
        self.stats['max_drawdown'] = max(self.stats['max_drawdown'], drawdown)
    
    def get_performance_report(self) -> Dict:
        """Génère un rapport de performance."""
        total_trades = self.stats['total_trades']
        
        if total_trades == 0:
            return {
                'status': 'no_trades',
                'capital': self.capital,
                'initial_capital': self.initial_capital
            }
        
        win_rate = self.stats['winning_trades'] / total_trades * 100
        
        winning_pnl = sum(t.pnl for t in self.trades if t.pnl > 0)
        losing_pnl = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')
        
        avg_win = winning_pnl / self.stats['winning_trades'] if self.stats['winning_trades'] > 0 else 0
        avg_loss = losing_pnl / self.stats['losing_trades'] if self.stats['losing_trades'] > 0 else 0
        
        return {
            'capital': round(self.capital, 2),
            'initial_capital': self.initial_capital,
            'total_return_pct': round((self.capital - self.initial_capital) / self.initial_capital * 100, 2),
            'total_pnl': round(self.stats['total_pnl'], 2),
            'total_trades': total_trades,
            'winning_trades': self.stats['winning_trades'],
            'losing_trades': self.stats['losing_trades'],
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'max_drawdown_pct': round(self.stats['max_drawdown'] * 100, 2),
            'open_positions': len(self.positions),
            'unrealized_pnl': round(sum(p.unrealized_pnl for p in self.positions.values()), 2)
        }
    
    def _save_trades(self):
        """Sauvegarde les trades dans un fichier JSON."""
        data = {
            'capital': self.capital,
            'stats': self.stats,
            'trades': [
                {
                    'id': t.id,
                    'symbol': t.symbol,
                    'side': t.side.value,
                    'quantity': t.quantity,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'entry_time': t.entry_time.isoformat(),
                    'exit_time': t.exit_time.isoformat(),
                    'pnl': t.pnl,
                    'pnl_percent': t.pnl_percent,
                    'exit_reason': t.exit_reason
                }
                for t in self.trades
            ]
        }
        
        with open(self.trades_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_open_positions_summary(self) -> List[Dict]:
        """Retourne un résumé des positions ouvertes."""
        return [
            {
                'symbol': p.symbol,
                'side': p.side.value,
                'quantity': p.quantity,
                'entry_price': round(p.entry_price, 4),
                'unrealized_pnl': round(p.unrealized_pnl, 2),
                'stop_loss': p.stop_loss,
                'take_profit': p.take_profit
            }
            for p in self.positions.values()
        ]


if __name__ == "__main__":
    print("=" * 60)
    print("TEST PAPER TRADING SIMULATOR")
    print("=" * 60)
    
    # Initialiser
    sim = PaperTradingSimulator(
        initial_capital=10000,
        commission_pct=0.001,
        slippage_pct=0.0005,
        save_trades=False
    )
    
    print(f"\nCapital initial: ${sim.capital}")
    
    # Simuler quelques trades
    current_time = datetime.now()
    
    # Trade 1: BUY
    order1 = sim.place_order(
        symbol="EURUSD",
        side=PositionSide.LONG,
        quantity=1000,
        stop_loss=1.0800,
        take_profit=1.0900
    )
    
    sim.execute_order(order1, 1.0850, current_time)
    print(f"\nTrade 1 exécuté: BUY EURUSD @ {order1.filled_price:.4f}")
    
    # Mise à jour avec un prix plus haut
    sim.update_positions({'EURUSD': 1.0880}, current_time + timedelta(hours=1))
    print(f"Position mise à jour, P&L non réalisé: ${sim.positions['EURUSD'].unrealized_pnl:.2f}")
    
    # TP atteint
    closed = sim.update_positions({'EURUSD': 1.0900}, current_time + timedelta(hours=2))
    if closed:
        print(f"Position fermée par TP, P&L: ${closed[0].pnl:.2f}")
    
    # Trade 2: SELL avec SL
    order2 = sim.place_order(
        symbol="GBPUSD",
        side=PositionSide.SHORT,
        quantity=1000,
        stop_loss=1.2700,
        take_profit=1.2500
    )
    
    sim.execute_order(order2, 1.2600, current_time + timedelta(hours=3))
    print(f"\nTrade 2 exécuté: SELL GBPUSD @ {order2.filled_price:.4f}")
    
    # SL atteint
    closed = sim.update_positions({'GBPUSD': 1.2700}, current_time + timedelta(hours=4))
    if closed:
        print(f"Position fermée par SL, P&L: ${closed[0].pnl:.2f}")
    
    # Rapport
    print("\n" + "=" * 40)
    print("RAPPORT DE PERFORMANCE")
    print("=" * 40)
    
    report = sim.get_performance_report()
    for key, value in report.items():
        print(f"  {key}: {value}")
