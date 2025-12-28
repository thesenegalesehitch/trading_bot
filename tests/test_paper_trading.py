"""
Tests unitaires pour le simulateur de paper trading.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.paper_trading import (
    PaperTradingSimulator,
    Order,
    Position,
    Trade,
    OrderType,
    OrderStatus,
    PositionSide
)


class TestPaperTradingSimulator:
    """Tests pour PaperTradingSimulator."""
    
    @pytest.fixture
    def simulator(self):
        return PaperTradingSimulator(
            initial_capital=10000,
            commission_pct=0.001,
            slippage_pct=0.0005,
            save_trades=False
        )
    
    def test_initialization(self, simulator):
        """Initialisation doit être correcte."""
        assert simulator.capital == 10000
        assert simulator.initial_capital == 10000
        assert len(simulator.positions) == 0
        assert len(simulator.trades) == 0
    
    def test_place_order(self, simulator):
        """Placement d'ordre doit créer un ordre."""
        order = simulator.place_order(
            symbol='EURUSD',
            side=PositionSide.LONG,
            quantity=1000,
            stop_loss=1.0800,
            take_profit=1.0900
        )
        
        assert isinstance(order, Order)
        assert order.symbol == 'EURUSD'
        assert order.side == PositionSide.LONG
        assert order.status == OrderStatus.PENDING
    
    def test_execute_market_order(self, simulator):
        """Exécution d'ordre marché doit fonctionner."""
        order = simulator.place_order(
            symbol='EURUSD',
            side=PositionSide.LONG,
            quantity=1000
        )
        
        success = simulator.execute_order(order, 1.0850)
        
        assert success
        assert order.status == OrderStatus.FILLED
        assert 'EURUSD' in simulator.positions
    
    def test_position_created_on_execution(self, simulator):
        """Position doit être créée après exécution."""
        order = simulator.place_order(
            symbol='EURUSD',
            side=PositionSide.LONG,
            quantity=1000
        )
        simulator.execute_order(order, 1.0850)
        
        position = simulator.positions['EURUSD']
        
        assert position.symbol == 'EURUSD'
        assert position.side == PositionSide.LONG
        assert position.quantity == 1000
    
    def test_commission_deducted(self, simulator):
        """Commission doit être déduite."""
        initial_capital = simulator.capital
        
        order = simulator.place_order(
            symbol='EURUSD',
            side=PositionSide.LONG,
            quantity=1000
        )
        simulator.execute_order(order, 1.0850)
        
        expected_commission = 1000 * 1.0850 * simulator.commission_pct
        assert simulator.capital < initial_capital
    
    def test_stop_loss_triggered(self, simulator):
        """Stop loss doit fermer la position."""
        order = simulator.place_order(
            symbol='EURUSD',
            side=PositionSide.LONG,
            quantity=1000,
            stop_loss=1.0800
        )
        
        current_time = datetime.now()
        simulator.execute_order(order, 1.0850, current_time)
        
        # Prix touche le SL
        closed_trades = simulator.update_positions(
            {'EURUSD': 1.0800},
            current_time + timedelta(hours=1)
        )
        
        assert len(closed_trades) == 1
        assert closed_trades[0].exit_reason == "Stop Loss"
        assert 'EURUSD' not in simulator.positions
    
    def test_take_profit_triggered(self, simulator):
        """Take profit doit fermer la position."""
        order = simulator.place_order(
            symbol='EURUSD',
            side=PositionSide.LONG,
            quantity=1000,
            take_profit=1.0900
        )
        
        current_time = datetime.now()
        simulator.execute_order(order, 1.0850, current_time)
        
        # Prix atteint le TP
        closed_trades = simulator.update_positions(
            {'EURUSD': 1.0900},
            current_time + timedelta(hours=1)
        )
        
        assert len(closed_trades) == 1
        assert closed_trades[0].exit_reason == "Take Profit"
        assert closed_trades[0].pnl > 0
    
    def test_short_position(self, simulator):
        """Position short doit fonctionner."""
        order = simulator.place_order(
            symbol='EURUSD',
            side=PositionSide.SHORT,
            quantity=1000,
            stop_loss=1.0900,
            take_profit=1.0750
        )
        
        current_time = datetime.now()
        simulator.execute_order(order, 1.0850, current_time)
        
        # Prix baisse = profit pour short
        closed_trades = simulator.update_positions(
            {'EURUSD': 1.0750},
            current_time + timedelta(hours=1)
        )
        
        assert len(closed_trades) == 1
        assert closed_trades[0].pnl > 0
    
    def test_close_all_positions(self, simulator):
        """Close all doit fermer toutes les positions."""
        # Ouvrir plusieurs positions
        for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
            order = simulator.place_order(
                symbol=symbol,
                side=PositionSide.LONG,
                quantity=1000
            )
            simulator.execute_order(order, 100)
        
        assert len(simulator.positions) == 3
        
        # Fermer toutes
        simulator.close_all_positions({
            'EURUSD': 101,
            'GBPUSD': 101,
            'USDJPY': 101
        })
        
        assert len(simulator.positions) == 0
    
    def test_max_positions_limit(self, simulator):
        """Limite de positions doit être respectée."""
        simulator.max_positions = 2
        
        # Ouvrir 2 positions
        for symbol in ['EURUSD', 'GBPUSD']:
            order = simulator.place_order(symbol=symbol, side=PositionSide.LONG, quantity=1000)
            simulator.execute_order(order, 100)
        
        # 3ème position doit être rejetée
        order3 = simulator.place_order(symbol='USDJPY', side=PositionSide.LONG, quantity=1000)
        success = simulator.execute_order(order3, 100)
        
        assert not success
        assert order3.status == OrderStatus.REJECTED
    
    def test_insufficient_capital_rejected(self, simulator):
        """Capital insuffisant doit rejeter l'ordre."""
        simulator.capital = 100
        
        order = simulator.place_order(
            symbol='EURUSD',
            side=PositionSide.LONG,
            quantity=10000  # Trop grand pour le capital
        )
        
        success = simulator.execute_order(order, 100)
        
        assert not success
        assert order.status == OrderStatus.REJECTED
    
    def test_performance_report(self, simulator):
        """Rapport de performance doit être généré."""
        # Faire quelques trades
        order1 = simulator.place_order(
            symbol='EURUSD',
            side=PositionSide.LONG,
            quantity=1000,
            take_profit=1.0900
        )
        simulator.execute_order(order1, 1.0850)
        simulator.update_positions({'EURUSD': 1.0900})
        
        report = simulator.get_performance_report()
        
        assert 'capital' in report
        assert 'total_trades' in report
        assert 'win_rate' in report
        assert report['total_trades'] == 1
    
    def test_statistics_updated(self, simulator):
        """Statistiques doivent être mises à jour."""
        # Trade gagnant
        order = simulator.place_order(
            symbol='EURUSD',
            side=PositionSide.LONG,
            quantity=1000,
            take_profit=1.0900
        )
        simulator.execute_order(order, 1.0850)
        simulator.update_positions({'EURUSD': 1.0900})
        
        assert simulator.stats['total_trades'] == 1
        assert simulator.stats['winning_trades'] == 1
        assert simulator.stats['total_pnl'] > 0


class TestPositionClass:
    """Tests pour la classe Position."""
    
    def test_update_pnl_long(self):
        """P&L doit être correct pour position long."""
        position = Position(
            symbol='EURUSD',
            side=PositionSide.LONG,
            quantity=1000,
            entry_price=1.0850,
            entry_time=datetime.now()
        )
        
        position.update_pnl(1.0900)  # Prix monte
        
        assert position.unrealized_pnl > 0
        assert position.max_profit > 0
    
    def test_update_pnl_short(self):
        """P&L doit être correct pour position short."""
        position = Position(
            symbol='EURUSD',
            side=PositionSide.SHORT,
            quantity=1000,
            entry_price=1.0850,
            entry_time=datetime.now()
        )
        
        position.update_pnl(1.0800)  # Prix baisse
        
        assert position.unrealized_pnl > 0  # Profit car short


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
