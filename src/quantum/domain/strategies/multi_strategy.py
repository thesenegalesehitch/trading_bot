"""
Moteur Multi-Stratégie.
Gère plusieurs stratégies de trading et alloue le capital dynamiquement.

Stratégies implémentées:
1. Trend Following (Momentum)
2. Mean Reversion
3. Breakout
4. Statistical Arbitrage
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import sys
import os


from quantum.domain.core.regime_detector import RegimeDetector, MarketRegime


class SignalType(Enum):
    """Type de signal."""
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    HOLD = "hold"


@dataclass
class TradeSignal:
    """Signal de trading généré par une stratégie."""
    signal_type: SignalType
    symbol: str
    strategy_name: str
    confidence: float  # 0-100
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


class BaseStrategy(ABC):
    """Classe de base pour toutes les stratégies."""
    
    def __init__(self, name: str, params: Dict = None):
        self.name = name
        self.params = params or {}
        self.is_active = True
        self.performance_score = 50.0  # Score initial neutre
    
    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        regime: MarketRegime
    ) -> Optional[TradeSignal]:
        """Génère un signal de trading."""
        pass
    
    @abstractmethod
    def get_suitable_regimes(self) -> List[MarketRegime]:
        """Retourne les régimes où cette stratégie performe bien."""
        pass
    
    def update_performance(self, pnl: float):
        """Met à jour le score de performance basé sur les résultats."""
        # EMA du score
        alpha = 0.1
        result_score = 100 if pnl > 0 else 0
        self.performance_score = alpha * result_score + (1 - alpha) * self.performance_score


class TrendFollowingStrategy(BaseStrategy):
    """
    Stratégie de suivi de tendance.
    
    Logique:
    - Entre dans la direction de la tendance sur pullback
    - Utilise EMA croisement + confirmation momentum
    """
    
    def __init__(self, params: Dict = None):
        default_params = {
            'fast_ema': 12,
            'slow_ema': 26,
            'signal_ema': 9,
            'atr_period': 14,
            'atr_multiplier': 2.0
        }
        super().__init__("TrendFollowing", {**default_params, **(params or {})})
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        regime: MarketRegime
    ) -> Optional[TradeSignal]:
        if len(df) < 50:
            return None
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # EMAs
        fast_ema = close.ewm(span=self.params['fast_ema']).mean()
        slow_ema = close.ewm(span=self.params['slow_ema']).mean()
        
        # MACD
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=self.params['signal_ema']).mean()
        
        # ATR pour stop loss
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.params['atr_period']).mean()
        
        current_price = close.iloc[-1]
        current_atr = atr.iloc[-1]
        
        # Conditions de signal
        ema_bullish = fast_ema.iloc[-1] > slow_ema.iloc[-1]
        ema_bearish = fast_ema.iloc[-1] < slow_ema.iloc[-1]
        
        macd_cross_up = macd.iloc[-1] > signal_line.iloc[-1] and macd.iloc[-2] <= signal_line.iloc[-2]
        macd_cross_down = macd.iloc[-1] < signal_line.iloc[-1] and macd.iloc[-2] >= signal_line.iloc[-2]
        
        # Pullback detection
        pullback_buy = (close.iloc[-1] < fast_ema.iloc[-1]) and ema_bullish
        pullback_sell = (close.iloc[-1] > fast_ema.iloc[-1]) and ema_bearish
        
        if (ema_bullish and macd_cross_up) or (ema_bullish and pullback_buy and macd.iloc[-1] > 0):
            return TradeSignal(
                signal_type=SignalType.BUY,
                symbol=df.attrs.get('symbol', 'UNKNOWN'),
                strategy_name=self.name,
                confidence=min(90, 60 + abs(macd.iloc[-1]) * 100),
                price=current_price,
                stop_loss=current_price - current_atr * self.params['atr_multiplier'],
                take_profit=current_price + current_atr * self.params['atr_multiplier'] * 2,
                metadata={'macd': float(macd.iloc[-1]), 'atr': float(current_atr)}
            )
        
        if (ema_bearish and macd_cross_down) or (ema_bearish and pullback_sell and macd.iloc[-1] < 0):
            return TradeSignal(
                signal_type=SignalType.SELL,
                symbol=df.attrs.get('symbol', 'UNKNOWN'),
                strategy_name=self.name,
                confidence=min(90, 60 + abs(macd.iloc[-1]) * 100),
                price=current_price,
                stop_loss=current_price + current_atr * self.params['atr_multiplier'],
                take_profit=current_price - current_atr * self.params['atr_multiplier'] * 2,
                metadata={'macd': float(macd.iloc[-1]), 'atr': float(current_atr)}
            )
        
        return None
    
    def get_suitable_regimes(self) -> List[MarketRegime]:
        return [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]


class MeanReversionStrategy(BaseStrategy):
    """
    Stratégie de retour à la moyenne.
    
    Logique:
    - Achète quand le prix est extrêmement bas (oversold)
    - Vend quand le prix est extrêmement haut (overbought)
    - Utilise Bollinger Bands + RSI
    """
    
    def __init__(self, params: Dict = None):
        default_params = {
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }
        super().__init__("MeanReversion", {**default_params, **(params or {})})
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        regime: MarketRegime
    ) -> Optional[TradeSignal]:
        if len(df) < 50:
            return None
        
        close = df['Close']
        
        # Bollinger Bands
        sma = close.rolling(self.params['bb_period']).mean()
        std = close.rolling(self.params['bb_period']).std()
        upper_band = sma + std * self.params['bb_std']
        lower_band = sma - std * self.params['bb_std']
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(self.params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.params['rsi_period']).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        current_price = close.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # BB position (0 = lower, 1 = upper)
        bb_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1] + 1e-10)
        
        # Conditions
        oversold = current_rsi < self.params['rsi_oversold'] and bb_position < 0.1
        overbought = current_rsi > self.params['rsi_overbought'] and bb_position > 0.9
        
        # Target = SMA
        target = sma.iloc[-1]
        
        if oversold:
            confidence = min(90, 50 + (self.params['rsi_oversold'] - current_rsi) * 2)
            return TradeSignal(
                signal_type=SignalType.BUY,
                symbol=df.attrs.get('symbol', 'UNKNOWN'),
                strategy_name=self.name,
                confidence=confidence,
                price=current_price,
                stop_loss=lower_band.iloc[-1] - std.iloc[-1],
                take_profit=target,
                metadata={'rsi': float(current_rsi), 'bb_position': float(bb_position)}
            )
        
        if overbought:
            confidence = min(90, 50 + (current_rsi - self.params['rsi_overbought']) * 2)
            return TradeSignal(
                signal_type=SignalType.SELL,
                symbol=df.attrs.get('symbol', 'UNKNOWN'),
                strategy_name=self.name,
                confidence=confidence,
                price=current_price,
                stop_loss=upper_band.iloc[-1] + std.iloc[-1],
                take_profit=target,
                metadata={'rsi': float(current_rsi), 'bb_position': float(bb_position)}
            )
        
        return None
    
    def get_suitable_regimes(self) -> List[MarketRegime]:
        return [MarketRegime.RANGING]


class BreakoutStrategy(BaseStrategy):
    """
    Stratégie de breakout.
    
    Logique:
    - Détecte les cassures de range/consolidation
    - Entre sur breakout avec confirmation volume
    """
    
    def __init__(self, params: Dict = None):
        default_params = {
            'lookback': 20,
            'volume_threshold': 1.5,  # 1.5x le volume moyen
            'atr_period': 14
        }
        super().__init__("Breakout", {**default_params, **(params or {})})
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        regime: MarketRegime
    ) -> Optional[TradeSignal]:
        if len(df) < 50:
            return None
        
        lookback = self.params['lookback']
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df.get('Volume', pd.Series(np.ones(len(df))))
        
        # Range des N dernières bougies (excluant la dernière)
        range_high = high.iloc[-lookback-1:-1].max()
        range_low = low.iloc[-lookback-1:-1].min()
        
        current_price = close.iloc[-1]
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_volume = volume.iloc[-1]
        avg_volume = volume.iloc[-lookback:].mean()
        
        # ATR
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.params['atr_period']).mean().iloc[-1]
        
        # Conditions de breakout
        volume_confirm = current_volume > avg_volume * self.params['volume_threshold']
        
        breakout_up = current_high > range_high
        breakout_down = current_low < range_low
        
        if breakout_up and volume_confirm:
            return TradeSignal(
                signal_type=SignalType.BUY,
                symbol=df.attrs.get('symbol', 'UNKNOWN'),
                strategy_name=self.name,
                confidence=min(85, 60 + (current_volume / avg_volume) * 10),
                price=current_price,
                stop_loss=range_low - atr,
                take_profit=current_price + (current_price - range_low),
                metadata={'range_high': float(range_high), 'volume_ratio': float(current_volume / avg_volume)}
            )
        
        if breakout_down and volume_confirm:
            return TradeSignal(
                signal_type=SignalType.SELL,
                symbol=df.attrs.get('symbol', 'UNKNOWN'),
                strategy_name=self.name,
                confidence=min(85, 60 + (current_volume / avg_volume) * 10),
                price=current_price,
                stop_loss=range_high + atr,
                take_profit=current_price - (range_high - current_price),
                metadata={'range_low': float(range_low), 'volume_ratio': float(current_volume / avg_volume)}
            )
        
        return None
    
    def get_suitable_regimes(self) -> List[MarketRegime]:
        return [MarketRegime.TRANSITION, MarketRegime.RANGING]


class MultiStrategyEngine:
    """
    Moteur qui gère plusieurs stratégies et alloue le capital.
    
    Fonctionnalités:
    - Rotation automatique selon le régime de marché
    - Pondération par performance historique
    - Agrégation des signaux multiples
    """
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.regime_detector = RegimeDetector()
        self.capital_allocation: Dict[str, float] = {}
        
        # Charger les stratégies par défaut
        self._load_default_strategies()
    
    def _load_default_strategies(self):
        """Charge les stratégies par défaut."""
        self.add_strategy(TrendFollowingStrategy())
        self.add_strategy(MeanReversionStrategy())
        self.add_strategy(BreakoutStrategy())
    
    def add_strategy(self, strategy: BaseStrategy):
        """Ajoute une stratégie au moteur."""
        self.strategies[strategy.name] = strategy
        self.capital_allocation[strategy.name] = 1.0 / len(self.strategies)
        self._rebalance_allocation()
    
    def remove_strategy(self, name: str):
        """Retire une stratégie du moteur."""
        if name in self.strategies:
            del self.strategies[name]
            del self.capital_allocation[name]
            self._rebalance_allocation()
    
    def _rebalance_allocation(self):
        """Rééquilibre l'allocation entre stratégies."""
        if not self.strategies:
            return
        
        # Allocation basée sur la performance
        total_score = sum(s.performance_score for s in self.strategies.values())
        
        if total_score > 0:
            for name, strategy in self.strategies.items():
                self.capital_allocation[name] = strategy.performance_score / total_score
    
    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> List[TradeSignal]:
        """
        Génère des signaux de toutes les stratégies actives.
        
        Args:
            df: DataFrame OHLCV
            symbol: Symbole du marché
        
        Returns:
            Liste de signaux pondérés par allocation
        """
        df.attrs['symbol'] = symbol
        
        # Détecter le régime
        regime_analysis = self.regime_detector.detect(df)
        current_regime = regime_analysis.current_regime
        
        signals = []
        
        for name, strategy in self.strategies.items():
            if not strategy.is_active:
                continue
            
            # Vérifier si le régime est adapté
            suitable_regimes = strategy.get_suitable_regimes()
            regime_factor = 1.0 if current_regime in suitable_regimes else 0.3
            
            try:
                signal = strategy.generate_signal(df, current_regime)
                
                if signal:
                    # Ajuster la confiance par le facteur de régime et l'allocation
                    signal.confidence *= regime_factor
                    signal.confidence *= self.capital_allocation.get(name, 1.0)
                    signal.metadata['regime'] = current_regime.value
                    signal.metadata['regime_factor'] = regime_factor
                    signal.metadata['allocation'] = self.capital_allocation.get(name, 0)
                    signals.append(signal)
            
            except Exception as e:
                print(f"Erreur stratégie {name}: {e}")
        
        return signals
    
    def get_consensus_signal(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        min_confidence: float = 50.0
    ) -> Optional[TradeSignal]:
        """
        Retourne un signal consensus basé sur toutes les stratégies.
        
        Agrège les signaux multiples en un signal unique.
        """
        signals = self.generate_signals(df, symbol)
        
        if not signals:
            return None
        
        # Séparer buy et sell
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        
        buy_confidence = sum(s.confidence for s in buy_signals)
        sell_confidence = sum(s.confidence for s in sell_signals)
        
        # Déterminer la direction dominante
        if buy_confidence > sell_confidence and buy_confidence >= min_confidence:
            # Agrégat buy
            avg_sl = np.mean([s.stop_loss for s in buy_signals if s.stop_loss])
            avg_tp = np.mean([s.take_profit for s in buy_signals if s.take_profit])
            
            return TradeSignal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                strategy_name="Consensus",
                confidence=buy_confidence / len(self.strategies),
                price=df['Close'].iloc[-1],
                stop_loss=avg_sl if avg_sl else None,
                take_profit=avg_tp if avg_tp else None,
                metadata={
                    'contributing_strategies': [s.strategy_name for s in buy_signals],
                    'total_signals': len(signals)
                }
            )
        
        elif sell_confidence > buy_confidence and sell_confidence >= min_confidence:
            avg_sl = np.mean([s.stop_loss for s in sell_signals if s.stop_loss])
            avg_tp = np.mean([s.take_profit for s in sell_signals if s.take_profit])
            
            return TradeSignal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                strategy_name="Consensus",
                confidence=sell_confidence / len(self.strategies),
                price=df['Close'].iloc[-1],
                stop_loss=avg_sl if avg_sl else None,
                take_profit=avg_tp if avg_tp else None,
                metadata={
                    'contributing_strategies': [s.strategy_name for s in sell_signals],
                    'total_signals': len(signals)
                }
            )
        
        return None
    
    def update_strategy_performance(self, strategy_name: str, pnl: float):
        """Met à jour la performance d'une stratégie après un trade."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].update_performance(pnl)
            self._rebalance_allocation()
    
    def get_status(self) -> Dict:
        """Retourne le statut de toutes les stratégies."""
        return {
            'strategies': {
                name: {
                    'active': s.is_active,
                    'performance_score': round(s.performance_score, 2),
                    'allocation': round(self.capital_allocation.get(name, 0) * 100, 1),
                    'suitable_regimes': [r.value for r in s.get_suitable_regimes()]
                }
                for name, s in self.strategies.items()
            },
            'total_strategies': len(self.strategies)
        }


if __name__ == "__main__":
    print("=" * 60)
    print("TEST MULTI-STRATEGY ENGINE")
    print("=" * 60)
    
    # Données de test
    np.random.seed(42)
    n = 200
    
    # Simuler un marché trending puis ranging
    trend = 100 + np.cumsum(np.random.randn(100) * 0.5 + 0.1)
    ranging = trend[-1] + np.cumsum(np.random.randn(100) * 0.3)
    close = np.concatenate([trend, ranging])
    
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    df = pd.DataFrame({
        'Open': close + np.random.randn(n) * 0.2,
        'High': close + abs(np.random.randn(n)) * 0.5,
        'Low': close - abs(np.random.randn(n)) * 0.5,
        'Close': close,
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    # Initialiser le moteur
    engine = MultiStrategyEngine()
    
    print("\n--- Statut des Stratégies ---")
    status = engine.get_status()
    for name, info in status['strategies'].items():
        print(f"  {name}:")
        print(f"    Allocation: {info['allocation']}%")
        print(f"    Régimes: {', '.join(info['suitable_regimes'])}")
    
    # Générer des signaux
    print("\n--- Signaux Générés ---")
    signals = engine.generate_signals(df, "EURUSD")
    
    for signal in signals:
        print(f"\n  {signal.strategy_name}:")
        print(f"    Type: {signal.signal_type.value}")
        print(f"    Confiance: {signal.confidence:.1f}%")
        print(f"    Prix: {signal.price:.4f}")
        if signal.stop_loss:
            print(f"    SL: {signal.stop_loss:.4f}")
        if signal.take_profit:
            print(f"    TP: {signal.take_profit:.4f}")
    
    # Signal consensus
    print("\n--- Signal Consensus ---")
    consensus = engine.get_consensus_signal(df, "EURUSD")
    if consensus:
        print(f"  Direction: {consensus.signal_type.value}")
        print(f"  Confiance: {consensus.confidence:.1f}%")
        print(f"  Stratégies: {consensus.metadata.get('contributing_strategies', [])}")
    else:
        print("  Aucun consensus")
