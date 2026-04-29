"""
Module de détection des régimes de marché.
Identifie si le marché est en tendance, range ou haute volatilité.

Régimes:
- TRENDING_UP: Tendance haussière claire
- TRENDING_DOWN: Tendance baissière claire
- RANGING: Consolidation, pas de direction claire
- HIGH_VOLATILITY: Volatilité explosive, danger
- TRANSITION: Changement de régime en cours
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
import os




class MarketRegime(Enum):
    """Types de régimes de marché."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    TRANSITION = "transition"
    UNKNOWN = "unknown"


@dataclass
class RegimeAnalysis:
    """Résultat de l'analyse de régime."""
    current_regime: MarketRegime
    confidence: float  # 0-100
    trend_strength: float  # -100 à +100
    volatility_percentile: float  # 0-100
    regime_duration: int  # Nombre de bougies dans ce régime
    previous_regime: Optional[MarketRegime] = None
    recommendation: str = ""


class RegimeDetector:
    """
    Détecteur de régimes de marché multi-méthodes.
    
    Combine plusieurs indicateurs pour identifier le régime:
    1. ADX pour la force de tendance
    2. ATR percentile pour la volatilité
    3. Hurst exponent pour trend/mean-reversion
    4. Structure highs/lows pour direction
    """
    
    def __init__(
        self,
        adx_period: int = 14,
        atr_period: int = 14,
        lookback: int = 50,
        trend_threshold: float = 20.0,  # ADX > 20 = trending (plus sensible)
        volatility_high_pct: float = 80,  # Percentile > 80 = high vol
        volatility_low_pct: float = 20   # Percentile < 20 = low vol
    ):
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.lookback = lookback
        self.trend_threshold = trend_threshold
        self.volatility_high_pct = volatility_high_pct
        self.volatility_low_pct = volatility_low_pct
        
        self._regime_history: List[MarketRegime] = []
    
    def detect(self, df: pd.DataFrame) -> RegimeAnalysis:
        """
        Détecte le régime de marché actuel.
        
        Args:
            df: DataFrame OHLCV
        
        Returns:
            RegimeAnalysis avec le régime détecté
        """
        if len(df) < self.lookback:
            return RegimeAnalysis(
                current_regime=MarketRegime.UNKNOWN,
                confidence=0,
                trend_strength=0,
                volatility_percentile=50,
                regime_duration=0,
                recommendation="Données insuffisantes"
            )
        
        # 1. Calculer ADX (force de tendance)
        adx, plus_di, minus_di = self._calculate_adx(df)
        
        # 2. Calculer ATR percentile (volatilité relative)
        atr_percentile = self._calculate_volatility_percentile(df)
        
        # 3. Analyser la structure des prix
        trend_direction = self._analyze_price_structure(df)
        
        # 4. Calculer le Hurst exponent simplifié
        hurst = self._simple_hurst(df['Close'].values[-self.lookback:])
        
        # Déterminer le régime
        regime, confidence = self._determine_regime(
            adx, plus_di, minus_di,
            atr_percentile, trend_direction, hurst
        )
        
        # Calculer la force de tendance (-100 à +100)
        # Utilise ADX combiné à l'écart DI
        di_diff = plus_di - minus_di
        trend_strength = (adx * di_diff / 50) if adx > 15 else 0
        trend_strength = max(-100, min(100, trend_strength))
        
        # Durée du régime actuel
        self._regime_history.append(regime)
        regime_duration = self._count_consecutive_regime(regime)
        
        # Régime précédent
        previous = None
        if len(self._regime_history) > regime_duration:
            previous = self._regime_history[-(regime_duration + 1)]
        
        # Recommandation
        recommendation = self._generate_recommendation(
            regime, confidence, atr_percentile, trend_strength
        )
        
        return RegimeAnalysis(
            current_regime=regime,
            confidence=round(confidence, 1),
            trend_strength=round(trend_strength, 2),
            volatility_percentile=round(atr_percentile, 1),
            regime_duration=regime_duration,
            previous_regime=previous,
            recommendation=recommendation
        )
    
    def _calculate_adx(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """Calcule ADX, +DI et -DI en utilisant le lissage de Wilder."""
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        
        n = self.adx_period
        
        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                abs(high[1:] - close[:-1]),
                abs(low[1:] - close[:-1])
            )
        )
        
        # +DM et -DM
        plus_dm = np.maximum(high[1:] - high[:-1], 0)
        minus_dm = np.maximum(low[:-1] - low[1:], 0)
        
        # Appliquer la logique DM : le plus grand gagne
        plus_dm_mask = (plus_dm > minus_dm) & (plus_dm > 0)
        minus_dm_mask = (minus_dm > plus_dm) & (minus_dm > 0)
        
        plus_dm = np.where(plus_dm_mask, plus_dm, 0)
        minus_dm = np.where(minus_dm_mask, minus_dm, 0)
        
        # Lissage de Wilder (RMA)
        def wilder_smoothing(arr, period):
            result = np.zeros_like(arr, dtype=float)
            if len(arr) == 0: return result
            result[0] = np.mean(arr[:period]) if len(arr) >= period else arr[0]
            alpha = 1 / period
            for i in range(1, len(arr)):
                result[i] = arr[i] * alpha + result[i-1] * (1 - alpha)
            return result
        
        atr = wilder_smoothing(tr, n)
        plus_di = 100 * wilder_smoothing(plus_dm, n) / (atr + 1e-10)
        minus_di = 100 * wilder_smoothing(minus_dm, n) / (atr + 1e-10)
        
        # DX et ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = wilder_smoothing(dx, n)
        
        return float(adx[-1]), float(plus_di[-1]), float(minus_di[-1])
    
    def _calculate_volatility_percentile(self, df: pd.DataFrame) -> float:
        """Calcule le percentile de volatilité actuelle."""
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        
        # True Range
        tr = np.zeros(len(df) - 1)
        for i in range(1, len(df)):
            tr[i-1] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Lissage Wilder pour ATR
        def wilder_smoothing(arr, period):
            result = np.zeros_like(arr, dtype=float)
            if len(arr) == 0: return result
            result[0] = np.mean(arr[:period]) if len(arr) >= period else arr[0]
            alpha = 1 / period
            for i in range(1, len(arr)):
                result[i] = arr[i] * alpha + result[i-1] * (1 - alpha)
            return result

        atr_series = wilder_smoothing(tr, self.atr_period)
        current_atr = atr_series[-1]
        
        # Comparer aux 252 dernières bougies
        historical_atr = atr_series[-252:] if len(atr_series) > 252 else atr_series
        
        percentile = (historical_atr < current_atr).sum() / len(historical_atr) * 100
        
        return percentile
    
    def _analyze_price_structure(self, df: pd.DataFrame) -> str:
        """Analyse la structure Higher Highs/Lows ou Lower Highs/Lows."""
        high = df['High'].values[-self.lookback:]
        low = df['Low'].values[-self.lookback:]
        
        # Trouver les swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(high) - 2):
            if high[i] > high[i-1] and high[i] > high[i-2] and \
               high[i] > high[i+1] and high[i] > high[i+2]:
                swing_highs.append(high[i])
            
            if low[i] < low[i-1] and low[i] < low[i-2] and \
               low[i] < low[i+1] and low[i] < low[i+2]:
                swing_lows.append(low[i])
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "neutral"
        
        # Higher Highs et Higher Lows = Uptrend
        hh = swing_highs[-1] > swing_highs[-2]
        hl = swing_lows[-1] > swing_lows[-2]
        
        # Lower Highs et Lower Lows = Downtrend
        lh = swing_highs[-1] < swing_highs[-2]
        ll = swing_lows[-1] < swing_lows[-2]
        
        if hh and hl:
            return "uptrend"
        elif lh and ll:
            return "downtrend"
        else:
            return "neutral"
    
    def _simple_hurst(self, prices: np.ndarray) -> float:
        """Calcul simplifié du Hurst exponent."""
        if len(prices) < 20:
            return 0.5
        
        n = len(prices)
        t = np.arange(1, n + 1)
        y = np.log(prices)
        
        # R/S analysis simplifié
        mean_y = np.mean(y)
        cumulative = np.cumsum(y - mean_y)
        r = np.max(cumulative) - np.min(cumulative)
        s = np.std(y)
        
        if s == 0:
            return 0.5
        
        rs = r / s
        hurst = np.log(rs) / np.log(n)
        
        return min(1, max(0, hurst))
    
    def _determine_regime(
        self,
        adx: float,
        plus_di: float,
        minus_di: float,
        volatility_pct: float,
        trend_direction: str,
        hurst: float
    ) -> Tuple[MarketRegime, float]:
        """Détermine le régime basé sur tous les indicateurs."""
        
        # Score de confiance
        confidence = 50.0
        
        # Haute volatilité override
        if volatility_pct > self.volatility_high_pct:
            confidence = 70 + (volatility_pct - 80) * 1.5
            return MarketRegime.HIGH_VOLATILITY, min(100, confidence)
        
        # Trending
        if adx > self.trend_threshold:
            confidence = 50 + adx
            
            if plus_di > minus_di and trend_direction in ["uptrend", "neutral"]:
                return MarketRegime.TRENDING_UP, min(100, confidence)
            elif minus_di > plus_di and trend_direction in ["downtrend", "neutral"]:
                return MarketRegime.TRENDING_DOWN, min(100, confidence)
            else:
                # Conflit entre DI et structure
                return MarketRegime.TRANSITION, confidence * 0.7
        
        # Ranging (faible ADX)
        if adx < 20 and hurst < 0.5:
            confidence = 50 + (20 - adx) * 2
            return MarketRegime.RANGING, min(100, confidence)
        
        # Transition
        if 20 <= adx <= self.trend_threshold:
            return MarketRegime.TRANSITION, 40 + adx
        
        return MarketRegime.UNKNOWN, 30
    
    def _count_consecutive_regime(self, current: MarketRegime) -> int:
        """Compte le nombre de bougies consécutives dans le même régime."""
        count = 0
        for regime in reversed(self._regime_history):
            if regime == current:
                count += 1
            else:
                break
        return count
    
    def _generate_recommendation(
        self,
        regime: MarketRegime,
        confidence: float,
        volatility: float,
        trend_strength: float
    ) -> str:
        """Génère une recommandation de trading."""
        
        if regime == MarketRegime.HIGH_VOLATILITY:
            return "⚠️ VOLATILITÉ EXTRÊME - Réduire la taille des positions ou éviter"
        
        if regime == MarketRegime.TRANSITION:
            return "⏳ TRANSITION - Attendre confirmation du nouveau régime"
        
        if regime == MarketRegime.RANGING:
            return "📊 RANGE - Stratégies mean-reversion, acheter bas / vendre haut"
        
        if regime == MarketRegime.TRENDING_UP:
            if confidence > 70:
                return "📈 TENDANCE HAUSSIÈRE FORTE - Acheter les pullbacks"
            return "📈 TENDANCE HAUSSIÈRE - Privilégier les longs"
        
        if regime == MarketRegime.TRENDING_DOWN:
            if confidence > 70:
                return "📉 TENDANCE BAISSIÈRE FORTE - Vendre les rallies"
            return "📉 TENDANCE BAISSIÈRE - Privilégier les shorts"
        
        return "❓ Régime incertain - Prudence recommandée"
    
    def get_regime_for_strategy(self, regime: MarketRegime) -> Dict:
        """
        Retourne les paramètres de stratégie adaptés au régime.
        """
        strategies = {
            MarketRegime.TRENDING_UP: {
                'strategy_type': 'trend_following',
                'direction_bias': 'long_only',
                'entry_method': 'pullback',
                'stop_multiplier': 1.5,
                'position_size_factor': 1.0,
                'indicators': ['EMA', 'MACD', 'ADX']
            },
            MarketRegime.TRENDING_DOWN: {
                'strategy_type': 'trend_following',
                'direction_bias': 'short_only',
                'entry_method': 'rally',
                'stop_multiplier': 1.5,
                'position_size_factor': 1.0,
                'indicators': ['EMA', 'MACD', 'ADX']
            },
            MarketRegime.RANGING: {
                'strategy_type': 'mean_reversion',
                'direction_bias': 'both',
                'entry_method': 'extremes',
                'stop_multiplier': 1.0,
                'position_size_factor': 0.8,
                'indicators': ['RSI', 'Bollinger', 'Stochastic']
            },
            MarketRegime.HIGH_VOLATILITY: {
                'strategy_type': 'reduced_exposure',
                'direction_bias': 'avoid',
                'entry_method': 'none',
                'stop_multiplier': 2.0,
                'position_size_factor': 0.3,
                'indicators': ['ATR', 'VIX']
            },
            MarketRegime.TRANSITION: {
                'strategy_type': 'wait',
                'direction_bias': 'neutral',
                'entry_method': 'confirmation',
                'stop_multiplier': 1.5,
                'position_size_factor': 0.5,
                'indicators': ['ADX', 'Volume']
            }
        }
        
        return strategies.get(regime, strategies[MarketRegime.TRANSITION])


if __name__ == "__main__":
    print("=" * 60)
    print("TEST REGIME DETECTOR")
    print("=" * 60)
    
    # Données de test - Tendance haussière
    np.random.seed(42)
    n = 200
    
    # Simuler une tendance puis un range
    trend = np.cumsum(np.random.randn(100) * 0.5 + 0.1)  # Uptrend
    ranging = trend[-1] + np.cumsum(np.random.randn(100) * 0.5)  # Range
    close = np.concatenate([trend, ranging])
    
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    df = pd.DataFrame({
        'Open': close + np.random.randn(n) * 0.2,
        'High': close + abs(np.random.randn(n)) * 0.5,
        'Low': close - abs(np.random.randn(n)) * 0.5,
        'Close': close,
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    detector = RegimeDetector()
    
    # Analyser à différents moments
    print("\n--- Analyse du Régime ---")
    
    for i in [50, 100, 150, 199]:
        result = detector.detect(df.iloc[:i+1])
        print(f"\nBougie {i+1}:")
        print(f"  Régime: {result.current_regime.value}")
        print(f"  Confiance: {result.confidence}%")
        print(f"  Force tendance: {result.trend_strength}")
        print(f"  Volatilité: {result.volatility_percentile}%")
        print(f"  Recommandation: {result.recommendation}")
    
    # Paramètres de stratégie
    print("\n--- Paramètres Stratégie Adaptés ---")
    final_result = detector.detect(df)
    strategy_params = detector.get_regime_for_strategy(final_result.current_regime)
    for k, v in strategy_params.items():
        print(f"  {k}: {v}")
