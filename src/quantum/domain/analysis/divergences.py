"""
Module de détection automatique des divergences.
Identifie les divergences bullish/bearish sur RSI, MACD, OBV et CCI.

Types de divergences:
- Regular Bullish: Prix fait un lower low, indicateur fait un higher low
- Regular Bearish: Prix fait un higher high, indicateur fait un lower high
- Hidden Bullish: Prix fait un higher low, indicateur fait un lower low
- Hidden Bearish: Prix fait un lower high, indicateur fait un higher high
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import sys
import os


from quantum.shared.config.settings import config


class DivergenceType(Enum):
    """Types de divergences."""
    REGULAR_BULLISH = "regular_bullish"
    REGULAR_BEARISH = "regular_bearish"
    HIDDEN_BULLISH = "hidden_bullish"
    HIDDEN_BEARISH = "hidden_bearish"


@dataclass
class Divergence:
    """Représente une divergence détectée."""
    type: DivergenceType
    indicator: str
    price_point1: Tuple[int, float]  # (index, value)
    price_point2: Tuple[int, float]
    indicator_point1: Tuple[int, float]
    indicator_point2: Tuple[int, float]
    strength: float  # 0-1
    timestamp: Optional[pd.Timestamp] = None


class DivergenceDetector:
    """
    Détecte automatiquement les divergences entre le prix et les indicateurs.
    
    Indicateurs supportés:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - OBV (On-Balance Volume)
    - CCI (Commodity Channel Index)
    - Stochastic
    """
    
    def __init__(
        self,
        lookback: int = 50,
        min_swing_distance: int = 5,
        strength_threshold: float = 0.3
    ):
        self.lookback = lookback
        self.min_swing_distance = min_swing_distance
        self.strength_threshold = strength_threshold
    
    def detect_all(self, df: pd.DataFrame) -> Dict[str, List[Divergence]]:
        """
        Détecte toutes les divergences sur tous les indicateurs.
        
        Args:
            df: DataFrame avec colonnes prix et indicateurs
        
        Returns:
            Dict avec divergences par indicateur
        """
        divergences = {}
        
        # RSI
        if 'rsi' in df.columns:
            rsi_divs = self.detect_divergence(df, 'rsi')
            if rsi_divs:
                divergences['rsi'] = rsi_divs
        
        # MACD
        if 'macd' in df.columns:
            macd_divs = self.detect_divergence(df, 'macd')
            if macd_divs:
                divergences['macd'] = macd_divs
        
        # MACD Histogram
        if 'macd_hist' in df.columns:
            hist_divs = self.detect_divergence(df, 'macd_hist')
            if hist_divs:
                divergences['macd_hist'] = hist_divs
        
        # OBV
        if 'obv' in df.columns:
            obv_divs = self.detect_divergence(df, 'obv')
            if obv_divs:
                divergences['obv'] = obv_divs
        
        # CCI
        if 'cci' in df.columns:
            cci_divs = self.detect_divergence(df, 'cci')
            if cci_divs:
                divergences['cci'] = cci_divs
        
        # Stochastic
        if 'stoch_k' in df.columns:
            stoch_divs = self.detect_divergence(df, 'stoch_k')
            if stoch_divs:
                divergences['stochastic'] = stoch_divs
        
        return divergences
    
    def detect_divergence(
        self,
        df: pd.DataFrame,
        indicator_col: str,
        price_col: str = 'Close'
    ) -> List[Divergence]:
        """
        Détecte les divergences entre le prix et un indicateur.
        
        Args:
            df: DataFrame
            indicator_col: Nom de la colonne indicateur
            price_col: Nom de la colonne prix
        
        Returns:
            Liste des divergences détectées
        """
        if indicator_col not in df.columns or price_col not in df.columns:
            return []
        
        df_work = df.tail(self.lookback).copy()
        
        if len(df_work) < self.min_swing_distance * 2:
            return []
        
        price = df_work[price_col].values
        indicator = df_work[indicator_col].values
        
        # Trouver les swing points
        price_highs = self._find_swing_highs(price)
        price_lows = self._find_swing_lows(price)
        ind_highs = self._find_swing_highs(indicator)
        ind_lows = self._find_swing_lows(indicator)
        
        divergences = []
        
        # Vérifier Regular Bullish (prix lower low, indicateur higher low)
        if len(price_lows) >= 2 and len(ind_lows) >= 2:
            div = self._check_regular_bullish(
                price, indicator, price_lows, ind_lows, df_work.index, indicator_col
            )
            if div:
                divergences.append(div)
        
        # Vérifier Regular Bearish (prix higher high, indicateur lower high)
        if len(price_highs) >= 2 and len(ind_highs) >= 2:
            div = self._check_regular_bearish(
                price, indicator, price_highs, ind_highs, df_work.index, indicator_col
            )
            if div:
                divergences.append(div)
        
        # Vérifier Hidden Bullish (prix higher low, indicateur lower low)
        if len(price_lows) >= 2 and len(ind_lows) >= 2:
            div = self._check_hidden_bullish(
                price, indicator, price_lows, ind_lows, df_work.index, indicator_col
            )
            if div:
                divergences.append(div)
        
        # Vérifier Hidden Bearish (prix lower high, indicateur higher high)
        if len(price_highs) >= 2 and len(ind_highs) >= 2:
            div = self._check_hidden_bearish(
                price, indicator, price_highs, ind_highs, df_work.index, indicator_col
            )
            if div:
                divergences.append(div)
        
        return divergences
    
    def _find_swing_highs(self, data: np.ndarray, window: int = 5) -> List[int]:
        """Trouve les swing highs."""
        highs = []
        for i in range(window, len(data) - window):
            if all(data[i] >= data[i-j] for j in range(1, window+1)) and \
               all(data[i] >= data[i+j] for j in range(1, window+1)):
                highs.append(i)
        return highs
    
    def _find_swing_lows(self, data: np.ndarray, window: int = 5) -> List[int]:
        """Trouve les swing lows."""
        lows = []
        for i in range(window, len(data) - window):
            if all(data[i] <= data[i-j] for j in range(1, window+1)) and \
               all(data[i] <= data[i+j] for j in range(1, window+1)):
                lows.append(i)
        return lows
    
    def _check_regular_bullish(
        self,
        price: np.ndarray,
        indicator: np.ndarray,
        price_lows: List[int],
        ind_lows: List[int],
        index: pd.Index,
        indicator_name: str
    ) -> Optional[Divergence]:
        """Vérifie une divergence bullish régulière."""
        # Prendre les deux derniers lows
        if len(price_lows) < 2:
            return None
        
        # Trouver les paires de lows les plus récentes
        p1, p2 = price_lows[-2], price_lows[-1]
        
        # Vérifier que le prix fait un lower low
        if price[p2] >= price[p1]:
            return None
        
        # Trouver les lows correspondants de l'indicateur
        i1 = self._find_closest_swing(ind_lows, p1)
        i2 = self._find_closest_swing(ind_lows, p2)
        
        if i1 is None or i2 is None:
            return None
        
        # Vérifier que l'indicateur fait un higher low
        if indicator[i2] <= indicator[i1]:
            return None
        
        # Calculer la force
        price_diff = abs(price[p2] - price[p1]) / price[p1]
        ind_diff = abs(indicator[i2] - indicator[i1]) / (abs(indicator[i1]) + 1e-10)
        strength = min((price_diff + ind_diff) / 0.1, 1.0)
        
        if strength < self.strength_threshold:
            return None
        
        return Divergence(
            type=DivergenceType.REGULAR_BULLISH,
            indicator=indicator_name,
            price_point1=(p1, price[p1]),
            price_point2=(p2, price[p2]),
            indicator_point1=(i1, indicator[i1]),
            indicator_point2=(i2, indicator[i2]),
            strength=strength,
            timestamp=index[p2] if isinstance(index[p2], pd.Timestamp) else None
        )
    
    def _check_regular_bearish(
        self,
        price: np.ndarray,
        indicator: np.ndarray,
        price_highs: List[int],
        ind_highs: List[int],
        index: pd.Index,
        indicator_name: str
    ) -> Optional[Divergence]:
        """Vérifie une divergence bearish régulière."""
        if len(price_highs) < 2:
            return None
        
        p1, p2 = price_highs[-2], price_highs[-1]
        
        # Vérifier que le prix fait un higher high
        if price[p2] <= price[p1]:
            return None
        
        i1 = self._find_closest_swing(ind_highs, p1)
        i2 = self._find_closest_swing(ind_highs, p2)
        
        if i1 is None or i2 is None:
            return None
        
        # Vérifier que l'indicateur fait un lower high
        if indicator[i2] >= indicator[i1]:
            return None
        
        price_diff = abs(price[p2] - price[p1]) / price[p1]
        ind_diff = abs(indicator[i2] - indicator[i1]) / (abs(indicator[i1]) + 1e-10)
        strength = min((price_diff + ind_diff) / 0.1, 1.0)
        
        if strength < self.strength_threshold:
            return None
        
        return Divergence(
            type=DivergenceType.REGULAR_BEARISH,
            indicator=indicator_name,
            price_point1=(p1, price[p1]),
            price_point2=(p2, price[p2]),
            indicator_point1=(i1, indicator[i1]),
            indicator_point2=(i2, indicator[i2]),
            strength=strength,
            timestamp=index[p2] if isinstance(index[p2], pd.Timestamp) else None
        )
    
    def _check_hidden_bullish(
        self,
        price: np.ndarray,
        indicator: np.ndarray,
        price_lows: List[int],
        ind_lows: List[int],
        index: pd.Index,
        indicator_name: str
    ) -> Optional[Divergence]:
        """Vérifie une divergence bullish cachée."""
        if len(price_lows) < 2:
            return None
        
        p1, p2 = price_lows[-2], price_lows[-1]
        
        # Vérifier que le prix fait un higher low
        if price[p2] <= price[p1]:
            return None
        
        i1 = self._find_closest_swing(ind_lows, p1)
        i2 = self._find_closest_swing(ind_lows, p2)
        
        if i1 is None or i2 is None:
            return None
        
        # Vérifier que l'indicateur fait un lower low
        if indicator[i2] >= indicator[i1]:
            return None
        
        price_diff = abs(price[p2] - price[p1]) / price[p1]
        ind_diff = abs(indicator[i2] - indicator[i1]) / (abs(indicator[i1]) + 1e-10)
        strength = min((price_diff + ind_diff) / 0.1, 1.0) * 0.8  # Hidden = légèrement moins fort
        
        if strength < self.strength_threshold:
            return None
        
        return Divergence(
            type=DivergenceType.HIDDEN_BULLISH,
            indicator=indicator_name,
            price_point1=(p1, price[p1]),
            price_point2=(p2, price[p2]),
            indicator_point1=(i1, indicator[i1]),
            indicator_point2=(i2, indicator[i2]),
            strength=strength,
            timestamp=index[p2] if isinstance(index[p2], pd.Timestamp) else None
        )
    
    def _check_hidden_bearish(
        self,
        price: np.ndarray,
        indicator: np.ndarray,
        price_highs: List[int],
        ind_highs: List[int],
        index: pd.Index,
        indicator_name: str
    ) -> Optional[Divergence]:
        """Vérifie une divergence bearish cachée."""
        if len(price_highs) < 2:
            return None
        
        p1, p2 = price_highs[-2], price_highs[-1]
        
        # Vérifier que le prix fait un lower high
        if price[p2] >= price[p1]:
            return None
        
        i1 = self._find_closest_swing(ind_highs, p1)
        i2 = self._find_closest_swing(ind_highs, p2)
        
        if i1 is None or i2 is None:
            return None
        
        # Vérifier que l'indicateur fait un higher high
        if indicator[i2] <= indicator[i1]:
            return None
        
        price_diff = abs(price[p2] - price[p1]) / price[p1]
        ind_diff = abs(indicator[i2] - indicator[i1]) / (abs(indicator[i1]) + 1e-10)
        strength = min((price_diff + ind_diff) / 0.1, 1.0) * 0.8
        
        if strength < self.strength_threshold:
            return None
        
        return Divergence(
            type=DivergenceType.HIDDEN_BEARISH,
            indicator=indicator_name,
            price_point1=(p1, price[p1]),
            price_point2=(p2, price[p2]),
            indicator_point1=(i1, indicator[i1]),
            indicator_point2=(i2, indicator[i2]),
            strength=strength,
            timestamp=index[p2] if isinstance(index[p2], pd.Timestamp) else None
        )
    
    def _find_closest_swing(self, swings: List[int], target: int, tolerance: int = 10) -> Optional[int]:
        """Trouve le swing le plus proche d'un index cible."""
        closest = None
        min_dist = float('inf')
        
        for swing in swings:
            dist = abs(swing - target)
            if dist < min_dist and dist <= tolerance:
                min_dist = dist
                closest = swing
        
        return closest
    
    def get_divergence_signal(self, df: pd.DataFrame) -> Dict:
        """
        Génère un signal de trading basé sur les divergences détectées.
        
        Returns:
            Dict avec signal, force et détails
        """
        all_divs = self.detect_all(df)
        
        if not all_divs:
            return {
                "signal": "NEUTRAL",
                "reason": "Aucune divergence détectée",
                "divergences": []
            }
        
        # Compter les divergences bullish vs bearish
        bullish_count = 0
        bearish_count = 0
        bullish_strength = 0
        bearish_strength = 0
        details = []
        
        for indicator, divs in all_divs.items():
            for div in divs:
                if div.type in [DivergenceType.REGULAR_BULLISH, DivergenceType.HIDDEN_BULLISH]:
                    bullish_count += 1
                    bullish_strength += div.strength
                    details.append({
                        "type": div.type.value,
                        "indicator": indicator,
                        "strength": round(div.strength, 2)
                    })
                else:
                    bearish_count += 1
                    bearish_strength += div.strength
                    details.append({
                        "type": div.type.value,
                        "indicator": indicator,
                        "strength": round(div.strength, 2)
                    })
        
        # Déterminer le signal
        if bullish_count > bearish_count and bullish_strength > bearish_strength:
            signal = "BUY"
            strength = bullish_strength / max(bullish_count, 1)
            reason = f"{bullish_count} divergence(s) bullish détectée(s)"
        elif bearish_count > bullish_count and bearish_strength > bullish_strength:
            signal = "SELL"
            strength = bearish_strength / max(bearish_count, 1)
            reason = f"{bearish_count} divergence(s) bearish détectée(s)"
        else:
            signal = "NEUTRAL"
            strength = 0
            reason = "Divergences mixtes"
        
        return {
            "signal": signal,
            "strength": round(strength, 2),
            "reason": reason,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "divergences": details
        }


if __name__ == "__main__":
    # Test
    print("=" * 60)
    print("TEST DÉTECTION DE DIVERGENCES")
    print("=" * 60)
    
    # Créer des données de test
    np.random.seed(42)
    n = 100
    
    # Prix avec tendance
    trend = np.cumsum(np.random.randn(n) * 0.5) + 100
    
    df = pd.DataFrame({
        'Close': trend,
        'rsi': 50 + np.cumsum(np.random.randn(n) * 2),  # RSI simulé
        'macd': np.cumsum(np.random.randn(n) * 0.1),
        'macd_hist': np.random.randn(n) * 0.05,
    }, index=pd.date_range(start='2024-01-01', periods=n, freq='1h'))
    
    # Détecter
    detector = DivergenceDetector(lookback=50)
    divs = detector.detect_all(df)
    
    print(f"\nDivergences détectées par indicateur:")
    for indicator, div_list in divs.items():
        print(f"\n{indicator}:")
        for div in div_list:
            print(f"  - {div.type.value}: force={div.strength:.2f}")
    
    # Signal
    signal = detector.get_divergence_signal(df)
    print(f"\n=== Signal ===")
    print(f"Signal: {signal['signal']}")
    print(f"Force: {signal['strength']}")
    print(f"Raison: {signal['reason']}")
