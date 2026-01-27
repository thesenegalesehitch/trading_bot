"""
Analyse Ichimoku Kinko Hyo - Filtre de tendance absolue.
Utilise le nuage (Kumo) comme indicateur principal.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import sys
import os


from quantum.shared.config.settings import config


class IchimokuAnalyzer:
    """
    Analyse Ichimoku avec focus sur le Kumo (nuage).
    
    Composantes:
    - Tenkan-sen (Conversion): (9-period high + low) / 2
    - Kijun-sen (Base): (26-period high + low) / 2
    - Senkou Span A: (Tenkan + Kijun) / 2, projeté 26 périodes
    - Senkou Span B: (52-period high + low) / 2, projeté 26 périodes
    - Chikou Span: Close projeté 26 périodes en arrière
    """
    
    def __init__(
        self,
        tenkan_period: int = None,
        kijun_period: int = None,
        senkou_b_period: int = None,
        displacement: int = None
    ):
        self.tenkan = tenkan_period or config.technical.ICHIMOKU_TENKAN
        self.kijun = kijun_period or config.technical.ICHIMOKU_KIJUN
        self.senkou_b = senkou_b_period or config.technical.ICHIMOKU_SENKOU_B
        self.displacement = displacement or config.technical.ICHIMOKU_DISPLACEMENT
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule tous les composants Ichimoku.
        
        Args:
            df: DataFrame OHLCV
        
        Returns:
            DataFrame avec colonnes Ichimoku ajoutées
        """
        result = df.copy()
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=self.tenkan).max()
        tenkan_low = low.rolling(window=self.tenkan).min()
        result['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=self.kijun).max()
        kijun_low = low.rolling(window=self.kijun).min()
        result['kijun_sen'] = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        result['senkou_span_a'] = ((result['tenkan_sen'] + result['kijun_sen']) / 2).shift(self.displacement)
        
        # Senkou Span B (Leading Span B)
        senkou_b_high = high.rolling(window=self.senkou_b).max()
        senkou_b_low = low.rolling(window=self.senkou_b).min()
        result['senkou_span_b'] = ((senkou_b_high + senkou_b_low) / 2).shift(self.displacement)
        
        # Chikou Span (Lagging Span)
        result['chikou_span'] = close.shift(-self.displacement)
        
        # Kumo (Cloud) boundaries
        result['kumo_top'] = result[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        result['kumo_bottom'] = result[['senkou_span_a', 'senkou_span_b']].min(axis=1)
        result['kumo_thickness'] = result['kumo_top'] - result['kumo_bottom']
        
        # Position relative au Kumo
        result['price_vs_kumo'] = self._price_vs_kumo(close, result['kumo_top'], result['kumo_bottom'])
        
        return result
    
    def _price_vs_kumo(self, close: pd.Series, kumo_top: pd.Series, kumo_bottom: pd.Series) -> pd.Series:
        """
        Détermine la position du prix par rapport au Kumo.
        
        Returns:
            1 = Au-dessus, -1 = En-dessous, 0 = Dans le nuage
        """
        position = pd.Series(0, index=close.index)
        position[close > kumo_top] = 1
        position[close < kumo_bottom] = -1
        return position
    
    def get_signal(self, df: pd.DataFrame) -> Dict:
        """
        Génère un signal de trading basé sur Ichimoku.
        
        Règles:
        - BUY: Prix > Kumo + Tenkan > Kijun + Kumo vert (A > B)
        - SELL: Prix < Kumo + Tenkan < Kijun + Kumo rouge (A < B)
        """
        ichi = self.calculate(df)
        
        if len(ichi) < self.senkou_b + self.displacement:
            return {"signal": "WAIT", "reason": "Données insuffisantes"}
        
        current = ichi.iloc[-1]
        close = current['Close'] if 'Close' in current else df['Close'].iloc[-1]
        
        # Conditions bullish
        price_above_kumo = close > current['kumo_top']
        tenkan_above_kijun = current['tenkan_sen'] > current['kijun_sen']
        kumo_green = current['senkou_span_a'] > current['senkou_span_b']
        
        # Conditions bearish
        price_below_kumo = close < current['kumo_bottom']
        tenkan_below_kijun = current['tenkan_sen'] < current['kijun_sen']
        kumo_red = current['senkou_span_a'] < current['senkou_span_b']
        
        # Scoring
        bullish_score = sum([price_above_kumo, tenkan_above_kijun, kumo_green])
        bearish_score = sum([price_below_kumo, tenkan_below_kijun, kumo_red])
        
        if bullish_score >= 2 and price_above_kumo:
            signal = "BUY"
            strength = bullish_score / 3
            reason = "Prix au-dessus du Kumo, tendance haussière"
        elif bearish_score >= 2 and price_below_kumo:
            signal = "SELL"
            strength = bearish_score / 3
            reason = "Prix en-dessous du Kumo, tendance baissière"
        elif close > current['kumo_bottom'] and close < current['kumo_top']:
            signal = "WAIT"
            strength = 0
            reason = "Prix dans le Kumo - zone d'incertitude"
        else:
            signal = "NEUTRAL"
            strength = 0.3
            reason = "Signaux mixtes"
        
        return {
            "signal": signal,
            "strength": strength,
            "reason": reason,
            "details": {
                "price": close,
                "kumo_top": current['kumo_top'],
                "kumo_bottom": current['kumo_bottom'],
                "tenkan": current['tenkan_sen'],
                "kijun": current['kijun_sen'],
                "kumo_color": "GREEN" if kumo_green else "RED",
                "kumo_thickness": current['kumo_thickness'],
                "price_vs_kumo": int(current['price_vs_kumo'])
            }
        }
    
    def get_kumo_filter(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Retourne le filtre Kumo simplifié pour filtrer les signaux.
        
        Returns:
            Tuple (direction autorisée, force du filtre)
            - "LONG_ONLY": Ne prendre que des longs
            - "SHORT_ONLY": Ne prendre que des shorts
            - "BOTH": Les deux directions ok
            - "NONE": Éviter de trader
        """
        signal = self.get_signal(df)
        details = signal.get("details", {})
        
        price_vs_kumo = details.get("price_vs_kumo", 0)
        kumo_color = details.get("kumo_color", "")
        strength = signal.get("strength", 0)
        
        if price_vs_kumo == 1:  # Au-dessus
            if kumo_color == "GREEN":
                return "LONG_ONLY", 1.0
            else:
                return "LONG_ONLY", 0.7
        elif price_vs_kumo == -1:  # En-dessous
            if kumo_color == "RED":
                return "SHORT_ONLY", 1.0
            else:
                return "SHORT_ONLY", 0.7
        else:  # Dans le nuage
            return "NONE", 0.0
    
    def calculate_kumo_position_score(self, df: pd.DataFrame) -> float:
        """
        Score de position par rapport au Kumo (-1 à +1).
        
        +1 = Prix bien au-dessus du Kumo
        -1 = Prix bien en-dessous du Kumo
        0 = Dans le Kumo ou proche
        """
        ichi = self.calculate(df)
        
        if ichi.empty:
            return 0.0
        
        current = ichi.iloc[-1]
        close = df['Close'].iloc[-1]
        
        kumo_mid = (current['kumo_top'] + current['kumo_bottom']) / 2
        kumo_thickness = current['kumo_thickness']
        
        if kumo_thickness == 0:
            return 0.0
        
        # Distance normalisée par l'épaisseur du kumo
        distance = (close - kumo_mid) / kumo_thickness
        
        # Limiter entre -1 et +1
        return np.clip(distance, -1, 1)
    
    def detect_kumo_breakout(self, df: pd.DataFrame, lookback: int = 5) -> Dict:
        """
        Détecte une cassure du Kumo récente.
        
        Args:
            df: DataFrame OHLCV
            lookback: Nombre de bougies à vérifier
        
        Returns:
            Info sur la cassure détectée
        """
        ichi = self.calculate(df)
        
        if len(ichi) < lookback + 1:
            return {"breakout": False}
        
        recent = ichi.tail(lookback + 1)
        
        # Vérifier cassure haussière (entrée au-dessus du Kumo)
        was_below = recent['price_vs_kumo'].iloc[:-1].min() <= 0
        is_above = recent['price_vs_kumo'].iloc[-1] == 1
        
        if was_below and is_above:
            return {
                "breakout": True,
                "direction": "BULLISH",
                "breakout_price": df['Close'].iloc[-1],
                "kumo_level": recent['kumo_top'].iloc[-1]
            }
        
        # Vérifier cassure baissière
        was_above = recent['price_vs_kumo'].iloc[:-1].max() >= 0
        is_below = recent['price_vs_kumo'].iloc[-1] == -1
        
        if was_above and is_below:
            return {
                "breakout": True,
                "direction": "BEARISH",
                "breakout_price": df['Close'].iloc[-1],
                "kumo_level": recent['kumo_bottom'].iloc[-1]
            }
        
        return {"breakout": False}


if __name__ == "__main__":
    # Test du module
    np.random.seed(42)
    n = 100
    
    # Simuler tendance haussière
    base = 100 + np.cumsum(np.random.randn(n) * 0.5 + 0.1)
    
    dates = pd.date_range(start='2024-01-01', periods=n, freq='1H')
    df = pd.DataFrame({
        'Open': base + np.random.randn(n) * 0.3,
        'High': base + abs(np.random.randn(n)) * 0.5,
        'Low': base - abs(np.random.randn(n)) * 0.5,
        'Close': base + np.random.randn(n) * 0.3,
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    # Analyser
    ichi = IchimokuAnalyzer()
    result = ichi.calculate(df)
    
    print("=== Analyse Ichimoku ===\n")
    print("Dernières valeurs:")
    cols = ['Close', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'price_vs_kumo']
    print(result[cols].tail())
    
    print("\n=== Signal ===")
    signal = ichi.get_signal(df)
    print(f"Signal: {signal['signal']}")
    print(f"Force: {signal['strength']:.2f}")
    print(f"Raison: {signal['reason']}")
    
    print("\n=== Filtre Kumo ===")
    direction, force = ichi.get_kumo_filter(df)
    print(f"Direction autorisée: {direction}")
    print(f"Force du filtre: {force:.2f}")
    
    print("\n=== Score Position ===")
    score = ichi.calculate_kumo_position_score(df)
    print(f"Score Kumo: {score:.3f}")
    
    print("\n=== Cassure Kumo ===")
    breakout = ichi.detect_kumo_breakout(df)
    print(f"Cassure détectée: {breakout['breakout']}")
