"""
Analyse de convergence multi-timeframe.
Un signal n'est valide que s'il est confirmé sur plusieurs unités de temps.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os


from quantum.shared.config.settings import config


class MultiTimeframeAnalyzer:
    """
    Analyse la convergence des signaux sur plusieurs timeframes.
    
    Un signal est plus fiable s'il apparaît simultanément sur:
    - M15 (court terme)
    - H1 (moyen terme)
    - H4 (long terme)
    
    Principe: Les grands timeframes définissent la tendance,
    les petits timeframes affinent les entrées.
    """
    
    def __init__(
        self,
        timeframes: List[str] = None,
        required_confirmations: int = None,
        weights: Dict[str, float] = None
    ):
        """
        Initialise l'analyseur multi-timeframe.
        
        Args:
            timeframes: Liste des timeframes à analyser
            required_confirmations: Nombre de TF requis pour valider
            weights: Poids de chaque timeframe dans le score
        """
        self.timeframes = timeframes or config.timeframes.TIMEFRAMES
        self.required_confirmations = required_confirmations or config.technical.REQUIRED_TF_CONFIRMATION
        self.weights = weights or config.timeframes.TIMEFRAME_WEIGHTS
    
    def analyze_trend(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyse la tendance sur tous les timeframes.
        
        Args:
            data: Dict {timeframe: DataFrame} avec colonnes OHLCV
        
        Returns:
            Analyse de tendance multi-TF
        """
        trends = {}
        
        for tf, df in data.items():
            if df is None or len(df) < 50:
                trends[tf] = {"trend": "UNKNOWN", "strength": 0}
                continue
            
            trend_info = self._analyze_single_tf_trend(df)
            trends[tf] = trend_info
        
        # Score de convergence
        convergence = self._calculate_convergence(trends)
        
        return {
            "timeframes": trends,
            "convergence": convergence
        }
    
    def _analyze_single_tf_trend(self, df: pd.DataFrame) -> Dict:
        """
        Analyse la tendance sur un seul timeframe.
        
        Utilise:
        - Position par rapport aux MAs
        - Momentum
        - Structure des plus hauts/bas
        """
        close = df['Close']
        
        # Moyennes mobiles
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        
        # MACD
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        
        current_close = close.iloc[-1]
        
        # Points pour déterminer la tendance
        points = 0
        max_points = 5
        
        # 1. Prix > SMA20
        if current_close > sma_20.iloc[-1]:
            points += 1
        else:
            points -= 1
        
        # 2. Prix > SMA50
        if current_close > sma_50.iloc[-1]:
            points += 1
        else:
            points -= 1
        
        # 3. SMA20 > SMA50 (Golden cross)
        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            points += 1
        else:
            points -= 1
        
        # 4. MACD > Signal
        if macd.iloc[-1] > macd_signal.iloc[-1]:
            points += 1
        else:
            points -= 1
        
        # 5. Higher highs and higher lows
        highs = df['High'].tail(20)
        lows = df['Low'].tail(20)
        
        if highs.iloc[-1] > highs.iloc[-10] and lows.iloc[-1] > lows.iloc[-10]:
            points += 1
        elif highs.iloc[-1] < highs.iloc[-10] and lows.iloc[-1] < lows.iloc[-10]:
            points -= 1
        
        # Normaliser le score
        strength = abs(points) / max_points
        
        if points > 1:
            trend = "BULLISH"
        elif points < -1:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"
        
        return {
            "trend": trend,
            "strength": strength,
            "points": points,
            "price_vs_sma20": "above" if current_close > sma_20.iloc[-1] else "below",
            "macd_momentum": "bullish" if macd.iloc[-1] > macd_signal.iloc[-1] else "bearish"
        }
    
    def _calculate_convergence(self, trends: Dict) -> Dict:
        """
        Calcule le score de convergence multi-TF.
        """
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        weighted_score = 0
        total_weight = 0
        
        for tf, info in trends.items():
            weight = self.weights.get(tf, 0.25)
            total_weight += weight
            
            if info["trend"] == "BULLISH":
                bullish_count += 1
                weighted_score += weight * info["strength"]
            elif info["trend"] == "BEARISH":
                bearish_count += 1
                weighted_score -= weight * info["strength"]
            else:
                neutral_count += 1
        
        # Normaliser
        if total_weight > 0:
            weighted_score /= total_weight
        
        # Déterminer la tendance globale
        if bullish_count >= self.required_confirmations:
            overall_trend = "BULLISH"
            is_confirmed = True
        elif bearish_count >= self.required_confirmations:
            overall_trend = "BEARISH"
            is_confirmed = True
        else:
            overall_trend = "MIXED"
            is_confirmed = False
        
        return {
            "overall_trend": overall_trend,
            "is_confirmed": is_confirmed,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "weighted_score": weighted_score,
            "confirmation_level": max(bullish_count, bearish_count) / len(trends)
        }
    
    def get_entry_signal(
        self,
        data: Dict[str, pd.DataFrame],
        signal_type: str = "trend_follow"
    ) -> Dict:
        """
        Génère un signal d'entrée basé sur la convergence multi-TF.
        
        Args:
            data: Dict {timeframe: DataFrame}
            signal_type: "trend_follow" ou "reversal"
        
        Returns:
            Signal d'entrée avec confiance
        """
        analysis = self.analyze_trend(data)
        convergence = analysis["convergence"]
        
        if not convergence["is_confirmed"]:
            return {
                "signal": "WAIT",
                "reason": "Pas de confirmation multi-TF",
                "confidence": 0,
                "analysis": analysis
            }
        
        # Signal de suivi de tendance
        if signal_type == "trend_follow":
            if convergence["overall_trend"] == "BULLISH":
                signal = "BUY"
            elif convergence["overall_trend"] == "BEARISH":
                signal = "SELL"
            else:
                signal = "WAIT"
        
        # Calculer la confiance
        confidence = convergence["confirmation_level"] * abs(convergence["weighted_score"]) * 100
        confidence = min(confidence * 1.5, 100)  # Ajustement
        
        return {
            "signal": signal,
            "reason": f"Convergence {convergence['overall_trend']} sur {max(convergence['bullish_count'], convergence['bearish_count'])} TF",
            "confidence": confidence,
            "weighted_score": convergence["weighted_score"],
            "analysis": analysis
        }
    
    def find_optimal_entry(
        self,
        data: Dict[str, pd.DataFrame],
        direction: str
    ) -> Dict:
        """
        Trouve le meilleur point d'entrée basé sur les petits TF.
        
        La tendance est définie par les grands TF,
        l'entrée est affinée par les petits TF.
        
        Args:
            data: Dict {timeframe: DataFrame}
            direction: "BUY" ou "SELL"
        
        Returns:
            Recommandation d'entrée
        """
        # Trouver le plus petit TF
        smallest_tf = self.timeframes[0]  # Ex: "15m"
        
        if smallest_tf not in data or data[smallest_tf] is None:
            return {"optimal_entry": "NOW", "reason": "Pas de données petit TF"}
        
        df = data[smallest_tf]
        close = df['Close']
        
        # Calculer RSI sur petit TF
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        current_price = close.iloc[-1]
        
        if direction == "BUY":
            # Pour un BUY, on cherche un pullback (RSI bas)
            if current_rsi < 30:
                return {
                    "optimal_entry": "NOW",
                    "reason": "RSI survendu - entrée optimale",
                    "entry_price": current_price,
                    "quality": "EXCELLENT"
                }
            elif current_rsi < 40:
                return {
                    "optimal_entry": "NOW",
                    "reason": "RSI acceptable pour entrée",
                    "entry_price": current_price,
                    "quality": "GOOD"
                }
            else:
                return {
                    "optimal_entry": "WAIT_PULLBACK",
                    "reason": f"RSI={current_rsi:.1f} - attendre un pullback",
                    "target_rsi": 40,
                    "quality": "POOR"
                }
        else:  # SELL
            if current_rsi > 70:
                return {
                    "optimal_entry": "NOW",
                    "reason": "RSI suracheté - entrée optimale",
                    "entry_price": current_price,
                    "quality": "EXCELLENT"
                }
            elif current_rsi > 60:
                return {
                    "optimal_entry": "NOW",
                    "reason": "RSI acceptable pour entrée",
                    "entry_price": current_price,
                    "quality": "GOOD"
                }
            else:
                return {
                    "optimal_entry": "WAIT_PULLBACK",
                    "reason": f"RSI={current_rsi:.1f} - attendre un rebond",
                    "target_rsi": 60,
                    "quality": "POOR"
                }
    
    def calculate_mtf_score(self, data: Dict[str, pd.DataFrame]) -> float:
        """
        Calcule un score normalisé (-1 à +1) de convergence multi-TF.
        
        +1 = Tous les TF sont bullish avec force maximale
        -1 = Tous les TF sont bearish avec force maximale
        0 = Mixte ou neutre
        """
        analysis = self.analyze_trend(data)
        return analysis["convergence"]["weighted_score"]


if __name__ == "__main__":
    # Test du module
    import numpy as np
    
    np.random.seed(42)
    n = 100
    
    # Créer des données de test pour différents TF
    # Simuler une tendance haussière
    base_trend = np.cumsum(np.random.randn(n) + 0.05)
    
    data = {}
    for tf in ["15m", "1h", "4h", "1d"]:
        df = pd.DataFrame({
            'Open': 100 + base_trend + np.random.randn(n) * 0.3,
            'High': 100 + base_trend + np.random.randn(n) * 0.3 + 0.5,
            'Low': 100 + base_trend + np.random.randn(n) * 0.3 - 0.5,
            'Close': 100 + base_trend + np.random.randn(n) * 0.3,
            'Volume': np.random.randint(1000, 10000, n)
        })
        data[tf] = df
    
    # Analyser
    analyzer = MultiTimeframeAnalyzer()
    
    print("=== Analyse Multi-Timeframe ===\n")
    
    analysis = analyzer.analyze_trend(data)
    print("Tendances par TF:")
    for tf, info in analysis["timeframes"].items():
        print(f"  {tf}: {info['trend']} (force: {info['strength']:.2f})")
    
    print(f"\nConvergence:")
    conv = analysis["convergence"]
    print(f"  Tendance globale: {conv['overall_trend']}")
    print(f"  Confirmé: {conv['is_confirmed']}")
    print(f"  Score pondéré: {conv['weighted_score']:.3f}")
    
    print("\n=== Signal d'entrée ===")
    signal = analyzer.get_entry_signal(data)
    print(f"  Signal: {signal['signal']}")
    print(f"  Raison: {signal['reason']}")
    print(f"  Confiance: {signal['confidence']:.1f}%")
    
    if signal['signal'] in ['BUY', 'SELL']:
        print("\n=== Entrée Optimale ===")
        entry = analyzer.find_optimal_entry(data, signal['signal'])
        for k, v in entry.items():
            print(f"  {k}: {v}")
