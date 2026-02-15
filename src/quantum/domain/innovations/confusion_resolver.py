"""
Confusion Resolver - Résout les contradictions d'indicateurs.
Phase 4: Innovations - Trade Advisor & Coach (FLAGSHP)

Cet outil est le FLAGSHP: il analyse les indicateurs contradictoires
et fournit une décision claire avec explication.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from quantum.domain.data.downloader import DataDownloader
from quantum.domain.data.feature_engine import TechnicalIndicators
from quantum.domain.core.regime_detector import RegimeDetector


class IndicatorSignal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


@dataclass
class IndicatorAnalysis:
    name: str
    signal: IndicatorSignal
    value: float
    weight: float  # Pondération dans la décision finale
    reasoning: str


@dataclass
class ConfusionResolution:
    # Signal final
    final_signal: str  # "BUY", "SELL", "NEUTRAL"
    confidence: float  # 0-100%
    
    # Régime de marché détecté
    regime: str
    
    # Analyse de chaque indicateur
    indicators: List[IndicatorAnalysis]
    
    # Pondération finale
    weighting: Dict[str, float]  # ex: {"ichimoku": 50%, "macd": 30%, "rsi": 10%, "support": 10%}
    
    # Explication détaillée
    explanation: str
    
    # Verdict
    verdict: str
    
    # Contexte supplémentaire
    market_context: Dict[str, Any]


class ConfusionResolver:
    """
    Le FLAGSHP: Resolve les contradictions d'indicateurs.
    
    Exemple de scénario:
    - RSI: OVERSOLD (achat)
    - MACD: CROSSDOWN (vente)
    - Ichimoku: TENDANCE BAISSIÈRE (vente)
    - Support: Test (achat)
    
    Le resolver analyse le régime de marché et pondère les indicateurs
    pour fournir une décision claire.
    """
    
    def __init__(self):
        self.downloader = DataDownloader()
        self.indicators = TechnicalIndicators()
        self.regime_detector = RegimeDetector()
    
    def resolve(
        self,
        symbol: str,
        timeframe: str = "1H"
    ) -> ConfusionResolution:
        """
        Analyse les indicateurs et résout les contradictions.
        
        Args:
            symbol: Symbole à analyser (ex: "EURUSD")
            timeframe: Timeframe (ex: "1H", "4H", "1D")
        
        Returns:
            ConfusionResolution avec la décision finale
        """
        # Télécharger les données
        try:
            interval_map = {"1H": "1h", "4H": "4h", "1D": "1d", "15M": "15m"}
            interval = interval_map.get(timeframe, "1h")
            
            df = self.downloader.download_data(
                symbol=symbol,
                interval=interval,
                years=1
            )
            
            if df is None or len(df) < 100:
                return self._create_error("Données insuffisantes")
                
        except Exception as e:
            return self._create_error(f"Erreur: {str(e)}")
        
        # Calculer les indicateurs
        indicators_df = self.indicators.calculate_all_indicators(df)
        
        # Détecter le régime de marché
        regime = self.regime_detector.detect_regime(df)
        
        # Analyser chaque indicateur
        indicator_analysis = []
        
        # RSI
        rsi = indicators_df.get('rsi')
        if rsi is not None and len(rsi) > 0:
            rsi_val = rsi.iloc[-1]
            if rsi_val < 30:
                rsi_signal = IndicatorSignal.BUY
                rsi_reasoning = "RSI en survente (<30) - opportunité d'achat"
                rsi_weight = 0.10  # Faible pondération en tendance
            elif rsi_val > 70:
                rsi_signal = IndicatorSignal.SELL
                rsi_reasoning = "RSI en surachat (>70) - opportunité de vente"
                rsi_weight = 0.10
            else:
                rsi_signal = IndicatorSignal.NEUTRAL
                rsi_reasoning = "RSI neutre"
                rsi_weight = 0.10
            
            indicator_analysis.append(IndicatorAnalysis(
                name="RSI",
                signal=rsi_signal,
                value=rsi_val,
                weight=rsi_weight,
                reasoning=rsi_reasoning
            ))
        
        # MACD
        macd = indicators_df.get('macd')
        macd_signal = indicators_df.get('macd_signal')
        if macd is not None and macd_signal is not None and len(macd) > 0:
            macd_val = macd.iloc[-1]
            macd_signal_val = macd_signal.iloc[-1]
            
            if macd_val > macd_signal_val:
                macd_signal_enum = IndicatorSignal.BUY
                macd_reasoning = "MACD au-dessus du signal - momentum haussier"
            else:
                macd_signal_enum = IndicatorSignal.SELL
                macd_reasoning = "MACD en dessous du signal - momentum baissier"
            
            indicator_analysis.append(IndicatorAnalysis(
                name="MACD",
                signal=macd_signal_enum,
                value=macd_val,
                weight=0.30,
                reasoning=macd_reasoning
            ))
        
        # Moving Averages (EMA)
        ema_20 = df['Close'].ewm(span=20).mean()
        ema_50 = df['Close'].ewm(span=50).mean()
        
        if len(ema_20) > 0 and len(ema_50) > 0:
            ema20_val = ema_20.iloc[-1]
            ema50_val = ema_50.iloc[-1]
            
            if ema20_val > ema50_val:
                ma_signal = IndicatorSignal.BUY
                ma_reasoning = "EMA20 > EMA50 - tendance haussière"
            else:
                ma_signal = IndicatorSignal.SELL
                ma_reasoning = "EMA20 < EMA50 - tendance baissière"
            
            indicator_analysis.append(IndicatorAnalysis(
                name="EMA_Cross",
                signal=ma_signal,
                value=ema20_val / ema50_val - 1,  # Différentiel en %
                weight=0.25,
                reasoning=ma_reasoning
            ))
        
        # Bollinger Bands
        bb = indicators_df.get('bb_upper'), indicators_df.get('bb_lower')
        if bb[0] is not None and bb[1] is not None and len(df) > 0:
            current_price = df['Close'].iloc[-1]
            bb_upper_val = bb[0].iloc[-1]
            bb_lower_val = bb[1].iloc[-1]
            
            if current_price <= bb_lower_val:
                bb_signal = IndicatorSignal.BUY
                bb_reasoning = "Prix près de la bande inférieure - support"
            elif current_price >= bb_upper_val:
                bb_signal = IndicatorSignal.SELL
                bb_reasoning = "Prix près de la bande supérieure - résistance"
            else:
                bb_signal = IndicatorSignal.NEUTRAL
                bb_reasoning = "Prix au milieu des bandes"
            
            indicator_analysis.append(IndicatorAnalysis(
                name="Bollinger",
                signal=bb_signal,
                value=(current_price - bb_lower_val) / (bb_upper_val - bb_lower_val) * 100,
                weight=0.15,
                reasoning=bb_reasoning
            ))
        
        # Support/Resistance levels
        recent_low = df['Low'].tail(20).min()
        recent_high = df['High'].tail(20).max()
        current_price = df['Close'].iloc[-1]
        
        # Calculer la distance aux niveaux
        dist_to_support = (current_price - recent_low) / current_price * 100
        dist_to_resistance = (recent_high - current_price) / current_price * 100
        
        if dist_to_support < dist_to_resistance and dist_to_support < 2:
            sr_signal = IndicatorSignal.BUY
            sr_reasoning = f"Prix près du support ({dist_to_support:.2f}%)"
        elif dist_to_resistance < 2:
            sr_signal = IndicatorSignal.SELL
            sr_reasoning = f"Prix près de la résistance ({dist_to_resistance:.2f}%)"
        else:
            sr_signal = IndicatorSignal.NEUTRAL
            sr_reasoning = "Prix entre support et résistance"
        
        indicator_analysis.append(IndicatorAnalysis(
            name="Support_Resistance",
            signal=sr_signal,
            value=dist_to_support,
            weight=0.20,
            reasoning=sr_reasoning
        ))
        
        # Ajuster les pondérations selon le régime
        adjusted_indicators = self._adjust_for_regime(indicator_analysis, regime)
        
        # Calculer le signal final
        final_signal, confidence, weighting = self._calculate_final_signal(adjusted_indicators, regime)
        
        # Générer l'explication
        explanation = self._generate_explanation(adjusted_indicators, regime, final_signal)
        
        # Générer le verdict
        verdict = self._generate_verdict(final_signal, regime, confidence)
        
        return ConfusionResolution(
            final_signal=final_signal,
            confidence=confidence,
            regime=regime,
            indicators=adjusted_indicators,
            weighting=weighting,
            explanation=explanation,
            verdict=verdict,
            market_context={
                'current_price': current_price,
                'recent_low': recent_low,
                'recent_high': recent_high,
                'timeframe': timeframe
            }
        )
    
    def _adjust_for_regime(
        self,
        indicators: List[IndicatorAnalysis],
        regime: str
    ) -> List[IndicatorAnalysis]:
        """
        Ajuste les pondérations selon le régime de marché.
        
        En tendance, les indicateurs de tendance ont plus de poids.
        En range, les oscillateurs ont plus de poids.
        """
        # Ajuster les poids selon le régime
        if regime in ["TRENDING_UP", "TRENDING_DOWN"]:
            # En tendance, EMA et MACD ont plus de poids
            for ind in indicators:
                if ind.name in ["EMA_Cross", "MACD"]:
                    ind.weight = min(1.0, ind.weight * 1.5)
                elif ind.name == "RSI":
                    ind.weight = ind.weight * 0.5  # RSI moins utile en tendance
        elif regime == "RANGING":
            # En range, RSI et Bollinger ont plus de poids
            for ind in indicators:
                if ind.name in ["RSI", "Bollinger"]:
                    ind.weight = min(1.0, ind.weight * 1.5)
                elif ind.name == "EMA_Cross":
                    ind.weight = ind.weight * 0.5  # Moins utile en range
        
        return indicators
    
    def _calculate_final_signal(
        self,
        indicators: List[IndicatorAnalysis],
        regime: str
    ) -> tuple:
        """Calcule le signal final pondéré."""
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        weighting = {}
        
        for ind in indicators:
            weight = ind.weight
            total_weight += weight
            weighting[ind.name] = f"{weight * 100:.0f}%"
            
            if ind.signal == IndicatorSignal.BUY:
                buy_score += weight
            elif ind.signal == IndicatorSignal.SELL:
                sell_score += weight
        
        # Normaliser
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
        
        # Déterminer le signal
        threshold = 0.4  # Seuil de confiance minimum
        
        if buy_score > threshold and buy_score > sell_score:
            final_signal = "BUY"
            confidence = buy_score * 100
        elif sell_score > threshold and sell_score > buy_score:
            final_signal = "SELL"
            confidence = sell_score * 100
        else:
            final_signal = "NEUTRAL"
            confidence = 50
        
        # Ajuster la confiance selon le régime
        if regime in ["TRENDING_UP", "TRENDING_DOWN"]:
            confidence = min(95, confidence + 10)  # Plus de confiance en tendance
        elif regime == "RANGING":
            confidence = max(40, confidence - 10)  # Moins de confiance en range
        
        return final_signal, confidence, weighting
    
    def _generate_explanation(
        self,
        indicators: List[IndicatorAnalysis],
        regime: str,
        final_signal: str
    ) -> str:
        """Génère l'explication détaillée."""
        lines = []
        
        lines.append(f"**Régime détecté: {regime}**")
        
        lines.append("\n**Analyse des indicateurs:**")
        for ind in indicators:
            emoji = "🟢" if ind.signal == IndicatorSignal.BUY else "🔴" if ind.signal == IndicatorSignal.SELL else "⚪"
            lines.append(f"{emoji} **{ind.name}**: {ind.reasoning}")
        
        lines.append(f"\n**Décision finale: {final_signal}**")
        
        # Ajouter le contexte selon le régime
        if regime == "TRENDING_DOWN":
            lines.append("\n*Note: En tendance baissière, les rebonds (RSI oversold) sont des opportunités de SHORT, pas de BUY*")
        elif regime == "TRENDING_UP":
            lines.append("\n*Note: En tendance haussière, les pullbacks (RSI oversold) sont des opportunités de BUY*")
        
        return "\n".join(lines)
    
    def _generate_verdict(
        self,
        final_signal: str,
        regime: str,
        confidence: float
    ) -> str:
        """Génère le verdict final."""
        if confidence >= 70:
            strength = "FORT"
        elif confidence >= 50:
            strength = "MODÉRÉ"
        else:
            strength = "FAIBLE"
        
        verdict = f"Signal {strength}"
        
        if confidence >= 70:
            verdict += " - Confiance élevée"
            if final_signal == "BUY":
                verdict += " - Entry agressive recommandée"
            elif final_signal == "SELL":
                verdict += " - Vente recommandée"
        elif confidence >= 50:
            verdict += " - Entry conservatrice recommandée"
        else:
            verdict += " - Attendre confirmation"
        
        return verdict
    
    def _create_error(self, message: str) -> ConfusionResolution:
        """Crée un résultat d'erreur."""
        return ConfusionResolution(
            final_signal="ERROR",
            confidence=0,
            regime="UNKNOWN",
            indicators=[],
            weighting={},
            explanation=f"Erreur: {message}",
            verdict=message,
            market_context={}
        )


def confusion_resolver_example():
    """Exemple d'utilisation du Confusion Resolver."""
    resolver = ConfusionResolver()
    
    # Résoudre les contradictions sur EURUSD
    result = resolver.resolve("EURUSD", "1H")
    
    print(f"\n{'='*60}")
    print(f"CONFUSION RESOLVER - FLAGSHP")
    print(f"{'='*60}")
    print(f"\nSignal final: {result.final_signal}")
    print(f"Confiance: {result.confidence:.0f}%")
    print(f"Régime: {result.regime}")
    
    print(f"\nPondération:")
    for name, weight in result.weighting.items():
        print(f"  {name}: {weight}")
    
    print(f"\n{result.explanation}")
    print(f"\n**VERDICT: {result.verdict}**")


if __name__ == "__main__":
    confusion_resolver_example()
