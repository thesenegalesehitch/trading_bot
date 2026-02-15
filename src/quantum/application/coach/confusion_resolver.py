# Innovation 4: Confusion Resolver Module (Flagship)
# Resolves indicator contradictions
# "En bear market, les oversold sont des chances de SHORT"

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

class TrendDirection(Enum):
    """Trend direction"""
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    RANGE = "RANGE"
    UNKNOWN = "UNKNOWN"

class MarketRegime(Enum):
    """Market regime classification"""
    BULL = "BULL"  # Strong uptrend
    BEAR = "BEAR"  # Strong downtrend
    NEUTRAL = "NEUTRAL"  # No clear direction
    VOLATILE = "VOLATILE"  # High volatility, unpredictable
    RECOVERING = "RECOVERING"  # Coming out of bear

class Signal(Enum):
    """Trading signal"""
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"

@dataclass
class IndicatorReading:
    """Single indicator reading"""
    name: str
    value: float
    signal: Signal
    weight: float  # Weight in final decision
    raw_value: str  # e.g., "OVERSOLD", "OVERBOUGHT", "BULLISH CROSS"
    
@dataclass
class MarketContext:
    """Market context at analysis time"""
    symbol: str
    current_price: float
    trend: TrendDirection
    regime: MarketRegime
    timeframe: str
    # Support/Resistance
    support: float
    resistance: float
    # Trend indicators
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None
    sma_50: Optional[float] = None
    # Momentum
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    # Volatility
    atr: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    # Ichimoku
    ichimoku_tenkan: Optional[float] = None
    ichimoku_kijun: Optional[float] = None
    ichimoku_cloud_top: Optional[float] = None
    ichimoku_cloud_bottom: Optional[float] = None
    # Volume
    volume: Optional[float] = None
    volume_ma: Optional[float] = None

@dataclass
class ConfusionResolverResult:
    """Result of confusion resolution"""
    final_signal: Signal
    confidence: float  # 0-100%
    regime: MarketRegime
    trend: TrendDirection
    # Weight breakdown
    indicator_weights: Dict[str, float]
    # Detailed analysis
    analysis: str
    explanation: str
    verdict: str
    # Contradictions found
    contradictions: List[Dict[str, str]]
    # Recommendations
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

class ConfusionResolver:
    """
    Resolves contradictions between trading indicators.
    
    Mission: "Résoudre les contradictions d'indicateurs"
    
    This is the flagship innovation that analyzes conflicting signals
    and provides a clear, actionable decision with explanations.
    """
    
    # Indicator weights by regime
    REGIME_WEIGHTS = {
        MarketRegime.BULL: {
            "EMA": 0.25,
            "SMA_50": 0.15,
            "RSI": 0.10,  # Less weight to oversold in bull
            "MACD": 0.20,
            "Ichimoku": 0.20,
            "Support": 0.10
        },
        MarketRegime.BEAR: {
            "EMA": 0.25,
            "SMA_50": 0.15,
            "RSI": 0.10,  # Less weight to overbought in bear
            "MACD": 0.20,
            "Ichimoku": 0.20,
            "Resistance": 0.10
        },
        MarketRegime.NEUTRAL: {
            "EMA": 0.15,
            "SMA_50": 0.10,
            "RSI": 0.20,
            "MACD": 0.15,
            "Support": 0.15,
            "Resistance": 0.15,
            "Volume": 0.10
        },
        MarketRegime.VOLATILE: {
            "EMA": 0.20,
            "RSI": 0.25,  # More weight to extremes
            "BB": 0.20,   # Bollinger Bands for volatility
            "ATR": 0.15,
            "Volume": 0.20
        },
        MarketRegime.RECOVERING: {
            "EMA": 0.20,
            "SMA_50": 0.15,
            "RSI": 0.15,
            "MACD": 0.20,
            "Ichimoku": 0.20,
            "Support": 0.10
        }
    }
    
    def __init__(self):
        self.analysis_cache: List[ConfusionResolverResult] = []
    
    def resolve(self, context: MarketContext) -> ConfusionResolverResult:
        """
        Main entry point to resolve indicator confusion.
        
        Args:
            context: Market context with all indicator readings
            
        Returns:
            ConfusionResolverResult with final signal and explanation
        """
        
        # Step 1: Determine regime and trend
        regime = self._determine_regime(context)
        trend = self._determine_trend(context)
        
        # Step 2: Collect all indicator readings
        indicators = self._collect_indicators(context, regime)
        
        # Step 3: Detect contradictions
        contradictions = self._detect_contradictions(indicators, trend, regime)
        
        # Step 4: Calculate weighted signal
        indicator_weights, weighted_score = self._calculate_weighted_signal(
            indicators, regime
        )
        
        # Step 5: Determine final signal
        final_signal = self._determine_signal(weighted_score)
        
        # Step 6: Calculate confidence
        confidence = self._calculate_confidence(
            indicators, contradictions, regime
        )
        
        # Step 7: Generate explanation
        explanation = self._generate_explanation(
            context, indicators, regime, trend, final_signal
        )
        
        # Step 8: Generate verdict
        verdict = self._generate_verdict(
            final_signal, confidence, regime, trend
        )
        
        # Step 9: Generate recommendations
        recommendations = self._generate_recommendations(
            context, indicators, regime, final_signal
        )
        
        result = ConfusionResolverResult(
            final_signal=final_signal,
            confidence=confidence,
            regime=regime,
            trend=trend,
            indicator_weights=indicator_weights,
            analysis=self._format_analysis(context, regime, trend),
            explanation=explanation,
            verdict=verdict,
            contradictions=contradictions,
            recommendations=recommendations
        )
        
        # Cache result
        self.analysis_cache.append(result)
        
        return result
    
    def _determine_regime(self, context: MarketContext) -> MarketRegime:
        """Determine market regime"""
        
        # Check EMA alignment
        ema_bullish = (
            context.ema_20 and 
            context.ema_50 and 
            context.ema_20 > context.ema_50
        )
        ema_bearish = (
            context.ema_20 and 
            context.ema_50 and 
            context.ema_20 < context.ema_50
        )
        
        # Check SMA 50 vs price
        price_above_sma = context.sma_50 and context.current_price > context.sma_50
        price_below_sma = context.sma_50 and context.current_price < context.sma_50
        
        # Check volatility
        high_volatility = False
        if context.atr and context.current_price:
            atr_percent = (context.atr / context.current_price) * 100
            high_volatility = atr_percent > 2.0  # > 2% ATR is high
        
        # Determine regime
        if high_volatility and not ema_bullish and not ema_bearish:
            return MarketRegime.VOLATILE
        
        if ema_bullish and price_above_sma:
            return MarketRegime.BULL
        
        if ema_bearish and price_below_sma:
            return MarketRegime.BEAR
        
        # Check if recovering (price below SMA but EMA turning)
        if price_below_sma and ema_bullish:
            return MarketRegime.RECOVERING
        
        return MarketRegime.NEUTRAL
    
    def _determine_trend(self, context: MarketContext) -> TrendDirection:
        """Determine trend direction"""
        
        # Use EMA crossover method
        if context.ema_20 and context.ema_50:
            if context.ema_20 > context.ema_50 * 1.01:  # 1% buffer
                return TrendDirection.UPTREND
            elif context.ema_20 < context.ema_50 * 0.99:
                return TrendDirection.DOWNTREND
        
        # Use price vs SMA
        if context.sma_50:
            if context.current_price > context.sma_50 * 1.02:
                return TrendDirection.UPTREND
            elif context.current_price < context.sma_50 * 0.98:
                return TrendDirection.DOWNTREND
        
        return TrendDirection.RANGE
    
    def _collect_indicators(
        self, 
        context: MarketContext,
        regime: MarketRegime
    ) -> List[IndicatorReading]:
        """Collect all indicator readings with their signals"""
        
        indicators = []
        
        # RSI
        if context.rsi is not None:
            if context.rsi < 30:
                signal = Signal.BUY
                raw = "OVERSOLD"
            elif context.rsi > 70:
                signal = Signal.SELL
                raw = "OVERBOUGHT"
            else:
                signal = Signal.NEUTRAL
                raw = f"{context.rsi:.1f}"
            
            indicators.append(IndicatorReading(
                name="RSI",
                value=context.rsi,
                signal=signal,
                weight=self.REGIME_WEIGHTS[regime].get("RSI", 0.15),
                raw_value=raw
            ))
        
        # MACD
        if context.macd is not None and context.macd_signal is not None:
            if context.macd > context.macd_signal:
                signal = Signal.BUY
                raw = "BULLISH CROSS"
            elif context.macd < context.macd_signal:
                signal = Signal.SELL
                raw = "BEARISH CROSS"
            else:
                signal = Signal.NEUTRAL
                raw = "NO CROSS"
            
            indicators.append(IndicatorReading(
                name="MACD",
                value=context.macd - context.macd_signal,
                signal=signal,
                weight=self.REGIME_WEIGHTS[regime].get("MACD", 0.15),
                raw_value=raw
            ))
        
        # EMA
        if context.ema_20 and context.ema_50:
            if context.ema_20 > context.ema_50:
                signal = Signal.BUY
                raw = "EMA BULLISH"
            else:
                signal = Signal.SELL
                raw = "EMA BEARISH"
            
            indicators.append(IndicatorReading(
                name="EMA",
                value=context.ema_20 / context.ema_50 - 1,
                signal=signal,
                weight=self.REGIME_WEIGHTS[regime].get("EMA", 0.20),
                raw_value=raw
            ))
        
        # Stochastic
        if context.stoch_k and context.stoch_d:
            if context.stoch_k < 20:
                signal = Signal.BUY
                raw = "OVERSOLD"
            elif context.stoch_k > 80:
                signal = Signal.SELL
                raw = "OVERBOUGHT"
            else:
                signal = Signal.NEUTRAL
                raw = f"K={context.stoch_k:.0f}, D={context.stoch_d:.0f}"
            
            indicators.append(IndicatorReading(
                name="Stochastic",
                value=context.stoch_k,
                signal=signal,
                weight=0.10,
                raw_value=raw
            ))
        
        # Support
        if context.support:
            distance = (context.current_price - context.support) / context.support * 100
            if distance < 2:  # Within 2% of support
                signal = Signal.BUY
                raw = f"NEAR SUPPORT ({distance:.1f}%)"
            else:
                signal = Signal.NEUTRAL
                raw = f"Support at {context.support}"
            
            indicators.append(IndicatorReading(
                name="Support",
                value=distance,
                signal=signal,
                weight=0.10,
                raw_value=raw
            ))
        
        # Resistance
        if context.resistance:
            distance = (context.resistance - context.current_price) / context.resistance * 100
            if distance < 2:  # Within 2% of resistance
                signal = Signal.SELL
                raw = f"NEAR RESISTANCE ({distance:.1f}%)"
            else:
                signal = Signal.NEUTRAL
                raw = f"Resistance at {context.resistance}"
            
            indicators.append(IndicatorReading(
                name="Resistance",
                value=distance,
                signal=signal,
                weight=0.10,
                raw_value=raw
            ))
        
        # Ichimoku
        if context.ichimoku_tenkan and context.ichimoku_kijun:
            # Tenkan-Kijun cross
            if context.ichimoku_tenkan > context.ichimoku_kijun:
                tk_signal = Signal.BUY
                tk_raw = "TK > KJ"
            else:
                tk_signal = Signal.SELL
                tk_raw = "TK < KJ"
            
            # Price vs cloud
            if context.ichimoku_cloud_top and context.ichimoku_cloud_bottom:
                if context.current_price > context.ichimoku_cloud_top:
                    cloud_signal = Signal.BUY
                    cloud_raw = "ABOVE CLOUD"
                elif context.current_price < context.ichimoku_cloud_bottom:
                    cloud_signal = Signal.SELL
                    cloud_raw = "BELOW CLOUD"
                else:
                    cloud_signal = Signal.NEUTRAL
                    cloud_raw = "IN CLOUD"
            
            # Combined Ichimoku signal
            combined = Signal.NEUTRAL
            if tk_signal == cloud_signal:
                combined = tk_signal
            elif tk_signal != Signal.NEUTRAL:
                combined = tk_signal
            
            indicators.append(IndicatorReading(
                name="Ichimoku",
                value=1 if combined == Signal.BUY else (-1 if combined == Signal.SELL else 0),
                signal=combined,
                weight=self.REGIME_WEIGHTS[regime].get("Ichimoku", 0.15),
                raw_value=f"{tk_raw}, {cloud_raw}"
            ))
        
        return indicators
    
    def _detect_contradictions(
        self,
        indicators: List[IndicatorReading],
        trend: TrendDirection,
        regime: MarketRegime
    ) -> List[Dict[str, str]]:
        """Detect contradictions between indicators"""
        
        contradictions = []
        
        # Count signals
        buy_signals = [i for i in indicators if i.signal == Signal.BUY]
        sell_signals = [i for i in indicators if i.signal == Signal.SELL]
        
        # Classic contradictions
        for buy in buy_signals:
            for sell in sell_signals:
                contradiction = {
                    "indicator_buy": buy.name,
                    "indicator_sell": sell.name,
                    "explanation": f"{buy.name} dit {buy.signal.value} mais {sell.name} dit {sell.signal.value}"
                }
                contradictions.append(contradiction)
        
        # Regime-specific contradictions
        if regime == MarketRegime.BEAR:
            rsi = next((i for i in indicators if i.name == "RSI"), None)
            if rsi and rsi.signal == Signal.BUY:
                contradictions.append({
                    "indicator_buy": "RSI",
                    "indicator_sell": "Regime",
                    "explanation": "RSI oversold en BEAR = opportunity de SHORT (pas de long!)"
                })
        
        if regime == MarketRegime.BULL:
            rsi = next((i for i in indicators if i.name == "RSI"), None)
            if rsi and rsi.signal == Signal.SELL:
                contradictions.append({
                    "indicator_buy": "Regime",
                    "indicator_sell": "RSI",
                    "explanation": "RSI overbought en BULL = pullback temporaire, maintient les longs"
                })
        
        return contradictions
    
    def _calculate_weighted_signal(
        self,
        indicators: List[IndicatorReading],
        regime: MarketRegime
    ) -> Tuple[Dict[str, float], float]:
        """Calculate weighted signal score"""
        
        weights = {}
        weighted_score = 0.0
        total_weight = 0.0
        
        for indicator in indicators:
            # Get regime weight
            reg_weight = self.REGIME_WEIGHTS[regime].get(indicator.name, 0.15)
            
            # Combine with indicator's own weight
            final_weight = (reg_weight + indicator.weight) / 2
            weights[indicator.name] = final_weight * 100
            
            # Calculate score: BUY = +1, NEUTRAL = 0, SELL = -1
            if indicator.signal == Signal.BUY:
                score = 1
            elif indicator.signal == Signal.SELL:
                score = -1
            else:
                score = 0
            
            weighted_score += score * final_weight
            total_weight += final_weight
        
        # Normalize to -1 to 1
        if total_weight > 0:
            weighted_score /= total_weight
        
        # Convert to 0-100 for weights dict
        weights = {k: v * 100 for k, v in weights.items()}
        
        return weights, weighted_score
    
    def _determine_signal(self, weighted_score: float) -> Signal:
        """Determine final signal from weighted score"""
        
        if weighted_score > 0.2:
            return Signal.BUY
        elif weighted_score < -0.2:
            return Signal.SELL
        else:
            return Signal.NEUTRAL
    
    def _calculate_confidence(
        self,
        indicators: List[IndicatorReading],
        contradictions: List[Dict[str, str]],
        regime: MarketRegime
    ) -> float:
        """Calculate confidence in the signal"""
        
        # Base confidence from signal strength
        buy_count = sum(1 for i in indicators if i.signal == Signal.BUY)
        sell_count = sum(1 for i in indicators if i.signal == Signal.SELL)
        total = len(indicators)
        
        if total == 0:
            return 50
        
        # Agreement ratio
        max_agreement = max(buy_count, sell_count)
        agreement_ratio = max_agreement / total
        
        # Reduce confidence for contradictions
        contradiction_penalty = len(contradictions) * 5
        
        # Regime confidence boost
        regime_boost = 10 if regime in [MarketRegime.BULL, MarketRegime.BEAR] else 0
        
        confidence = (agreement_ratio * 80) + regime_boost - contradiction_penalty
        
        return min(95, max(30, confidence))
    
    def _generate_explanation(
        self,
        context: MarketContext,
        indicators: List[IndicatorReading],
        regime: MarketRegime,
        trend: TrendDirection,
        final_signal: Signal
    ) -> str:
        """Generate human-readable explanation"""
        
        lines = []
        
        # Regime explanation
        regime_explanations = {
            MarketRegime.BULL: "Prix en tendance HAUSSIÈRE (EMA 20 > EMA 50)",
            MarketRegime.BEAR: "Prix en tendance BAISSIÈRE (EMA 20 < EMA 50)",
            MarketRegime.NEUTRAL: "Pas de tendance claire",
            MarketRegime.VOLATILE: "Haute volatilité - prudence recommandée",
            MarketRegime.RECOVERING: "Prix se remet d'une tendance baissière"
        }
        
        lines.append(f"📊 RÉGIME: {regime_explanations[regime]}")
        
        # Trend explanation
        lines.append(f"📈 TENDANCE: {trend.value}")
        
        # Indicator analysis
        lines.append("\n📉 INDICATEURS:")
        for ind in indicators:
            emoji = "🟢" if ind.signal == Signal.BUY else ("🔴" if ind.signal == Signal.SELL else "⚪")
            lines.append(f"  {emoji} {ind.name}: {ind.raw_value} (poids: {ind.weight*100:.0f}%)")
        
        # Final decision
        signal_emoji = "🟢" if final_signal == Signal.BUY else ("🔴" if final_signal == Signal.SELL else "⚪")
        lines.append(f"\n{signal_emoji} DÉCISION: {final_signal.value}")
        
        # Special explanations for regime-based decisions
        if regime == MarketRegime.BEAR:
            rsi = next((i for i in indicators if i.name == "RSI"), None)
            if rsi and rsi.signal == Signal.BUY:
                lines.append("\n💡 Note: Bien que RSI oversold, en bear market les")
                lines.append("   rebonds sont des opportunités de SHORT!")
        
        elif regime == MarketRegime.BULL:
            rsi = next((i for i in indicators if i.name == "RSI"), None)
            if rsi and rsi.signal == Signal.SELL:
                lines.append("\n💡 Note: Bien que RSI overbought, en bull market les")
                lines.append("   pullbacks sont des opportunités d'ACHAT!")
        
        return "\n".join(lines)
    
    def _format_analysis(
        self,
        context: MarketContext,
        regime: MarketRegime,
        trend: TrendDirection
    ) -> str:
        """Format short analysis string"""
        return f"{regime.value} | {trend.value} | {context.symbol} @ {context.current_price}"
    
    def _generate_verdict(
        self,
        final_signal: Signal,
        confidence: float,
        regime: MarketRegime,
        trend: TrendDirection
    ) -> str:
        """Generate verdict message"""
        
        if confidence > 80:
            confidence_msg = "TRÈS FORTE"
        elif confidence > 60:
            confidence_msg = "FORTE"
        elif confidence > 40:
            confidence_msg = "MODÉRÉE"
        else:
            confidence_msg = "FAIBLE"
        
        verdict = f"Signal {final_signal.value} avec confiance {confidence_msg} ({confidence:.0f}%)"
        
        # Add counter-intuitive warning
        if regime == MarketRegime.BEAR and final_signal == Signal.BUY:
            verdict += " - ⚠️ Contre-intuitif mais correct en bear market!"
        elif regime == MarketRegime.BULL and final_signal == Signal.SELL:
            verdict += " - ⚠️ Contre-intuitif mais correct en bull market!"
        
        return verdict
    
    def _generate_recommendations(
        self,
        context: MarketContext,
        indicators: List[IndicatorReading],
        regime: MarketRegime,
        final_signal: Signal
    ) -> List[str]:
        """Generate trade recommendations"""
        
        recs = []
        
        # Entry recommendation
        if final_signal == Signal.BUY:
            entry = context.support * 1.02 if context.support else context.current_price * 0.98
            recs.append(f"Entry: ~{entry:.5f} (2% au-dessus du support)")
        elif final_signal == Signal.SELL:
            entry = context.resistance * 0.98 if context.resistance else context.current_price * 1.02
            recs.append(f"Entry: ~{entry:.5f} (2% sous la résistance)")
        
        # Stop loss
        if final_signal == Signal.BUY:
            sl = context.support * 0.98 if context.support else context.current_price * 0.95
            recs.append(f"Stop Loss: ~{sl:.5f} (sous le support)")
        else:
            sl = context.resistance * 1.02 if context.resistance else context.current_price * 1.05
            recs.append(f"Stop Loss: ~{sl:.5f} (au-dessus de la résistance)")
        
        # Take profit
        if final_signal == Signal.BUY:
            tp = context.resistance if context.resistance else context.current_price * 1.05
            recs.append(f"Take Profit: ~{tp:.5f}")
        else:
            tp = context.support if context.support else context.current_price * 0.95
            recs.append(f"Take Profit: ~{tp:.5f}")
        
        # Risk/Reward
        if final_signal != Signal.NEUTRAL:
            recs.append("Risk/Reward: Vise minimum 1:2")
        
        # Timeframe reminder
        recs.append(f"Timeframe recommandé: {context.timeframe}")
        
        return recs
    
    def to_dict(self, result: ConfusionResolverResult) -> Dict[str, Any]:
        """Convert result to dictionary"""
        
        return {
            "signal_final": result.final_signal.value,
            "confiance": f"{result.confidence:.0f}%",
            "régime": result.regime.value,
            "tendance": result.trend.value,
            "ponderation": result.indicator_weights,
            "explication": result.explanation,
            "verdict": result.verdict,
            "contradictions": result.contradictions,
            "recommandations": result.recommendations
        }
    
    def to_json(self, result: ConfusionResolverResult) -> str:
        """Convert result to JSON"""
        return json.dumps(self.to_dict(result), indent=2, ensure_ascii=False)
    
    def format_report(self, result: ConfusionResolverResult) -> str:
        """Format a detailed report"""
        
        emoji = "🟢" if result.final_signal == Signal.BUY else ("🔴" if result.final_signal == Signal.SELL else "⚪")
        
        report = f"""
╔══════════════════════════════════════════════════════════╗
║         🎭 CONFUSION RESOLVER - FLAGSHIP 🎭            ║
╠══════════════════════════════════════════════════════════╣

{emoji} SIGNAL: {result.final_signal.value}
📊 Confidence: {result.confidence:.0f}%

══════════════════════════════════════════════════════════
                        ANALYSE
══════════════════════════════════════════════════════════

{result.explanation}

══════════════════════════════════════════════════════════
                      VERDICT
══════════════════════════════════════════════════════════

{result.verdict}

"""
        
        if result.contradictions:
            report += "⚠️ CONTRADICTIONS DÉTECTÉES:\n"
            for c in result.contradictions:
                report += f"  • {c.get('explanation', '')}\n"
            report += "\n"
        
        report += "══════════════════════════════════════════════════════════\n"
        report += "                    RECOMMANDATIONS\n"
        report += "══════════════════════════════════════════════════════════\n\n"
        
        for rec in result.recommendations:
            report += f"  📌 {rec}\n"
        
        report += "\n" + "═" * 60 + "\n"
        
        return report
