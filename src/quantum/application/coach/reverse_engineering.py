# Innovation 1: Reverse Engineering Module
# Transforms each winning trade into a lesson
# "Voici le setup qui T'AVAIT DONNÉ ce trade"

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class TradeContext:
    """Context of a trade at entry time (not after)"""
    symbol: str
    entry_price: float
    exit_price: float
    side: str  # 'BUY' or 'SELL'
    entry_time: datetime
    exit_time: datetime
    timeframe: str
    # Indicators at entry
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    volume: Optional[float] = None
    bb_position: Optional[float] = None  # Position in Bollinger Bands (0-100)
    # Market context
    trend: Optional[str] = None  # 'UPTREND', 'DOWNTREND', 'RANGE'
    support_near: Optional[float] = None
    resistance_near: Optional[float] = None

@dataclass
class SetupIdentified:
    """Identified trading setup from the winning trade"""
    name: str  # e.g., "Bull Flag", "Head and Shoulders", "RSI Oversold"
    timeframe: str
    indicators_used: List[str]
    reliability_score: float  # Historical win rate of this setup
    description: str

@dataclass
class ReverseEngineeringResult:
    """Result of reverse engineering analysis"""
    setup_identified: SetupIdentified
    indicators_useful: List[str]  # Indicators that contributed to the decision
    indicators_misleading: List[str]  # Indicators that could have led to wrong decision
    lesson: str
    historical_winrate: float
    optimal_entry_offset: float  # How much earlier/later optimal entry would have been
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

class ReverseEngineering:
    """
    Transforms winning trades into actionable lessons.
    
    Mission: "Transformer chaque trade winner en leçon"
    
    This agent analyzes what conditions were present AT THE MOMENT of entry
    (not with hindsight) and identifies the setup that led to the winning trade.
    """
    
    # Known trading setups with their typical indicator patterns
    SETUP_PATTERNS = {
        "Bull Flag": {
            "indicators": ["EMA 20", "EMA 50", "Volume", "RSI"],
            "pattern": "Strong upward move + consolidation + breakout",
            "reliability": 0.72
        },
        "Bear Flag": {
            "indicators": ["EMA 20", "EMA 50", "Volume", "RSI"],
            "pattern": "Strong downward move + consolidation + breakdown",
            "reliability": 0.68
        },
        "RSI Oversold Reversal": {
            "indicators": ["RSI", "Support", "Volume"],
            "pattern": "RSI < 30 + price at support + volume increase",
            "reliability": 0.65
        },
        "RSI Overbought Breakdown": {
            "indicators": ["RSI", "Resistance", "Volume"],
            "pattern": "RSI > 70 + price at resistance + volume increase",
            "reliability": 0.62
        },
        "Moving Average Crossover": {
            "indicators": ["EMA 20", "EMA 50", "MACD"],
            "pattern": "Fast EMA crosses above slow EMA (bullish)",
            "reliability": 0.70
        },
        "Breakout from Range": {
            "indicators": ["Support", "Resistance", "Volume"],
            "pattern": "Price breaks resistance with volume",
            "reliability": 0.75
        },
        "Double Bottom": {
            "indicators": ["Support", "RSI", "Volume"],
            "pattern": "Two touches at support + RSI divergence",
            "reliability": 0.73
        },
        "Double Top": {
            "indicators": ["Resistance", "RSI", "Volume"],
            "pattern": "Two touches at resistance + RSI divergence",
            "reliability": 0.71
        },
        "Ichimoku Cloud Breakout": {
            "indicators": ["Ichimoku", "RSI", "Volume"],
            "pattern": "Price breaks above cloud + Tenkan crosses Kijun",
            "reliability": 0.78
        },
        "MACD Divergence": {
            "indicators": ["MACD", "Price", "RSI"],
            "pattern": "Price makes lower low, MACD makes higher low",
            "reliability": 0.66
        }
    }
    
    def __init__(self):
        self.analysis_cache: Dict[str, List[ReverseEngineeringResult]] = {}
    
    def analyze_trade(self, trade_context: TradeContext) -> ReverseEngineeringResult:
        """
        Main entry point to analyze a winning trade.
        
        Args:
            trade_context: The trade context with entry/exit and indicators
            
        Returns:
            ReverseEngineeringResult with identified setup and lessons
        """
        # Step 1: Identify the setup that was used
        setup = self._identify_setup(trade_context)
        
        # Step 2: Determine which indicators were useful
        useful_indicators, misleading_indicators = self._analyze_indicators(
            trade_context, setup
        )
        
        # Step 3: Generate the lesson
        lesson = self._generate_lesson(trade_context, setup)
        
        # Step 4: Calculate optimal entry
        optimal_offset = self._calculate_optimal_entry(trade_context, setup)
        
        # Step 5: Generate recommendations
        recommendations = self._generate_recommendations(
            trade_context, setup, useful_indicators
        )
        
        result = ReverseEngineeringResult(
            setup_identified=setup,
            indicators_useful=useful_indicators,
            indicators_misleading=misleading_indicators,
            lesson=lesson,
            historical_winrate=setup.reliability_score,
            optimal_entry_offset=optimal_offset,
            recommendations=recommendations
        )
        
        # Cache the result
        self._cache_result(trade_context.symbol, result)
        
        return result
    
    def _identify_setup(self, ctx: TradeContext) -> SetupIdentified:
        """Identify the trading setup from the trade context"""
        
        # Check for Bull Flag pattern
        if ctx.side == "BUY" and ctx.trend == "UPTREND":
            if ctx.ema_20 and ctx.ema_50 and ctx.ema_20 > ctx.ema_50:
                if ctx.volume and ctx.entry_price < ctx.ema_20 * 1.02:
                    return SetupIdentified(
                        name="Bull Flag",
                        timeframe=ctx.timeframe,
                        indicators_used=["EMA 20", "EMA 50", "Volume"],
                        reliability_score=0.72,
                        description="Strong upward move followed by consolidation"
                    )
        
        # Check for RSI Oversold Reversal
        if ctx.rsi and ctx.rsi < 35:
            if ctx.support_near and ctx.entry_price <= ctx.support_near * 1.02:
                return SetupIdentified(
                    name="RSI Oversold Reversal",
                    timeframe=ctx.timeframe,
                    indicators_used=["RSI", "Support", "Volume"],
                    reliability_score=0.65,
                    description="RSI oversold at support level"
                )
        
        # Check for RSI Overbought Breakdown
        if ctx.rsi and ctx.rsi > 65:
            if ctx.resistance_near and ctx.entry_price >= ctx.resistance_near * 0.98:
                return SetupIdentified(
                    name="RSI Overbought Breakdown",
                    timeframe=ctx.timeframe,
                    indicators_used=["RSI", "Resistance", "Volume"],
                    reliability_score=0.62,
                    description="RSI overbought at resistance level"
                )
        
        # Check for Moving Average Crossover
        if ctx.macd and ctx.macd_signal:
            if ctx.side == "BUY" and ctx.macd > ctx.macd_signal:
                return SetupIdentified(
                    name="MACD Bullish Crossover",
                    timeframe=ctx.timeframe,
                    indicators_used=["MACD", "EMA 20", "EMA 50"],
                    reliability_score=0.70,
                    description="MACD line crosses above signal line"
                )
            elif ctx.side == "SELL" and ctx.macd < ctx.macd_signal:
                return SetupIdentified(
                    name="MACD Bearish Crossover",
                    timeframe=ctx.timeframe,
                    indicators_used=["MACD", "EMA 20", "EMA 50"],
                    reliability_score=0.68,
                    description="MACD line crosses below signal line"
                )
        
        # Check for breakout from range
        if ctx.support_near and ctx.resistance_near:
            range_width = ctx.resistance_near - ctx.support_near
            if ctx.entry_price >= ctx.resistance_near - range_width * 0.1:
                return SetupIdentified(
                    name="Breakout from Range",
                    timeframe=ctx.timeframe,
                    indicators_used=["Support", "Resistance", "Volume"],
                    reliability_score=0.75,
                    description="Price breaks out of consolidation zone"
                )
            elif ctx.entry_price <= ctx.support_near + range_width * 0.1:
                return SetupIdentified(
                    name="Breakdown from Range",
                    timeframe=ctx.timeframe,
                    indicators_used=["Support", "Resistance", "Volume"],
                    reliability_score=0.73,
                    description="Price breaks down from consolidation zone"
                )
        
        # Default: Simple trend following
        if ctx.trend == "UPTREND" and ctx.side == "BUY":
            return SetupIdentified(
                name="Uptrend Continuation",
                timeframe=ctx.timeframe,
                indicators_used=["EMA 20", "EMA 50"],
                reliability_score=0.60,
                description="Buying in an uptrend"
            )
        elif ctx.trend == "DOWNTREND" and ctx.side == "SELL":
            return SetupIdentified(
                name="Downtrend Continuation",
                timeframe=ctx.timeframe,
                indicators_used=["EMA 20", "EMA 50"],
                reliability_score=0.58,
                description="Selling in a downtrend"
            )
        
        # Fallback
        return SetupIdentified(
            name="Price Action",
            timeframe=ctx.timeframe,
            indicators_used=["Price"],
            reliability_score=0.50,
            description="Simple price action trade"
        )
    
    def _analyze_indicators(
        self, 
        ctx: TradeContext, 
        setup: SetupIdentified
    ) -> tuple[List[str], List[str]]:
        """Determine which indicators were useful vs misleading"""
        
        useful = []
        misleading = []
        
        # RSI analysis
        if "RSI" in setup.indicators_used:
            if ctx.rsi:
                if ctx.side == "BUY" and ctx.rsi < 50:
                    useful.append("RSI (oversold for long)")
                elif ctx.side == "SELL" and ctx.rsi > 50:
                    useful.append("RSI (overbought for short)")
                elif ctx.rsi > 70 or ctx.rsi < 30:
                    useful.append("RSI (extreme zone)")
                else:
                    misleading.append("RSI (neutral zone)")
        
        # EMA analysis
        if "EMA 20" in setup.indicators_used or "EMA 50" in setup.indicators_used:
            if ctx.ema_20 and ctx.ema_50:
                if ctx.side == "BUY" and ctx.ema_20 > ctx.ema_50:
                    useful.append("EMA Crossover (bullish)")
                elif ctx.side == "SELL" and ctx.ema_20 < ctx.ema_50:
                    useful.append("EMA Crossover (bearish)")
        
        # Volume analysis
        if "Volume" in setup.indicators_used and ctx.volume:
            useful.append("Volume confirmation")
        
        # Support/Resistance
        if "Support" in setup.indicators_used and ctx.support_near:
            if ctx.side == "BUY" and ctx.entry_price >= ctx.support_near * 0.98:
                useful.append("Support level")
        
        if "Resistance" in setup.indicators_used and ctx.resistance_near:
            if ctx.side == "SELL" and ctx.entry_price <= ctx.resistance_near * 1.02:
                useful.append("Resistance level")
        
        return useful, misleading
    
    def _generate_lesson(self, ctx: TradeContext, setup: SetupIdentified) -> str:
        """Generate the main lesson from the trade"""
        
        pnl_percent = ((ctx.exit_price - ctx.entry_price) / ctx.entry_price) * 100
        if ctx.side == "SELL":
            pnl_percent = -pnl_percent
        
        lesson = f"La prochaine fois: attends {setup.name}"
        
        if "Flag" in setup.name:
            lesson += " + confirmation de breakout avec volume"
        elif "RSI" in setup.name:
            lesson += " + confirmation au support/résistance"
        elif "Crossover" in setup.name:
            lesson += " + confluence avec trend"
        
        lesson += f" - Ce setup a un winrate de {setup.reliability_score:.0%}"
        
        return lesson
    
    def _calculate_optimal_entry(
        self, 
        ctx: TradeContext, 
        setup: SetupIdentified
    ) -> float:
        """Calculate optimal entry offset (in pips/percentage)"""
        
        # For now, suggest a conservative offset based on setup
        optimal_offsets = {
            "Bull Flag": -0.5,  # Enter slightly earlier
            "Bear Flag": 0.5,
            "RSI Oversold Reversal": -1.0,  # Wait for more oversold
            "RSI Overbought Breakdown": 1.0,
            "Breakout from Range": 0.2,  # Enter on breakout
            "Default": 0.0
        }
        
        for pattern, offset in optimal_offsets.items():
            if pattern in setup.name:
                return offset
        
        return 0.0
    
    def _generate_recommendations(
        self,
        ctx: TradeContext,
        setup: SetupIdentified,
        useful_indicators: List[str]
    ) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Setup-specific recommendations
        if "Bull Flag" in setup.name:
            recommendations.append("Attends une consolidation après un fort mouvement")
            recommendations.append("Entre quand le prix casse le plus haut de consolidation")
        elif "RSI Oversold" in setup.name:
            recommendations.append("Combine RSI oversold avec un support horizontal")
            recommendations.append("Attends une augmentation de volume")
        elif "Breakout" in setup.name:
            recommendations.append("Entre au breakout avec volume > 1.5x moyenne")
        else:
            recommendations.append(f"Focus sur: {', '.join(useful_indicators[:3])}")
        
        # Risk management
        recommendations.append(f"Stop loss: sous le support (minimum 2% pour {ctx.symbol})")
        recommendations.append(f"Take profit: 2-3x le risque")
        
        return recommendations
    
    def _cache_result(self, symbol: str, result: ReverseEngineeringResult):
        """Cache the analysis result"""
        if symbol not in self.analysis_cache:
            self.analysis_cache[symbol] = []
        self.analysis_cache[symbol].append(result)
    
    def get_historical_analysis(self, symbol: str) -> List[ReverseEngineeringResult]:
        """Get all cached analysis for a symbol"""
        return self.analysis_cache.get(symbol, [])
    
    def to_dict(self, result: ReverseEngineeringResult) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            "setup_identifié": result.setup_identified.name,
            "timeframe": result.setup_identified.timeframe,
            "indicateurs_utiles": result.indicators_useful,
            "indicateurs_misleading": result.indicators_misleading,
            "leçon": result.lesson,
            "fiabilité": f"{result.historical_winrate:.0%}",
            "optimal_entry_offset": f"{result.optimal_entry_offset:.2f}%",
            "recommandations": result.recommendations,
            "timestamp": result.timestamp.isoformat()
        }
    
    def to_json(self, result: ReverseEngineeringResult) -> str:
        """Convert result to JSON string"""
        return json.dumps(self.to_dict(result), indent=2, ensure_ascii=False)
