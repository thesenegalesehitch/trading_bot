# Innovation 5: Auto-Post-Mortem Module
# Generates automatic analysis after each trade
# "Post-Mortem Auto #47"

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class CompletedTrade:
    """A completed trade with all details"""
    id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    timeframe: str
    # Indicators at entry
    rsi_entry: Optional[float] = None
    macd_entry: Optional[float] = None
    macd_signal_entry: Optional[float] = None
    ema_20_entry: Optional[float] = None
    ema_50_entry: Optional[float] = None
    volume_entry: Optional[float] = None
    # Trade management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    # Context
    trend_at_entry: Optional[str] = None
    support_near: Optional[float] = None
    resistance_near: Optional[float] = None
    # User's reasoning
    user_reasoning: Optional[str] = None
    
@dataclass
class WhatWasGood:
    """What was good about the trade"""
    direction_correct: bool = False
    reasoning_correct: bool = False
    stop_loss_present: bool = False
    good_timing: bool = False
    good_timeframe: bool = False
    good_risk_reward: bool = False
    details: List[str] = field(default_factory=list)
    
@dataclass
class WhatWentWrong:
    """What went wrong in the trade"""
    entry_too_early: bool = False
    entry_too_late: bool = False
    entry_too_high: bool = False
    entry_too_low: bool = False
    wrong_timeframe: bool = False
    sl_too_tight: bool = False
    sl_too_loose: bool = False
    no_stop_loss: bool = False
    tp_too_ambitious: bool = False
    tp_too_small: bool = False
    emotional_trade: bool = False
    ignored_indicators: bool = False
    details: List[str] = field(default_factory=list)

@dataclass
class Improvement:
    """Suggested improvement for future trades"""
    entry: Optional[str] = None
    stop_loss: Optional[str] = None
    take_profit: Optional[str] = None
    timeframe: Optional[str] = None
    reasoning: Optional[str] = None
    
@dataclass
class AutoPostMortemResult:
    """Complete post-mortem result"""
    trade_id: str
    # Results
    result: str  # 'WIN', 'LOSS'
    pips: float
    pips_percent: float
    duration_minutes: float
    # Analysis
    what_was_good: WhatWasGood
    what_went_wrong: WhatWentWrong
    improvement: Improvement
    lessons: List[str]
    # Trade details
    entry_analysis: str
    timing_analysis: str
    risk_management_analysis: str
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

class AutoPostMortem:
    """
    Generates automatic post-mortem analysis after each trade.
    
    Mission: "Générer analyse automatique après chaque trade"
    
    Analyzes what was correct, what went wrong, and provides
    actionable improvements.
    """
    
    def __init__(self):
        self.analysis_history: List[AutoPostMortemResult] = []
    
    def analyze(self, trade: CompletedTrade) -> AutoPostMortemResult:
        """
        Main entry point to analyze a completed trade.
        
        Args:
            trade: The completed trade with all details
            
        Returns:
            AutoPostMortemResult with detailed analysis
        """
        
        # Calculate basic results
        pips, pips_percent = self._calculate_pips(trade)
        duration = self._calculate_duration(trade)
        
        result_type = "WIN" if pips > 0 else "LOSS"
        
        # Analyze what was good
        what_was_good = self._analyze_what_was_good(trade, pips > 0)
        
        # Analyze what went wrong
        what_went_wrong = self._analyze_what_went_wrong(trade, pips)
        
        # Generate improvement suggestions
        improvement = self._generate_improvement(trade, what_was_good, what_went_wrong)
        
        # Generate lessons
        lessons = self._generate_lessons(what_was_good, what_went_wrong)
        
        # Generate detailed analysis
        entry_analysis = self._analyze_entry(trade, what_was_good, what_went_wrong)
        timing_analysis = self._analyze_timing(trade)
        risk_analysis = self._analyze_risk_management(trade, what_was_good, what_went_wrong)
        
        # Create result
        post_mortem = AutoPostMortemResult(
            trade_id=trade.id,
            result=result_type,
            pips=pips,
            pips_percent=pips_percent,
            duration_minutes=duration,
            what_was_good=what_was_good,
            what_went_wrong=what_went_wrong,
            improvement=improvement,
            lessons=lessons,
            entry_analysis=entry_analysis,
            timing_analysis=timing_analysis,
            risk_management_analysis=risk_analysis
        )
        
        # Cache result
        self.analysis_history.append(post_mortem)
        
        return post_mortem
    
    def _calculate_pips(self, trade: CompletedTrade) -> tuple[float, float]:
        """Calculate pips gained/lost"""
        
        if trade.side == "BUY":
            pips = (trade.exit_price - trade.entry_price) * 10000
            pips_percent = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
        else:  # SELL
            pips = (trade.entry_price - trade.exit_price) * 10000
            pips_percent = ((trade.entry_price - trade.exit_price) / trade.entry_price) * 100
        
        return pips, pips_percent
    
    def _calculate_duration(self, trade: CompletedTrade) -> float:
        """Calculate trade duration in minutes"""
        
        duration = trade.exit_time - trade.entry_time
        return duration.total_seconds() / 60
    
    def _analyze_what_was_good(
        self, 
        trade: CompletedTrade, 
        is_win: bool
    ) -> WhatWasGood:
        """Analyze what was good about the trade"""
        
        good = WhatWasGood()
        
        # Check direction
        if is_win:
            if trade.side == "BUY" and trade.trend_at_entry == "UPTREND":
                good.direction_correct = True
                good.details.append("Direction achat correcte avec la tendance")
            elif trade.side == "SELL" and trade.trend_at_entry == "DOWNTREND":
                good.direction_correct = True
                good.details.append("Direction vente correcte avec la tendance")
        
        # Check reasoning
        if trade.user_reasoning:
            good.reasoning_correct = True
            good.details.append(f"Raisonnement: {trade.user_reasoning}")
        
        # Check stop loss
        if trade.stop_loss:
            good.stop_loss_present = True
            good.details.append("Stop loss présent (bonne pratique)")
        
        # Check timing (simple heuristic)
        if trade.rsi_entry:
            if trade.side == "BUY" and trade.rsi_entry < 40:
                good.good_timing = True
                good.details.append(f"Timing correct: RSI à {trade.rsi_entry:.0f} (pas trop haut)")
            elif trade.side == "SELL" and trade.rsi_entry > 60:
                good.good_timing = True
                good.details.append(f"Timing correct: RSI à {trade.rsi_entry:.0f} (pas trop bas)")
        
        return good
    
    def _analyze_what_went_wrong(
        self, 
        trade: CompletedTrade, 
        pips: float
    ) -> WhatWentWrong:
        """Analyze what went wrong in the trade"""
        
        wrong = WhatWentWrong()
        
        if pips < 0:  # Losing trade
            # Check entry timing
            if trade.rsi_entry:
                if trade.side == "BUY" and trade.rsi_entry > 60:
                    wrong.entry_too_late = True
                    wrong.details.append(f"Entry trop tard: RSI déjà à {trade.rsi_entry:.0f}")
                elif trade.side == "SELL" and trade.rsi_entry < 40:
                    wrong.entry_too_late = True
                    wrong.details.append(f"Entry trop tard: RSI déjà à {trade.rsi_entry:.0f}")
            
            # Check entry price vs support/resistance
            if trade.side == "BUY" and trade.resistance_near:
                distance = (trade.entry_price - trade.resistance_near) / trade.resistance_near * 100
                if distance > -5:  # Within 5% of resistance
                    wrong.entry_too_high = True
                    wrong.details.append(f"Entry trop près de la résistance (5%)")
            
            if trade.side == "SELL" and trade.support_near:
                distance = (trade.entry_price - trade.support_near) / trade.support_near * 100
                if distance < 5:  # Within 5% of support
                    wrong.entry_too_low = True
                    wrong.details.append(f"Entry trop près du support (5%)")
            
            # Check stop loss
            if not trade.stop_loss:
                wrong.no_stop_loss = True
                wrong.details.append("Pas de stop loss!")
            elif trade.stop_loss and trade.entry_price:
                sl_distance = abs(trade.entry_price - trade.stop_loss) / trade.entry_price * 100
                if sl_distance < 1.0:  # Less than 1% SL
                    wrong.sl_too_tight = True
                    wrong.details.append(f"Stop loss trop serré: {sl_distance:.1f}%")
            
            # Check take profit
            if trade.take_profit and trade.stop_loss:
                tp_distance = abs(trade.take_profit - trade.entry_price) / trade.entry_price * 100
                sl_distance = abs(trade.entry_price - trade.stop_loss) / trade.entry_price * 100
                if tp_distance < sl_distance:
                    wrong.tp_too_small = True
                    wrong.details.append("Take profit plus petit que le stop loss")
            
            # Check timeframe
            if trade.timeframe in ['5m', '15m']:
                wrong.wrong_timeframe = True
                wrong.details.append(f"Timeframe {trade.timeframe} trop court pour ce trade")
        
        return wrong
    
    def _generate_improvement(
        self,
        trade: CompletedTrade,
        good: WhatWasGood,
        wrong: WhatWentWrong
    ) -> Improvement:
        """Generate improvement suggestions"""
        
        improvement = Improvement()
        
        # Entry improvement
        if wrong.entry_too_high:
            improvement.entry = f"{trade.entry_price * 0.98:.5f} (10 pips plus bas)"
        elif wrong.entry_too_low:
            improvement.entry = f"{trade.entry_price * 1.02:.5f} (10 pips plus haut)"
        
        # Stop loss improvement
        if wrong.sl_too_tight:
            if trade.side == "BUY":
                improvement.stop_loss = f"{trade.entry_price * 0.97:.5f} (40 pips)"
            else:
                improvement.stop_loss = f"{trade.entry_price * 1.03:.5f} (40 pips)"
        elif not trade.stop_loss:
            if trade.side == "BUY":
                improvement.stop_loss = f"{trade.entry_price * 0.95:.5f} (50 pips minimum)"
            else:
                improvement.stop_loss = f"{trade.entry_price * 1.05:.5f} (50 pips minimum)"
        
        # Take profit improvement
        if wrong.tp_too_small:
            if trade.side == "BUY":
                improvement.take_profit = f"{trade.entry_price * 1.03:.5f} (35 pips au-dessus)"
            else:
                improvement.take_profit = f"{trade.entry_price * 0.97:.5f} (35 pips en-dessous)"
        
        # Timeframe improvement
        if wrong.wrong_timeframe:
            improvement.timeframe = "1H pour la direction, 15min pour le timing"
        
        # Reasoning improvement
        if not good.reasoning_correct:
            improvement.reasoning = "Définis ton signal AVANT d'entrer"
        
        return improvement
    
    def _generate_lessons(
        self,
        good: WhatWasGood,
        wrong: WhatWentWrong
    ) -> List[str]:
        """Generate lessons from the trade"""
        
        lessons = []
        
        # From what was good
        if good.direction_correct:
            lessons.append("Trade avec la tendance = plus de chances de succès")
        
        if good.stop_loss_present:
            lessons.append("Toujours utiliser un stop loss")
        
        # From what went wrong
        if wrong.entry_too_high or wrong.entry_too_low:
            lessons.append("Attends un pullback vers le support/résistance AVANT d'entrer")
        
        if wrong.sl_too_tight:
            lessons.append(f"Stop loss minimum: 50 pips sur {getattr(trade, 'symbol', 'ce trade')}")
        
        if wrong.wrong_timeframe:
            lessons.append("Utilise 1H pour la direction, 15min pour le timing")
        
        if wrong.tp_too_small:
            lessons.append("Take profit: vise minimum 2x le risque")
        
        # General lessons
        if not lessons:
            lessons.append("Analyse complète - continue comme ça!")
        
        return lessons
    
    def _analyze_entry(
        self,
        trade: CompletedTrade,
        good: WhatWasGood,
        wrong: WhatWentWrong
    ) -> str:
        """Analyze the entry"""
        
        analysis = []
        
        if good.direction_correct:
            analysis.append("✅ Direction correcte")
        else:
            analysis.append("❌ Direction à revoir")
        
        if good.good_timing:
            analysis.append(f"✅ Timing bon (RSI: {trade.rsi_entry})")
        elif wrong.entry_too_late:
            analysis.append(f"❌ Timing:.entry trop tard (RSI: {trade.rsi_entry})")
        
        if wrong.entry_too_high:
            analysis.append(f"❌ Entry 5% trop près de la résistance")
        if wrong.entry_too_low:
            analysis.append(f"❌ Entry 5% trop près du support")
        
        return "\n".join(analysis) if analysis else "Analyse non disponible"
    
    def _analyze_timing(self, trade: CompletedTrade) -> str:
        """Analyze timing (time of day, etc.)"""
        
        hour = trade.entry_time.hour
        
        # London/New York overlap is best for forex
        if 13 <= hour <= 16:
            timing = "✅ Bon horaire: London/New York overlap"
        elif 8 <= hour <= 12:
            timing = "✅ Horaire correct: Session London"
        elif 0 <= hour <= 7:
            timing = "⚠️ Attention: Session asiatique (moins volatile)"
        else:
            timing = "⚠️ Horaire à risque: fin de session"
        
        duration = self._calculate_duration(trade)
        
        if duration < 30:
            timing += "\n⚠️ Trade très court (<30 min): sorti trop tôt?"
        elif duration > 480:
            timing += f"\nℹ️ Trade long ({duration/60:.1f}h): position hold"
        
        return timing
    
    def _analyze_risk_management(
        self,
        trade: CompletedTrade,
        good: WhatWasGood,
        wrong: WhatWentWrong
    ) -> str:
        """Analyze risk management"""
        
        analysis = []
        
        if good.stop_loss_present:
            analysis.append("✅ Stop loss présent")
            
            # Check risk/reward
            if trade.stop_loss and trade.take_profit:
                sl_dist = abs(trade.entry_price - trade.stop_loss)
                tp_dist = abs(trade.take_profit - trade.entry_price)
                rr = tp_dist / sl_dist if sl_dist > 0 else 0
                
                if rr >= 2:
                    analysis.append(f"✅ Bon Risk/Reward: 1:{rr:.1f}")
                elif rr >= 1:
                    analysis.append(f"⚠️ Risk/Reward faible: 1:{rr:.1f}")
                else:
                    analysis.append(f"❌ Mauvais Risk/Reward: 1:{rr:.1f}")
        else:
            analysis.append("❌ Pas de stop loss!")
        
        if wrong.sl_too_tight:
            analysis.append("❌ Stop loss trop serré")
        
        if wrong.tp_too_small:
            analysis.append("❌ Take profit trop petit")
        
        return "\n".join(analysis) if analysis else "Analyse non disponible"
    
    def to_dict(self, result: AutoPostMortemResult) -> Dict[str, Any]:
        """Convert result to dictionary"""
        
        return {
            "trade_id": result.trade_id,
            "resultat": result.result,
            "pips": f"{result.pips:+.0f}",
            "pourcentage": f"{result.pips_percent:+.2f}%",
            "duree_minutes": result.duration_minutes,
            "ce_qui_etait_bon": {
                "direction": result.what_was_good.direction_correct,
                "reasoning": result.what_was_good.reasoning_correct,
                "stop_loss": result.what_was_good.stop_loss_present,
                "details": result.what_was_good.details
            },
            "ce_qui_a_foire": {
                "entry": result.what_went_wrong.details,
                "timing": result.what_went_wrong.wrong_timeframe,
                "sl": result.what_went_wrong.sl_too_tight
            },
            "lecons": result.lessons,
            "amelioration": {
                "entry": result.improvement.entry,
                "stop_loss": result.improvement.stop_loss,
                "take_profit": result.improvement.take_profit,
                "timeframe": result.improvement.timeframe
            }
        }
    
    def to_json(self, result: AutoPostMortemResult) -> str:
        """Convert result to JSON"""
        return json.dumps(self.to_dict(result), indent=2, ensure_ascii=False)
    
    def format_report(self, result: AutoPostMortemResult) -> str:
        """Format a detailed post-mortem report"""
        
        result_emoji = "✅" if result.result == "WIN" else "❌"
        
        report = f"""
╔══════════════════════════════════════════════════════════╗
║              📊 POST-MORTEM AUTO #{result.trade_id}              ║
╚══════════════════════════════════════════════════════════╝

📊 RÉSULTAT: {result_emoji} {result.result} ({result.pips:+.0f} pips / {result.pips_percent:+.2f}%)
⏱️ Durée: {result.duration_minutes:.0f} minutes

══════════════════════════════════════════════════════════
                 ✅ CE QUI ÉTAIT BON
══════════════════════════════════════════════════════════
"""
        
        if result.what_was_good.details:
            for detail in result.what_was_good.details:
                report += f"• {detail}\n"
        else:
            report += "• Aucun point positif identifié\n"
        
        report += """
══════════════════════════════════════════════════════════
                 ❌ CE QUI A FOIRÉ
══════════════════════════════════════════════════════════
"""
        
        if result.what_went_wrong.details:
            for detail in result.what_went_wrong.details:
                report += f"• {detail}\n"
        else:
            report += "• Trade impeccable\n"
        
        report += """
══════════════════════════════════════════════════════════
                    📈 ENTRY
══════════════════════════════════════════════════════════
"""
        report += result.entry_analysis + "\n"
        
        report += """
══════════════════════════════════════════════════════════
                   ⏰ TIMING
══════════════════════════════════════════════════════════
"""
        report += result.timing_analysis + "\n"
        
        report += """
══════════════════════════════════════════════════════════
               💰 RISK MANAGEMENT
══════════════════════════════════════════════════════════
"""
        report += result.risk_management_analysis + "\n"
        
        report += """
══════════════════════════════════════════════════════════
                    📚 LEÇONS
══════════════════════════════════════════════════════════
"""
        
        for i, lesson in enumerate(result.lessons, 1):
            report += f"{i}. {lesson}\n"
        
        report += """
══════════════════════════════════════════════════════════
                  🔄 AMÉLIORATION
══════════════════════════════════════════════════════════
"""
        
        if result.improvement.entry:
            report += f"Entry: {result.improvement.entry}\n"
        if result.improvement.stop_loss:
            report += f"Stop Loss: {result.improvement.stop_loss}\n"
        if result.improvement.take_profit:
            report += f"Take Profit: {result.improvement.take_profit}\n"
        if result.improvement.timeframe:
            report += f"Timeframe: {result.improvement.timeframe}\n"
        
        report += "\n" + "═" * 60 + "\n"
        
        return report
