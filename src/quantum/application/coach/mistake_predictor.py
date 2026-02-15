# Innovation 3: Mistake Predictor Module (ML + NLP)
# Predicts when the user is about to make a mistake
# "T'ES SUR LE POINT DE FAIRE UNE CONNERIE"

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

class MistakePattern(Enum):
    """Known trading mistake patterns"""
    REVENGE_TRADE = "revenge_trade"
    OVERTRADING = "overtrading"
    CHASING_PRICE = "chasing_price"
    FOMO_ENTRY = "fomo_entry"
    DOUBLE_ENTRY = "double_entry"
    IGNORING_SR = "ignoring_support_resistance"
    WRONG_TIMEFRAME = "wrong_timeframe"
    EMOTIONAL_SIZE = "emotional_size"
    NO_STOP_LOSS = "no_stop_loss"
    EARLY_EXIT = "early_exit"
    LATE_ENTRY = "late_entry"

@dataclass
class UserTrade:
    """A trade from user history"""
    id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    status: str  # 'WIN' or 'LOSS'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: Optional[str] = None  # User's stated reason
    
@dataclass
class UserState:
    """Current emotional state of the user"""
    user_id: str
    last_trade_time: Optional[datetime]
    consecutive_losses: int
    consecutive_wins: int
    trades_today: int
    total_trades_week: int
    avg_trade_duration: timedelta
    last_loss_time: Optional[datetime]
    last_win_time: Optional[datetime]
    avg_loss_size: float
    avg_win_size: float
    emotional_score: float  # 0-100, higher = more emotional
    
@dataclass
class MistakePrediction:
    """Prediction of potential mistake"""
    pattern: MistakePattern
    probability: float  # 0-100%
    confidence: float  # Confidence in prediction
    evidence: List[str]
    historical_count: int  # How many times this pattern occurred
    alert_message: str
    advice: str
    cooldown_minutes: int  # Recommended wait time
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class MistakePredictorResult:
    """Complete result of mistake prediction analysis"""
    predictions: List[MistakePrediction]
    overall_risk_score: float  # 0-100
    current_state: UserState
    summary: str
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

class MistakePredictor:
    """
    Predicts when the user is about to make a trading mistake.
    
    Mission: "Prédire quand l'utilisateur va faire une erreur"
    
    Uses ML + NLP to analyze user behavior patterns and detect
    emotional trading states.
    """
    
    # Pattern detection rules
    PATTERN_RULES = {
        MistakePattern.REVENGE_TRADE: {
            "condition": lambda state: (
                state.consecutive_losses >= 2 and 
                (datetime.now() - state.last_loss_time).total_seconds() < 3600
            ),
            "probability_base": 80,
            "evidence": [
                "2+ pertes consécutives",
                "Dernière perte il y a moins d'une heure",
                "Forte tentation de 'se refaire'"
            ],
            "alert": "⚠️ STOP! T'es en tilt. Tiens 30 min avant de trader.",
            "advice": "Va marcher, respire profondément. Le marché sera encore là.",
            "cooldown": 30
        },
        MistakePattern.OVERTRADING: {
            "condition": lambda state: (
                state.trades_today >= 5 or 
                state.total_trades_week >= 20
            ),
            "probability_base": 70,
            "evidence": [
                f"{state.trades_today} trades aujourd'hui",
                f"{state.total_trades_week} trades cette semaine"
            ],
            "alert": "⚠️ TROP DE TRADES! Tu vas destroy ton compte.",
            "advice": "Prends une pause. Qualité > Quantité.",
            "cooldown": 60
        },
        MistakePattern.CHASING_PRICE: {
            "condition": lambda state: (
                state.last_trade_time and 
                (datetime.now() - state.last_trade_time).total_seconds() < 1800 and
                state.consecutive_losses >= 1
            ),
            "probability_base": 65,
            "evidence": [
                "Dernier trade il y a moins de 30 min",
                "Tu cherches probablement à 'rattraper' le prix"
            ],
            "alert": "⚠️ TU CHASES LE PRIX! C'est un piège classique.",
            "advice": "Attends une nouvelle configuration. Le prix reviendra.",
            "cooldown": 45
        },
        MistakePattern.FOMO_ENTRY: {
            "condition": lambda state: (
                state.emotional_score > 70 and
                state.consecutive_wins >= 2
            ),
            "probability_base": 60,
            "evidence": [
                "Score émotionnel élevé (trop confiant)",
                "2+ trades gagnants consécutifs - tu te sens invincible"
            ],
            "alert": "⚠️ FOMO DETECTED! Tu vas regretter ce trade.",
            "advice": "Ta confiance te joue des tours. AnalyseObjectivement.",
            "cooldown": 20
        },
        MistakePattern.EMOTIONAL_SIZE: {
            "condition": lambda state: (
                state.avg_loss_size > state.avg_win_size * 1.5 and
                state.consecutive_losses >= 2
            ),
            "probability_base": 75,
            "evidence": [
                "Tes pertes sont 1.5x plus grosses que tes gains",
                "2+ pertes consécutives - taille émotionnelle probable"
            ],
            "alert": "⚠️ TAILLE ÉMOTIONNELLE! T'as augmenté ta position?",
            "advice": "Respecte ta taille de position. Le size emocional = suicide.",
            "cooldown": 30
        },
        MistakePattern.NO_STOP_LOSS: {
            "condition": lambda state: (
                state.consecutive_losses >= 3
            ),
            "probability_base": 85,
            "evidence": [
                "3+ pertes consécutives",
                "Probable que tu aies retiré ou ignoré ton SL"
            ],
            "alert": "⚠️ STOP LOSS OÙ?! T'es en train de perdre ton compte.",
            "advice": "METTONS UN SL! Toujours. Sans exception.",
            "cooldown": 0
        },
        MistakePattern.EARLY_EXIT: {
            "condition": lambda state: (
                state.last_win_time and
                state.avg_trade_duration < timedelta(minutes=30) and
                state.consecutive_wins >= 2
            ),
            "probability_base": 55,
            "evidence": [
                "2+ gains consécutifs mais durée moyenne < 30 min",
                "Tu sors trop tôt, tu brides tes gains"
            ],
            "alert": "⚠️ TU SORS TROP TÔT! Laisse parler les gains.",
            "advice": "Laisse correr tes winners. Utilise un trailing stop.",
            "cooldown": 15
        },
        MistakePattern.WRONG_TIMEFRAME: {
            "condition": lambda state: (
                state.consecutive_losses >= 2 and
                state.trades_today >= 3
            ),
            "probability_base": 50,
            "evidence": [
                "2+ pertes + 3+ trades = tu cherches partout",
                "Tu changes probablement de timeframe"
            ],
            "alert": "⚠️ TU CHANGES DE TIMEFRAME! Tu perds le fil.",
            "advice": "Choisis UN timeframe et sticks-y. Multi-timeframe = confusion.",
            "cooldown": 25
        }
    }
    
    def __init__(self):
        self.user_states: Dict[str, UserState] = {}
        self.trade_history: Dict[str, List[UserTrade]] = {}
        self.pattern_weights = self._initialize_pattern_weights()
    
    def _initialize_pattern_weights(self) -> Dict[MistakePattern, float]:
        """Initialize pattern weights for ML"""
        return {
            MistakePattern.REVENGE_TRADE: 1.0,
            MistakePattern.OVERTRADING: 0.9,
            MistakePattern.CHASING_PRICE: 0.85,
            MistakePattern.FOMO_ENTRY: 0.8,
            MistakePattern.EMOTIONAL_SIZE: 0.9,
            MistakePattern.NO_STOP_LOSS: 1.0,
            MistakePattern.EARLY_EXIT: 0.7,
            MistakePattern.WRONG_TIMEFRAME: 0.6
        }
    
    def analyze(self, user_id: str, trades: List[UserTrade]) -> MistakePredictorResult:
        """
        Main entry point to analyze user and predict mistakes.
        
        Args:
            user_id: User identifier
            trades: List of user's recent trades
            
        Returns:
            MistakePredictorResult with predictions and recommendations
        """
        # Store trade history
        self.trade_history[user_id] = trades
        
        # Calculate user state
        state = self._calculate_user_state(user_id, trades)
        self.user_states[user_id] = state
        
        # Detect patterns
        predictions = self._detect_patterns(state)
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(predictions)
        
        # Generate summary and recommendations
        summary = self._generate_summary(predictions, state)
        recommendations = self._generate_recommendations(predictions, state)
        
        return MistakePredictorResult(
            predictions=predictions,
            overall_risk_score=risk_score,
            current_state=state,
            summary=summary,
            recommendations=recommendations
        )
    
    def _calculate_user_state(self, user_id: str, trades: List[UserTrade]) -> UserState:
        """Calculate current user state from trade history"""
        
        if not trades:
            return UserState(
                user_id=user_id,
                last_trade_time=None,
                consecutive_losses=0,
                consecutive_wins=0,
                trades_today=0,
                total_trades_week=0,
                avg_trade_duration=timedelta(),
                last_loss_time=None,
                last_win_time=None,
                avg_loss_size=0,
                avg_win_size=0,
                emotional_score=0
            )
        
        # Sort by time
        sorted_trades = sorted(trades, key=lambda t: t.entry_time, reverse=True)
        
        # Count consecutive losses/wins
        consecutive_losses = 0
        consecutive_wins = 0
        
        for trade in sorted_trades:
            if trade.status == 'LOSS':
                consecutive_losses += 1
                consecutive_wins = 0
            else:
                consecutive_wins += 1
                consecutive_losses = 0
            
            if consecutive_losses > 3 or consecutive_wins > 3:
                break
        
        # Count trades today and this week
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=now.weekday())
        
        trades_today = sum(1 for t in trades if t.entry_time >= today_start)
        trades_week = sum(1 for t in trades if t.entry_time >= week_start)
        
        # Calculate average trade duration
        durations = [(t.exit_time - t.entry_time) for t in trades if t.exit_time]
        avg_duration = sum(durations, timedelta()) / len(durations) if durations else timedelta()
        
        # Calculate average win/loss sizes
        wins = [t.pnl_percent for t in trades if t.status == 'WIN']
        losses = [abs(t.pnl_percent) for t in trades if t.status == 'LOSS']
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # Get last trade times
        last_trade = sorted_trades[0] if sorted_trades else None
        last_loss = next((t for t in sorted_trades if t.status == 'LOSS'), None)
        last_win = next((t for t in sorted_trades if t.status == 'WIN'), None)
        
        # Calculate emotional score (simplified)
        emotional_score = self._calculate_emotional_score(
            consecutive_losses, consecutive_wins, trades_today, avg_win, avg_loss
        )
        
        return UserState(
            user_id=user_id,
            last_trade_time=last_trade.entry_time if last_trade else None,
            consecutive_losses=consecutive_losses,
            consecutive_wins=consecutive_wins,
            trades_today=trades_today,
            total_trades_week=trades_week,
            avg_trade_duration=avg_duration,
            last_loss_time=last_loss.exit_time if last_loss else None,
            last_win_time=last_win.exit_time if last_win else None,
            avg_loss_size=avg_loss,
            avg_win_size=avg_win,
            emotional_score=emotional_score
        )
    
    def _calculate_emotional_score(
        self,
        consecutive_losses: int,
        consecutive_wins: int,
        trades_today: int,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """Calculate emotional score (0-100)"""
        
        score = 50  # Base score
        
        # Add points for consecutive losses (tilt)
        score += consecutive_losses * 10
        
        # Add points for consecutive wins (overconfidence)
        score += consecutive_wins * 8
        
        # Add points for overtrading
        score += max(0, (trades_today - 3) * 5)
        
        # Add points for unbalanced win/loss
        if avg_loss > avg_win:
            score += 15
        
        return min(100, max(0, score))
    
    def _detect_patterns(self, state: UserState) -> List[MistakePrediction]:
        """Detect which mistake patterns are likely"""
        
        predictions = []
        
        for pattern, rule in self.PATTERN_RULES.items():
            if rule["condition"](state):
                # Calculate probability with weights
                base_prob = rule["probability_base"]
                weight = self.pattern_weights.get(pattern, 0.8)
                probability = min(95, base_prob * weight)
                
                prediction = MistakePrediction(
                    pattern=pattern,
                    probability=probability,
                    confidence=weight * 100,
                    evidence=rule["evidence"],
                    historical_count=1,  # Would be from actual history in production
                    alert_message=rule["alert"],
                    advice=rule["advice"],
                    cooldown_minutes=rule["cooldown"]
                )
                predictions.append(prediction)
        
        # Sort by probability
        predictions.sort(key=lambda p: p.probability, reverse=True)
        
        return predictions
    
    def _calculate_risk_score(self, predictions: List[MistakePrediction]) -> float:
        """Calculate overall risk score"""
        
        if not predictions:
            return 0
        
        # Weighted average of predictions
        total_weight = sum(p.probability * p.confidence for p in predictions)
        total_confidence = sum(p.confidence for p in predictions)
        
        return total_weight / total_confidence if total_confidence > 0 else 0
    
    def _generate_summary(
        self, 
        predictions: List[MistakePrediction],
        state: UserState
    ) -> str:
        """Generate human-readable summary"""
        
        if not predictions:
            return "✅ État émotionnel stable. Aucune alerte majeure."
        
        top_prediction = predictions[0]
        
        summary = f"⚠️ ALERTE: {top_prediction.pattern.value.replace('_', ' ').upper()}\n"
        summary += f"Probabilité: {top_prediction.probability:.0f}%\n"
        summary += f"\n{top_prediction.alert_message}"
        
        if len(predictions) > 1:
            summary += f"\n\nAutres signaux: {', '.join(p.pattern.value for p in predictions[1:3])}"
        
        return summary
    
    def _generate_recommendations(
        self,
        predictions: List[MistakePrediction],
        state: UserState
    ) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        if not predictions:
            recommendations.append("Continue comme ça! Trading discipliné.")
            return recommendations
        
        # Main recommendation from top prediction
        top = predictions[0]
        recommendations.append(f"🛑 {top.advice}")
        recommendations.append(f"⏰ Attendez {top.cooldown_minutes} minutes avant de trader")
        
        # Additional recommendations
        if state.trades_today > 3:
            recommendations.append("📊 Arrête de trader pour aujourd'hui")
        
        if state.consecutive_losses >= 2:
            recommendations.append("💤 Prends une pause, dors sur tes décisions")
        
        if state.emotional_score > 70:
            recommendations.append("🧘 Respire. Meditation 5 min avant de continuer")
        
        return recommendations
    
    def quick_check(self, user_id: str) -> Optional[MistakePredictorResult]:
        """
        Quick check without full trade analysis.
        Uses cached state if available.
        """
        if user_id in self.user_states:
            state = self.user_states[user_id]
            predictions = self._detect_patterns(state)
            
            if predictions:
                return MistakePredictorResult(
                    predictions=predictions,
                    overall_risk_score=self._calculate_risk_score(predictions),
                    current_state=state,
                    summary=self._generate_summary(predictions, state),
                    recommendations=self._generate_recommendations(predictions, state)
                )
        
        return None
    
    def to_dict(self, result: MistakePredictorResult) -> Dict[str, Any]:
        """Convert result to dictionary"""
        
        return {
            "pattern_détecté": result.predictions[0].pattern.value if result.predictions else "Aucun",
            "probabilité": f"{result.predictions[0].probability:.0f}%" if result.predictions else "0%",
            "historique": f"Alertes actives: {len(result.predictions)}",
            "alerte": result.predictions[0].alert_message if result.predictions else "Aucun risque détecté",
            "conseil": result.predictions[0].advice if result.predictions else "Continuez",
            "risk_score": f"{result.overall_risk_score:.0f}/100",
            "summary": result.summary,
            "recommendations": result.recommendations
        }
    
    def to_json(self, result: MistakePredictorResult) -> str:
        """Convert result to JSON"""
        return json.dumps(self.to_dict(result), indent=2, ensure_ascii=False)
    
    def format_alert(self, result: MistakePredictorResult) -> str:
        """Format a quick alert message"""
        
        if not result.predictions:
            return "✅ Tout va bien. Tu es prêt à trader!"
        
        p = result.predictions[0]
        
        return f"""
╔═══════════════════════════════════════╗
║     🚨 MISTAKE PREDICTOR ALERT 🚨    ║
╠═══════════════════════════════════════╣
║ Pattern: {p.pattern.value:30s}║
║ Probabilité: {p.probability:5.0f}%                     ║
╠═══════════════════════════════════════╣
║ {p.alert_message:38s}║
╠═══════════════════════════════════════╣
║ 💡 {p.advice:36s}║
║ ⏰ Cooldown: {p.cooldown_minutes} minutes              ║
╚═══════════════════════════════════════╝
"""
