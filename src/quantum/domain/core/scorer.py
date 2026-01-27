"""
Module de scoring multi-critères pour les signaux de trading.
Combine tous les indicateurs en un score unique pondéré.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from quantum.shared.config.settings import config


class SignalStrength(Enum):
    AVOID = "avoid"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class ScoreComponent:
    name: str
    value: float
    weight: float
    direction: str
    details: Optional[Dict] = None


@dataclass
class TradingScore:
    total_score: float
    direction: str
    strength: SignalStrength
    confidence: float
    components: List[ScoreComponent]
    recommendation: str


class MultiCriteriaScorer:
    def __init__(self, custom_weights: Dict[str, float] = None):
        self.category_weights = custom_weights or {
            'technical': 0.25,
            'ml': 0.20,
            'onchain': 0.20,
            'social': 0.15,
            'statistical': 0.10,
            'risk': 0.10
        }
    
    def calculate_score(self, technical_data, ml_data, onchain_data, statistical_data, social_data=None, risk_data=None) -> TradingScore:
        components = [
            self._calculate_technical_score(technical_data),
            self._calculate_ml_score(ml_data),
            self._calculate_onchain_score(onchain_data),
            self._calculate_statistical_score(statistical_data),
        ]
        
        if social_data:
            components.append(self._calculate_social_score(social_data))
        else:
            components.append(ScoreComponent("social", 50, self.category_weights['social'], "neutral"))

        if risk_data:
            components.append(self._calculate_risk_score(risk_data))
        else:
            components.append(ScoreComponent("risk", 50, self.category_weights['risk'], "neutral"))
            
        return self._compute_final_score(components)

    def _calculate_social_score(self, data: Dict) -> ScoreComponent:
        score = data.get('score', 50)
        direction = "bullish" if score >= 60 else "bearish" if score <= 40 else "neutral"
        return ScoreComponent("social", score, self.category_weights['social'], direction)

    def _calculate_onchain_score(self, data: Dict) -> ScoreComponent:
        score = 50
        bullish = 0
        bearish = 0
        total = 0
        
        if not data or not data.get('is_active', False):
            return ScoreComponent("onchain", 50, self.category_weights['onchain'], "neutral")
            
        mempool = data.get('mempool', {})
        pressure = mempool.get('pressure_score', 0)
        if pressure > 30: bullish += 2
        elif pressure < -30: bearish += 2
        total += 2
        
        cc = data.get('cross_chain', {})
        if cc.get('is_bullish'): bullish += 2
        elif cc.get('is_bearish'): bearish += 2
        total += 2
        
        if total > 0:
            score = 50 + ((bullish - bearish) / total * 50)
            
        direction = "bullish" if score >= 60 else "bearish" if score <= 40 else "neutral"
        return ScoreComponent("onchain", score, self.category_weights['onchain'], direction)

    def _calculate_technical_score(self, data):
        # Placeholder for brevity, similar logic as before
        return ScoreComponent("technical", 50, self.category_weights['technical'], "neutral")

    def _calculate_ml_score(self, data):
        prob = data.get('probability', 0.5)
        score = 50 + (prob - 0.5) * 100
        direction = "bullish" if prob > 0.55 else "bearish" if prob < 0.45 else "neutral"
        return ScoreComponent("ml", score, self.category_weights['ml'], direction)

    def _calculate_statistical_score(self, data):
        return ScoreComponent("statistical", 50, self.category_weights['statistical'], "neutral")

    def _calculate_risk_score(self, data):
        return ScoreComponent("risk", 100, self.category_weights['risk'], "neutral")

    def _compute_final_score(self, components: List[ScoreComponent]) -> TradingScore:
        """Calcule le score final vectorisé et détermine la direction institutionnelle."""
        values = np.array([c.value for c in components])
        weights = np.array([c.weight for c in components])
        
        # Calcul du score pondéré (NumPy vectorization)
        total_score = np.average(values, weights=weights)
        
        # Analyse des biais directionnels
        bullish_sum = sum(c.weight for c in components if c.direction == "bullish")
        bearish_sum = sum(c.weight for c in components if c.direction == "bearish")
        
        # Logique de décision institutionnelle (Confirmation multi-couches)
        if bullish_sum > 0.6 and bearish_sum < 0.2:
            direction = "STRONG_BUY"
        elif bullish_sum > bearish_sum * 1.5:
            direction = "BUY"
        elif bearish_sum > 0.6 and bullish_sum < 0.2:
            direction = "STRONG_SELL"
        elif bearish_sum > bullish_sum * 1.5:
            direction = "SELL"
        else:
            direction = "NEUTRAL"

        return TradingScore(
            total_score=round(float(total_score), 1),
            direction=direction,
            strength=self._calculate_strength(total_score, direction),
            confidence=round(float(np.max([bullish_sum, bearish_sum])), 2),
            components=components,
            recommendation=f"Direction: {direction} | Alpha Confidence: {bullish_sum - bearish_sum:.2f}"
        )

    def _calculate_strength(self, score: float, direction: str) -> SignalStrength:
        if direction == "NEUTRAL": return SignalStrength.AVOID
        abs_deviation = abs(score - 50)
        if abs_deviation > 30: return SignalStrength.VERY_STRONG
        if abs_deviation > 15: return SignalStrength.STRONG
        return SignalStrength.MODERATE
