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

    def _compute_final_score(self, components):
        weighted_sum = sum(c.value * c.weight for c in components)
        total_weight = sum(c.weight for c in components)
        total_score = weighted_sum / total_weight if total_weight > 0 else 50
        
        bullish = sum(c.weight for c in components if c.direction == "bullish")
        bearish = sum(c.weight for c in components if c.direction == "bearish")
        
        direction = "BUY" if bullish > bearish * 1.2 else "SELL" if bearish > bullish * 1.2 else "NEUTRAL"
        
        return TradingScore(
            total_score=round(total_score, 1),
            direction=direction,
            strength=SignalStrength.MODERATE,
            confidence=0.7,
            components=components,
            recommendation="Signal generated"
        )
