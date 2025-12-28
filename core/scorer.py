"""
Module de scoring multi-critères pour les signaux de trading.
Combine tous les indicateurs en un score unique pondéré.

Score final: 0-100
- 0-30: Éviter
- 30-50: Faible
- 50-70: Modéré  
- 70-85: Fort
- 85-100: Très fort
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import config


class SignalStrength(Enum):
    """Force du signal."""
    AVOID = "avoid"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class ScoreComponent:
    """Composant d'un score."""
    name: str
    value: float  # 0-100
    weight: float  # Poids dans le score final
    direction: str  # "bullish", "bearish", "neutral"
    details: Optional[Dict] = None


@dataclass
class TradingScore:
    """Score de trading final."""
    total_score: float
    direction: str  # "BUY", "SELL", "NEUTRAL"
    strength: SignalStrength
    confidence: float  # 0-1
    components: List[ScoreComponent]
    recommendation: str


class MultiCriteriaScorer:
    """
    Système de scoring multi-critères pour évaluer les opportunités de trading.
    
    Catégories de critères:
    1. Analyse Technique (40%)
    2. Machine Learning (25%)
    3. Analyse Statistique (15%)
    4. Sentiment (10%)
    5. Risque/Contexte (10%)
    """
    
    def __init__(self, custom_weights: Dict[str, float] = None):
        # Poids par défaut des catégories
        self.category_weights = custom_weights or {
            'technical': 0.40,
            'ml': 0.25,
            'statistical': 0.15,
            'sentiment': 0.10,
            'risk': 0.10
        }
    
    def calculate_score(
        self,
        technical_data: Dict,
        ml_data: Dict,
        statistical_data: Dict,
        sentiment_data: Dict = None,
        risk_data: Dict = None
    ) -> TradingScore:
        """
        Calcule le score global de trading.
        
        Args:
            technical_data: Données d'analyse technique
            ml_data: Prédictions ML
            statistical_data: Analyse statistique
            sentiment_data: Données de sentiment (optionnel)
            risk_data: Évaluation du risque (optionnel)
        
        Returns:
            TradingScore avec score final et détails
        """
        components = []
        
        # 1. Score Technique
        tech_score = self._calculate_technical_score(technical_data)
        components.append(tech_score)
        
        # 2. Score ML
        ml_score = self._calculate_ml_score(ml_data)
        components.append(ml_score)
        
        # 3. Score Statistique
        stat_score = self._calculate_statistical_score(statistical_data)
        components.append(stat_score)
        
        # 4. Score Sentiment
        if sentiment_data:
            sent_score = self._calculate_sentiment_score(sentiment_data)
            components.append(sent_score)
        else:
            components.append(ScoreComponent(
                name="sentiment", value=50, weight=self.category_weights['sentiment'],
                direction="neutral", details={"status": "not_available"}
            ))
        
        # 5. Score Risque
        if risk_data:
            risk_score = self._calculate_risk_score(risk_data)
            components.append(risk_score)
        else:
            components.append(ScoreComponent(
                name="risk", value=50, weight=self.category_weights['risk'],
                direction="neutral", details={"status": "not_available"}
            ))
        
        # Calculer le score final
        return self._compute_final_score(components)
    
    def _calculate_technical_score(self, data: Dict) -> ScoreComponent:
        """Calcule le score d'analyse technique."""
        score = 50  # Neutre par défaut
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        # RSI (0-100)
        if 'rsi' in data:
            rsi = data['rsi']
            if rsi < 30:
                bullish_signals += 2
            elif rsi < 40:
                bullish_signals += 1
            elif rsi > 70:
                bearish_signals += 2
            elif rsi > 60:
                bearish_signals += 1
            total_signals += 2
        
        # MACD
        if 'macd_signal' in data:
            if data['macd_signal'] == 'bullish':
                bullish_signals += 2
            elif data['macd_signal'] == 'bearish':
                bearish_signals += 2
            total_signals += 2
        
        # Ichimoku
        if 'kumo_position' in data:
            position = data['kumo_position']
            if position == 'above':
                bullish_signals += 2
            elif position == 'below':
                bearish_signals += 2
            total_signals += 2
        
        # Multi-timeframe
        if 'mtf_score' in data:
            mtf = data['mtf_score']
            if mtf > 0:
                bullish_signals += abs(mtf)
            else:
                bearish_signals += abs(mtf)
            total_signals += 3
        
        # Order Blocks
        if 'order_block' in data:
            ob = data['order_block']
            if ob == 'bullish':
                bullish_signals += 1.5
            elif ob == 'bearish':
                bearish_signals += 1.5
            total_signals += 1.5
        
        # Divergences
        if 'divergence' in data:
            div = data['divergence']
            if 'bullish' in div:
                bullish_signals += 2
            elif 'bearish' in div:
                bearish_signals += 2
            total_signals += 2
        
        # Wyckoff 
        if 'wyckoff_phase' in data:
            phase = data['wyckoff_phase']
            if phase in ['accumulation', 'markup']:
                bullish_signals += 2
            elif phase in ['distribution', 'markdown']:
                bearish_signals += 2
            total_signals += 2
        
        # Calculer le score
        if total_signals > 0:
            net_score = (bullish_signals - bearish_signals) / total_signals
            score = 50 + (net_score * 50)  # Scale to 0-100
            score = max(0, min(100, score))
        
        # Déterminer la direction
        if score >= 60:
            direction = "bullish"
        elif score <= 40:
            direction = "bearish"
        else:
            direction = "neutral"
        
        return ScoreComponent(
            name="technical",
            value=score,
            weight=self.category_weights['technical'],
            direction=direction,
            details={
                "bullish_signals": bullish_signals,
                "bearish_signals": bearish_signals,
                "indicators_used": total_signals
            }
        )
    
    def _calculate_ml_score(self, data: Dict) -> ScoreComponent:
        """Calcule le score ML."""
        if not data or 'probability' not in data:
            return ScoreComponent(
                name="ml", value=50, weight=self.category_weights['ml'],
                direction="neutral", details={"status": "no_prediction"}
            )
        
        probability = data['probability']
        
        # Convertir prob (0.5-1.0 pour BUY, 0.0-0.5 pour SELL) en score 0-100
        if probability >= 0.5:
            score = 50 + (probability - 0.5) * 100
            direction = "bullish"
        else:
            score = 50 - (0.5 - probability) * 100
            direction = "bearish"
        
        # Ajuster avec consensus si disponible
        if 'consensus' in data:
            consensus = data['consensus']  # 0-100%
            score = score * 0.7 + consensus * 0.3
        
        score = max(0, min(100, score))
        
        return ScoreComponent(
            name="ml",
            value=score,
            weight=self.category_weights['ml'],
            direction=direction,
            details={
                "probability": probability,
                "model": data.get('model', 'ensemble'),
                "features_used": data.get('features_count', 0)
            }
        )
    
    def _calculate_statistical_score(self, data: Dict) -> ScoreComponent:
        """Calcule le score statistique."""
        score = 50
        signals = []
        
        # Z-Score
        if 'zscore' in data:
            zscore = data['zscore']
            if zscore < -2:
                score += 15
                signals.append("oversold")
            elif zscore > 2:
                score -= 15
                signals.append("overbought")
            elif zscore < -1:
                score += 8
            elif zscore > 1:
                score -= 8
        
        # Hurst Exponent
        if 'hurst' in data:
            hurst = data['hurst']
            if hurst > 0.55:
                score += 10  # Tendance confirmée
                signals.append("trending")
            elif hurst < 0.45:
                score += 5  # Mean-reversion possible
                signals.append("mean_reverting")
        
        # Cointegration
        if 'cointegration_signal' in data:
            if data['cointegration_signal'] == 'opportunity':
                score += 10
                signals.append("arbitrage_opportunity")
        
        score = max(0, min(100, score))
        
        if score >= 60:
            direction = "bullish"
        elif score <= 40:
            direction = "bearish"
        else:
            direction = "neutral"
        
        return ScoreComponent(
            name="statistical",
            value=score,
            weight=self.category_weights['statistical'],
            direction=direction,
            details={"signals": signals}
        )
    
    def _calculate_sentiment_score(self, data: Dict) -> ScoreComponent:
        """Calcule le score de sentiment."""
        score = 50
        
        # Score agrégé du sentiment
        if 'aggregated_score' in data:
            agg = data['aggregated_score']  # -1 à +1
            score = 50 + (agg * 50)
        
        # Fear & Greed (contrarian)
        if 'fear_greed' in data:
            fg = data['fear_greed']  # 0-100
            if fg < 25:  # Extreme Fear = Buy opportunity
                score += 10
            elif fg > 75:  # Extreme Greed = Sell signal
                score -= 10
        
        score = max(0, min(100, score))
        
        if score >= 60:
            direction = "bullish"
        elif score <= 40:
            direction = "bearish"
        else:
            direction = "neutral"
        
        return ScoreComponent(
            name="sentiment",
            value=score,
            weight=self.category_weights['sentiment'],
            direction=direction,
            details=data
        )
    
    def _calculate_risk_score(self, data: Dict) -> ScoreComponent:
        """Calcule le score de risque (inversé - haut = bon)."""
        score = 100  # Part de 100 et pénalise
        
        # VaR
        if 'var_95' in data:
            var = abs(data['var_95'])  # En %
            if var > 3:
                score -= 20
            elif var > 2:
                score -= 10
            elif var > 1:
                score -= 5
        
        # Circuit breaker
        if 'circuit_breaker_active' in data and data['circuit_breaker_active']:
            score -= 40
        
        # Calendrier économique
        if 'economic_blackout' in data and data['economic_blackout']:
            score -= 30
        
        # Corrélation portefeuille
        if 'correlation_risk' in data:
            corr_risk = data['correlation_risk']  # "LOW", "MEDIUM", "HIGH"
            if corr_risk == "HIGH":
                score -= 15
            elif corr_risk == "MEDIUM":
                score -= 5
        
        score = max(0, min(100, score))
        
        return ScoreComponent(
            name="risk",
            value=score,
            weight=self.category_weights['risk'],
            direction="neutral",
            details=data
        )
    
    def _compute_final_score(self, components: List[ScoreComponent]) -> TradingScore:
        """Calcule le score final pondéré."""
        # Score pondéré
        weighted_sum = 0
        total_weight = 0
        
        bullish_votes = 0
        bearish_votes = 0
        
        for comp in components:
            weighted_sum += comp.value * comp.weight
            total_weight += comp.weight
            
            if comp.direction == "bullish":
                bullish_votes += comp.weight
            elif comp.direction == "bearish":
                bearish_votes += comp.weight
        
        total_score = weighted_sum / total_weight if total_weight > 0 else 50
        
        # Déterminer la direction globale
        if bullish_votes > bearish_votes * 1.5:
            direction = "BUY"
        elif bearish_votes > bullish_votes * 1.5:
            direction = "SELL"
        else:
            direction = "NEUTRAL"
        
        # Confidence
        votes_diff = abs(bullish_votes - bearish_votes) / (bullish_votes + bearish_votes + 0.01)
        confidence = min(votes_diff * 2, 1.0)
        
        # Force du signal
        if total_score >= 85:
            strength = SignalStrength.VERY_STRONG
        elif total_score >= 70:
            strength = SignalStrength.STRONG
        elif total_score >= 50:
            strength = SignalStrength.MODERATE
        elif total_score >= 30:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.AVOID
        
        # Générer la recommandation
        recommendation = self._generate_recommendation(
            total_score, direction, strength, confidence, components
        )
        
        return TradingScore(
            total_score=round(total_score, 1),
            direction=direction,
            strength=strength,
            confidence=round(confidence, 2),
            components=components,
            recommendation=recommendation
        )
    
    def _generate_recommendation(
        self,
        score: float,
        direction: str,
        strength: SignalStrength,
        confidence: float,
        components: List[ScoreComponent]
    ) -> str:
        """Génère une recommandation textuelle."""
        if strength == SignalStrength.AVOID:
            return "⛔ ÉVITER - Conditions défavorables. Attendre un meilleur setup."
        
        if strength == SignalStrength.WEAK:
            return f"⚠️ Signal FAIBLE ({direction}) - Confirmation supplémentaire requise."
        
        if direction == "NEUTRAL":
            return "➖ NEUTRE - Pas de biais directionnel clair. Rester en dehors."
        
        # Trouver le composant le plus fort
        strongest = max(components, key=lambda c: abs(c.value - 50))
        
        if strength == SignalStrength.VERY_STRONG:
            return f"🚀 SIGNAL TRÈS FORT {direction} (Score: {score:.0f}/100)\n" \
                   f"Confiance: {confidence*100:.0f}%\n" \
                   f"Principal facteur: {strongest.name} ({strongest.value:.0f})"
        
        if strength == SignalStrength.STRONG:
            return f"✅ SIGNAL FORT {direction} (Score: {score:.0f}/100)\n" \
                   f"Confiance: {confidence*100:.0f}%"
        
        return f"📊 Signal MODÉRÉ {direction} (Score: {score:.0f}/100)"


if __name__ == "__main__":
    print("=" * 60)
    print("TEST MULTI-CRITERIA SCORER")
    print("=" * 60)
    
    scorer = MultiCriteriaScorer()
    
    # Données de test - Scénario bullish
    technical = {
        'rsi': 35,
        'macd_signal': 'bullish',
        'kumo_position': 'above',
        'mtf_score': 2,
        'order_block': 'bullish',
        'divergence': 'regular_bullish',
        'wyckoff_phase': 'accumulation'
    }
    
    ml = {
        'probability': 0.82,
        'consensus': 75,
        'model': 'ensemble'
    }
    
    statistical = {
        'zscore': -1.5,
        'hurst': 0.6,
        'cointegration_signal': 'opportunity'
    }
    
    sentiment = {
        'aggregated_score': 0.3,
        'fear_greed': 28
    }
    
    risk = {
        'var_95': 1.5,
        'circuit_breaker_active': False,
        'economic_blackout': False,
        'correlation_risk': 'LOW'
    }
    
    result = scorer.calculate_score(technical, ml, statistical, sentiment, risk)
    
    print(f"\n=== Résultat ===")
    print(f"Score Total: {result.total_score}/100")
    print(f"Direction: {result.direction}")
    print(f"Force: {result.strength.value}")
    print(f"Confiance: {result.confidence*100:.0f}%")
    print(f"\nRecommandation:\n{result.recommendation}")
    
    print(f"\n=== Composants ===")
    for comp in result.components:
        print(f"  {comp.name}: {comp.value:.1f} ({comp.direction})")
