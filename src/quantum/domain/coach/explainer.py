"""
LLM Explainer - Génère des explications naturelles pour les signaux de trading.
Phase 3: Coach Features - Trade Advisor & Coach
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class SignalDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"
    WAIT = "WAIT"


@dataclass
class ExplanationContext:
    symbol: str
    direction: SignalDirection
    confidence: float
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    indicators: Dict[str, Any]
    regime: str
    timeframe: str


class LLMExplainer:
    """
    Génère des explications naturelles pour les signaux de trading.
    
    Utilise un template-based approach pour générer des explanations
    (peut être amélioré avec une vraie LLM API).
    """
    
    def __init__(self, use_openai: bool = False):
        self.use_openai = use_openai
        if use_openai:
            self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
    
    def explain_signal(self, context: ExplanationContext) -> str:
        """
        Génère une explication pour un signal de trading.
        """
        if self.use_openai and self.openai_api_key:
            return self._explain_with_llm(context)
        else:
            return self._explain_with_template(context)
    
    def _explain_with_llm(self, context: ExplanationContext) -> str:
        """Génère une explication avec OpenAI API."""
        try:
            import openai
            
            prompt = self._build_prompt(context)
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Tu es un coach de trading expert. Explique les signaux de trading de manière claire et pédagogique."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            # Fallback to template if API fails
            return self._explain_with_template(context)
    
    def _explain_with_template(self, context: ExplanationContext) -> str:
        """Génère une explanation avec des templates."""
        direction_text = "ACHAT" if context.direction == SignalDirection.BUY else "VENTE" if context.direction == SignalDirection.SELL else "NEUTRE"
        
        explanation = f"""
🎯 **SIGNAL: {direction_text} sur {context.symbol}**

**Contexte:**
- Timeframe: {context.timeframe}
- Régime de marché: {context.regime}
- Confiance: {context.confidence:.0f}%

"""
        
        # Expliquer les indicateurs clés
        explanation += self._explain_indicators(context)
        
        # Expliquer les niveaux
        if context.entry_price and context.stop_loss and context.take_profit:
            explanation += self._explain_levels(context)
        
        # Ajouter une recommandation
        explanation += self._generate_recommendation(context)
        
        return explanation
    
    def _explain_indicators(self, context: ExplanationContext) -> str:
        """Explique les indicateurs utilisés."""
        explanation = "**Analyse des indicateurs:**\n"
        
        indicators = context.indicators
        
        # RSI
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi < 30:
                rsi_text = "RSI en survente (achats opportunités)"
            elif rsi > 70:
                rsi_text = "RSI en surachat (ventes opportunités)"
            else:
                rsi_text = "RSI neutre"
            explanation += f"• RSI: {rsi:.1f} - {rsi_text}\n"
        
        # MACD
        if 'macd' in indicators:
            macd = indicators['macd']
            macd_signal = indicators.get('macd_signal', 0)
            if macd > macd_signal:
                macd_text = "MACD haussier (momentum positif)"
            else:
                macd_text = "MACD baissier (momentum négatif)"
            explanation += f"• MACD: {macd:.5f} - {macd_text}\n"
        
        # EMA
        if 'ema_20' in indicators and 'ema_50' in indicators:
            ema20 = indicators['ema_20']
            ema50 = indicators['ema_50']
            if ema20 > ema50:
                explanation += f"• Tendance: HAUSSIÈRE (EMA20 > EMA50)\n"
            else:
                explanation += f"• Tendance: BAISSIÈRE (EMA20 < EMA50)\n"
        
        # Bollinger
        if 'bb_position' in indicators:
            bb_pos = indicators['bb_position']
            if bb_pos < 20:
                bb_text = "Prix près de la bande inférieure (support)"
            elif bb_pos > 80:
                bb_text = "Prix près de la bande supérieure (résistance)"
            else:
                bb_text = "Prix au milieu des bandes"
            explanation += f"• Bollinger: {bb_text}\n"
        
        explanation += "\n"
        return explanation
    
    def _explain_levels(self, context: ExplanationContext) -> str:
        """Explique les niveaux d'entrée, SL et TP."""
        if not all([context.entry_price, context.stop_loss, context.take_profit]):
            return ""
        
        entry = context.entry_price
        sl = context.stop_loss
        tp = context.take_profit
        
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = reward / risk if risk > 0 else 0
        
        explanation = "**Niveaux de trading:**\n"
        explanation += f"• Entry: {entry:.5f}\n"
        explanation += f"• Stop Loss: {sl:.5f} (risque: {risk*10000:.1f} pips)\n"
        explanation += f"• Take Profit: {tp:.5f} (récompense: {reward*10000:.1f} pips)\n"
        explanation += f"• Ratio Risk/Reward: 1:{rr:.2f}\n\n"
        
        return explanation
    
    def _generate_recommendation(self, context: ExplanationContext) -> str:
        """Génère une recommandation basée sur le contexte."""
        recommendation = "**Recommandation:**\n"
        
        if context.direction == SignalDirection.BUY:
            if context.confidence >= 70:
                recommendation += "✅ Signal fort - Entry agressive recommandée\n"
            elif context.confidence >= 50:
                recommendation += "⚠️ Signal modéré - Entry conservatrice recommandée\n"
            else:
                recommendation += "❌ Signal faible - Attendre confirmation\n"
        elif context.direction == SignalDirection.SELL:
            if context.confidence >= 70:
                recommendation += "✅ Signal fort - Vente recommandée\n"
            elif context.confidence >= 50:
                recommendation += "⚠️ Signal modéré - Vente conservatrice\n"
            else:
                recommendation += "❌ Signal faible - Attendre confirmation\n"
        else:
            recommendation += "⏸️ Pas de signal clair - Patienter\n"
        
        # Ajouter conseil basé sur le régime
        if context.regime == "TRENDING_UP":
            recommendation += "\n💡 En tendance haussière, privilégiez lesachats sur pullback"
        elif context.regime == "TRENDING_DOWN":
            recommendation += "\n💡 En tendance baissière, privilégiez lesventes sur rallies"
        elif context.regime == "RANGING":
            recommendation += "\n💡 En range, tradez les borders (support/résistance)"
        
        return recommendation
    
    def _build_prompt(self, context: ExplanationContext) -> str:
        """Construit le prompt pour la LLM."""
        prompt = f"""
Explique le signal de trading suivant de manière pédagogique:

Symbol: {context.symbol}
Direction: {context.direction.value}
Confiance: {context.confidence}%
Régime de marché: {context.regime}
Timeframe: {context.timeframe}

"""
        if context.entry_price:
            prompt += f"Prix d'entrée suggéré: {context.entry_price}\n"
        if context.stop_loss:
            prompt += f"Stop Loss: {context.stop_loss}\n"
        if context.take_profit:
            prompt += f"Take Profit: {context.take_profit}\n"
        
        prompt += "\nIndicateurs:\n"
        for key, value in context.indicators.items():
            prompt += f"- {key}: {value}\n"
        
        prompt += """
Explique:
1. Pourquoi ce signal est généré
2. Ce que chaque indicateur suggère
3. Les risques et opportunités
4. Un conseil pratique pour le trader
"""
        return prompt


def explain_signal_example():
    """Exemple d'utilisation de l'explainer."""
    explainer = LLMExplainer()
    
    context = ExplanationContext(
        symbol="EURUSD",
        direction=SignalDirection.BUY,
        confidence=75,
        entry_price=1.0850,
        stop_loss=1.0820,
        take_profit=1.0910,
        indicators={
            'rsi': 35.5,
            'macd': 0.0015,
            'macd_signal': 0.0010,
            'ema_20': 1.0845,
            'ema_50': 1.0830,
            'bb_position': 35
        },
        regime="TRENDING_UP",
        timeframe="1H"
    )
    
    explanation = explainer.explain_signal(context)
    print(explanation)


if __name__ == "__main__":
    explain_signal_example()
