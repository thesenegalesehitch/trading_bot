"""
Mistake Predictor - Prédit quand l'utilisateur va faire une erreur.
Phase 4: Innovations - Trade Advisor & Coach

Cet outil utilise l'historique des trades pour identifier les patterns
qui mènent à des erreurs et alerter l'utilisateur.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum

from quantum.domain.coach.history import TradeHistory, Trade, TradeOutcome


class MistakePattern(Enum):
    REVENGE_TRADE = "revenge_trade"  # Trade de vengeance après une perte
    OVERTRADING = "overtrading"  # Trop de trades
    CHASING_PRICE = "chasing_price"  # Courir après le prix
    INCREASING_SIZE = "increasing_size"  # Augmenter la taille après wins
    IGNORING_SIGNALS = "ignoring_signals"  # Ignorer les signaux
    EMOTIONAL_ENTRY = "emotional_entry"  # Entry émotionnelle


@dataclass
class MistakePrediction:
    pattern: MistakePattern
    probability: float  # 0-100%
    evidence: List[str]
    alert_message: str
    recommendation: str
    cooldown_minutes: int  # Minutes à attendre avant de trader


class MistakePredictor:
    """
    Prédit les erreurs de trading basées sur les patterns historiques.
    
    Patterns détectés:
    - Revenge Trading: Trade après 2+ pertes consécutives
    - Overtrading: Plus de X trades par jour
    - Chasing Price: Entrer après un mouvement important
    - Increasing Size: Augmenter la taille après gains
    - Ignoring Signals: Trader sans signal
    - Emotional Entry: Entrer pendant forte émotion
    """
    
    def __init__(self, trade_history: TradeHistory):
        self.history = trade_history
        self._load_patterns()
    
    def _load_patterns(self):
        """Charge les patterns de mistake depuis l'historique."""
        # Analyse des patterns fréquents
        self.pattern_stats = {
            MistakePattern.REVENGE_TRADE: self._analyze_revenge_trading(),
            MistakePattern.OVERTRADING: self._analyze_overtrading(),
            MistakePattern.CHASING_PRICE: self._analyze_chasing(),
            MistakePattern.INCREASING_SIZE: self._analyze_size_increase(),
        }
    
    def predict_mistake(self) -> Optional[MistakePrediction]:
        """
        Prédit si l'utilisateur est sur le point de faire une erreur.
        
        Returns:
            MistakePrediction si un pattern est détecté, None sinon
        """
        # Vérifier chaque pattern
        predictions = []
        
        # 1. Revenge Trading
        revenge = self._check_revenge_trading()
        if revenge:
            predictions.append(revenge)
        
        # 2. Overtrading
        overtrade = self._check_overtrading()
        if overtrade:
            predictions.append(overtrade)
        
        # 3. Chasing Price
        chase = self._check_chasing_price()
        if chase:
            predictions.append(chase)
        
        # 4. Increasing Size
        size_issue = self._check_increasing_size()
        if size_issue:
            predictions.append(size_issue)
        
        # Retourner le pattern avec la plus haute probabilité
        if predictions:
            return max(predictions, key=lambda x: x.probability)
        
        return None
    
    def _analyze_revenge_trading(self) -> Dict[str, Any]:
        """Analyse le pattern de revenge trading."""
        recent_trades = self.history.get_recent_trades(20)
        
        revenge_count = 0
        for i in range(len(recent_trades) - 1):
            # Si ce trade est une perte et le suivant est dans la direction opposée
            if (recent_trades[i].outcome == TradeOutcome.LOSS and 
                i + 1 < len(recent_trades) and
                recent_trades[i].direction != recent_trades[i + 1].direction):
                revenge_count += 1
        
        return {
            'count': revenge_count,
            'rate': revenge_count / len(recent_trades) if recent_trades else 0
        }
    
    def _analyze_overtrading(self) -> Dict[str, Any]:
        """Analyse le pattern d'overtrading."""
        # Compter les trades par jour
        trades_per_day = defaultdict(int)
        for trade in self.history.trades:
            if trade.entry_time:
                day = trade.entry_time.date()
                trades_per_day[day] += 1
        
        if not trades_per_day:
            return {'avg_per_day': 0, 'max_per_day': 0}
        
        return {
            'avg_per_day': sum(trades_per_day.values()) / len(trades_per_day),
            'max_per_day': max(trades_per_day.values())
        }
    
    def _analyze_chasing(self) -> Dict[str, Any]:
        """Analyse le pattern de chasing price."""
        # Pour l'instant, return données vides
        return {'chase_count': 0}
    
    def _analyze_size_increase(self) -> Dict[str, Any]:
        """Analyse le pattern d'augmentation de taille."""
        # Analyser la taille des positions après des wins
        return {'increase_count': 0}
    
    def _check_revenge_trading(self) -> Optional[MistakePrediction]:
        """Vérifie si l'utilisateur est en train de faire un revenge trade."""
        recent_trades = self.history.get_recent_trades(5)
        
        if len(recent_trades) < 2:
            return None
        
        # Vérifier les 2 dernières pertes consécutives
        consecutive_losses = 0
        for trade in recent_trades[:3]:
            if trade.outcome == TradeOutcome.LOSS:
                consecutive_losses += 1
            else:
                break
        
        if consecutive_losses >= 2:
            # Calculer la probabilité
            probability = min(95, 60 + consecutive_losses * 15)
            
            # Compter les revenge trades dans l'historique
            revenge_stats = self._analyze_revenge_trading()
            historical_count = revenge_stats.get('count', 0)
            
            evidence = [
                f"{consecutive_losses} pertes consécutives récentes",
                f"Historique: {historical_count} revenge trades identifiés",
                "Tendance à trader directement après une perte"
            ]
            
            alert = "⚠️ STOP! T'ES SUR LE POINT DE FAIRE UNE CONNERIE"
            
            if consecutive_losses >= 3:
                recommendation = "TU ES EN TILT! Va marcher, reviens dans 30 min"
                cooldown = 30
            else:
                recommendation = "Attends 15 minutes avant de trader"
                cooldown = 15
            
            return MistakePrediction(
                pattern=MistakePattern.REVENGE_TRADE,
                probability=probability,
                evidence=evidence,
                alert_message=alert,
                recommendation=recommendation,
                cooldown_minutes=cooldown
            )
        
        return None
    
    def _check_overtrading(self) -> Optional[MistakePrediction]:
        """Vérifie si l'utilisateur fait du sur-trading."""
        today = datetime.now().date()
        today_trades = [
            t for t in self.history.trades 
            if t.entry_time and t.entry_time.date() == today
        ]
        
        if len(today_trades) >= 5:
            stats = self._analyze_overtrading()
            
            probability = min(90, 50 + len(today_trades) * 8)
            
            evidence = [
                f"{len(today_trades)} trades aujourd'hui",
                f"Moyenne historique: {stats.get('avg_per_day', 0):.1f}/jour",
                "Tendance à trader excessivement"
            ]
            
            return MistakePrediction(
                pattern=MistakePattern.OVERTRADING,
                probability=probability,
                evidence=evidence,
                alert_message="⚠️ SUR-TRADING DÉTECTÉ",
                recommendation="Prends une pause. Plus de trades ne veut pas dire plus de profits.",
                cooldown_minutes=60
            )
        
        return None
    
    def _check_chasing_price(self) -> Optional[MistakePrediction]:
        """
        Vérifie si l'utilisateur chase le prix.
        (Note: Cette fonctionnalité nécessiterait des données en temps réel)
        """
        # Placeholder - nécessite l'implémentation avec des données temps réel
        return None
    
    def _check_increasing_size(self) -> Optional[MistakePrediction]:
        """Vérifie si l'utilisateur augmente sa taille après des wins."""
        recent_trades = self.history.get_recent_trades(5)
        
        if len(recent_trades) < 2:
            return None
        
        # Vérifier si les derniers trades sont winners et si la taille augmente
        increasing = False
        consecutive_wins = 0
        
        for trade in recent_trades[:3]:
            if trade.outcome == TradeOutcome.WIN:
                consecutive_wins += 1
            else:
                break
        
        if consecutive_wins >= 2:
            # Calculer l'augmentation de taille
            if len(recent_trades) >= 2:
                size_change = (recent_trades[0].quantity - recent_trades[1].quantity) / recent_trades[1].quantity * 100
                
                if size_change > 20:  # Plus de 20% d'augmentation
                    return MistakePrediction(
                        pattern=MistakePattern.INCREASING_SIZE,
                        probability=min(85, 50 + consecutive_wins * 10),
                        evidence=[
                            f"{consecutive_wins} wins consécutifs",
                            f"Taille augmentée de {size_change:.0f}%",
                            "Tendance à devenir agressif après des gains"
                        ],
                        alert_message="⚠️ TU DEVIENS AGRESSIF!",
                        recommendation="Garde la même taille de position. Les wins consécutifs ne garantissent pas le prochain.",
                        cooldown_minutes=30
                    )
        
        return None
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques des patterns."""
        return {
            'revenge_trading': self._analyze_revenge_trading(),
            'overtrading': self._analyze_overtrading(),
            'chasing_price': self._analyze_chasing(),
            'size_increase': self._analyze_size_increase()
        }
    
    def get_risk_score(self) -> int:
        """
        Calcule un score de risque global (0-100).
        
        0-30: Risque faible
        31-60: Risque modéré
        61-100: Risque élevé
        """
        score = 0
        
        # Ajouter les points pour chaque pattern
        revenge = self._analyze_revenge_trading()
        score += min(30, revenge.get('rate', 0) * 100)
        
        overtrade = self._analyze_overtrading()
        if overtrade.get('avg_per_day', 0) > 3:
            score += 20
        if overtrade.get('max_per_day', 0) > 5:
            score += 10
        
        # Obtenir une prédiction
        prediction = self.predict_mistake()
        if prediction:
            score += int(prediction.probability * 0.3)
        
        return min(100, int(score))


def mistake_predictor_example():
    """Exemple d'utilisation du Mistake Predictor."""
    # Créer un historique avec des trades exemple
    history = TradeHistory()
    
    # Ajouter des trades avec des patterns
    from quantum.domain.coach.history import Trade, TradeOutcome
    import random
    
    # Simuler des pertes consécutives (revenge trading pattern)
    for i in range(3):
        trade = Trade(
            id=f"trade_{i}",
            symbol="EURUSD",
            direction="BUY" if i % 2 == 0 else "SELL",
            entry_price=1.0850,
            exit_price=1.0830,
            entry_time=datetime.now() - timedelta(hours=i*2),
            exit_time=datetime.now() - timedelta(hours=i*2+1),
            stop_loss=1.0820,
            take_profit=1.0900,
            quantity=1.0,
            outcome=TradeOutcome.LOSS,
            pnl=-2.0,
            pnl_pips=-20,
            reasoning="Signal",
            notes="",
            setup_name="Test",
            timeframe="1H",
            validation_score=50,
            tags=[],
            metadata={}
        )
        history.add_trade(trade)
    
    # Créer le predictor
    predictor = MistakePredictor(history)
    
    # Tester la prédiction
    prediction = predictor.predict_mistake()
    
    print(f"\n{'='*60}")
    print(f"MISTAKE PREDICTOR")
    print(f"{'='*60}")
    
    if prediction:
        print(f"Pattern détecté: {prediction.pattern.value}")
        print(f"Probabilité: {prediction.probability:.0f}%")
        print(f"Message: {prediction.alert_message}")
        print(f"Recommandation: {prediction.recommendation}")
        print(f"Cooldown: {prediction.cooldown_minutes} minutes")
    else:
        print("Aucun pattern détecté")
    
    print(f"\nScore de risque global: {predictor.get_risk_score()}/100")


if __name__ == "__main__":
    mistake_predictor_example()
