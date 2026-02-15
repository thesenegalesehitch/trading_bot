"""
Auto-Post-Mortem - Génère une analyse automatique après chaque trade.
Phase 4: Innovations - Trade Advisor & Coach

Cet outil génère une analyse détaillée automatiquement après la fermeture d'un trade.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from quantum.domain.data.downloader import DataDownloader
from quantum.domain.data.feature_engine import TechnicalIndicators
from quantum.domain.core.regime_detector import RegimeDetector
from quantum.domain.coach.history import Trade, TradeOutcome


@dataclass
class PostMortemAnalysis:
    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    outcome: TradeOutcome
    
    # Ce qui était bon
    what_was_good: List[str]
    
    # Ce qui a foiré
    what_failed: List[str]
    
    # Leçons
    lessons: List[str]
    
    # Améliorations suggérées
    improvements: Dict[str, Any]
    
    # Score global
    overall_score: int  # 0-100
    
    # Rapport formaté
    report: str


class AutoPostMortem:
    """
    Génère une analyse automatique post-trade.
    
    Analyse:
    - Le timing de l'entrée
    - Le placement du Stop Loss
    - Le ratio Risk/Reward
    - L'alignement avec la tendance
    - Les conditions du marché
    """
    
    def __init__(self):
        self.downloader = DataDownloader()
        self.indicators = TechnicalIndicators()
        self.regime_detector = RegimeDetector()
    
    def analyze(self, trade: Trade) -> PostMortemAnalysis:
        """
        Génère une analyse post-mortem pour un trade.
        """
        if trade.outcome == TradeOutcome.OPEN:
            return self._create_error("Le trade est encore ouvert")
        
        # Télécharger les données autour du trade
        try:
            df = self.downloader.download_data(
                symbol=trade.symbol,
                interval="1h",
                years=1
            )
        except Exception:
            df = None
        
        # Analyser différents aspects
        what_was_good = []
        what_failed = []
        lessons = []
        improvements = {}
        
        # Analyse du timing d'entrée
        timing_analysis = self._analyze_timing(trade, df)
        what_was_good.extend(timing_analysis.get('good', []))
        what_failed.extend(timing_analysis.get('failed', []))
        lessons.extend(timing_analysis.get('lessons', []))
        
        # Analyse du Stop Loss
        sl_analysis = self._analyze_stop_loss(trade, df)
        what_was_good.extend(sl_analysis.get('good', []))
        what_failed.extend(sl_analysis.get('failed', []))
        lessons.extend(sl_analysis.get('lessons', []))
        
        # Analyse du Risk/Reward
        rr_analysis = self._analyze_risk_reward(trade)
        what_was_good.extend(rr_analysis.get('good', []))
        what_failed.extend(rr_analysis.get('failed', []))
        improvements.update(rr_analysis.get('improvements', {}))
        
        # Analyse de l'alignement tendance
        trend_analysis = self._analyze_trend_alignment(trade, df)
        what_was_good.extend(trend_analysis.get('good', []))
        what_failed.extend(trend_analysis.get('failed', []))
        lessons.extend(trend_analysis.get('lessons', []))
        
        # Calculer le score global
        overall_score = self._calculate_score(trade, what_was_good, what_failed)
        
        # Générer le rapport
        report = self._generate_report(trade, what_was_good, what_failed, lessons, improvements, overall_score)
        
        return PostMortemAnalysis(
            trade_id=trade.id,
            symbol=trade.symbol,
            direction=trade.direction,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            outcome=trade.outcome,
            what_was_good=what_was_good,
            what_failed=what_failed,
            lessons=lessons,
            improvements=improvements,
            overall_score=overall_score,
            report=report
        )
    
    def _analyze_timing(self, trade: Trade, df: Optional[pd.DataFrame]) -> Dict[str, List[str]]:
        """Analyse le timing de l'entrée."""
        result = {'good': [], 'failed': [], 'lessons': []}
        
        if df is None:
            return result
        
        # Trouver les données autour de l'entrée
        entry_time = trade.entry_time
        df_entry = df[df.index >= entry_time - timedelta(hours=24)]
        df_entry = df_entry[df_entry.index <= entry_time + timedelta(hours=1)]
        
        if len(df_entry) < 5:
            return result
        
        # Calculer les indicateurs au moment de l'entrée
        try:
            indicators = self.indicators.calculate_all_indicators(df_entry)
            
            # Vérifier RSI
            rsi = indicators.get('rsi')
            if rsi is not None:
                last_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
                
                if trade.direction == "BUY":
                    if last_rsi < 40:
                        result['good'].append(f"RSI OK: Entry avec RSI={last_r:.1f} (zone opportunité)")
                    elif last_rsi > 70:
                        result['failed'].append(f"RSI trop haut: {last_rsi:.1f} (suracheté)")
                        result['lessons'].append("Éviter les achats quand RSI > 70")
                else:
                    if last_rsi > 60:
                        result['good'].append(f"RSI OK: Entry avec RSI={last_rsi:.1f} (zone opportunité)")
                    elif last_rsi < 30:
                        result['failed'].append(f"RSI trop bas: {last_rsi:.1f} (survendu)")
                        result['lessons'].append("Éviter les ventes quand RSI < 30")
            
            # Vérifier MACD
            macd = indicators.get('macd')
            macd_signal = indicators.get('macd_signal')
            if macd is not None and macd_signal is not None:
                if len(macd) > 0 and len(macd_signal) > 0:
                    if trade.direction == "BUY":
                        if macd.iloc[-1] > macd_signal.iloc[-1]:
                            result['good'].append("MACD haussier au moment de l'entrée")
                        else:
                            result['failed'].append("MACD baissier au moment de l'entrée")
                    else:
                        if macd.iloc[-1] < macd_signal.iloc[-1]:
                            result['good'].append("MACD baissier au moment de l'entrée")
                        else:
                            result['failed'].append("MACD haussier au moment de l'entrée")
        
        except Exception:
            pass
        
        return result
    
    def _analyze_stop_loss(self, trade: Trade, df: Optional[pd.DataFrame]) -> Dict[str, List[str]]:
        """Analyse le placement du Stop Loss."""
        result = {'good': [], 'failed': [], 'lessons': []}
        
        if df is None:
            return result
        
        # Calculer la distance du SL en pips
        sl_distance = abs(trade.entry_price - trade.stop_loss) * 10000
        
        # Analyser le SL
        if sl_distance < 20:
            result['failed'].append(f"SL trop serré: {sl_distance:.1f} pips")
            result['lessons'].append(f"SL minimum recommandé: 20-30 pips pour {trade.symbol}")
        elif sl_distance > 100:
            result['failed'].append(f"SL trop large: {sl_distance:.1f} pips")
            result['lessons'].append("SL trop large = risque de compte trop important")
        else:
            result['good'].append(f"SL acceptable: {sl_distance:.1f} pips")
        
        # Vérifier si le SL était sous un support/résistance
        entry_time = trade.entry_time
        df_near = df[(df.index >= entry_time - timedelta(hours=24)) & (df.index <= entry_time)]
        
        if len(df_near) > 0:
            recent_low = df_near['Low'].min()
            recent_high = df_near['High'].max()
            
            if trade.direction == "BUY":
                sl_buffer = (trade.entry_price - trade.stop_loss) / trade.entry_price * 100
                if trade.stop_loss > recent_low * 1.005:
                    result['failed'].append("SL pas assez sous le support")
                else:
                    result['good'].append("SL bien placé sous le support")
            else:
                if trade.stop_loss < recent_high * 0.995:
                    result['failed'].append("SL pas assez au-dessus de la résistance")
                else:
                    result['good'].append("SL bien placé au-dessus de la résistance")
        
        return result
    
    def _analyze_risk_reward(self, trade: Trade) -> Dict[str, List[str]]:
        """Analyse le ratio Risk/Reward."""
        result = {'good': [], 'failed': [], 'improvements': {}}
        
        risk = abs(trade.entry_price - trade.stop_loss)
        reward = abs(trade.take_profit - trade.entry_price)
        
        if risk > 0:
            rr_ratio = reward / risk
            
            if rr_ratio >= 2:
                result['good'].append(f"Excellent R:R: 1:{rr_ratio:.2f}")
            elif rr_ratio >= 1.5:
                result['good'].append(f"Bon R:R: 1:{rr_ratio:.2f}")
            elif rr_ratio >= 1:
                result['failed'].append(f"R:R insuffisant: 1:{rr_ratio:.2f}")
                result['lessons'].append("Viser un R:R minimum de 1:1.5")
            else:
                result['failed'].append(f"Mauvais R:R: 1:{rr_ratio:.2f}")
                result['lessons'].append("Trade non rentable mathématiquement")
            
            # Suggérer une amélioration
            if trade.take_profit:
                suggested_tp = trade.entry_price + (trade.entry_price - trade.stop_loss) * 2
                if trade.direction == "SELL":
                    suggested_tp = trade.entry_price - (trade.stop_loss - trade.entry_price) * 2
                
                result['improvements'] = {
                    'current_tp': trade.take_profit,
                    'suggested_tp': round(suggested_tp, 5),
                    'current_rr': round(rr_ratio, 2),
                    'suggested_rr': 2.0
                }
        
        return result
    
    def _analyze_trend_alignment(self, trade: Trade, df: Optional[pd.DataFrame]) -> Dict[str, List[str]]:
        """Analyse l'alignement avec la tendance."""
        result = {'good': [], 'failed': [], 'lessons': []}
        
        if df is None:
            return result
        
        try:
            # Calculer les EMAs
            ema_20 = df['Close'].ewm(span=20).mean()
            ema_50 = df['Close'].ewm(span=50).mean()
            
            # Vérifier la tendance au moment de l'entrée
            entry_idx = df.index[df.index >= trade.entry_time]
            if len(entry_idx) > 0:
                idx = entry_idx[0]
                ema20_at_entry = ema_20.loc[:idx].iloc[-1] if len(ema_20.loc[:idx]) > 0 else ema_20.iloc[-1]
                ema50_at_entry = ema_50.loc[:idx].iloc[-1] if len(ema_50.loc[:idx]) > 0 else ema_50.iloc[-1]
                
                is_bullish = ema20_at_entry > ema50_at_entry
                
                if trade.direction == "BUY":
                    if is_bullish:
                        result['good'].append("Entry dans le sens de la tendance haussière")
                    else:
                        result['failed'].append("Entry CONTRE la tendance haussière")
                        result['lessons'].append("Préférer les achats en tendance haussière")
                else:
                    if not is_bullish:
                        result['good'].append("Entry dans le sens de la tendance baissière")
                    else:
                        result['failed'].append("Entry CONTRE la tendance baissière")
                        result['lessons'].append("Préférer les ventes en tendance baissière")
        
        except Exception:
            pass
        
        return result
    
    def _calculate_score(self, trade: Trade, what_good: List[str], what_failed: List[str]) -> int:
        """Calcule le score global du trade."""
        score = 50  # Score de base
        
        # Bonus pour ce qui était bon
        score += len(what_good) * 5
        
        # Malus pour ce qui a foiré
        score -= len(what_failed) * 8
        
        # Bonus/malus selon le résultat
        if trade.outcome == TradeOutcome.WIN:
            score += 15
        else:
            score -= 10
        
        # Bonus pour le R:R
        if trade.take_profit and trade.stop_loss:
            risk = abs(trade.entry_price - trade.stop_loss)
            reward = abs(trade.take_profit - trade.entry_price)
            if risk > 0:
                rr = reward / risk
                if rr >= 2:
                    score += 10
                elif rr >= 1.5:
                    score += 5
        
        return max(0, min(100, score))
    
    def _generate_report(
        self,
        trade: Trade,
        what_good: List[str],
        what_failed: List[str],
        lessons: List[str],
        improvements: Dict[str, Any],
        score: int
    ) -> str:
        """Génère le rapport formaté."""
        report = []
        report.append("=" * 60)
        report.append("           POST-MORTEM AUTO")
        report.append("=" * 60)
        report.append(f"\n📊 RÉSULTAT: {trade.pnl_pips:.1f} pips ({'WIN' if trade.outcome == TradeOutcome.WIN else 'LOSS'})")
        
        report.append("\n✅ CE QUI ÉTAIT BON:")
        if what_good:
            for item in what_good:
                report.append(f"  • {item}")
        else:
            report.append("  (Rien à signaler)")
        
        report.append("\n❌ CE QUI A FOIRÉ:")
        if what_failed:
            for item in what_failed:
                report.append(f"  • {item}")
        else:
            report.append("  (Rien à signaler)")
        
        if lessons:
            report.append("\n📈 LEÇONS:")
            for i, lesson in enumerate(lessons, 1):
                report.append(f"  {i}. {lesson}")
        
        if improvements:
            report.append("\n🔄 AMÉLIORATION:")
            report.append(f"  Entry: {improvements.get('current_tp', 'N/A')} → {improvements.get('suggested_tp', 'N/A')}")
            report.append(f"  R:R: 1:{improvements.get('current_rr', 'N/A')} → 1:{improvements.get('suggested_rr', 'N/A')}")
        
        report.append(f"\n📊 SCORE: {score}/100")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def _create_error(self, message: str) -> PostMortemAnalysis:
        """Crée un résultat d'erreur."""
        return PostMortemAnalysis(
            trade_id="ERROR",
            symbol="ERROR",
            direction="ERROR",
            entry_price=0,
            exit_price=0,
            outcome=TradeOutcome.BREAKEVEN,
            what_was_good=[],
            what_failed=[message],
            lessons=[],
            improvements={},
            overall_score=0,
            report=f"ERROR: {message}"
        )


def postmortem_example():
    """Exemple d'utilisation du Post-Mortem."""
    from quantum.domain.coach.history import Trade
    
    # Créer un trade exemple
    trade = Trade(
        id="trade_001",
        symbol="EURUSD",
        direction="BUY",
        entry_price=1.0850,
        exit_price=1.0820,
        entry_time=datetime(2026, 2, 10, 10, 0),
        exit_time=datetime(2026, 2, 10, 15, 0),
        stop_loss=1.0820,
        take_profit=1.0910,
        quantity=1.0,
        outcome=TradeOutcome.LOSS,
        pnl=-2.76,
        pnl_pips=-30,
        reasoning="Signal BUY sur rebond",
        notes="SL hit",
        setup_name="Bull Flag",
        timeframe="1H",
        validation_score=60,
        tags=["trend"],
        metadata={}
    )
    
    analyzer = AutoPostMortem()
    result = analyzer.analyze(trade)
    
    print(result.report)


if __name__ == "__main__":
    postmortem_example()
