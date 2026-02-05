"""
ICT/SMC Full Setup Detector - Séquence Complète Sweep → FVG Tap → MSS → IFVG

Logique de Détection:
1. Contextual Sweep: Prise de liquidité sur PDH/PDL ou HOD/LOD
2. The Tap: Prix touche un FVG HTF après le sweep
3. The Displacement (MSS): Cassure de structure locale avec bougie impulsive
4. IFVG Entry: FVG créé pendant le move et sekarang transpercé (inversion)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import sys
import os


from quantum.shared.config.settings import config
from quantum.domain.analysis.smc import SmartMoneyConceptsAnalyzer, FairValueGap


# Killzones UTC
KILLZONE_LONDON = (8, 11)  # 8h-11h UTC
KILLZONE_NY = (13, 16)     # 13h-16h UTC


@dataclass
class LiquidityLevel:
    """Représente un niveau de liquidité (PDH/PDL/HOD/LOD)."""
    type: str  # 'PDH', 'PDL', 'HOD', 'LOD'
    price: float
    timestamp: datetime
    is_swept: bool = False
    sweep_candle: int = None


@dataclass
class SweepEvent:
    """Représente un événement de sweep de liquidité."""
    liquidity_level: LiquidityLevel
    sweep_timestamp: datetime
    sweep_price_high: float
    sweep_price_low: float
    direction: str  # 'BULLISH' (sweep LOD) ou 'BEARISH' (sweep HOD)
    sweep_candle: int = None  # Ajouté pour compatibilité


@dataclass
class FVGTap:
    """Représente un tap sur un FVG HTF."""
    fvg: FairValueGap
    tap_timestamp: datetime
    tap_price: float
    htf_timeframe: str


@dataclass
class MSSEvent:
    """Représente un Market Structure Shift (cassure de structure)."""
    timestamp: datetime
    direction: str  # 'BULLISH' ou 'BEARISH'
    broken_high: float  # Swing high cassé (pour bullish MSS)
    broken_low: float   # Swing low cassé (pour bearish MSS)
    impulsive_candle_size: float
    is_valid: bool = True


@dataclass
class IFVGEntry:
    """Représente une zone d'entrée IFVG (Inverted Fair Value Gap)."""
    original_fvg: FairValueGap  # FVG créé pendant le move
    inversion_timestamp: datetime
    entry_price: float  # Médian 50% du FVG
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    risk_reward: float
    is_valid: bool = True


@dataclass
class FullSetupTrade:
    """Représente un trade Full Setup complet."""
    setup_id: str
    symbol: str
    direction: str  # 'BUY' ou 'SELL'
    
    # Séquence détectée
    sweep: SweepEvent
    fvg_tap: FVGTap
    mss: MSSEvent
    ifvg_entry: IFVGEntry
    
    # Confluence
    killzone: str  # 'LONDON' ou 'NY'
    volume_spike_confirmed: bool
    confluence_score: float
    
    # Métadonnées
    detected_at: datetime
    timeframe: str
    confidence: float
    
    def to_dict(self) -> Dict:
        """Serialize pour affichage/alerte."""
        return {
            "setup_id": self.setup_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "killzone": self.killzone,
            "entry": self.ifvg_entry.entry_price,
            "stop_loss": self.ifvg_entry.stop_loss,
            "take_profits": [
                self.ifvg_entry.target_1,
                self.ifvg_entry.target_2,
                self.ifvg_entry.target_3
            ],
            "risk_reward": self.ifvg_entry.risk_reward,
            "confidence": self.confidence,
            "volume_spike": self.volume_spike_confirmed,
            "detected_at": self.detected_at.isoformat(),
            "timeframe": self.timeframe,
            "sequence": {
                "swept_level": self.sweep.liquidity_level.type,
                "sweep_price": self.sweep.sweep_price_high if self.direction == "SELL" else self.sweep.sweep_price_low,
                "htf_fvg_tap": f"{self.fvg_tap.htf_timeframe} FVG",
                "mss_type": self.mss.direction,
                "ifvg_quality": "HIGH" if self.confidence > 80 else "MEDIUM" if self.confidence > 50 else "LOW"
            }
        }


class KillZoneAnalyzer:
    """Analyse des Killzones (London/NY sessions)."""
    
    @staticmethod
    def get_current_killzone(dt: datetime) -> Optional[str]:
        """Retourne la killzone actuelle si active."""
        hour = dt.hour
        
        # Londres
        if KILLZONE_LONDON[0] <= hour < KILLZONE_LONDON[1]:
            return "LONDON"
        
        # New York
        if KILLZONE_NY[0] <= hour < KILLZONE_NY[1]:
            return "NY"
        
        return None
    
    @staticmethod
    def is_in_killzone(dt: datetime) -> bool:
        """Vérifie si on est dans une killzone."""
        return KillZoneAnalyzer.get_current_killzone(dt) is not None
    
    @staticmethod
    def get_killzone_color(zone: str) -> int:
        """Retourne la couleur Discord pour la killzone."""
        colors = {
            "LONDON": 0x3498DB,  # Bleu
            "NY": 0xE74C3C      # Rouge
        }
        return colors.get(zone, 0x95A5A6)


class VolumeSpikeDetector:
    """Détecte les pics de volume."""
    
    def __init__(self, lookback: int = 10, spike_multiplier: float = 1.5):
        """
        Args:
            lookback: Nombre de bougies pour la moyenne
            spike_multiplier: Multiplicateur pour détecter le spike (1.5 = 150%)
        """
        self.lookback = lookback
        self.spike_multiplier = spike_multiplier
    
    def calculate_avg_volume(self, df: pd.DataFrame) -> float:
        """Calcule le volume moyen des N dernières bougies."""
        if len(df) < self.lookback + 1:
            return df['Volume'].mean()
        return df['Volume'].iloc[-self.lookback-1:-1].mean()
    
    def is_volume_spike(self, df: pd.DataFrame, candle_index: int = -1) -> Tuple[bool, float]:
        """
        Vérifie si la bougie a un volume spike.
        
        Returns:
            (is_spike, ratio)
        """
        if len(df) < self.lookback + 2:
            return False, 0.0
        
        avg_volume = self.calculate_avg_volume(df)
        current_volume = df['Volume'].iloc[candle_index]
        
        if avg_volume == 0:
            return False, 0.0
        
        ratio = current_volume / avg_volume
        return ratio >= self.spike_multiplier, ratio
    
    def get_volume_score(self, df: pd.DataFrame) -> float:
        """Score de volume normalisé (0-1)."""
        if len(df) < self.lookback + 1:
            return 0.0
        
        avg = self.calculate_avg_volume(df)
        current = df['Volume'].iloc[-1]
        
        if avg == 0:
            return 0.0
        
        return min(current / avg, 3.0) / 3.0  # Max score à 3x la moyenne


class LiquidityDetector:
    """Détecte les niveaux de liquidité (PDH/PDL/HOD/LOD)."""
    
    def __init__(self, session_hours: int = 24):
        """
        Args:
            session_hours: Nombre d'heures pour définir la session précédente
        """
        self.session_hours = session_hours
    
    def get_session_levels(self, df: pd.DataFrame) -> Tuple[float, float, float, float, datetime]:
        """
        Calcule les niveaux de liquidité de la session.
        
        Returns:
            (PDH, PDL, HOD, LOD, session_start)
        """
        now = datetime.now()
        session_start = now - timedelta(hours=self.session_hours)
        
        # Filtrer les données de la session
        session_df = df[df.index >= session_start]
        
        if len(session_df) < 2:
            # Fallback: utiliser tout l'historique
            pdh = df['High'].max()
            pdl = df['Low'].min()
            hod = df['High'].iloc[-20:].max() if len(df) >= 20 else df['High'].max()
            lod = df['Low'].iloc[-20:].min() if len(df) >= 20 else df['Low'].min()
            return pdh, pdl, hod, lod, session_start
        
        pdh = session_df['High'].max()
        pdl = session_df['Low'].min()
        hod = session_df['High'].iloc[-10:].max() if len(session_df) >= 10 else pdh
        lod = session_df['Low'].iloc[-10:].min() if len(session_df) >= 10 else pdl
        
        return pdh, pdl, hod, lod, session_start
    
    def detect_sweeps(
        self,
        df: pd.DataFrame,
        pdh: float,
        pdl: float,
        hod: float,
        lod: float,
        tolerance: float = 0.0001  # 0.01% de tolérance
    ) -> List[SweepEvent]:
        """
        Détecte les sweeps de liquidité.
        
        Un sweep haussier = prix descend sous le LOD et rebondit
        Un sweep baissier = prix monte au-dessus du HOD et redescend
        """
        sweeps = []
        
        for i in range(5, len(df)):
            high = df['High'].iloc[i]
            low = df['Low'].iloc[i]
            close = df['Close'].iloc[i]
            
            # Sweep baissier (HOD sweep)
            if high >= hod * (1 + tolerance) and close < hod:
                sweeps.append(SweepEvent(
                    liquidity_level=LiquidityLevel(
                        type="HOD",
                        price=hod,
                        timestamp=df.index[i],
                        is_swept=True,
                        sweep_candle=i
                    ),
                    sweep_timestamp=df.index[i],
                    sweep_price_high=high,
                    sweep_price_low=low,
                    direction="BEARISH"
                ))
            
            # Sweep haussier (LOD sweep)
            if low <= pdl * (1 - tolerance) and close > pdl:
                sweeps.append(SweepEvent(
                    liquidity_level=LiquidityLevel(
                        type="LOD",
                        price=pdl,
                        timestamp=df.index[i],
                        is_swept=True,
                        sweep_candle=i
                    ),
                    sweep_timestamp=df.index[i],
                    sweep_price_high=high,
                    sweep_price_low=low,
                    direction="BULLISH"
                ))
        
        return sweeps[-5:]  # Garder seulement les 5 derniers sweeps


class FVGTapDetector:
    """Détecte les taps sur FVGs de timeframe supérieur."""
    
    def __init__(self, smc_analyzer: SmartMoneyConceptsAnalyzer):
        self.smc = smc_analyzer
    
    def detect_htf_fvg_taps(
        self,
        df_ltf: pd.DataFrame,
        df_htf: pd.DataFrame,
        sweep_event: SweepEvent
    ) -> List[FVGTap]:
        """
        Détecte les taps sur FVGs HTF après un sweep.
        
        Le prix doit toucher un FVG du timeframe supérieur
        immédiatement après le sweep.
        """
        taps = []
        
        # Analyser les FVGs du HTF
        self.smc.analyze(df_htf)
        htf_fvgs = self.smc.fvg_list
        
        # Chercher les FVGs non remplis
        unfilled_fvgs = [f for f in htf_fvgs if not f.is_filled]
        
        # Fix: sweep_candle peut être 0 (falsy), donc utiliser 'is not None'
        sweep_idx = sweep_event.sweep_candle if sweep_event.sweep_candle is not None else len(df_ltf) - 1
        
        # Vérifier les bougies après le sweep
        for i in range(sweep_idx + 1, min(sweep_idx + 10, len(df_ltf))):
            high = df_ltf['High'].iloc[i]
            low = df_ltf['Low'].iloc[i]
            
            for fvg in unfilled_fvgs:
                # Vérifier si le prix touche le FVG
                if sweep_event.direction == "BEARISH":
                    # Prix monte et touche le bas d'un FVG bullish
                    if fvg.type == "BULLISH" and low <= fvg.high <= high:
                        taps.append(FVGTap(
                            fvg=fvg,
                            tap_timestamp=df_ltf.index[i],
                            tap_price=(fvg.high + fvg.low) / 2,
                            htf_timeframe="H4"  # À ajuster selon le HTF utilisé
                        ))
                else:
                    # Prix descend et touche le haut d'un FVG bearish
                    if fvg.type == "BEARISH" and low <= fvg.low <= high:
                        taps.append(FVGTap(
                            fvg=fvg,
                            tap_timestamp=df_ltf.index[i],
                            tap_price=(fvg.high + fvg.low) / 2,
                            htf_timeframe="H4"
                        ))
        
        return taps[:3]  # Max 3 taps


class MSSDetector:
    """Détecte les Market Structure Shifts (cassures de structure)."""
    
    def detect_mss(
        self,
        df: pd.DataFrame,
        direction: str,
        sweep_event: SweepEvent
    ) -> Optional[MSSEvent]:
        """
        Détecte un MSS après un sweep.
        
        Le MSS est validé si:
        - Le prix casse la structure locale (swing high/low)
        - La bougie de cassure est impulsive (corps > 60% de la range)
        """
        swing_window = 5
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        opens = df['Open'].values
        
        # Fix: sweep_candle peut être 0 (falsy), donc utiliser 'is not None'
        sweep_idx = sweep_event.sweep_candle if sweep_event.sweep_candle is not None else len(df) - 5
        
        if direction == "BEARISH":
            # Chercher un swing high à casser
            for i in range(sweep_idx + 2, len(df)):
                # Calculer la bougie impulsive
                body = abs(closes[i] - opens[i])
                total_range = highs[i] - lows[i] + 1e-10
                impulsive_ratio = body / total_range
                
                # Vérifier la cassure du swing high
                recent_highs = highs[max(0, i-10):i]
                if len(recent_highs) > 3:
                    swing_high = min(recent_highs)  # Swing high le plus récent
                    
                    # Si le prix casse ce swing high avec une bougie impulsive
                    if highs[i] > swing_high and impulsive_ratio > 0.6:
                        # Vérifier que la bougie ferme sous le swing high
                        if closes[i] < swing_high:
                            return MSSEvent(
                                timestamp=df.index[i],
                                direction="BEARISH",
                                broken_high=swing_high,
                                broken_low=0,
                                impulsive_candle_size=impulsive_ratio,
                                is_valid=True
                            )
        
        else:  # BULLISH
            # Chercher un swing low à casser
            for i in range(sweep_idx + 2, len(df)):
                body = abs(closes[i] - opens[i])
                total_range = highs[i] - lows[i] + 1e-10
                impulsive_ratio = body / total_range
                
                recent_lows = lows[max(0, i-10):i]
                if len(recent_lows) > 3:
                    swing_low = max(recent_lows)  # Swing low le plus récent
                    
                    if lows[i] < swing_low and impulsive_ratio > 0.6:
                        if closes[i] > swing_low:
                            return MSSEvent(
                                timestamp=df.index[i],
                                direction="BULLISH",
                                broken_high=0,
                                broken_low=swing_low,
                                impulsive_candle_size=impulsive_ratio,
                                is_valid=True
                            )
        
        return None


class IFVGDetector:
    """Détecte les IFVGs (Inverted Fair Value Gaps) pour l'entrée."""
    
    def detect_ifvg_entry(
        self,
        df: pd.DataFrame,
        direction: str,
        mss_event: MSSEvent,
        min_rr: float = 2.0
    ) -> Optional[IFVGEntry]:
        """
        Détecte l'IFVG et calcule les niveaux de trade.
        
        L'IFVG est le FVG créé pendant le move initial (le displacement)
        qui est maintenant transpercé lors de l'inversion.
        
        Args:
            df: Données OHLCV
            direction: 'BUY' ou 'SELL'
            mss_event: L'événement MSS qui a validé la structure
            min_rr: Ratio risque/récompense minimum (default 1:2)
        
        Returns:
            IFVGEntry avec niveaux de trade ou None si RR insuffisant
        """
        if not mss_event.is_valid:
            return None
        
        # Détecter les FVGs créés pendant le move (entre sweep et MSS)
        smc = SmartMoneyConceptsAnalyzer()
        smc.analyze(df)
        
        move_start_idx = mss_event.timestamp
        if isinstance(move_start_idx, (int, float)):
            move_start_idx = max(0, int(move_start_idx) - 5)
        
        # Chercher les FVGs créés pendant le move
        for i in range(max(0, len(df) - 20), len(df)):
            h1, l3 = df['High'].iloc[i-2], df['Low'].iloc[i]
            l1, h3 = df['Low'].iloc[i-2], df['High'].iloc[i]
            
            # Bullish FVG
            if l3 > h1:
                fvg_high, fvg_low = l3, h1
                
                if direction == "SELL":
                    # L'inversion transperce le FVG bullish
                    current_low = df['Low'].iloc[-1]
                    current_high = df['High'].iloc[-1]
                    
                    if current_high > fvg_high:  # FVG transpercé
                        entry = (fvg_high + fvg_low) / 2  # Médian 50%
                        sl = df['High'].iloc[max(0, i-5):i+5].max() + 0.0001  # Swing high + buffer
                        
                        # Cibles basées sur la liquidité opposée
                        tp1 = entry - (entry - sl) * 1.5  # 1.5R
                        tp2 = entry - (entry - sl) * 2.5  # 2.5R
                        tp3 = df['Low'].min() - 0.0001  # Prochain support/liquidité
                        
                        risk = entry - sl
                        reward = entry - tp1
                        rr = reward / abs(risk) if risk != 0 else 0
                        
                        if rr >= min_rr:
                            return IFVGEntry(
                                original_fvg=FairValueGap(
                                    type="BULLISH",
                                    high=fvg_high,
                                    low=fvg_low,
                                    timestamp=df.index[i],
                                    is_filled=False,
                                    size_percent=((fvg_high - fvg_low) / fvg_low) * 100
                                ),
                                inversion_timestamp=df.index[-1],
                                entry_price=entry,
                                stop_loss=sl,
                                target_1=tp1,
                                target_2=tp2,
                                target_3=tp3,
                                risk_reward=rr,
                                is_valid=True
                            )
            
            # Bearish FVG
            if h3 < l1:
                fvg_high, fvg_low = l1, h3
                
                if direction == "BUY":
                    current_low = df['Low'].iloc[-1]
                    current_high = df['High'].iloc[-1]
                    
                    if current_low < fvg_low:  # FVG transpercé
                        entry = (fvg_high + fvg_low) / 2
                        sl = df['Low'].iloc[max(0, i-5):i+5].min() - 0.0001
                        
                        tp1 = entry + (sl - entry) * 1.5
                        tp2 = entry + (sl - entry) * 2.5
                        tp3 = df['High'].max() + 0.0001
                        
                        risk = sl - entry
                        reward = tp1 - entry
                        rr = reward / abs(risk) if risk != 0 else 0
                        
                        if rr >= min_rr:
                            return IFVGEntry(
                                original_fvg=FairValueGap(
                                    type="BEARISH",
                                    high=fvg_high,
                                    low=fvg_low,
                                    timestamp=df.index[i],
                                    is_filled=False,
                                    size_percent=((fvg_high - fvg_low) / fvg_low) * 100
                                ),
                                inversion_timestamp=df.index[-1],
                                entry_price=entry,
                                stop_loss=sl,
                                target_1=tp1,
                                target_2=tp2,
                                target_3=tp3,
                                risk_reward=rr,
                                is_valid=True
                            )
        
        return None


class ICTFullSetupDetector:
    """
    Détecteur complet de la séquence ICT Full Setup.
    
    Séquence: Sweep → FVG Tap → MSS → IFVG Entry
    
    Filtres:
    - Killzones: Londres (8-11h UTC) ou NY (13-16h UTC)
    - Volume Spike: > 150% de la moyenne des 10 dernières bougies
    - RR minimum: 1:2
    """
    
    def __init__(
        self,
        session_hours: int = 24,
        min_rr: float = 2.0,
        volume_spike_multiplier: float = 1.5
    ):
        """
        Args:
            session_hours: Heures pour définir la session de liquidité
            min_rr: Ratio risque/récompense minimum (default 1:2)
            volume_spike_multiplier: Seuil de volume spike (1.5 = 150%)
        """
        self.session_hours = session_hours
        self.min_rr = min_rr
        
        # Composants
        self.liquidity_detector = LiquidityDetector(session_hours)
        self.volume_detector = VolumeSpikeDetector(lookback=10, spike_multiplier=volume_spike_multiplier)
        self.smc = SmartMoneyConceptsAnalyzer()
        self.fvg_tap_detector = FVGTapDetector(self.smc)
        self.mss_detector = MSSDetector()
        self.ifvg_detector = IFVGDetector()
    
    def detect_full_setup(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "15m",
        df_htf: pd.DataFrame = None
    ) -> List[FullSetupTrade]:
        """
        Détecte les setups Full Setup ICT.
        
        Args:
            df: DataFrame OHLCV du timeframe d'analyse
            symbol: Symbole trading
            timeframe: Timeframe analysé
            df_htf: DataFrame du timeframe supérieur (optionnel)
        
        Returns:
            Liste des trades Full Setup détectés
        """
        trades = []
        now = datetime.now()
        
        # Vérifier la killzone
        killzone = KillZoneAnalyzer.get_current_killzone(now)
        if not killzone:
            return []  # Pas de signal hors killzone
        
        # Vérifier volume spike sur la dernière bougie
        volume_spike, volume_ratio = self.volume_detector.is_volume_spike(df)
        
        # 1. Détecter les niveaux de liquidité
        pdh, pdl, hod, lod, session_start = self.liquidity_detector.get_session_levels(df)
        
        # 2. Détecter les sweeps
        sweeps = self.liquidity_detector.detect_sweeps(df, pdh, pdl, hod, lod)
        
        for sweep in sweeps:
            # 3. Détecter le FVG Tap (HTF)
            htf_data = df_htf if df_htf is not None else df
            fvg_taps = self.fvg_tap_detector.detect_htf_fvg_taps(df, htf_data, sweep)
            
            if not fvg_taps:
                continue
            
            # 4. Détecter le MSS
            mss = self.mss_detector.detect_mss(df, sweep.direction, sweep)
            
            if not mss or not mss.is_valid:
                continue
            
            # 5. Détecter l'IFVG et calculer le trade
            direction = "BUY" if sweep.direction == "BULLISH" else "SELL"
            ifvg = self.ifvg_detector.detect_ifvg_entry(df, direction, mss, self.min_rr)
            
            if not ifvg or not ifvg.is_valid:
                continue
            
            # Calculer le score de confluence
            confluence = self._calculate_confluence(
                sweep=sweep,
                fvg_tap=fvg_taps[0],
                mss=mss,
                ifvg=ifvg,
                volume_spike=volume_spike,
                volume_ratio=volume_ratio,
                killzone=killzone
            )
            
            # Créer le trade
            trade = FullSetupTrade(
                setup_id=f"{symbol}_{timeframe}_{now.strftime('%Y%m%d%H%M%S')}",
                symbol=symbol,
                direction=direction,
                sweep=sweep,
                fvg_tap=fvg_taps[0],
                mss=mss,
                ifvg_entry=ifvg,
                killzone=killzone,
                volume_spike_confirmed=volume_spike,
                confluence_score=confluence,
                detected_at=now,
                timeframe=timeframe,
                confidence=min(confluence * 100, 95)
            )
            
            trades.append(trade)
        
        return trades
    
    def _calculate_confluence(
        self,
        sweep: SweepEvent,
        fvg_tap: FVGTap,
        mss: MSSEvent,
        ifvg: IFVGEntry,
        volume_spike: bool,
        volume_ratio: float,
        killzone: str
    ) -> float:
        """Calcule le score de confluence (0-1)."""
        score = 0.0
        
        # Killzone: +0.2
        score += 0.2
        
        # Volume spike: +0.3
        if volume_spike:
            score += 0.3
        else:
            score += min(volume_ratio * 0.2, 0.2)
        
        # MSS impulsif: +0.2
        if mss.impulsive_candle_size > 0.7:
            score += 0.2
        elif mss.impulsive_candle_size > 0.6:
            score += 0.1
        
        # RR excellent (>3): +0.1
        if ifvg.risk_reward > 3:
            score += 0.1
        elif ifvg.risk_reward > 2:
            score += 0.05
        
        return min(score, 1.0)
    
    def scan_symbol(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframes: List[str] = ["15m", "1h"]
    ) -> Dict[str, List[FullSetupTrade]]:
        """
        Scan un symbole sur plusieurs timeframes.
        
        Returns:
            Dict {timeframe: List[trades]}
        """
        results = {}
        
        for tf in timeframes:
            # Créer un DataFrame pour ce timeframe (simplifié)
            # En production, utiliser des données réelles pour chaque TF
            tf_trades = self.detect_full_setup(df, symbol, tf)
            
            if tf_trades:
                results[tf] = tf_trades
        
        return results


class ICTAlertFormatter:
    """Formate les alertes ICT pour Discord/Telegram."""
    
    @staticmethod
    def _get_value(trade, key, default=None):
        """Récupère une valeur d'un dict ou d'un objet."""
        if isinstance(trade, dict):
            return trade.get(key, default)
        return getattr(trade, key, default)
    
    @staticmethod
    def format_discord_embed(trade_data: Dict) -> Dict:
        """Crée un embed Discord riche pour un trade Full Setup."""
        
        # Support both dict and FullSetupTrade object
        direction = ICTAlertFormatter._get_value(trade_data, 'direction')
        symbol = ICTAlertFormatter._get_value(trade_data, 'symbol')
        killzone = ICTAlertFormatter._get_value(trade_data, 'killzone')
        volume_spike = ICTAlertFormatter._get_value(trade_data, 'volume_spike')
        confidence = ICTAlertFormatter._get_value(trade_data, 'confidence')
        setup_id = ICTAlertFormatter._get_value(trade_data, 'setup_id')
        
        # Accès aux données imbriquées
        if isinstance(trade_data, dict):
            entry_price = trade_data.get('entry')
            stop_loss = trade_data.get('stop_loss')
            risk_reward = trade_data.get('risk_reward')
            tps = trade_data.get('take_profits', [0, 0, 0])
            htf_timeframe = trade_data.get('sequence', {}).get('htf_fvg_tap', 'HTF').replace(' FVG', '')
            swept_level = trade_data.get('sequence', {}).get('swept_level', 'N/A')
            detected_at = trade_data.get('detected_at')
            if isinstance(detected_at, str):
                detected_at = datetime.fromisoformat(detected_at.replace('Z', '+00:00'))
        else:
            entry_price = trade_data.ifvg_entry.entry_price
            stop_loss = trade_data.ifvg_entry.stop_loss
            risk_reward = trade_data.ifvg_entry.risk_reward
            tps = [trade_data.ifvg_entry.target_1, trade_data.ifvg_entry.target_2, trade_data.ifvg_entry.target_3]
            htf_timeframe = trade_data.fvg_tap.htf_timeframe
            swept_level = trade_data.sweep.liquidity_level.type
            detected_at = trade_data.detected_at
        
        direction_emoji = "🟢" if direction == "BUY" else "🔴"
        direction_color = 0x2ECC71 if direction == "BUY" else 0xE74C3C
        
        embed = {
            "title": f"{direction_emoji} ICT Full Setup: {symbol} | {direction}",
            "color": direction_color,
            "fields": [
                {
                    "name": "🎯 Entry",
                    "value": f"`{entry_price:.5f}`",
                    "inline": True
                },
                {
                    "name": "🛑 Stop Loss",
                    "value": f"`{stop_loss:.5f}`",
                    "inline": True
                },
                {
                    "name": "📈 Risk/Reward",
                    "value": f"`1:{risk_reward:.1f}`",
                    "inline": True
                },
                {
                    "name": "🎯 Take Profits",
                    "value": f"TP1: `{tps[0]:.5f}`\n"
                             f"TP2: `{tps[1]:.5f}`\n"
                             f"TP3: `{tps[2]:.5f}`",
                    "inline": False
                },
                {
                    "name": "📊 Confluence",
                    "value": f"Killzone: **{killzone}**\n"
                             f"Volume Spike: **{'✅' if volume_spike else '❌'}**\n"
                             f"Confiance: **{confidence:.0f}%**\n"
                             f"FVG Tap: **{htf_timeframe}**",
                    "inline": False
                },
                {
                    "name": "🔄 Séquence",
                    "value": f"Sweep: **{swept_level}**\n"
                             f"MSS: **{direction}**\n"
                             f"Setup ID: `{setup_id}`",
                    "inline": False
                }
            ],
            "footer": {
                "text": f"Quantum Trading System | Detected: {detected_at.strftime('%H:%M:%S')} UTC"
            },
            "timestamp": detected_at.isoformat()
        }
        
        # Ajouter les targets sous forme de Price Targets
        if direction == "BUY":
            embed["fields"].append({
                "name": "📍 Distance aux Targets",
                "value": f"TP1: +{((tps[0] - entry_price) / entry_price * 100):.2f}%\n"
                         f"TP2: +{((tps[1] - entry_price) / entry_price * 100):.2f}%\n"
                         f"TP3: +{((tps[2] - entry_price) / entry_price * 100):.2f}%",
                "inline": True
            })
        else:
            embed["fields"].append({
                "name": "📍 Distance aux Targets",
                "value": f"TP1: -{((entry_price - tps[0]) / entry_price * 100):.2f}%\n"
                         f"TP2: -{((entry_price - tps[1]) / entry_price * 100):.2f}%\n"
                         f"TP3: -{((entry_price - tps[2]) / entry_price * 100):.2f}%",
                "inline": True
            })
        
        return embed
    
    @staticmethod
    def format_telegram_message(trade_data: Dict) -> str:
        """Crée un message Telegram formaté pour un trade Full Setup."""
        
        # Support both dict and FullSetupTrade object
        direction = ICTAlertFormatter._get_value(trade_data, 'direction')
        symbol = ICTAlertFormatter._get_value(trade_data, 'symbol')
        killzone = ICTAlertFormatter._get_value(trade_data, 'killzone')
        volume_spike = ICTAlertFormatter._get_value(trade_data, 'volume_spike')
        confidence = ICTAlertFormatter._get_value(trade_data, 'confidence')
        setup_id = ICTAlertFormatter._get_value(trade_data, 'setup_id')
        
        # Accès aux données imbriquées
        if isinstance(trade_data, dict):
            entry_price = trade_data.get('entry')
            stop_loss = trade_data.get('stop_loss')
            risk_reward = trade_data.get('risk_reward')
            tps = trade_data.get('take_profits', [0, 0, 0])
            htf_timeframe = trade_data.get('sequence', {}).get('htf_fvg_tap', 'HTF').replace(' FVG', '')
            swept_level = trade_data.get('sequence', {}).get('swept_level', 'N/A')
            detected_at = trade_data.get('detected_at')
            if isinstance(detected_at, str):
                detected_at = datetime.fromisoformat(detected_at.replace('Z', '+00:00'))
        else:
            entry_price = trade_data.ifvg_entry.entry_price
            stop_loss = trade_data.ifvg_entry.stop_loss
            risk_reward = trade_data.ifvg_entry.risk_reward
            tps = [trade_data.ifvg_entry.target_1, trade_data.ifvg_entry.target_2, trade_data.ifvg_entry.target_3]
            htf_timeframe = trade_data.fvg_tap.htf_timeframe
            swept_level = trade_data.sweep.liquidity_level.type
            detected_at = trade_data.detected_at
        
        direction_emoji = "🟢" if direction == "BUY" else "🔴"
        
        message = (
            f"{direction_emoji} *ICT FULL SETUP DETECTED*\n\n"
            f"📈 *Symbol:* {symbol}\n"
            f"🎯 *Direction:* {direction}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 *Trade Levels*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"• Entry: `{entry_price:.5f}`\n"
            f"• Stop Loss: `{stop_loss:.5f}`\n"
            f"• TP1: `{tps[0]:.5f}`\n"
            f"• TP2: `{tps[1]:.5f}`\n"
            f"• TP3: `{tps[2]:.5f}`\n\n"
            f"📈 *Risk/Reward:* `1:{risk_reward:.1f}`\n\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🔍 *Confluence*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"• Killzone: {killzone}\n"
            f"• Volume Spike: {'✅' if volume_spike else '❌'}\n"
            f"• FVG Tap: {htf_timeframe}\n"
            f"• Swept: {swept_level}\n"
            f"• Confiance: {confidence:.0f}%\n\n"
            f"⏰ *Detected:* {detected_at.strftime('%H:%M:%S')} UTC\n"
            f"🆔 *ID:* `{setup_id}`"
        )
        
        return message


# Fonctions utilitaires pour compatibilité avec l'existant
def detect_ict_full_setup(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "15m",
    min_rr: float = 2.0
) -> List[Dict]:
    """
    Fonction utilitaire pour détecter les Full Setups ICT.
    
    Args:
        df: DataFrame OHLCV
        symbol: Symbole trading
        timeframe: Timeframe d'analyse
        min_rr: Ratio RR minimum
    
    Returns:
        Liste des trades formatés en dict
    """
    detector = ICTFullSetupDetector(min_rr=min_rr)
    trades = detector.detect_full_setup(df, symbol, timeframe)
    
    return [trade.to_dict() for trade in trades]


if __name__ == "__main__":
    # Test du module ICT Full Setup
    print("=" * 70)
    print("TEST MODULE ICT FULL SETUP DETECTOR")
    print("=" * 70)
    
    # Créer des données de test
    import numpy as np
    
    np.random.seed(42)
    n = 200
    
    # Simuler un mouvement avec sweep + FVG tap + MSS + IFVG
    base = np.cumsum(np.random.randn(n) * 0.5)
    
    # Créer un sweep (prix teste le low puis rebondit)
    base[-50:-40] = base[-50] - np.linspace(0, 2, 10)  # Drop
    base[-40:-30] = base[-40] + np.linspace(0, 3, 10)  # Rebond avec FVG
    
    df = pd.DataFrame({
        'Open': 100 + base + np.random.randn(n) * 0.2,
        'High': 100 + base + np.random.randn(n) * 0.3 + 0.5,
        'Low': 100 + base + np.random.randn(n) * 0.3 - 0.5,
        'Close': 100 + base + np.random.randn(n) * 0.2,
        'Volume': np.random.randint(1000, 10000, n)
    }, index=pd.date_range('2024-01-01', periods=n, freq='15min'))
    
    print(f"\nDonnées de test créées: {len(df)} bougies")
    print(f"Prix actuel: {df['Close'].iloc[-1]:.5f}")
    
    # Tester le detector
    detector = ICTFullSetupDetector()
    
    print("\n--- Vérification Killzone ---")
    now = datetime.now()
    killzone = KillZoneAnalyzer.get_current_killzone(now)
    print(f"Killzone actuelle: {killzone}")
    
    print("\n--- Analyse Volume ---")
    volume_detector = VolumeSpikeDetector()
    is_spike, ratio = volume_detector.is_volume_spike(df)
    print(f"Volume Spike: {is_spike} (ratio: {ratio:.2f}x)")
    print(f"Score Volume: {volume_detector.get_volume_score(df):.2f}")
    
    print("\n--- Détection Liquidité ---")
    liquidity = LiquidityDetector()
    pdh, pdl, hod, lod, _ = liquidity.get_session_levels(df)
    print(f"PDH: {pdh:.5f}, PDL: {pdl:.5f}")
    print(f"HOD: {hod:.5f}, LOD: {lod:.5f}")
    
    print("\n--- Détection Sweeps ---")
    sweeps = liquidity.detect_sweeps(df, pdh, pdl, hod, lod)
    print(f"Sweeps détectés: {len(sweeps)}")
    
    for sweep in sweeps:
        print(f"  - {sweep.direction} sweep à {sweep.sweep_timestamp}")
    
    print("\n--- Détection Full Setup ---")
    trades = detector.detect_full_setup(df, "BTCUSDT", "15m")
    print(f"Trades Full Setup détectés: {len(trades)}")
    
    for trade in trades:
        print(f"\n🎯 Trade détecté:")
        print(f"  Direction: {trade.direction}")
        print(f"  Killzone: {trade.killzone}")
        print(f"  Entry: {trade.ifvg_entry.entry_price:.5f}")
        print(f"  SL: {trade.ifvg_entry.stop_loss:.5f}")
        print(f"  RR: 1:{trade.ifvg_entry.risk_reward:.1f}")
        print(f"  Confiance: {trade.confidence:.0f}%")
    
    # Test formatting
    if trades:
        print("\n--- Format Discord Embed ---")
        embed = ICTAlertFormatter.format_discord_embed(trades[0])
        print(f"Embed title: {embed['title']}")
        print(f"Embed color: #{embed['color']:06x}")
        
        print("\n--- Format Telegram ---")
        tg_msg = ICTAlertFormatter.format_telegram_message(trades[0])
        print(tg_msg)
    
    print("\n" + "=" * 70)
    print("FIN DES TESTS")
    print("=" * 70)
