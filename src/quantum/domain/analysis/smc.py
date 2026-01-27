"""
Smart Money Concepts (SMC) - Analyse institutionnelle.
Détecte Order Blocks, Fair Value Gaps et structures SMC.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys
import os


from quantum.shared.config.settings import config


@dataclass
class OrderBlock:
    """Représente un Order Block."""
    type: str
    high: float
    low: float
    timestamp: object
    is_valid: bool
    strength: float


@dataclass
class FairValueGap:
    """Représente un Fair Value Gap."""
    type: str
    high: float
    low: float
    timestamp: object
    is_filled: bool
    size_percent: float


class SmartMoneyConceptsAnalyzer:
    """Analyse SMC: Order Blocks, FVG, BOS, CHoCH."""
    
    def __init__(self, lookback: int = None, min_fvg_percent: float = None):
        self.lookback = lookback or config.technical.ORDER_BLOCK_LOOKBACK
        self.min_fvg_percent = min_fvg_percent or config.technical.FVG_MIN_GAP_PERCENT
        self.order_blocks: List[OrderBlock] = []
        self.fvg_list: List[FairValueGap] = []
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyse SMC complète."""
        self.order_blocks = []
        self.fvg_list = []
        
        structure = self._detect_market_structure(df)
        self._detect_order_blocks(df)
        self._detect_fair_value_gaps(df)
        
        return {
            "market_structure": structure,
            "order_blocks": self._serialize_obs(),
            "fair_value_gaps": self._serialize_fvgs(),
            "current_analysis": self._get_current_analysis(df)
        }
    
    def _detect_market_structure(self, df: pd.DataFrame) -> Dict:
        """Détecte HH, HL, LH, LL."""
        highs = df['High'].values
        lows = df['Low'].values
        swing_highs = self._find_swings(highs, True)
        swing_lows = self._find_swings(lows, False)
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {"trend": "UNDEFINED"}
        
        hh = highs[swing_highs[-1]] > highs[swing_highs[-2]]
        hl = lows[swing_lows[-1]] > lows[swing_lows[-2]]
        
        if hh and hl:
            trend = "BULLISH"
        elif not hh and not hl:
            trend = "BEARISH"
        else:
            trend = "CONSOLIDATION"
        
        return {"trend": trend, "hh": hh, "hl": hl}
    
    def _find_swings(self, data: np.ndarray, is_high: bool, window: int = 5) -> List[int]:
        """Trouve swing points."""
        swings = []
        for i in range(window, len(data) - window):
            left = data[i-window:i]
            right = data[i+1:i+window+1]
            if is_high:
                if all(data[i] >= left) and all(data[i] >= right):
                    swings.append(i)
            else:
                if all(data[i] <= left) and all(data[i] <= right):
                    swings.append(i)
        return swings
    
    def _detect_order_blocks(self, df: pd.DataFrame):
        """Détecte Order Blocks."""
        close = df['Close'].values
        open_p = df['Open'].values
        high = df['High'].values
        low = df['Low'].values
        
        for i in range(2, min(len(df) - 4, self.lookback)):
            impulse = close[i+3] - close[i] if i + 3 < len(df) else 0
            impulse_pct = abs(impulse / close[i]) * 100 if close[i] != 0 else 0
            
            if impulse_pct < 0.5:
                continue
            
            if impulse > 0 and close[i] < open_p[i]:
                self.order_blocks.append(OrderBlock(
                    "BULLISH", high[i], low[i], df.index[i], True, min(impulse_pct/2, 1)
                ))
            elif impulse < 0 and close[i] > open_p[i]:
                self.order_blocks.append(OrderBlock(
                    "BEARISH", high[i], low[i], df.index[i], True, min(impulse_pct/2, 1)
                ))
        
        # Invalider OB traversés
        current = close[-1]
        for ob in self.order_blocks:
            if ob.type == "BULLISH" and current < ob.low:
                ob.is_valid = False
            elif ob.type == "BEARISH" and current > ob.high:
                ob.is_valid = False
    
    def _detect_fair_value_gaps(self, df: pd.DataFrame):
        """Détecte Fair Value Gaps."""
        for i in range(2, len(df)):
            h1, l3 = df['High'].iloc[i-2], df['Low'].iloc[i]
            l1, h3 = df['Low'].iloc[i-2], df['High'].iloc[i]
            
            if l3 > h1:  # Bullish FVG
                gap_pct = ((l3 - h1) / ((l3 + h1) / 2)) * 100
                if gap_pct >= self.min_fvg_percent:
                    self.fvg_list.append(FairValueGap(
                        "BULLISH", l3, h1, df.index[i], False, gap_pct
                    ))
            
            if h3 < l1:  # Bearish FVG
                gap_pct = ((l1 - h3) / ((l1 + h3) / 2)) * 100
                if gap_pct >= self.min_fvg_percent:
                    self.fvg_list.append(FairValueGap(
                        "BEARISH", l1, h3, df.index[i], False, gap_pct
                    ))
    
    def _get_current_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyse situation actuelle."""
        price = df['Close'].iloc[-1]
        valid_obs = [ob for ob in self.order_blocks if ob.is_valid]
        
        signal, reason = "WAIT", "Pas de structure SMC active"
        
        for ob in valid_obs[-10:]:
            if ob.type == "BULLISH" and ob.low <= price <= ob.high:
                signal, reason = "BUY", f"Dans OB bullish ({ob.low:.5f})"
                break
            elif ob.type == "BEARISH" and ob.low <= price <= ob.high:
                signal, reason = "SELL", f"Dans OB bearish ({ob.high:.5f})"
                break
        
        return {"price": price, "signal": signal, "reason": reason}
    
    def _serialize_obs(self) -> List[Dict]:
        return [{"type": ob.type, "high": ob.high, "low": ob.low, "valid": ob.is_valid}
                for ob in self.order_blocks if ob.is_valid][-10:]
    
    def _serialize_fvgs(self) -> List[Dict]:
        return [{"type": f.type, "high": f.high, "low": f.low, "filled": f.is_filled}
                for f in self.fvg_list if not f.is_filled][-10:]
    
    def get_ob_proximity_score(self, df: pd.DataFrame) -> float:
        """Score de proximité aux OB (-1 à +1)."""
        price = df['Close'].iloc[-1]
        for ob in [o for o in self.order_blocks if o.is_valid][-20:]:
            if ob.low <= price <= ob.high:
                return ob.strength if ob.type == "BULLISH" else -ob.strength
        return 0.0
