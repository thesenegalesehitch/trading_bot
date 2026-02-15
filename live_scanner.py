#!/usr/bin/env python3
"""
Scanner Live - Analyse en temps reel des opportunites de trading
==============================================================

Ce script analyse les marches en temps reel et detecte les opportunites
basees sur les concepts ICT/SMC.

Installation:
    pip install yfinance pandas pytz

Usage:
    python live_scanner.py
    python live_scanner.py --symbol BTC-USD
    python live_scanner.py --timeframe 15m
"""

import argparse
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Couleurs pour le terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
BOLD = '\033[1m'
END = '\033[0m'


def get_data(symbol: str, timeframe: str = "1h") -> Optional[Dict]:
    """Recupere les donnees depuis yfinance."""
    try:
        import yfinance as yf
        
        # Map timeframe
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"
        }
        
        interval = interval_map.get(timeframe, "1h")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5d", interval=interval)
        
        if df.empty:
            return None
            
        return {
            "df": df,
            "symbol": symbol,
            "timeframe": interval,
            "current_price": df['Close'].iloc[-1],
            "high": df['High'].iloc[-1],
            "low": df['Low'].iloc[-1],
            "open": df['Open'].iloc[-1],
            "volume": df['Volume'].iloc[-1]
        }
    except ImportError:
        print(f"{RED}Erreur: Installe yfinance avec: pip install yfinance{END}")
        return None
    except Exception as e:
        print(f"{RED}Erreur lors du telechargement: {e}{END}")
        return None


def detect_fvg(df) -> List[Dict]:
    """Detecte les Fair Value Gaps."""
    fvgs = []
    
    for i in range(2, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        prev2 = df.iloc[i-2]
        
        # FVG haussier: Low actuel > High d'il y a 2 bougies
        if current['Low'] > prev2['High']:
            fvgs.append({
                "type": "BULLISH",
                "start": prev2['High'],
                "end": current['Low'],
                "size": current['Low'] - prev2['High'],
                "index": i
            })
        
        # FVG baissier: High actuel < Low d'il y a 2 bougies
        if current['High'] < prev2['Low']:
            fvgs.append({
                "type": "BEARISH",
                "start": prev2['Low'],
                "end": current['High'],
                "size": prev2['Low'] - current['High'],
                "index": i
            })
    
    return fvgs[-5:]  # Retourne les 5 derniers


def detect_order_blocks(df, lookback: int = 10) -> List[Dict]:
    """Detecte les Order Blocks."""
    blocks = []
    
    for i in range(lookback, len(df)):
        # Bearish OB: Apres une monte, un range serré
        if df['Close'].iloc[i] < df['Open'].iloc[i]:  # Bougie baissiere
            # Chercher le plus bas recent
            recent_lows = df['Low'].iloc[i-lookback:i].min()
            if df['Low'].iloc[i] <= recent_lows * 1.001:  # Proche du low
                blocks.append({
                    "type": "BEARISH",
                    "zone_low": df['Low'].iloc[i],
                    "zone_high": df['High'].iloc[i],
                    "index": i
                })
    
    return blocks[-5:]


def detect_mss(df) -> Optional[Dict]:
    """Detecte les Market Structure Shifts."""
    if len(df) < 20:
        return None
    
    # Prix sous le dernier swing low?
    recent_lows = []
    for i in range(10, len(df)-1):
        if df['Low'].iloc[i] < df['Low'].iloc[i-1] and df['Low'].iloc[i] < df['Low'].iloc[i+1]:
            recent_lows.append(df['Low'].iloc[i])
    
    if len(recent_lows) > 1:
        current_low = df['Low'].iloc[-1]
        last_swing_low = recent_lows[-1]
        
        if current_low < last_swing_low:
            return {
                "type": "BEARISH",
                "swing_low": last_swing_low,
                "current": current_low,
                "break": last_swing_low - current_low
            }
        
        # Prix au-dessus du dernier swing high?
        recent_highs = []
        for i in range(10, len(df)-1):
            if df['High'].iloc[i] > df['High'].iloc[i-1] and df['High'].iloc[i] > df['High'].iloc[i+1]:
                recent_highs.append(df['High'].iloc[i])
        
        if len(recent_highs) > 1:
            current_high = df['High'].iloc[-1]
            last_swing_high = recent_highs[-1]
            
            if current_high > last_swing_high:
                return {
                    "type": "BULLISH",
                    "swing_high": last_swing_high,
                    "current": current_high,
                    "break": current_high - last_swing_high
                }
    
    return None


def get_killzone_status() -> Dict:
    """Retourne le status de la killzone actuelle."""
    from datetime import datetime
    import pytz
    
    # UTC time
    utc_dt = datetime.now(pytz.UTC)
    hour = utc_dt.hour
    
    # London: 7h-11h UTC
    # NY: 12h-16h UTC
    # Asia: 0h-4h UTC
    
    if 7 <= hour <= 11:
        return {"active": True, "name": "LONDON", "best_for": "Breakouts"}
    elif 12 <= hour <= 16:
        return {"active": True, "name": "NY", "best_for": "Reversions"}
    elif 0 <= hour <= 4:
        return {"active": True, "name": "ASIA", "best_for": "Range trading"}
    else:
        return {"active": False, "name": "OFF-HOURS", "best_for": "Wait"}


def analyze_market(data: Dict) -> None:
    """Affiche l'analyse complete du marche."""
    df = data['df']
    symbol = data['symbol']
    
    print(f"\n{BOLD}{'='*60}{END}")
    print(f"{BOLD}ANALYSE LIVE - {symbol} ({data['timeframe']}){END}")
    print(f"{BOLD}{'='*60}{END}")
    
    # Prix actuel
    price = data['current_price']
    change = ((price - df['Open'].iloc[-1]) / df['Open'].iloc[-1]) * 100
    
    color = GREEN if change >= 0 else RED
    print(f"\nPrix: {price:.5f} ({color}{change:+.2f}%{END})")
    
    # Killzone
    killzone = get_killzone_status()
    kz_color = GREEN if killzone['active'] else YELLOW
    print(f"Killzone: {kz_color}{killzone['name']}{END} - {killzone['best_for']}")
    
    # FVG
    print(f"\n{BOLD}Fair Value Gaps:{END}")
    fvgs = detect_fvg(df)
    if fvgs:
        for fvg in fvgs:
            fvg_type = fvg['type']
            fvg_color = GREEN if fvg_type == "BULLISH" else RED
            print(f"  {fvg_color}{fvg_type}{END}: {fvg['start']:.5f} - {fvg['end']:.5f}")
    else:
        print(f"  {YELLOW}Aucun FVG detecte{END}")
    
    # Order Blocks
    print(f"\n{BOLD}Order Blocks:{END}")
    blocks = detect_order_blocks(df)
    if blocks:
        for block in blocks:
            block_color = RED if block['type'] == "BEARISH" else GREEN
            print(f"  {block_color}{block['type']}{END}: {block['zone_low']:.5f} - {block['zone_high']:.5f}")
    else:
        print(f"  {YELLOW}Aucun OB recent{END}")
    
    # MSS
    print(f"\n{BOLD}Market Structure Shift:{END}")
    mss = detect_mss(df)
    if mss:
        mss_color = RED if mss['type'] == "BEARISH" else GREEN
        print(f"  {mss_color}MSS {mss['type']}{END}")
        if mss['type'] == "BEARISH":
            print(f"     Prix sous le swing low: {mss['swing_low']:.5f}")
        else:
            print(f"     Prix au-dessus du swing high: {mss['swing_high']:.5f}")
    else:
        print(f"  {YELLOW}Pas de MSS detecte{END}")
    
    # Resume du signal
    print(f"\n{BOLD}Resume:{END}")
    
    # Compter les signaux
    bullish_signals = 0
    bearish_signals = 0
    
    if fvgs:
        for fvg in fvgs:
            if fvg['type'] == "BULLISH":
                bullish_signals += 1
            else:
                bearish_signals += 1
    
    if mss:
        if mss['type'] == "BULLISH":
            bullish_signals += 1
        else:
            bearish_signals += 1
    
    if bullish_signals > bearish_signals:
        print(f"  {GREEN}Tendance haussiere ({bullish_signals} signaux){END}")
    elif bearish_signals > bullish_signals:
        print(f"  {RED}Tendance baissiere ({bearish_signaux} signaux){END}")
    else:
        print(f"  {YELLOW}Neutre{END}")
    
    print(f"\n{BOLD}{'='*60}{END}\n")


def main():
    parser = argparse.ArgumentParser(description="Scanner Live - Analyse ICT/SMC")
    parser.add_argument("--symbol", "-s", default="EURUSD=X", help="Symbole a analyser (ex: BTC-USD, EURUSD=X)")
    parser.add_argument("--timeframe", "-t", default="1h", help="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)")
    parser.add_argument("--watch", "-w", action="store_true", help="Mode watch - Analyse continue")
    
    args = parser.parse_args()
    
    print(f"{BLUE}Scanner Live - Concepts ICT/SMC{END}")
    print(f"Symbole: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    
    data = get_data(args.symbol, args.timeframe)
    
    if data is None:
        print(f"{RED}Erreur: Impossible de recuperer les donnees{END}")
        sys.exit(1)
    
    analyze_market(data)
    
    if args.watch:
        import time
        print(f"\n{YELLOW}Mode watch active - Ctrl+C pour arreter{END}\n")
        try:
            while True:
                time.sleep(60)  # Update chaque minute
                data = get_data(args.symbol, args.timeframe)
                if data:
                    analyze_market(data)
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Arrete{END}")


if __name__ == "__main__":
    main()
