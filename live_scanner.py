#!/usr/bin/env python3
"""
Scanner Live - Analyse en temps réel des opportunités de trading
================================================================

Ce script analyse les marchés en temps réel et détecte les opportunités
basées sur les concepts ICT/SMC.

Usage:
    python live_scanner.py
    python live_scanner.py --symbol BTC-USD
    python live_scanner.py --timeframe 15m
"""

import argparse
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def get_data(symbol: str, timeframe: str = "1h") -> Optional[Dict]:
    """Récupère les données depuis yfinance."""
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
    except Exception as e:
        print(f"{Colors.RED}Erreur lors du téléchargement: {e}{Colors.END}")
        return None


def detect_fvg(df) -> List[Dict]:
    """Détecte les Fair Value Gaps."""
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
    """Détecte les Order Blocks."""
    blocks = []
    
    for i in range(lookback, len(df)):
        # Bearish OB: Après une montée, un range serré
        if df['Close'].iloc[i] < df['Open'].iloc[i]:  # Bougie baissière
            # Chercher le plus bas récent
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
    """Détecte les Market Structure Shifts."""
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
    import pytz
    from datetime import datetime
    
    # UTC time
    utc_dt = datetime.now(pytz.UTC)
    hour = utc_dt.hour
    
    # London: 7h-11h UTC (8h-12h CET)
    # NY: 12h-16h UTC (13h-17h CET)  
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
    """Affiche l'analyse complète du marché."""
    df = data['df']
    symbol = data['symbol']
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}📊 ANALYSE LIVE - {symbol} ({data['timeframe']}){Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")
    
    # Prix actuel
    price = data['current_price']
    change = ((price - df['Open'].iloc[-1]) / df['Open'].iloc[-1]) * 100
    
    color = Colors.GREEN if change >= 0 else Colors.RED
    print(f"\n💰 Prix: {price:.5f} ({color}{change:+.2f}%{Colors.END})")
    
    # Killzone
    killzone = get_killzone_status()
    kz_color = Colors.GREEN if killzone['active'] else Colors.YELLOW
    print(f"🕐 Killzone: {kz_color}{killzone['name']}{Colors.END} - {killzone['best_for']}")
    
    # FVG
    print(f"\n{Colors.BOLD}📊 Fair Value Gaps:{Colors.END}")
    fvgs = detect_fvg(df)
    if fvgs:
        for fvg in fvgs:
            fvg_type = fvg['type']
            fvg_color = Colors.GREEN if fvg_type == "BULLISH" else Colors.RED
            print(f"  {fvg_color}{fvg_type}{Colors.END}: {fvg['start']:.5f} - {fvg['end']:.5f} (size: {fvg['size']:.5f})")
    else:
        print(f"  {Colors.YELLOW}Aucun FVG détecté{Colors.END}")
    
    # Order Blocks
    print(f"\n{Colors.BOLD}📦 Order Blocks:{Colors.END}")
    blocks = detect_order_blocks(df)
    if blocks:
        for block in blocks:
            block_color = Colors.RED if block['type'] == "BEARISH" else Colors.GREEN
            print(f"  {block_color}{block['type']}{Colors.END}: {block['zone_low']:.5f} - {block['zone_high']:.5f}")
    else:
        print(f"  {Colors.YELLOW}Aucun OB récent{Colors.END}")
    
    # MSS
    print(f"\n{Colors.BOLD}🔄 Market Structure Shift:{Colors.END}")
    mss = detect_mss(df)
    if mss:
        mss_color = Colors.RED if mss['type'] == "BEARISH" else Colors.GREEN
        print(f"  {mss_color}⚠️ MSS {mss['type']}{Colors.END}")
        if mss['type'] == "BEARISH":
            print(f"     Prix sous le swing low: {mss['swing_low']:.5f}")
        else:
            print(f"     Prix au-dessus du swing high: {mss['swing_high']:.5f}")
    else:
        print(f"  {Colors.YELLOW}Pas de MSS détecté{Colors.END}")
    
    # Résumé du signal
    print(f"\n{Colors.BOLD}🎯 Résumé:{Colors.END}")
    
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
        print(f"  {Colors.GREEN}📈 Tendance haussière ({bullish_signals} signaux){Colors.END}")
    elif bearish_signals > bullish_signals:
        print(f"  {Colors.RED}📉 Tendance baissière ({bearish_signals} signaux){Colors.END}")
    else:
        print(f"  {Colors.YELLOW}⏸️ Neutre{Colors.END}")
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}\n")


def main():
    parser = argparse.ArgumentParser(description="Scanner Live - Analyse ICT/SMC")
    parser.add_argument("--symbol", "-s", default="EURUSD=X", help="Symbole à analyser (ex: BTC-USD, EURUSD=X)")
    parser.add_argument("--timeframe", "-t", default="1h", help="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)")
    parser.add_argument("--watch", "-w", action="store_true", help="Mode watch - Analyse continue")
    
    args = parser.parse_args()
    
    print(f"{Colors.BLUE}🔍 Scanner Live - Concepts ICT/SMC{Colors.END}")
    print(f"Symbole: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    
    data = get_data(args.symbol, args.timeframe)
    
    if data is None:
        print(f"{Colors.RED}Erreur: Impossible de récupérer les données{Colors.END}")
        sys.exit(1)
    
    analyze_market(data)
    
    if args.watch:
        import time
        print(f"\n{Colors.YELLOW}Mode watch activé - Ctrl+C pour arrêter{Colors.END}\n")
        try:
            while True:
                time.sleep(60)  # Update chaque minute
                data = get_data(args.symbol, args.timeframe)
                if data:
                    analyze_market(data)
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Arrêté{Colors.END}")


if __name__ == "__main__":
    main()
