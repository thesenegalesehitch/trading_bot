#!/usr/bin/env python3
"""
ICT Full Setup Scanner - Démarrage Rapide
==========================================

Ce script permet de démarrer rapidement un scan ICT Full Setup.

Usage:
    python3 run_ict_scanner.py --symbol BTCUSDT --timeframe 15m
    python3 run_ict_scanner.py --help
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ajout du path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from quantum.domain.analysis.ict_full_setup import (
    ICTFullSetupDetector,
    KillZoneAnalyzer,
    VolumeSpikeDetector
)


def generate_sample_data(symbol: str, timeframe: str, n_candles: int = 200) -> pd.DataFrame:
    """Génère des données OHLCV sample pour le test."""
    np.random.seed(42)
    
    # Simulation de mouvement de prix
    base = np.cumsum(np.random.randn(n_candles) * 0.5)
    
    # Créer des candles réalistes
    df = pd.DataFrame({
        'Open': 100 + base + np.random.randn(n_candles) * 0.2,
        'High': 100 + base + np.random.randn(n_candles) * 0.3 + 0.5,
        'Low': 100 + base + np.random.randn(n_candles) * 0.3 - 0.5,
        'Close': 100 + base + np.random.randn(n_candles) * 0.2,
        'Volume': np.random.randint(1000, 10000, n_candles)
    }, index=pd.date_range('2024-01-15', periods=n_candles, freq='15min'))
    
    return df


def print_banner():
    """Affiche la bannière du projet."""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║     ██████╗ ██████╗ ███████╗ █████╗  ██████╗██╗  ██╗           ║
║    ██╔════╝██╔═══██╗██╔════╝██╔══██╗██╔════╝██║  ██║           ║
║    ██║     ██║   ██║█████╗ ███████║██║     ███████║           ║
║    ██║     ██║   ██║██╔══╝ ██╔══██║██║     ██╔══██║           ║
║    ╚██████╗╚██████╔╝███████╗██║  ██║╚██████╗██║  ██║           ║
║     ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝           ║
║                                                                   ║
║         ICT FULL SETUP DETECTOR v1.0.0                            ║
║         Méthodologie ICT / SMC                                    ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)


def print_status(df: pd.DataFrame, symbol: str, timeframe: str):
    """Affiche le statut actuel."""
    now = datetime.utcnow()
    killzone = KillZoneAnalyzer.get_current_killzone(now)
    
    print(f"\n📊 STATUT DU SCANNER")
    print(f"   Symbole: {symbol}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Bougies: {len(df)}")
    print(f"   Prix Actuel: {df['Close'].iloc[-1]:.5f}")
    print(f"   Killzone: {killzone or 'Hors zone'}")
    print(f"   Horaire UTC: {now.strftime('%H:%M:%S')}")
    

def main():
    parser = argparse.ArgumentParser(
        description='ICT Full Setup Scanner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python3 run_ict_scanner.py --symbol BTCUSDT --timeframe 15m
  python3 run_ict_scanner.py --symbol EURUSD --timeframe 1h --live
  python3 run_ict_scanner.py --symbol ETHUSDT --timeframe 15m --test
        """
    )
    
    parser.add_argument('--symbol', '-s', type=str, default='BTCUSDT',
                        help='Symbole à scanner (default: BTCUSDT)')
    parser.add_argument('--timeframe', '-t', type=str, default='15m',
                        help='Timeframe (default: 15m)')
    parser.add_argument('--test', action='store_true',
                        help='Mode test avec données sample')
    parser.add_argument('--live', action='store_true',
                        help='Mode live (nécessite connexion API)')
    parser.add_argument('--min-rr', type=float, default=2.0,
                        help='Ratio Risk/Reward minimum (default: 2.0)')
    
    args = parser.parse_args()
    
    # Afficher la bannière
    print_banner()
    
    # Charger ou générer les données
    if args.test or not args.live:
        print("\n🔧 Mode test - Génération de données sample...")
        df = generate_sample_data(args.symbol, args.timeframe)
    else:
        print("\n🌐 Mode live - Connexion aux données...")
        # Ici vous pouvez implémenter la connexion à une API
        print("⚠️  Mode live non implémenté - Utilisation données sample")
        df = generate_sample_data(args.symbol, args.timeframe)
    
    # Afficher le statut
    print_status(df, args.symbol, args.timeframe)
    
    # Initialiser le detector
    print(f"\n🎯 INITIALISATION")
    detector = ICTFullSetupDetector(min_rr=args.min_rr)
    print(f"   RR Minimum: 1:{args.min_rr}")
    print(f"   Volume Spike Multiplier: 1.5x")
    
    # Analyse Volume
    volume_detector = VolumeSpikeDetector()
    is_spike, ratio = volume_detector.is_volume_spike(df)
    print(f"\n   Volume Spike: {'✅' if is_spike else '❌'} ({ratio:.2f}x)")
    
    # Détecter les setups
    print(f"\n🔍 RECHERCHE DE SETUPS ICT...")
    trades = detector.detect_full_setup(df, args.symbol, args.timeframe)
    
    if trades:
        print(f"\n✅ {len(trades)} SETUP(S) TROUVÉ(S)!")
        print("=" * 60)
        
        for i, trade in enumerate(trades, 1):
            print(f"\n📌 SETUP #{i}")
            print(f"   Direction: {trade.direction}")
            print(f"   Entry: {trade.ifvg_entry.entry_price:.5f}")
            print(f"   Stop Loss: {trade.ifvg_entry.stop_loss:.5f}")
            print(f"   Take Profit 1: {trade.ifvg_entry.target_1:.5f}")
            print(f"   Risk/Reward: 1:{trade.ifvg_entry.risk_reward:.1f}")
            print(f"   Confiance: {trade.confidence:.0f}%")
            print(f"   Killzone: {trade.killzone}")
            print(f"   Volume Spike: {'✅' if trade.volume_spike_confirmed else '❌'}")
    else:
        print("\n❌ Aucun setup ICT Full Setup détecté")
        print("\n💡 Suggestions:")
        print("   - Vérifiez que vous êtes en killzone (8-11h ou 13-16h UTC)")
        print("   - Augmentez le nombre de bougies")
        print("   - Baissez le seuil RR (--min-rr 1.5)")
    
    print("\n" + "=" * 60)
    print("Pour plus d'informations, consultez: docs/ICT_FULL_SETUP_GUIDE.md")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Arrêt du scanner par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)
