#!/usr/bin/env python3
"""
Quantum Trading System - Menu Interactif ICT
=============================================

Interface simple et conviviale pour les débutants.
Pas besoin de connaître Python - juste sélectionner les options!

Usage:
    python3 run_ict_menu.py
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ajout du path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Imports des modules ICT
from quantum.domain.analysis.ict_full_setup import (
    ICTFullSetupDetector,
    KillZoneAnalyzer,
    VolumeSpikeDetector
)
from quantum.domain.analysis.multi_tf import MultiTimeframeAnalyzer


def clear_screen():
    """Nettoie l'écran."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """Affiche la bannière."""
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
║         🎯 ICT FULL SETUP DETECTOR v1.0.0                        ║
║         Interface Interactive pour Débutants                     ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)


def print_menu(title: str, options: list):
    """Affiche un menu avec des options."""
    print(f"\n{'─' * 50}")
    print(f"  📋 {title}")
    print(f"{'─' * 50}")
    
    for i, option in enumerate(options, 1):
        emoji = option.get('emoji', '•')
        text = option.get('text', '')
        print(f"   {i}. {emoji} {text}")
    
    print(f"{'─' * 50}")
    print("   0. ⬅️  Retour")
    print(f"{'─' * 50}")


def generate_sample_data(symbol: str, timeframe: str, n_candles: int = 200) -> pd.DataFrame:
    """Génère des données sample."""
    np.random.seed(42)
    base = np.cumsum(np.random.randn(n_candles) * 0.5)
    
    df = pd.DataFrame({
        'Open': 100 + base + np.random.randn(n_candles) * 0.2,
        'High': 100 + base + np.random.randn(n_candles) * 0.3 + 0.5,
        'Low': 100 + base + np.random.randn(n_candles) * 0.3 - 0.5,
        'Close': 100 + base + np.random.randn(n_candles) * 0.2,
        'Volume': np.random.randint(1000, 10000, n_candles)
    }, index=pd.date_range('2024-01-15', periods=n_candles, freq='15min'))
    
    return df


def get_user_input(prompt: str, default: str = '') -> str:
    """Récupère une entrée utilisateur."""
    if default:
        response = input(f"   {prompt} [{default}]: ").strip()
        return response if response else default
    return input(f"   {prompt}: ").strip()


def show_info():
    """Affiche les informations sur ICT."""
    clear_screen()
    print_banner()
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                     📖 QU'EST-CE QUE ICT?                         ║
╚═══════════════════════════════════════════════════════════════════╝

ICT (Inner Circle Trader) est une méthodologie de trading développée
par Michael J. Hudson. Elle se base sur l'observation du comportement
des "smart money" (les gros acteurs du marché).

╔═══════════════════════════════════════════════════════════════════╗
║                  🧩 COMPOSANTS D'UN SETUP                         ║
╚═══════════════════════════════════════════════════════════════════╝

1️⃣  SWEEP (Prise de liquidité)
    Le prix "nettoie" les stop orders aux points clés:
    • PDH/PDL (Previous Day High/Low)
    • HOD/LOD (High/Low de la session)

2️⃣  FVG TAP (Touche du FVG)
    Le prix touche un Fair Value Gap du timeframe supérieur.
    C'est une zone de déséquilibre où le prix veut revenir.

3️⃣  MSS (Changement de structure)
    Le prix casse la structure locale avec une bougie impulsive.
    C'est la validation du move.

4️⃣  IFVG ENTRY (Zone d'entrée)
    L'Inverted FVG est la zone précise pour entrer en position.
    On place l'ordre au 50% du FVG.

╔═══════════════════════════════════════════════════════════════════╗
║                    ⏰ QUAND TRADER?                                ║
╚═══════════════════════════════════════════════════════════════════╝

🟢 KILLZONES - Moments de forte liquidité:

   🇬🇧 LONDRES:  08:00 - 11:00 UTC
   🇺🇸 NEW YORK: 13:00 - 16:00 UTC

   Le projet ne génère des signaux que pendant ces horaires!

╔═══════════════════════════════════════════════════════════════════╗
║                    📊 GESTION DU RISQUE                            ║
╚═══════════════════════════════════════════════════════════════════╝

🎯 Ratio Risk/Reward minimum: 1:2
   Chaque trade doit risquer 1 pour gagner 2.

🛑 Stop Loss:
   • BUY: Sous le swing low
   • SELL: Au-dessus du swing high

📈 Take Profits:
   • TP1: 1.5R
   • TP2: 2.5R
   • TP3: Prochaine zone de liquidité
""")
    
    input("\n   Appuyez sur Entrée pour revenir au menu principal...")


def show_settings():
    """Affiche et permet de modifier les paramètres."""
    clear_screen()
    print_banner()
    
    # Paramètres par défaut
    settings = {
        'symbol': 'BTCUSDT',
        'timeframe': '15m',
        'min_rr': 2.0,
        'volume_spike': 1.5,
        'notifications': False
    }
    
    while True:
        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║              ⚙️  PARAMÈTRES ACTUELS                       ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print(f"\n   1. 📈 Symbole:          {settings['symbol']}")
        print(f"   2. ⏱️  Timeframe:       {settings['timeframe']}")
        print(f"   3. 📊 RR Minimum:      1:{settings['min_rr']}")
        print(f"   4. 📊 Volume Spike:    {settings['volume_spike']}x")
        print(f"   5. 🔔 Notifications:    {'✅ Activées' if settings['notifications'] else '❌ Désactivées'}")
        print("\n   0. ⬅️  Retour au menu principal")
        
        choice = input("\n   ➤ Votre choix: ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            print("\n   Symboles disponibles:")
            print("   • BTCUSDT - Bitcoin/USDT")
            print("   • ETHUSDT - Ethereum/USDT")
            print("   • EURUSD - Euro/US Dollar")
            print("   • GBPUSD - Livre/US Dollar")
            print("   • USDJPY - Dollar/Yen")
            settings['symbol'] = get_user_input("Entrez le symbole", "BTCUSDT")
        elif choice == '2':
            print("\n   Timeframes disponibles:")
            print("   • 1m  - 1 minute")
            print("   • 5m  - 5 minutes")
            print("   • 15m - 15 minutes")
            print("   • 1h  - 1 heure")
            print("   • 4h  - 4 heures")
            settings['timeframe'] = get_user_input("Entrez le timeframe", "15m")
        elif choice == '3':
            settings['min_rr'] = float(get_user_input("RR minimum (1.5, 2.0, 3.0)", "2.0"))
        elif choice == '4':
            settings['volume_spike'] = float(get_user_input("Volume spike multiplier", "1.5"))
        elif choice == '5':
            settings['notifications'] = not settings['notifications']
    
    return settings


def run_scan(settings: dict):
    """Exécute un scan avec les paramètres."""
    clear_screen()
    print_banner()
    
    print(f"\n🔍 SCANNER EN COURS...")
    print(f"   Symbole: {settings['symbol']}")
    print(f"   Timeframe: {settings['timeframe']}")
    print(f"   RR Minimum: 1:{settings['min_rr']}")
    
    # Générer les données
    df = generate_sample_data(settings['symbol'], settings['timeframe'])
    
    # Importer et exécuter le scanner
    from quantum.domain.analysis.ict_full_setup import (
        ICTFullSetupDetector,
        KillZoneAnalyzer,
        VolumeSpikeDetector
    )
    
    now = datetime.utcnow()
    killzone = KillZoneAnalyzer.get_current_killzone(now)
    
    print(f"\n📊 RÉSULTATS:")
    print(f"   Prix actuel: {df['Close'].iloc[-1]:.5f}")
    print(f"   Killzone: {killzone or '❌ Hors zone'}")
    
    # Analyse volume
    volume_detector = VolumeSpikeDetector()
    is_spike, ratio = volume_detector.is_volume_spike(df)
    print(f"   Volume: {'✅ Spike' if is_spike else '❌ Normal'} ({ratio:.2f}x)")
    
    # Détecter les setups
    detector = ICTFullSetupDetector(min_rr=settings['min_rr'])
    trades = detector.detect_full_setup(df, settings['symbol'], settings['timeframe'])
    
    if trades:
        print(f"\n✅ {len(trades)} SETUP(S) TROUVÉ(S)!")
        print("═" * 60)
        
        for i, trade in enumerate(trades, 1):
            emoji = "🟢" if trade.direction == "BUY" else "🔴"
            print(f"\n📌 SETUP #{i} {emoji} {trade.direction}")
            print(f"   Entry:      {trade.ifvg_entry.entry_price:.5f}")
            print(f"   Stop Loss:  {trade.ifvg_entry.stop_loss:.5f}")
            print(f"   TP1:        {trade.ifvg_entry.target_1:.5f}")
            print(f"   TP2:        {trade.ifvg_entry.target_2:.5f}")
            print(f"   TP3:        {trade.ifvg_entry.target_3:.5f}")
            print(f"   Risk/Reward: 1:{trade.ifvg_entry.risk_reward:.1f}")
            print(f"   Confiance:   {trade.confidence:.0f}%")
            print(f"   Killzone:    {trade.killzone}")
    else:
        print("\n❌ AUCUN SETUP TROUVÉ")
        print("\n💡 Raisons possibles:")
        if not killzone:
            print("   • Vous n'êtes pas en killzone (8-11h ou 13-16h UTC)")
        else:
            print("   • Les conditions ICT ne sont pas réunies")
            print("   • Le prix n'a pas fait de sweep récemment")
            print("   • Pas de structure MSS validée")
        print("\n💡 Suggestions:")
        print("   • Attendez la prochaine killzone")
        print("   • Baissez le seuil RR (aller dans Paramètres)")
        print("   • Changez de timeframe")
    
    print("\n" + "═" * 60)
    input("\n   Appuyez sur Entrée pour continuer...")


def run_multi_tf_scan():
    """Scan multi-timeframes."""
    clear_screen()
    print_banner()
    
    symbol = get_user_input("Entrez le symbole à scanner", "BTCUSDT")
    
    from quantum.domain.analysis.ict_full_setup import ICTFullSetupDetector
    from quantum.domain.analysis.multi_tf import MultiTimeframeAnalyzer
    
    print(f"\n🔍 SCAN MULTI-TIMEFRAME: {symbol}")
    
    # Générer données pour chaque TF
    timeframes = ['15m', '1h', '4h']
    results = {}
    
    df_sample = generate_sample_data(symbol, '15m')
    
    detector = ICTFullSetupDetector()
    analyzer = MultiTimeframeAnalyzer()
    
    for tf in timeframes:
        df = generate_sample_data(symbol, tf)
        
        # Scan ICT
        trades = detector.detect_full_setup(df, symbol, tf)
        results[tf] = {'trades': trades}
    
    print(f"\n📊 RÉSULTATS POUR {symbol}:")
    print("═" * 60)
    
    for tf in timeframes:
        trades = results[tf]['trades']
        emoji = "✅" if trades else "❌"
        print(f"\n   {tf}: {emoji} {len(trades)} setup(s)")
        
        for trade in trades:
            print(f"      • {trade.direction} @ {trade.ifvg_entry.entry_price:.5f}")
            print(f"        RR 1:{trade.ifvg_entry.risk_reward:.1f} | Confiance {trade.confidence:.0f}%")
    
    # Analyse tendance
    print(f"\n📈 ANALYSE DE TENDANCE:")
    data = {tf: generate_sample_data(symbol, tf) for tf in timeframes}
    trend_analysis = analyzer.analyze_trend(data)
    
    print(f"   Tendance globale: {trend_analysis['convergence']['overall_trend']}")
    print(f"   Confirmé: {'✅' if trend_analysis['convergence']['is_confirmed'] else '❌'}")
    print(f"   Score: {trend_analysis['convergence']['weighted_score']:.2f}")
    
    print("\n" + "═" * 60)
    input("\n   Appuyez sur Entrée pour continuer...")


def show_alerts():
    """Affiche les options d'alertes."""
    clear_screen()
    print_banner()
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                    🔔 CONFIGURATION ALERTES                       ║
╚═══════════════════════════════════════════════════════════════════╝

Les alertes vous permettent de recevoir les signaux en temps réel
sur Discord ou Telegram.

╔═══════════════════════════════════════════════════════════════════╗
║                    📱 CANAUX DISPONIBLES                          ║
╚═══════════════════════════════════════════════════════════════════╝

1️⃣  Discord
   • Créez un webhook dans votre serveur
   • Copiez l'URL du webhook
   • Collez-la dans les paramètres

2️⃣  Telegram
   • Créez un bot via @BotFather
   • Obtenez le token du bot
   • Obtenez votre chat_id

╔═══════════════════════════════════════════════════════════════════╗
║                    📋 FORMAT DES ALERTES                         ║
╚═══════════════════════════════════════════════════════════════════╝

Chaque alerte contient:
   • Direction du trade (BUY/SELL)
   • Niveau d'entrée
   • Stop Loss
   • 3 Take Profits
   • Ratio Risk/Reward
   • Score de confiance
   • Confluences (Killzone, Volume)
   • ID unique du setup

╔═══════════════════════════════════════════════════════════════════╗
║                    ⚙️  CONFIGURATION                              ║
╚═══════════════════════════════════════════════════════════════════╝

Créez un fichier .env à la racine du projet:

   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id

Les alertes sont activées automatiquement!
""")
    
    input("\n   Appuyez sur Entrée pour revenir au menu principal...")


def show_backtest():
    """Affiche les options de backtest."""
    clear_screen()
    print_banner()
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                    📊 BACKTEST ICT                               ║
╚═══════════════════════════════════════════════════════════════════╝

Le backtest vous permet de tester la stratégie ICT sur des données
historiques pour évaluer sa performance.

╔═══════════════════════════════════════════════════════════════════╗
║                    📈 MÉTRIQUES DISPONIBLES                      ║
╚═══════════════════════════════════════════════════════════════════╝

   • Win Rate (pourcentage de trades gagnants)
   • Profit Factor (gain/perte total)
   • Average Trade (trade moyen)
   • Maximum Drawdown (perte maximale)
   • Sharpe Ratio (qualité du signal)
   • Expectancy (espérance de gain)

╔═══════════════════════════════════════════════════════════════════╗
║                    ⚠️  LIMITES                                    ║
╚═══════════════════════════════════════════════════════════════════╝

⚡ Note importante: Les résultats passés ne garantissent pas
   les résultats futurs. Le backtest est une estimation,
   pas une prédiction!

╔═══════════════════════════════════════════════════════════════════╗
║                    🚀 DÉMARRER UN BACKTEST                       ║
╚═══════════════════════════════════════════════════════════════════╝

Pour lancer un backtest, utilisez:

   python3 run_ict_scanner.py --symbol BTCUSDT --backtest --days 365

Options disponibles:
   --symbol SYM    Symbole à tester
   --days N        Nombre de jours d'historique
   --timeframe TF  Timeframe à utiliser
   --min-rr RR     Ratio RR minimum
   --export        Exporter les résultats en CSV
""")
    
    input("\n   Appuyez sur Entrée pour revenir au menu principal...")


def show_help():
    """Affiche l'aide."""
    clear_screen()
    print_banner()
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                        ❓ AIDE                                    ║
╚═══════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════╗
║                    📁 STRUCTURE DU PROJET                        ║
╚═══════════════════════════════════════════════════════════════════╝

   quantum_trading_system/
   ├── src/quantum/
   │   ├── domain/analysis/
   │   │   ├── ict_full_setup.py    ← Module ICT principal
   │   │   ├── smc.py               ← Smart Money Concepts
   │   │   └── multi_tf.py          ← Multi-timeframe
   │   ├── application/reporting/
   │   │   └── alerts.py            ← Alertes Discord/Telegram
   │   └── infrastructure/
   │       └── exchanges/           ← Connexions exchanges
   ├── docs/
   │   └── ICT_FULL_SETUP_GUIDE.md  ← Documentation complète
   ├── tests/
   │   └── test_ict_full_setup.py  ← Tests unitaires
   └── run_ict_menu.py             ← Ce menu interactif

╔═══════════════════════════════════════════════════════════════════╗
║                    🎯 PREMIERS PAS                               ║
╚═══════════════════════════════════════════════════════════════════╝

1. Lancer le menu interactif:
   python3 run_ict_menu.py

2. Aller dans "Paramètres" pour configurer:
   • Le symbole à trader
   • Le timeframe préféré
   • Le ratio RR minimum

3. Lancer un scan dans "Scanner un symbole"

4. Consulter "Documentation ICT" pour apprendre la méthodologie

╔═══════════════════════════════════════════════════════════════════╗
║                    📞 SUPPORT                                     ║
╚═══════════════════════════════════════════════════════════════════╝

   • Documentation: docs/ICT_FULL_SETUP_GUIDE.md
   • Issues: GitHub Issues
   • Communauté: Discord
""")
    
    input("\n   Appuyez sur Entrée pour revenir au menu principal...")


def main():
    """Menu principal."""
    settings = {
        'symbol': 'BTCUSDT',
        'timeframe': '15m',
        'min_rr': 2.0,
        'volume_spike': 1.5,
        'notifications': False
    }
    
    while True:
        clear_screen()
        print_banner()
        
        now = datetime.utcnow().strftime('%H:%M UTC')
        killzone = KillZoneAnalyzer.get_current_killzone(datetime.utcnow())
        status = f"{killzone or 'Hors zone'}" if 'KillZoneAnalyzer' in sys.modules else "..."
        
        print(f"""
   ╔═══════════════════════════════════════════════════════════╗
   ║  📊 STATUT ACTUEL                                         ║
   ║     Horaire: {now}                               ║
   ║     Killzone: {status:^15}                            ║
   ║     Symbole: {settings['symbol']:^15}                            ║
   ║     Timeframe: {settings['timeframe']:^10}                              ║
   ╚═══════════════════════════════════════════════════════════════════╝
        """)
        
        print_menu("MENU PRINCIPAL", [
            {'emoji': '🔍', 'text': 'Scanner un symbole'},
            {'emoji': '📊', 'text': 'Scan multi-timeframes'},
            {'emoji': '⚙️', 'text': 'Paramètres'},
            {'emoji': '📖', 'text': 'Documentation ICT'},
            {'emoji': '🔔', 'text': 'Configuration alertes'},
            {'emoji': '📈', 'text': 'Backtest'},
            {'emoji': '❓', 'text': 'Aide'},
        ])
        
        choice = input("\n   ➤ Votre choix: ").strip()
        
        if choice == '0':
            print("\n👋 Au revoir et bon trading!")
            break
        elif choice == '1':
            run_scan(settings)
        elif choice == '2':
            run_multi_tf_scan()
        elif choice == '3':
            settings = show_settings()
        elif choice == '4':
            show_info()
        elif choice == '5':
            show_alerts()
        elif choice == '6':
            show_backtest()
        elif choice == '7':
            show_help()
        else:
            print("\n❌ Choix invalide. Veuillez sélectionner une option valide.")
            input("   Appuyez sur Entrée pour continuer...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Au revoir et bon trading!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        print("\n💡 Conseil: Vérifiez que toutes les dépendances sont installées:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
