# Guide d'Utilisation - ICT Full Setup Detector

## Table des Matières

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Utilisation de Base](#utilisation-de-base)
5. [Concepts ICT](#concepts-ict)
6. [Filtres et Confluences](#filtres-et-confluences)
7. [Alertes et Notifications](#alertes-et-notifications)
8. [Exemples d'Utilisation](#exemples-dutilisation)
9. [Dépannage](#dépannage)
10. [Référence API](#référence-api)

---

## Introduction

Le module **ICT Full Setup Detector** est un système de détection automatique de trades basé sur la méthodologie **ICT (Inner Circle Trader)** et **SMC (Smart Money Concepts)**.

### Qu'est-ce qu'un Full Setup ICT ?

Un "Full Setup" est une configuration de trade complète basée sur la séquence :

```
Sweep → FVG Tap → MSS → IFVG Entry
```

Cette séquence représente le comportement institutionnel du marché où les "smart money" prennent des positions en suivant des patterns spécifiques.

---

## Installation

### Prérequis

```bash
# Cloner le projet
git clone https://github.com/thesenegalesehitch/quantum_trading_system.git
cd quantum_trading_system

# Installer les dépendances
pip install -r requirements.txt
```

### Vérification de l'Installation

```python
from quantum.domain.analysis.ict_full_setup import ICTFullSetupDetector

# Créer une instance
detector = ICTFullSetupDetector()

print("✅ Module ICT loaded successfully!")
```

---

## Configuration

### Variables d'Environnement

Créez un fichier `.env` à la racine du projet :

```env
# Trading Configuration
SYMBOLS=BTCUSDT,ETHUSDT,EURUSD
TIMEFRAMES=15m,1h,4h

# ICT Settings
MIN_RR=2.0
VOLUME_SPIKE_MULTIPLIER=1.5
SESSION_HOURS=24

# Killzones
KILLZONE_LONDON_START=8
KILLZONE_LONDON_END=11
KILLZONE_NY_START=13
KILLZONE_NY_END=16

# Notifications
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Configuration Programmatique

```python
from quantum.domain.analysis.ict_full_setup import ICTFullSetupDetector

# Configuration par défaut
detector = ICTFullSetupDetector(
    session_hours=24,           # Heures pour la session de liquidité
    min_rr=2.0,                # Ratio risque/récompense minimum
    volume_spike_multiplier=1.5 # Seuil de volume spike (150%)
)
```

---

## Utilisation de Base

### 1. Détection Simple

```python
import pandas as pd
from quantum.domain.analysis.ict_full_setup import detect_ict_full_setup

# Charger vos données OHLCV
df = pd.read_csv('btcusdt_15m.csv', index_col='timestamp')

# Détecter les setups
trades = detect_ict_full_setup(
    df=df,
    symbol='BTCUSDT',
    timeframe='15m',
    min_rr=2.0
)

# Afficher les résultats
for trade in trades:
    print(f"Direction: {trade['direction']}")
    print(f"Entry: {trade['entry']}")
    print(f"Stop Loss: {trade['stop_loss']}")
    print(f"Risk/Reward: 1:{trade['risk_reward']}")
```

### 2. Utilisation Avancée

```python
from quantum.domain.analysis.ict_full_setup import (
    ICTFullSetupDetector,
    KillZoneAnalyzer,
    VolumeSpikeDetector
)

# Créer le detector avec configuration personnalisée
detector = ICTFullSetupDetector(
    session_hours=24,
    min_rr=2.0,
    volume_spike_multiplier=1.5
)

# Analyser un symbole
trades = detector.detect_full_setup(
    df=df,
    symbol='EURUSD',
    timeframe='15m',
    df_htf=df_h4  # Optionnel: données timeframe supérieur
)

# Scanner plusieurs timeframes
results = detector.scan_symbol(
    df=df,
    symbol='BTCUSDT',
    timeframes=['15m', '1h', '4h']
)
```

---

## Concepts ICT

### 1. Contextual Sweep (Prise de Liquidité)

Le Sweep détecte quand le prix "nettoie" la liquidité aux points clés :

- **PDH/PDL**: Previous Day High/Low
- **HOD/LOD**: High/Low de la session en cours

```python
from quantum.domain.analysis.ict_full_setup import LiquidityDetector

detector = LiquidityDetector(session_hours=24)
pdh, pdl, hod, lod, _ = detector.get_session_levels(df)

sweeps = detector.detect_sweeps(df, pdh, pdl, hod, lod)

for sweep in sweeps:
    print(f"Type: {sweep.direction}")
    print(f"Level Swept: {sweep.liquidity_level.type}")
```

### 2. FVG Tap (Touche du FVG HTF)

Après le sweep, le prix doit toucher un Fair Value Gap du timeframe supérieur :

```python
from quantum.domain.analysis.ict_full_setup import FVGTapDetector

fvg_detector = FVGTapDetector(smc_analyzer)
taps = fvg_detector.detect_htf_fvg_taps(df_ltf, df_htf, sweep_event)

for tap in taps:
    print(f"FVG Type: {tap.fvg.type}")
    print(f"HTF: {tap.htf_timeframe}")
```

### 3. MSS (Market Structure Shift)

Le MSS valide la cassure de structure avec une bougie impulsive :

```python
from quantum.domain.analysis.ict_full_setup import MSSDetector

mss_detector = MSSDetector()
mss = mss_detector.detect_mss(df, direction, sweep_event)

if mss:
    print(f"MSS Direction: {mss.direction}")
    print(f"Impulsive Candle: {mss.impulsive_candle_size:.1%}")
```

### 4. IFVG Entry (Inverted FVG)

L'IFVG est la zone d'entrée précise :

```python
from quantum.domain.analysis.ict_full_setup import IFVGDetector

ifvg_detector = IFVGDetector()
ifvg = ifvg_detector.detect_ifvg_entry(df, direction, mss_event, min_rr=2.0)

if ifvg:
    print(f"Entry: {ifvg.entry_price}")
    print(f"Stop Loss: {ifvg.stop_loss}")
    print(f"RR: 1:{ifvg.risk_reward}")
```

---

## Filtres et Confluences

### 1. Killzones

Les signaux ne sont validés que pendant les heures de forte liquidité :

| Killzone | Horaire UTC | Activité |
|-----------|-------------|----------|
| **Londres** | 08:00 - 11:00 | Ouverture européenne |
| **New York** | 13:00 - 16:00 | Ouverture US |

```python
from quantum.domain.analysis.ict_full_setup import KillZoneAnalyzer
from datetime import datetime

# Vérifier la killzone actuelle
now = datetime.utcnow()
killzone = KillZoneAnalyzer.get_current_killzone(now)

if killzone:
    print(f"🟢 Killzone active: {killzone}")
else:
    print("🔴 Hors killzone - pas de signaux")
```

### 2. Volume Spike

La bougie de signal doit avoir un volume significatif :

```python
from quantum.domain.analysis.ict_full_setup import VolumeSpikeDetector

volume_detector = VolumeSpikeDetector(
    lookback=10,           # Nombre de bougies pour la moyenne
    spike_multiplier=1.5   # Seuil (150%)
)

is_spike, ratio = volume_detector.is_volume_spike(df)

print(f"Volume Spike: {is_spike}")
print(f"Ratio: {ratio:.2f}x la moyenne")
```

### 3. Risk/Reward

Seuls les trades avec RR ≥ 2.0 sont proposés :

```python
detector = ICTFullSetupDetector(min_rr=2.0)

# RR minimum = 1:2
# RR excellent = 1:3+
```

---

## Alertes et Notifications

### Configuration Discord

```python
from quantum.application.reporting.alerts import AlertManager

manager = AlertManager()

# Envoyer un signal ICT
manager.send_ict_full_setup_signal(trade_data)
```

### Format d'Alerte Discord

```json
{
  "title": "🟢 ICT Full Setup: BTCUSDT | BUY",
  "color": 0x2ECC71,
  "fields": [
    {"name": "🎯 Entry", "value": "50000.00", "inline": true},
    {"name": "🛑 Stop Loss", "value": "49500.00", "inline": true},
    {"name": "📈 Risk/Reward", "value": "1:2.5", "inline": true},
    {"name": "📊 Confluence", "value": "Killzone: LONDON\nVolume Spike: ✅", "inline": false}
  ]
}
```

### Format Telegram

```
🟢 *ICT FULL SETUP DETECTED*

📈 *Symbol:* BTCUSDT
🎯 *Direction:* BUY

━━━━━━━━━━━━━━━━━━━━
📊 *Trade Levels*
━━━━━━━━━━━━━━━━━━━━
• Entry: `50000.00`
• Stop Loss: `49500.00`
• TP1: `51000.00`
• TP2: `52000.00`
• TP3: `53000.00`

📈 *Risk/Reward:* `1:2.5`

⏰ *Detected:* 10:30:00 UTC
```

---

## Exemples d'Utilisation

### Exemple 1: Scan Complet

```python
import pandas as pd
from quantum.domain.analysis.ict_full_setup import ICTFullSetupDetector
from quantum.application.reporting.alerts import AlertManager

# Configuration
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'EURUSD']
TIMEFRAMES = ['15m', '1h']

detector = ICTFullSetupDetector(min_rr=2.0)
alert_manager = AlertManager()

# Scanner chaque symbole
for symbol in SYMBOLS:
    for tf in TIMEFRAMES:
        # Charger les données
        df = load_data(symbol, tf)
        
        # Détecter les setups
        trades = detector.detect_full_setup(df, symbol, tf)
        
        # Envoyer les alertes
        for trade in trades:
            alert_manager.send_ict_full_setup_signal(trade.to_dict())
            print(f"🎯 Signal {symbol} {tf}: {trade.direction}")
```

### Exemple 2: Analyse en Temps Réel

```python
from quantum.domain.analysis.ict_full_setup import (
    ICTFullSetupDetector,
    KillZoneAnalyzer,
    VolumeSpikeDetector
)
from datetime import datetime

class ICTRealTimeScanner:
    def __init__(self):
        self.detector = ICTFullSetupDetector()
        self.volume_detector = VolumeSpikeDetector()
    
    def analyze_tick(self, df, symbol):
        now = datetime.utcnow()
        
        # Vérifier killzone
        killzone = KillZoneAnalyzer.get_current_killzone(now)
        if not killzone:
            return None
        
        # Vérifier volume spike
        is_spike, ratio = self.volume_detector.is_volume_spike(df)
        
        # Détecter les setups
        trades = self.detector.detect_full_setup(df, symbol, '15m')
        
        if trades:
            return {
                'killzone': killzone,
                'volume_spike': is_spike,
                'volume_ratio': ratio,
                'trades': trades
            }
        
        return None
```

### Exemple 3: Backtesting

```python
import pandas as pd
from quantum.domain.analysis.ict_full_setup import detect_ict_full_setup

def backtest_ict_strategy(df, symbol):
    """Backtest de la stratégie ICT Full Setup."""
    
    # Paramètres
    min_rr = 2.0
    win_count = 0
    total_trades = 0
    
    # Simuler les trades
    for i in range(100, len(df)):
        # Utiliser les données jusqu'à maintenant
        test_df = df.iloc[:i]
        
        trades = detect_ict_full_setup(
            test_df, 
            symbol, 
            '15m', 
            min_rr=min_rr
        )
        
        if trades:
            total_trades += 1
            # Logique de simulation de trade...
    
    # Résultats
    if total_trades > 0:
        win_rate = win_count / total_trades * 100
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Trades: {total_trades}")
```

---

## Dépannage

### Problèmes Courants

#### 1. Pas de signaux détectés

**Cause possible**: Hors killzone

```python
from quantum.domain.analysis.ict_full_setup import KillZoneAnalyzer

# Vérifier l'heure actuelle
now = datetime.utcnow()
print(f"Heure UTC: {now.hour}:{now.minute}")
print(f"Killzone: {KillZoneAnalyzer.get_current_killzone(now)}")
```

**Solution**: Attendre les horaires de killzone (8-11h ou 13-16h UTC)

#### 2. Volume toujours normal

**Cause possible**: Données de volume incorrectes

```python
# Vérifier les données de volume
print(df['Volume'].describe())
print(f"Volume moyen: {df['Volume'].mean()}")
print(f"Volume dernière bougie: {df['Volume'].iloc[-1]}")
```

#### 3. Erreur de configuration

```python
# Vérifier la configuration
from quantum.shared.config.settings import config

print(f"MIN_RR: {config.technical.MIN_RR}")
print(f"KILLZONE_LONDON: {config.timeframes.KILLZONE_LONDON}")
```

### Logs de Débogage

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('quantum.analysis.ict')

# Activer les logs détaillés
logger.setLevel(logging.DEBUG)
```

---

## Référence API

### ICTFullSetupDetector

```python
class ICTFullSetupDetector:
    def __init__(self, session_hours=24, min_rr=2.0, volume_spike_multiplier=1.5)
    
    def detect_full_setup(self, df, symbol, timeframe='15m', df_htf=None) -> List[FullSetupTrade]
    
    def scan_symbol(self, df, symbol, timeframes=['15m', '1h']) -> Dict[str, List[FullSetupTrade]]
```

### KillZoneAnalyzer

```python
class KillZoneAnalyzer:
    @staticmethod
    def get_current_killzone(dt) -> Optional[str]
    
    @staticmethod
    def is_in_killzone(dt) -> bool
    
    @staticmethod
    def get_killzone_color(zone) -> int
```

### VolumeSpikeDetector

```python
class VolumeSpikeDetector:
    def __init__(self, lookback=10, spike_multiplier=1.5)
    
    def calculate_avg_volume(self, df) -> float
    
    def is_volume_spike(self, df, candle_index=-1) -> Tuple[bool, float]
    
    def get_volume_score(self, df) -> float
```

### FullSetupTrade

```python
@dataclass
class FullSetupTrade:
    setup_id: str
    symbol: str
    direction: str  # 'BUY' ou 'SELL'
    sweep: SweepEvent
    fvg_tap: FVGTap
    mss: MSSEvent
    ifvg_entry: IFVGEntry
    killzone: str
    volume_spike_confirmed: bool
    confluence_score: float
    detected_at: datetime
    timeframe: str
    confidence: float
    
    def to_dict(self) -> Dict
```

---

## Bonnes Pratiques

### 1. Gestion du Risque

```python
# Toujours utiliser le stop loss
for trade in trades:
    print(f"Stop Loss: {trade['stop_loss']}")
    
    # Calculer la taille de position
    risk_per_trade = 100  # $ par trade
    account_balance = 10000
    risk_percent = risk_per_trade / account_balance
    
    # Ne jamais risquer plus de 1-2% par trade
    assert risk_percent <= 0.02
```

### 2. Multi-Timeframe

```python
# Confirmer sur plusieurs timeframes
results = detector.scan_symbol(df, symbol, ['15m', '1h', '4h'])

# Vérifier la convergence
for tf, trades in results.items():
    if trades:
        print(f"{tf}: {len(trades)} signaux")
```

### 3. Journalisation

```python
import logging

logger = logging.getLogger('ict_strategy')

for trade in trades:
    logger.info(f"Signal: {trade.direction} {trade.symbol}")
    logger.info(f"Entry: {trade.ifvg_entry.entry_price}")
    logger.info(f"RR: 1:{trade.ifvg_entry.risk_reward}")
```

---

## Support

- **Documentation**: [README.md](../README.md)
- **Issues**: GitHub Issues
- **Discord**: Communauté Quantum Trading

---

*Mis à jour: Février 2025*
*Version: 1.0.0*
