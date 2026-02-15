# 🚀 Guide de Lancement Rapide

## Installation

```bash
# 1. Cloner le projet
git clone https://github.com/thesenegalesehitch/trading_bot.git
cd trading_bot

# 2. Creer un environnement virtuel (recommande)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# 3. Installer les dependances minimales
pip install yfinance pandas pytz

# Optionnel: Si vous voulez tous les indicateurs
pip install pandas-ta
```

---

## Lancer le Projet

### Option 1: Scanner Live (Analyse en temps reel)

```bash
# Analyser un symbole specifique
python live_scanner.py --symbol EURUSD=X

# Avec un autre symbole
python live_scanner.py --symbol BTC-USD
python live_scanner.py --symbol GOLD

# Avec un autre timeframe
python live_scanner.py --symbol EURUSD=X --timeframe 15m
python live_scanner.py --symbol ETH-USD --timeframe 5m

# Mode continu (mise a jour chaque minute)
python live_scanner.py --watch --symbol EURUSD=X
```

### Option 2: Menu Interactif ICT

```bash
python run_ict_menu.py
```

Ce menu vous guide pas a pas pour apprendre les concepts ICT.

### Option 3: Scanner ICT

```bash
python run_ict_scanner.py
```

Analyse automatique des graphiques avec detection ICT.

### Option 4: Interface Streamlit

```bash
# Installer streamlit si necessaire
pip install streamlit

# Lancer l'interface graphique
streamlit run src/quantum/application/ui/streamlit_app.py
```

---

## Symboles Disponibles

| Type | Symboles |
|------|----------|
| Forex | EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X |
| Crypto | BTC-USD, ETH-USD, SOL-USD |
| Metaux | GOLD (XAU/USD), SILVER |
| Indices | ^GSPC (S&P500), ^IXIC (NASDAQ) |

---

## Timeframes

- `1m` - 1 minute
- `5m` - 5 minutes
- `15m` - 15 minutes
- `1h` - 1 heure (defaut)
- `4h` - 4 heures
- `1d` - 1 jour

---

## Exemples Pratiques

```bash
# Analyse rapide EURUSD en 1H
python live_scanner.py -s EURUSD=X -t 1h

# Analyse crypto en 15 minutes
python live_scanner.py -s BTC-USD -t 15m

# Suivre l'or toute la journee
python live_scanner.py -s GOLD -t 15m --watch
```

---

## Depannage

### Erreur "No module named 'yfinance'"
```bash
pip install yfinance
```

### Erreur "Permission denied" sur Windows
```bash
# Utiliser PowerShell
powershell -Command "& {./venv/Scripts/Activate.ps1}"
```

### Probleme avec pandas-ta
```bash
# Si pandas-ta ne sinstalle pas, installez juste les basics
pip install yfinance pandas pytz
# Le scanner fonctionne sans pandas-ta
```

---

*Projet gratuit pour apprendre le trading ICT/SMC*
