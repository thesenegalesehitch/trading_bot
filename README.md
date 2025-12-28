# 🚀 Quantum Trading System

> **Système de trading quantitatif haute précision pour EUR/USD, XAU/USD et crypto**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

## 📋 Table des Matières

- [Présentation](#-présentation)
- [Fonctionnalités](#-fonctionnalités)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Utilisation](#-utilisation)
- [Architecture](#-architecture)
- [Sources de Données](#-sources-de-données)
- [Indicateurs et Analyses](#-indicateurs-et-analyses)
- [Machine Learning](#-machine-learning)
- [Gestion du Risque](#-gestion-du-risque)
- [Alertes et Notifications](#-alertes-et-notifications)
- [FAQ](#-faq)

---

## 🎯 Présentation

Le **Quantum Trading System** est un système de trading algorithmique complet qui combine:

- 📊 **Analyse statistique avancée** (Co-intégration, Hurst, Z-Score)
- 📈 **Analyse technique multi-timeframe** (Ichimoku, SMC, Wyckoff)
- 🤖 **Machine Learning** (Ensemble XGBoost + LightGBM + CatBoost)
- 🛡️ **Gestion du risque robuste** (VaR, Kelly Criterion, Portfolio)
- 🔔 **Alertes multi-canal** (Telegram, Discord, Email)

### Points forts

✅ **7+ sources de données gratuites** avec fallback automatique  
✅ **Ensemble de modèles ML** avec calibration des probabilités  
✅ **Backtesting Monte Carlo** avec 10,000+ simulations  
✅ **Kelly Criterion dynamique** ajusté au drawdown  
✅ **Détection automatique des divergences** RSI/MACD  
✅ **Analyse Wyckoff** (accumulation/distribution)  

---

## ⚡ Fonctionnalités

### Sources de Données
| Source | Type | Limite Gratuite |
|--------|------|-----------------|
| Yahoo Finance | Forex, Actions | Illimité |
| Alpha Vantage | Forex, Crypto | 25/jour |
| Polygon.io | Tous | 5/min |
| Finnhub | Forex, Actions | 60/min |
| FRED | Économique | 120/min |
| Binance | Crypto | Illimité |
| CCXT | 100+ exchanges | Variable |

### Analyses Techniques
- 📊 **Multi-Timeframe**: 15m, 1h, 4h, 1d avec convergence
- ☁️ **Ichimoku Kumo**: Filtre de tendance
- 💰 **Smart Money Concepts**: Order Blocks, FVG
- 📉 **Wyckoff**: Phases d'accumulation/distribution
- ↔️ **Divergences**: RSI, MACD, OBV automatiques
- 🌊 **Elliott Wave**: Vagues impulsives et correctives

### Machine Learning
- 🌲 **Ensemble de modèles**: XGBoost + LightGBM + CatBoost + RF
- 🎯 **Optimisation bayésienne**: via Optuna
- 📈 **Walk-Forward**: Validation robuste
- 🔧 **SHAP**: Feature importance

### Gestion du Risque
- 📉 **Value at Risk**: Historique, Paramétrique, Monte Carlo
- 📊 **Kelly Criterion**: Position sizing optimal
- 🔄 **Portfolio**: Max Sharpe, Min Variance, Risk Parity
- 🚨 **Circuit Breaker**: Arrêt automatique

---

## 🛠️ Installation

### Prérequis
- Python 3.9 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation rapide

```bash
# 1. Cloner le repository
git clone https://github.com/votre-repo/quantum_trading_system.git
cd quantum_trading_system

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. (Optionnel) Installer les dépendances avancées
pip install lightgbm catboost optuna ta-lib
```

### Installation des dépendances optionnelles

```bash
# Pour le deep learning (LSTM)
pip install tensorflow

# Pour les alertes
pip install python-telegram-bot discord-webhook

# Pour les visualisations avancées
pip install plotly dash
```

---

## ⚙️ Configuration

### 1. Variables d'environnement

Créez un fichier `.env` à la racine du projet:

```env
# === Sources de Données ===
ALPHA_VANTAGE_API_KEY=votre_clé_ici
POLYGON_API_KEY=votre_clé_ici
FINNHUB_API_KEY=votre_clé_ici
FRED_API_KEY=votre_clé_ici
NEWSAPI_KEY=votre_clé_ici

# === Alertes Telegram ===
TELEGRAM_BOT_TOKEN=votre_token_bot
TELEGRAM_CHAT_ID=votre_chat_id

# === Alertes Discord ===
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# === Alertes Email ===
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=votre_email@gmail.com
SMTP_PASSWORD=votre_mot_de_passe_app
EMAIL_FROM=votre_email@gmail.com
EMAIL_TO=destinataire@email.com
```

### 2. Configuration du système

Modifiez `config/settings.py` selon vos besoins:

```python
# Symboles à trader
ACTIVE_SYMBOLS = ["EURUSD=X", "GC=F", "BTCUSDT"]

# Risque par trade
RISK_PER_TRADE = 0.01  # 1%

# Seuil ML minimum
MIN_PROBABILITY_THRESHOLD = 0.85  # 85%
```

---

## 🚀 Utilisation

### Commandes principales

```bash
# Analyser un symbole
python main.py --mode analyze --symbol EURUSD=X

# Générer un signal de trading
python main.py --mode signal --symbol EURUSD=X

# Exécuter un backtest
python main.py --mode backtest --symbol EURUSD=X

# Entraîner le modèle ML
python main.py --mode train --symbol EURUSD=X

# Analyser la corrélation EUR/USD vs Gold
python main.py --mode correlation
```

### Forcer le téléchargement des données

```bash
python main.py --mode analyze --symbol EURUSD=X --download
```

### Exemples de code

#### Analyse d'un symbole

```python
from main import QuantumTradingSystem

# Initialiser le système
system = QuantumTradingSystem()

# Charger les données
system.load_data("EURUSD=X")

# Analyser
analysis = system.analyze_symbol("EURUSD=X")
print(f"Signal: {analysis['combined_signal']}")
print(f"Confiance: {analysis['confidence']}%")
```

#### Utiliser le ML Ensemble

```python
from ml.ensemble import EnsembleClassifier, EnsembleConfig
import pandas as pd

# Configurer l'ensemble
config = EnsembleConfig(
    use_xgboost=True,
    use_lightgbm=True,
    use_catboost=True,
    calibrate_probabilities=True
)

# Créer et entraîner
ensemble = EnsembleClassifier(config)
metrics = ensemble.train(X_train, y_train)

# Prédire
signal = ensemble.predict_signal(X_new)
print(f"Signal: {signal['signal']}")
print(f"Probabilité: {signal['probability']}%")
```

#### Calculer le Value at Risk

```python
from risk.var_calculator import VaRCalculator

# Calculer le VaR
var_calc = VaRCalculator(confidence_level=0.95, horizon_days=1)
result = var_calc.calculate_monte_carlo_var(returns, portfolio_value=10000)

print(f"VaR 95% 1 jour: ${result.var_value}")
print(f"CVaR (Expected Shortfall): ${result.cvar}")
```

#### Envoyer des alertes

```python
from reporting.alerts import AlertManager, AlertLevel

# Initialiser
manager = AlertManager()

# Envoyer un signal
manager.send_signal(
    symbol="EURUSD=X",
    signal="BUY",
    price=1.0850,
    confidence=87.5,
    stop_loss=1.0820,
    take_profit=1.0920
)
```

---

## 🏗️ Architecture

```
quantum_trading_system/
│
├── main.py                 # Point d'entrée principal
├── requirements.txt        # Dépendances Python
├── README.md              # Ce fichier
│
├── config/
│   └── settings.py        # Configuration centralisée
│
├── data/
│   ├── downloader.py      # Téléchargement des données
│   ├── data_sources.py    # Sources multiples avec fallback
│   ├── sentiment.py       # Analyse de sentiment
│   ├── kalman_filter.py   # Lissage des prix
│   └── feature_engine.py  # Création des features
│
├── core/
│   ├── cointegration.py   # Analyse de co-intégration
│   ├── hurst.py           # Exposant de Hurst
│   └── zscore.py          # Z-Score de Bollinger
│
├── analysis/
│   ├── ichimoku.py        # Analyse Ichimoku
│   ├── smc.py             # Smart Money Concepts
│   ├── wyckoff.py         # Analyse Wyckoff
│   ├── divergences.py     # Détection des divergences
│   └── multi_tf.py        # Multi-timeframe
│
├── ml/
│   ├── model.py           # Classificateur signal
│   ├── ensemble.py        # Ensemble de modèles
│   ├── optimizer.py       # Optimisation bayésienne
│   ├── features.py        # Préparation ML
│   └── trainer.py         # Entraînement avec CV
│
├── risk/
│   ├── manager.py         # Gestionnaire de risque
│   ├── var_calculator.py  # Value at Risk
│   ├── portfolio.py       # Gestion portefeuille
│   ├── circuit_breaker.py # Arrêt d'urgence
│   └── calendar.py        # Calendrier économique
│
├── backtest/
│   ├── engine.py          # Moteur de backtest
│   └── monte_carlo.py     # Simulation Monte Carlo
│
└── reporting/
    ├── interface.py       # Affichage console
    └── alerts.py          # Alertes multi-canal
```

---

## 📡 Sources de Données

### Obtenir les clés API gratuites

#### Alpha Vantage
1. Aller sur https://www.alphavantage.co/support/#api-key
2. S'inscrire avec email
3. Recevoir la clé immédiatement

#### Polygon.io
1. Aller sur https://polygon.io/
2. Créer un compte gratuit
3. Copier la clé API depuis le dashboard

#### Finnhub
1. Aller sur https://finnhub.io/
2. S'inscrire gratuitement
3. Obtenir la clé dans Settings

#### FRED
1. Aller sur https://fred.stlouisfed.org/docs/api/api_key.html
2. Créer un compte
3. Demander une clé API

#### NewsAPI
1. Aller sur https://newsapi.org/
2. S'inscrire gratuitement
3. Obtenir la clé (100 requêtes/jour)

---

## 📊 Indicateurs et Analyses

### Analyse Statistique

| Indicateur | Description | Usage |
|------------|-------------|-------|
| **Co-intégration** | Relation long-terme entre actifs | Arbitrage |
| **Hurst Exponent** | Persistance de tendance | Régime |
| **Z-Score** | Distance à la moyenne | Mean-reversion |

### Analyse Technique

| Indicateur | Description | Signal |
|------------|-------------|--------|
| **Ichimoku** | Nuage de tendance | Filtre direction |
| **SMC** | Order Blocks, FVG | Zones institutionnelles |
| **Wyckoff** | Accumulation/Distribution | Phase de marché |
| **Divergences** | RSI/MACD vs Prix | Retournement |

---

## 🤖 Machine Learning

### Modèles utilisés

1. **XGBoost**: Gradient boosting optimisé
2. **LightGBM**: Boosting plus rapide
3. **CatBoost**: Gestion des catégorielles
4. **Random Forest**: Ensemble d'arbres

### Features utilisées

- Indicateurs techniques (RSI, MACD, ATR...)
- Features temporelles cycliques (heure, jour, mois)
- Z-Score et Hurst
- Position Ichimoku
- Multi-timeframe score

### Validation

- **Walk-Forward Optimization**: Évite le surapprentissage
- **Purged K-Fold CV**: Respecte l'ordre temporel
- **Monte Carlo**: 10,000+ simulations

---

## 🛡️ Gestion du Risque

### Position Sizing

```
Taille = (Capital × Risk%) / |Entry - StopLoss|
```

Avec le **Kelly Criterion dynamique**:
- Full Kelly basé sur win rate et R:R
- Fractional Kelly (demi-Kelly) pour sécurité
- Ajustement automatique selon le drawdown

### Value at Risk (VaR)

3 méthodes disponibles:
1. **Historique**: Distribution empirique
2. **Paramétrique**: Assume normalité
3. **Monte Carlo**: 10,000 simulations

### Circuit Breaker

Arrêt automatique si:
- Drawdown > 5%
- 3 pertes consécutives
- Perte journalière > 2%

---

## 🔔 Alertes et Notifications

### Telegram

1. Créer un bot via @BotFather
2. Envoyer `/newbot` et suivre les instructions
3. Copier le token
4. Envoyer un message au bot
5. Obtenir le chat_id via `https://api.telegram.org/bot<TOKEN>/getUpdates`

### Discord

1. Aller dans les paramètres du serveur
2. Intégrations → Webhooks → Nouveau Webhook
3. Copier l'URL du Webhook

### Email (Gmail)

1. Activer l'A2F sur Google
2. Créer un mot de passe d'application
3. Utiliser ce mot de passe dans `SMTP_PASSWORD`

---

## ❓ FAQ

### Le système peut-il trader automatiquement ?

Non, ce système génère des **signaux** et des **analyses**. Il ne passe pas d'ordres automatiquement. C'est à vous de décider d'exécuter les trades.

### Quelle est la différence avec un bot de trading ?

Un bot exécute automatiquement. Ce système est un **assistant d'analyse** qui vous aide à prendre de meilleures décisions.

### Les API gratuites sont-elles suffisantes ?

Oui, pour un usage personnel. Le système utilise le caching et le rate limiting intelligent pour rester dans les limites gratuites.

### Comment améliorer la précision ?

1. Entraîner le ML sur plus de données
2. Ajuster les hyperparamètres via `optimizer.py`
3. Combiner plusieurs signaux
4. Filtrer par conditions de marché

### Le système fonctionne-t-il sur Windows ?

Oui, le système est compatible Windows, macOS et Linux.

---

## 📜 Licence

MIT License - Copyright (c) 2026 Alexandre Albert Ndour

Voir le fichier `LICENSE` pour plus de détails.

---

## ⚠️ Avertissement

Ce logiciel est fourni à titre **éducatif uniquement**. Le trading comporte des risques importants de perte. Les performances passées ne garantissent pas les résultats futurs.

**N'investissez jamais plus que ce que vous pouvez vous permettre de perdre.**

---

## 👨‍💻 Auteur

**Alexandre Albert Ndour**

- Conception et développement complet du système
- Architecture logicielle et algorithmes
- Documentation et tests

---

## 📧 Support

Pour toute question ou suggestion, ouvrez une issue sur le repository GitHub.

---

<p align="center">
  <i>Conçu et développé avec ❤️ par <b>Alexandre Albert Ndour</b></i><br>
  <i>Copyright © 2026 Alexandre Albert Ndour. All Rights Reserved.</i>
</p>

<!-- Signature: QUFOLVFUUy0yMDI0 - Alexandre Albert Ndour - Quantum Trading System -->

