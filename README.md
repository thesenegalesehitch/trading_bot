# 🚀 Quantum Trading System v3.0 (Autonomous Grade)

> **Moteur de trading autonome unifiant Intelligence Technique, Machine Learning, On-Chain et Psychologie Sociale (Twitter/X).**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Architecture](https://img.shields.io/badge/Architecture-Clean%20Institutional-orange.svg)
![Status](https://img.shields.io/badge/Status-Autonomous%20Live-brightgreen.svg)

---

## 🏛️ Vision & Architecture

Le **Quantum Trading System v3** franchit l'étape de l'autonomie. Il ne se contente plus d'analyser, il exécute sur les marchés mondiaux tout en capturant le pouls psychologique des réseaux sociaux.

- **Intelligence Totale (Alpha Engine v3)** : Intégration du sentiment social (Twitter/X) pour anticiper les mouvements de foule.
- **Exécution Native (Live Bridge)** : Connecteurs directs vers **Binance** (Crypto) et **Interactive Brokers** (Forex/Futures) pour un trading sans intermédiaire.
- **Garde-fous Institutionnels** : `ExecutionManager` couplé au `CircuitBreaker` pour une sécurité transactionnelle maximale.

---

## ⚡ Innovations Majeures (v3.0)

### 🧩 Alpha Engine v3.0
Pondération de décision mise à jour :
- **Technique (25%)** : Ichimoku, SMC, Wyckoff.
- **Machine Learning (20%)** : XGBoost / LightGBM.
- **On-Chain Intelligence (20%)** : Mempool, Whale Alerts.
- **IA Sociale (15%)** : Sentiment Twitter/X en temps réel. [NEW]
- **Statistique (10%)** : Co-intégration, Hurst.
- **Risque (10%)** : Black-Litterman Sizing.

### 🏦 Connectivité Transactionnelle
- **Binance API** : Support Spot/Testnet pour la crypto.
- **IBKR (ib_insync)** : Exécution Forex/Or via TWS/Gateway.

### 🖥️ Dashboard v3
Visualisation des flux de sentiment Twitter et monitoring des ordres réels exécutés.

---

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

## 🏗️ Structure du Projet

```text
src/quantum/
├── domain/             # Logique métier pure (Logic, Models, Rules)
│   ├── analysis/       # Ichimoku, SMC, Wyckoff, Divergences
│   ├── core/           # Hurst, Cointegration, Scorer
│   ├── ml/             # Trainer, Classifier, Features
│   ├── risk/           # Portfolio (Black-Litterman), Circuit Breaker
│   └── strategies/     # Multi-Strategy Engine
├── application/        # Cas d'utilisation & Orchestration
│   ├── backtest/       # Simulations & Monte-Carlo
│   └── reporting/      # Alertes (Telegram, Discord), Scan Coordinator
├── infrastructure/     # Détails techniques & Connecteurs
│   ├── api/            # Serveur Fast API (optionnel)
│   ├── db/             # Cache Redis, Database Migrations
│   └── ui/             # Dashboard Streamlit
└── shared/             # Utilitaires transverses
    ├── config/         # Paramètres centralisés
    ├── utils/          # Logger structuré
    └── web3/           # Intelligence On-Chain (Oracle, Mempool)
```

---

## 🛠️ Utilisation Rapide

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Installer Redis pour le caching (optionnel mais recommandé)
```

### Commandes AI-Ready
```bash
# Analyser un actif avec l'Alpha Engine complet
python main.py --mode analyze --symbol BTC-USD

# Lancer le scan multi-actifs parallèle
python main.py --mode scan

# Entraîner le modèle ML pour un symbole spécifique
python main.py --mode train --symbol EURUSD=X

# Lancer le Dashboard Streamlit
streamlit run src/quantum/infrastructure/ui/dashboard.py
```

---

## 🛡️ Gestion du Risque : Black-Litterman
Contrairement aux modèles classiques, notre optimiseur **Black-Litterman** combine l'équilibre du marché avec les "vues" propriétaires de notre Alpha Engine. 
- **Rendement attendu** = Équilibre Marché + Confiance Alpha.
- **Résultat** : Des tailles de positions plus stables et une protection contre les spikes de corrélation.

---

## 📡 Sources de Données
Le système interroge dynamiquement :
- **Yahoo Finance** : Historique large.
- **Alpha Vantage & Polygon** : Flux temps réel.
- **Web3 Engine** : Mempool Ethereum et Staking sentiment.

---

## 👨‍💻 Auteur & Licence
**Alexandre Albert Ndour** - Concevoir l'avenir du trading quantique.
MIT License - Copyright (c) 2026.

---
<p align="center">
  <i>Propulsé par la fusion de l'intelligence humaine et artificielle.</i>
</p>

<!-- Fin du README v2.0 -->

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

