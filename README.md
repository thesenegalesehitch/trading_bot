# ICT Trading Education Tool 🎓

> Outil gratuit pour apprendre les concepts de trading ICT, SMC, Wyckoff, Ichimoku et autres stratégies.

**Pour ceux qui n'ont pas les moyens de payer des formations chères.**

---

## C'est quoi ce projet ?

Un outil **gratuit et open-source** qui te permet de:
- 📊 **Visualiser** les concepts ICT en temps réel sur les graphiques
- 📚 **Apprendre** avec des explications claires de chaque concept
- 🧪 **Pratiquer** dans un mode sandbox sans risque
- 🔍 **Analyser** tes trades passés pour comprendre tes erreurs

---

## Concepts enseignés

| Concept | Description |
|---------|-------------|
| **ICT** | Inner Circle Trader - Order Blocks, FVG, MSS |
| **SMC** | Smart Money Concepts - Smart Money vs Dumb Money |
| **Wyckoff** | Phases d'accumulation et distribution |
| **Ichimoku** | Nuage de tendance japonais |
| **Divergences** | RSI, MACD, OBV |

---

## Installation

```bash
# Clone le projet
git clone https://github.com/thesenegalesehitch/trading_bot.git
cd trading_bot

# Crée un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installe les dépendances
pip install -r requirements.txt
```

---

## Utilisation rapide

### Mode Menu Interactif (Pour débutants)

```bash
python run_ict_menu.py
```

Ce menu interactif te guide pas à pas:
1. Choisis un symbole (EURUSD, BTC, etc.)
2. Choisis un timeframe
3. Apprends chaque concept ICT étape par étape

### Mode Scanner

```bash
python run_ict_scanner.py
```

Analyse automatiquement les graphiques et affiche:
- Order Blocks détectés
- Fair Value Gaps (FVG)
- Market Structure Shifts (MSS)
- Signaux de tendance

### Interface Streamlit

```bash
streamlit run src/quantum/application/ui/streamlit_app.py
```

Interface graphique pour:
- Visualisation interactive
- Backtest de stratégies
- Analyse de trades

---

## Pour qui ?

- ✅ **Débutants** qui veulent apprendre le trading
- ✅ **Traders autodidactes** sans budget pour les formations
- ✅ **Ceux qui veulent comprendre** les concepts ICT/SMC
- ❌ **Ceux qui cherchent un bot de trading rentable** — Ce n'est PAS un robot qui gagne de l'argent

---

## Gratuit et Open Source

Ce projet est **100% gratuit**. Pourquoi ?
- Le trading est déjà assez difficile financièrement
- Les formations coûtent souvent $500-$5000
- Tout le monde devrait avoir accès à l'éducation

**Contribue** en partageant, en Forkant, en améliorant !

---

## Avertissement

⚠️ **Ceci est un outil pédagogique, pas un conseil financier.**
- Le trading comporte des risques importants
- Ne trade jamais avec de l'argent que tu ne peux pas perdre
- backtest ≠ résultats futurs

---

## Prochaines étapes

1. Lance `python run_ict_menu.py` pour commencer
2. Lis les docs dans `docs/`
3. Explore les indicateurs dans `src/quantum/domain/analysis/`

---

**Fait avec ❤️ pour l'éducation trading accessible à tous.**
