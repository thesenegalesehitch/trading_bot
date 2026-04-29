# 🌌 Quantum Trading System (Evolution v2.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)

**Quantum** est une plateforme de trading institutionnelle 100% gratuite et open-source. Elle permet aux traders particuliers d'utiliser les mêmes concepts algorithmiques que les banques (SMC/ICT) tout en offrant un parcours éducatif complet.

![Quantum Dashboard](https://images.unsplash.com/photo-1611974717482-58100010887e?auto=format&fit=crop&q=80&w=1200)

## 🚀 Fonctionnalités Clés

*   **⚡ Scanner ICT/SMC Ultra-Rapide** : Détection automatique des Fair Value Gaps (FVG), Order Blocks (OB) et Market Structure Shifts (MSS).
*   **🎓 Académie Quantum** : Un cursus complet sur la psychologie des marchés et la gestion des risques avec quizz interactifs.
*   **📊 Analyse Institutionnelle** : Graphiques professionnels via `lightweight-charts` avec superposition des zones algorithmiques.
*   **🛡️ Gestion des Risques** : Calculateur de position, Kelly Criterion et Value at Risk (VaR).
*   **📓 Journal de Trading & Backtesting** : Validez vos stratégies et traquez votre discipline émotionnelle.

## 🛠️ Installation Locale (Local-First)

### Pré-requis
- Python 3.13+
- Node.js 20+
- Docker (Optionnel)

### Installation Rapide

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/votre-username/quantum-trading-system.git
   cd quantum-trading-system
   ```

2. **Configuration Backend** :
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   # Modifiez le .env avec vos clés si nécessaire
   ```

3. **Initialisation de la DB** :
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/src
   python scripts/init_db.py
   python scripts/academy_init.py
   ```

4. **Lancement Frontend** :
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## 🐋 Docker (Recommandé pour Production Locale)

```bash
docker-compose up --build
```

## 🤝 Contribuer

Le projet est **Open-Source**. Nous accueillons toutes les contributions :
- Amélioration des algorithmes de détection (SMC/ICT).
- Ajout de nouveaux cours dans l'Académie.
- Optimisation des performances du scanner.

## 📄 Licence

Ce projet est sous licence **MIT**. Vous êtes libre de l'utiliser, de le modifier et de le distribuer gratuitement.

---
*Créé avec ❤️ pour la communauté des traders quantitatifs.*
