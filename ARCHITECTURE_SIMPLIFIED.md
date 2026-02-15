# Architecture Simplifiée — Recommandations

> Ce document décrit l'architecture cible pour simplifier le projet.
> Les modules marqués comme "DEPRECATED" peuvent être supprimés progressivement.

---

## Structure Actuelle vs Cible

### Structure Actuelle (50+ modules)

```
src/quantum/
├── domain/
│   ├── analysis/       # 6 analyseurs (ICT, SMC, Wyckoff, Ichimoku, Divergences, Intermarket)
│   ├── core/           # 5 modules (Hurst, Cointegration, Z-Score, Regime, Scorer)
│   ├── ml/             # 5 modules (service, trainer, optimizer, ensemble, features)
│   ├── risk/           # 6 modules (manager, portfolio, circuit_breaker, var, calendar)
│   ├── strategies/     # 1 module (multi_strategy)
│   ├── coach/          # 3 modules (explainer, history, validator)
│   ├── innovations/    # 4 modules (confusion_resolver, mistake_predictor, etc.)
│   └── data/           # 6 modules (downloader, realtime, sentiment, etc.)
├── application/
│   ├── backtest/       # 5 modules (engine, monte_carlo, paper_trading, trading_costs)
│   ├── reporting/      # 4 modules (alerts, interface, scan_coordinator)
│   └── execution/      # 1 module (service)
├── infrastructure/
│   ├── api/            # 1 module
│   ├── db/             # 4 modules (cache, migrations, models, secrets)
│   ├── exchanges/      # 2 stubs (binance, ibkr)
│   └── ui/             # 1 module (dashboard)
└── shared/
    ├── config/         # 1 module
    ├── utils/          # 2 modules
    ├── social/         # 1 module
    └── web3/           # 8 modules (engine, analyzers, clients, hooks, models, utils)
```

---

## Structure Cible (Outil Pédagogique)

### Phase 1: Conserver (Core)

```
src/quantum/
├── domain/
│   ├── analysis/       # ← CONSERVER (valeur pédagogique)
│   │   ├── ichimoku.py
│   │   ├── smc.py
│   │   ├── divergences.py
│   │   └── wyckoff.py
│   ├── core/           # ← CONSERVER
│   │   ├── hurst.py
│   │   ├── zscore.py
│   │   └── regime_detector.py
│   └── data/           # ← CONSERVER
│       ├── downloader.py
│       └── feature_engine.py
├── application/
│   ├── backtest/      # ← CONSERVER (valeur pédagogique)
│   │   └── engine.py
│   └── reporting/     # ← CONSERVER
│       └── alerts.py
└── shared/
    └── config/
        └── settings.py
```

### Phase 2: Déprécier (À supprimer progressivement)

| Module | Raison de dépréciation |
|--------|----------------------|
| `domain/coach/*` | Non fonctionnel, nécessite LLM externe |
| `domain/innovations/*` | Complexe, pas validé |
| `domain/strategies/multi_strategy.py` | Trop complexe, non testé |
| `domain/ml/service.py` | ML désactivé (pas de données réelles) |
| `domain/ml/trainer.py` | Dépend de service.py |
| `domain/ml/optimizer.py` | Non utilisé |
| `domain/risk/portfolio.py` | Dupliqué dans manager.py |
| `domain/risk/calendar.py` | Non fonctionnel |
| `application/backtest/monte_carlo.py` | Doublon avec engine.py |
| `application/backtest/paper_trading.py` | Stub non fonctionnel |
| `application/backtest/trading_costs.py` | Doublon |
| `application/execution/*` | Stub non fonctionnel |
| `infrastructure/api/*` | Non déployé |
| `infrastructure/db/*` | Pas de base de données |
| `infrastructure/exchanges/*` | Stubs vides |
| `shared/web3/*` | Complexe, non testé |
| `shared/social/*` | Non fonctionnel |

---

## Plan de Simplification

### Step 1: Marquer comme dépréciés

Créer un fichier `DEPRECATED.py` qui marque les modules à supprimer:

```python
"""
DEPRECATED MODULES - À supprimer dans la version 2.0
====================================================

Les modules suivants sont dépréciés et seront supprimés
dans une future version.

Pour chaque module, la raison et un remplacement suggéré:
"""

DEPRECATED_MODULES = {
    "domain/coach": {
        "status": "DEPRECATED",
        "reason": "Nécessite LLM externe (OpenAI), non fonctionnel",
        "replacement": "Supprimer ou séparer dans un package optional"
    },
    "domain/innovations": {
        "status": "DEPRECATED", 
        "reason": "Complexe, jamais validé en conditions réelles",
        "replacement": "Supprimer"
    },
    "shared/web3": {
        "status": "DEPRECATED",
        "reason": "Non testé, complexité inutilisée",
        "replacement": "Supprimer ou déplacer vers optional"
    },
    # ... etc
}
```

### Step 2: Supprimer progressivement

1. Créer une branche `refactor/remove-deprecated`
2. Supprimer les modules un par un
3. Tester que le système core fonctionne toujours
4. Commiter chaque suppression

### Step 3: Réorganiser

Une fois les modules supprimés, restructurer:
- `src/indicators/` pour les analyseurs
- `src/data/` pour le downloader et features
- `src/backtest/` pour le moteur de backtest

---

## Commandes Git pour la simplification

```bash
# Identifier les modules non importés
grep -r "from quantum.domain.coach" src/
grep -r "from quantum.domain.innovations" src/
grep -r "from quantum.shared.web3" src/

# Supprimer les modules dépréciés (exemple)
rm -rf src/quantum/domain/coach/
rm -rf src/quantum/domain/innovations/
rm -rf src/quantum/shared/web3/
```

---

## Critères de Décision

Un module devrait être CONSERVÉ si:
- ✅ Il a une valeur pédagogique claire
- ✅ Il est utilisé par d'autres modules
- ✅ Il fonctionne sans configuration complexe
- ✅ Il peut être réutilisé dans l'outil éducatif

Un module devrait être SUPPRIMÉ si:
- ❌ Il ne fonctionne pas (stub, NotImplementedError)
- ❌ Il nécessite des API externes non configurées
- ❌ Il n'est jamais importé par d'autres modules
- ❌ Il est trop complexe pour sa valeur

---

*Document créé dans le cadre du pivot vers un outil pédagogique.*
*Dernière mise à jour: Février 2026*
