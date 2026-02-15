# ⚠️ LIMITATIONS DU SYSTÈME — DOCUMENTATION OBLIGATOIRE

> **AVERTISSEMENT IMPORTANT**: Ce document est obligatoire. Lisez-le avant d'utiliser ce système.
> Le non-respect de ces limitations peut entraîner des pertes financières importantes.

---

## 🔴 STATUT ACTUEL DU PROJET

Ce projet est un **prototype en développement**. Il n'est PAS:
- Un système de trading rentable
- Un produit financier homologué
- Un substitut à un conseil financier professionnel
- Une garantie de gains

---

## 🔴 LIMITATIONS CONNUES

### 1. Machine Learning — NON ENTRÉINÉ SUR DONNÉES RÉELLES

**Statut**: ❌ DÉSACTIVÉ

Les modèles ML sont entraînementnés sur des données synthétiques aléatoires (`np.random.randn()`).
Cela produit des modèles **sans capacité prédictive**.

**Situation actuelle**:
- `predict_signal()` retourne `NEUTRE` par défaut
- Les modèles nécessitent un entraînement sur des données réelles

**Pour utiliser le ML**:
1. Obtenez des données OHLCV réelles (via yfinance, votre broker, etc.)
2. Calculez des indicateurs techniques (RSI, MACD, ATR, etc.)
3. Définissez une cible (ex: `1` si le prix monte dans 24h, `0` sinon)
4. Appelez `ml_service.update_models(df_avec_features)`

**Avertissement**: Les marchés financiers sont semi-efficients. Même avec des données réelles,
il n'y a aucune garantie que le ML produira des prédictions profitables.

---

### 2. Données Historiques — QUANTITÉ LIMITÉE

| Timeframe | Données max disponibles |
|-----------|------------------------|
| 1 minute | ~60 jours |
| 15 minutes | ~60 jours |
| 1 heure | ~700 jours (~2 ans) |
| 1 jour | 10+ ans |

**Problème**: Pas assez de données pour valider des stratégies sur le long terme.

**Conséquence**: Les backtests peuvent être sujets à:
- Surapprentissage (overfitting)
- Biais de sélection
- Résultats non transposables

**Recommandation**: Testez les stratégies sur plusieurs instruments et timeframes
avant de les utiliser en production.

---

### 3. Backtests — COÛTS IRRÉALISTES PAR DÉFAUT

Les paramètres par défaut sous-estiment les coûts réels:

| Paramètre | Défaut actuel | Réalité |
|-----------|---------------|---------|
| Commission | 0.01% | 0.1-0.5% |
| Slippage | 0.01% | 0.05-0.5% |
| Spread (EURUSD) | 1 pip | 0.5-2 pips |

**Impact**: Les performances affichées en backtest peuvent surestimer
les résultats réels de **20-50%**.

**Solution**: Ajustez les paramètres dans `config/settings.py`:
```python
commission: float = 0.001  # 0.1% instead of 0.01%
slippage: float = 0.0005  # 0.05% minimum
```

---

### 4. Stratégies ICT/SMC — NON VALIDÉES EMPIRIQUEMENT

Les concepts ICT (SMC) implémentés:
- Fair Value Gaps (FVG)
- Order Blocks
- Market Structure Shifts (MSS)
- Killzones

**Problème**: Ces concepts sont popularisés par des traders YouTube/Instagram.
Aucune étude académique peer-reviewed ne valide leur efficacité.

**Avertissement**:
- Les FVGs détectés peuvent être des artefacts statistiques
- Les "killzones" n'ont pas de base scientifique prouvée
- Les Order Blocks sont subjectifs et non quantifiables

**Recommandation**: Testez thoroughly sur papier avant d'utiliser avec capital réel.

---

### 5. Risk Manager — DONNÉES DE FALLBACK SYNTHÉTIQUES

Si aucune donnée réelle n'est disponible, le système utilise:
- Distributions Student-t (queues grasses)
- Effets ARCH (volatility clustering)
- Changements de régime simulés

**Limitation**: Ces données restent des simulations.
Les métriques de risque (VaR, CVaR) sont des estimations, pas des garanties.

---

### 6. Connectivité de Trading — STUB SEULEMENT

**Statut**: ❌ NON FONCTIONNEL

Les fichiers suivants ne sont que des stubs:
- `binance_client.py` — NON implémenté
- `ibkr_client.py` — NON implémenté

**Implications**:
- Le système ne peut pas trader automatiquement
- Mode "backtest" uniquement par défaut
- Aucune exécution d'ordres réels

---

### 7. Web3/Mempool — COMPLEXE ET NON TESTÉ

Les modules Web3 nécessitent:
- Accès à un node Ethereum (QuickNode, Infura)
- Configuration d'API keys
- Connaissance technique blockchain

**Avertissement**: Ces fonctionnalités n'ont pas été testées en conditions réelles.
Les métriques on-chain peuvent être obsolètes ou inexactes.

---

## 🟡 RECOMMANDATIONS D'UTILISATION

### Pour le développement:
1. **Jamais** utiliser en live trading sans validation complète
2. **Toujours** tester sur papier (paper trading) d'abord
3. **Toujours** vérifier les signaux avec votre propre analyse
4. **Jamais** investir plus que ce que vous pouvez perdre

### Pour la validation:
1. Testez sur 10+ ans de données hors-échantillon
2. Validez sur plusieurs instruments
3. Testez en conditions de marché variées (bull, bear, volatile)
4. Comparez aux benchmarks (buy & hold, stratégies simples)

### Pour la production:
1. Commencez avec un capital que vous pouvez perdre à 100%
2. Implémentez un stop-loss strict
3. Surveillez activement les positions
4. Documentez chaque trade et son raisonnement

---

## 🔵 LIMITES DE RESPONSABILITÉ

**Ce logiciel est fourni "tel quel", sans garantie d'aucune sorte.**

L'auteur ne peut être tenu responsable de:
- Pertes financières résultant de l'utilisation de ce système
- Bugs ou erreurs dans le code
- Données incorrectes ou obsolètes
- Décisions de trading basées sur les signaux du système

**L'utilisateur assume l'entière responsabilité de ses décisions de trading.**

---

## 📝 CHECKLIST AVANT UTILISATION

- [ ] J'ai lu et compris ce document
- [ ] Je comprends que ce système est un prototype
- [ ] Je ne l'utilise pas avec de l'argent réel sans validation préalable
- [ ] Je comprends les limitations du ML
- [ ] Je sais que les backtests peuvent surestimer les performances
- [ ] Je suis conscient que les stratégies ICT/SMC ne sont pas validées
- [ ] J'ai les connaissances nécessaires pour trader

---

*Document généré dans le cadre du pivot vers un outil pédagogique.*
*Version: 1.0 - Février 2026*
