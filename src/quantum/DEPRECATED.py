"""
DEPRECATED MODULES - Quantum Trading System
===========================================

Ce fichier marque les modules dépréciés qui seront supprimés
dans la version 2.0.

Pour chaque module: raison de dépréciation et remplacement suggéré.
"""

DEPRECATED_MODULES = {
    # === COACH MODULES ===
    # Nécessitent LLM externe (OpenAI), non fonctionnels sans configuration
    "domain/coach": {
        "status": "DEPRECATED",
        "reason": "Nécessite LLM externe (OpenAI API), non fonctionnel sans clés API",
        "replacement": "Supprimer ou séparer dans un package optional",
        "severity": "HIGH",
    },
    "domain/coach/__init__": {
        "status": "DEPRECATED",
        "reason": "Dépend de coach/explainer.py",
        "replacement": "Supprimer",
    },
    "domain/coach/explainer.py": {
        "status": "DEPRECATED",
        "reason": "Appelle OpenAI API, non fonctionnel",
        "replacement": "Supprimer ou implémenter sans LLM",
    },
    "domain/coach/history.py": {
        "status": "DEPRECATED",
        "reason": "Dépend de coach/explainer.py",
        "replacement": "Supprimer",
    },
    "domain/coach/validator.py": {
        "status": "DEPRECATED",
        "reason": "Non fonctionnel, nécessite données utilisateur",
        "replacement": "Supprimer",
    },

    # === INNOVATIONS MODULES ===
    # Complexes, jamais validés en conditions réelles
    "domain/innovations": {
        "status": "DEPRECATED",
        "reason": "Complexe, jamais validé empiriquement",
        "replacement": "Supprimer",
        "severity": "HIGH",
    },
    "domain/innovations/confusion_resolver.py": {
        "status": "DEPRECATED",
        "reason": "Logique复杂, pas de validation",
        "replacement": "Supprimer",
    },
    "domain/innovations/mistake_predictor.py": {
        "status": "DEPRECATED",
        "reason": "Prédit erreurs utilisateur, non testé",
        "replacement": "Supprimer",
    },
    "domain/innovations/postmortem.py": {
        "status": "DEPRECATED",
        "reason": "Non intégré au flux principal",
        "replacement": "Supprimer",
    },
    "domain/innovations/whatif_simulator.py": {
        "status": "DEPRECATED",
        "reason": "Non fonctionnel, nécessite données historiques",
        "replacement": "Supprimer",
    },

    # === WEB3 MODULES ===
    # Non testés, complexité inutilisée
    "shared/web3": {
        "status": "DEPRECATED",
        "reason": "Non testé, nécessite QuickNode/Infura",
        "replacement": "Supprimer ou déplacer vers optional",
        "severity": "MEDIUM",
    },
    "shared/web3/engine.py": {
        "status": "DEPRECATED",
        "reason": "Non fonctionnel sans API blockchain",
        "replacement": "Supprimer",
    },
    "shared/web3/settings.py": {
        "status": "DEPRECATED",
        "reason": "Dépend de engine.py",
        "replacement": "Supprimer",
    },
    "shared/web3/analyzers": {
        "status": "DEPRECATED",
        "reason": "Non testés",
        "replacement": "Supprimer",
    },
    "shared/web3/clients": {
        "status": "DEPRECATED",
        "reason": "Non fonctionnels",
        "replacement": "Supprimer",
    },
    "shared/web3/hooks": {
        "status": "DEPRECATED",
        "reason": "Non utilisés",
        "replacement": "Supprimer",
    },
    "shared/web3/models": {
        "status": "DEPRECATED",
        "reason": "Dépend de web3 engine",
        "replacement": "Supprimer",
    },
    "shared/web3/utils": {
        "status": "DEPRECATED",
        "reason": "Non utilisés",
        "replacement": "Supprimer",
    },

    # === EXECUTION STUBS ===
    # Non fonctionnels (stubs vides)
    "application/execution": {
        "status": "DEPRECATED",
        "reason": "Stub vide, pas d'implémentation",
        "replacement": "Supprimer",
        "severity": "HIGH",
    },
    "infrastructure/exchanges/binance_client.py": {
        "status": "DEPRECATED",
        "reason": "Stub vide, pas de code",
        "replacement": "Supprimer",
    },
    "infrastructure/exchanges/ibkr_client.py": {
        "status": "DEPRECATED",
        "reason": "Stub vide, pas de code",
        "replacement": "Supprimer",
    },

    # === SOCIAL MODULES ===
    # Non fonctionnels
    "shared/social": {
        "status": "DEPRECATED",
        "reason": "Twitter API require OAuth, non configuré",
        "replacement": "Supprimer",
        "severity": "MEDIUM",
    },
    "shared/social/twitter_client.py": {
        "status": "DEPRECATED",
        "reason": "Stub non fonctionnel",
        "replacement": "Supprimer",
    },

    # === UNUSED ML MODULES ===
    # ML désactivé, ces modules ne sont plus utilisés
    "domain/ml/model.py": {
        "status": "DEPRECATED",
        "reason": "ML désactivé (pas de données réelles)",
        "replacement": "Supprimer",
    },
    "domain/ml/optimizer.py": {
        "status": "DEPRECATED",
        "reason": "Non utilisé sans modèles entraînés",
        "replacement": "Supprimer",
    },
    "domain/ml/trainer.py": {
        "status": "DEPRECATED",
        "reason": "ML désactivé",
        "replacement": "Supprimer",
    },
    "domain/ml/ensemble.py": {
        "status": "DEPRECATED",
        "reason": "ML désactivé",
        "replacement": "Supprimer",
    },

    # === UNUSED BACKTEST MODULES ===
    # Doublons, non nécessaires
    "application/backtest/monte_carlo.py": {
        "status": "DEPRECATED",
        "reason": "Doublon avec engine.py",
        "replacement": "Intégrer à engine.py ou supprimer",
    },
    "application/backtest/paper_trading.py": {
        "status": "DEPRECATED",
        "reason": "Stub non fonctionnel",
        "replacement": "Supprimer",
    },
    "application/backtest/trading_costs.py": {
        "status": "DEPRECATED",
        "reason": "Doublon, peut être intégré",
        "replacement": "Supprimer",
    },

    # === UNUSED RISK MODULES ===
    # Doublons avec manager.py
    "domain/risk/calendar.py": {
        "status": "DEPRECATED",
        "reason": "Non fonctionnel, dates codées en dur",
        "replacement": "Supprimer",
    },
    "domain/risk/portfolio.py": {
        "status": "DEPRECATED",
        "reason": "Doublon avec manager.py",
        "replacement": "Intégrer à manager.py",
    },
    "domain/risk/circuit_breaker.py": {
        "status": "DEPRECATED",
        "reason": "Non utilisé dans le flux principal",
        "replacement": "Supprimer",
    },
    "domain/risk/var_calculator.py": {
        "status": "DEPRECATED",
        "reason": "Doublon avec manager.py",
        "replacement": "Intégrer à manager.py",
    },

    # === UNUSED API/DB MODULES ===
    # Pas déployés
    "infrastructure/api": {
        "status": "DEPRECATED",
        "reason": "Jamais déployé",
        "replacement": "Supprimer",
        "severity": "MEDIUM",
    },
    "infrastructure/db/migrations.py": {
        "status": "DEPRECATED",
        "reason": "Pas de base de données",
        "replacement": "Supprimer",
    },
    "infrastructure/db/models.py": {
        "status": "DEPRECATED",
        "reason": "Pas de base de données",
        "replacement": "Supprimer",
    },
    "infrastructure/db/secrets.py": {
        "status": "DEPRECATED",
        "reason": "Pas utilisé",
        "replacement": "Supprimer",
    },
}


def get_deprecated_modules():
    """Retourne la liste des modules dépréciés."""
    return DEPRECATED_MODULES


def get_deprecated_by_severity(severity: str) -> dict:
    """Retourne les modules par sévérité (HIGH, MEDIUM, LOW)."""
    return {
        k: v for k, v in DEPRECATED_MODULES.items() 
        if v.get("severity") == severity
    }


def count_deprecated() -> dict:
    """Compte les modules par sévérité."""
    return {
        "HIGH": len(get_deprecated_by_severity("HIGH")),
        "MEDIUM": len(get_deprecated_by_severity("MEDIUM")),
        "LOW": len([m for m in DEPRECATED_MODULES.values() 
                   if m.get("severity") not in ["HIGH", "MEDIUM"]]),
        "TOTAL": len(DEPRECATED_MODULES)
    }


if __name__ == "__main__":
    print("=" * 60)
    print("MODULES DÉPRÉCIÉS - Quantum Trading System")
    print("=" * 60)
    
    counts = count_deprecated()
    print(f"\nTotal modules dépréciés: {counts['TOTAL']}")
    print(f"  - HIGH severity:   {counts['HIGH']}")
    print(f"  - MEDIUM severity: {counts['MEDIUM']}")
    print(f"  - LOW severity:    {counts['LOW']}")
    
    print("\nModules HIGH severity (à supprimer en priorité):")
    for module, info in get_deprecated_by_severity("HIGH").items():
        print(f"  - {module}")
        print(f"    Raison: {info['reason']}")
