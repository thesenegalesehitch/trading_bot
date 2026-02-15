"""
Trade Validator - Valide les trades saisis par l'utilisateur et fournit un feedback structuré.
Phase 3: Coach Features - Trade Advisor & Coach
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from quantum.domain.data.downloader import DataDownloader
from quantum.domain.data.feature_engine import TechnicalIndicators
from quantum.domain.core.regime_detector import RegimeDetector


class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


@dataclass
class ValidationIssue:
    severity: ValidationSeverity
    category: str
    message: str
    recommendation: str
    details: Optional[Dict] = None


@dataclass
class TradeValidation:
    is_valid: bool
    score: float  # 0-100
    issues: List[ValidationIssue]
    summary: str
    improvements: List[str]
    market_context: Dict[str, Any]


class TradeValidator:
    """
    Valide les trades saisis par l'utilisateur et fournit un feedback structuré.
    
    Vérifications effectuées:
    - Validité des prix (entry, SL, TP)
    - Alignement avec la tendance du marché
    - Qualité du risk/reward
    - Conformité avec le régime de marché
    - Taille de position raisonnable
    """
    
    def __init__(self):
        self.downloader = DataDownloader()
        self.indicators = TechnicalIndicators()
        self.regime_detector = RegimeDetector()
    
    def validate_trade(
        self,
        symbol: str,
        direction: str,  # "BUY" or "SELL"
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: Optional[float] = None,
        account_balance: Optional[float] = None,
        risk_per_trade: float = 2.0  # % du compte
    ) -> TradeValidation:
        """
        Valide un trade et retourne un feedback structuré.
        """
        issues = []
        improvements = []
        score = 100
        
        # Récupérer les données de marché
        try:
            df = self.downloader.download_data(
                symbol=symbol,
                interval="1h",
                years=1
            )
            if df is None or len(df) < 100:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="data",
                    message="Données de marché insuffisantes",
                    recommendation="Vérifier la disponibilité des données"
                ))
                score -= 10
                df = None
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="data",
                message=f"Erreur lors de la récupération des données: {str(e)}",
                recommendation="Vérifier la connexion internet et le symbol"
            ))
            score -= 20
            df = None
        
        # Validation des prix
        price_issues = self._validate_prices(direction, entry_price, stop_loss, take_profit)
        issues.extend(price_issues)
        score -= len(price_issues) * 10
        
        # Validation du Risk/Reward
        rr_issues = self._validate_risk_reward(entry_price, stop_loss, take_profit)
        issues.extend(rr_issues)
        score -= len(rr_issues) * 15
        
        # Validation du contexte de marché si données disponibles
        market_context = {}
        if df is not None:
            # Détecter le régime de marché
            regime = self.regime_detector.detect_regime(df)
            market_context['regime'] = regime
            
            # Vérifier l'alignement avec la tendance
            trend_issues = self._validate_alignment(df, direction, entry_price)
            issues.extend(trend_issues)
            score -= len(trend_issues) * 15
            
            # Vérifier les niveaux techniques
            level_issues = self._validate_levels(df, direction, entry_price, stop_loss, take_profit)
            issues.extend(level_issues)
            score -= len(level_issues) * 10
        
        # Validation de la taille de position
        if account_balance and position_size:
            size_issues = self._validate_position_size(
                entry_price, stop_loss, account_balance, risk_per_trade, position_size
            )
            issues.extend(size_issues)
            score -= len(size_issues) * 10
        
        # Calculer le score final
        score = max(0, min(100, score))
        is_valid = score >= 50 and not any(i.severity == ValidationSeverity.ERROR for i in issues)
        
        # Générer les améliorations
        improvements = self._generate_improvements(issues, market_context)
        
        # Générer le résumé
        summary = self._generate_summary(score, is_valid, issues)
        
        return TradeValidation(
            is_valid=is_valid,
            score=score,
            issues=issues,
            summary=summary,
            improvements=improvements,
            market_context=market_context
        )
    
    def _validate_prices(
        self,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> List[ValidationIssue]:
        """Valide la cohérence des prix."""
        issues = []
        
        if direction == "BUY":
            if entry_price <= stop_loss:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="prices",
                    message="Pour un BUY, le Stop Loss doit être INFÉRIEUR au prix d'entrée",
                    recommendation="Déplacer le SL en dessous de l'entrée"
                ))
            if entry_price >= take_profit:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="prices",
                    message="Pour un BUY, le Take Profit doit être SUPÉRIEUR au prix d'entrée",
                    recommendation="Déplacer le TP au-dessus de l'entrée"
                ))
        else:  # SELL
            if entry_price >= stop_loss:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="prices",
                    message="Pour un SELL, le Stop Loss doit être SUPÉRIEUR au prix d'entrée",
                    recommendation="Déplacer le SL au-dessus de l'entrée"
                ))
            if entry_price <= take_profit:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="prices",
                    message="Pour un SELL, le Take Profit doit être INFÉRIEUR au prix d'entrée",
                    recommendation="Déplacer le TP en dessous de l'entrée"
                ))
        
        return issues
    
    def _validate_risk_reward(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> List[ValidationIssue]:
        """Valide le ratio risk/reward."""
        issues = []
        
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="risk_reward",
                message="Risque nul - Stop Loss non défini correctement",
                recommendation="Définir un Stop Loss"
            ))
            return issues
        
        rr_ratio = reward / risk
        
        if rr_ratio < 1:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="risk_reward",
                message=f"Risk/Reward insuffisant: 1:{rr_ratio:.2f} (minimum recommandé: 1:1.5)",
                recommendation="Soit rapproche le SL, soit éloigne le TP pour obtenir au moins 1:1.5"
            ))
        elif rr_ratio < 1.5:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="risk_reward",
                message=f"Risk/Reward modéré: 1:{rr_ratio:.2f} (optimal: > 1:2)",
                recommendation="Envisager un meilleur ratio pour améliorer la rentabilité"
            ))
        
        return issues
    
    def _validate_alignment(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float
    ) -> List[ValidationIssue]:
        """Valide l'alignement avec la tendance du marché."""
        issues = []
        
        # Calculer les EMAs
        ema_20 = df['Close'].ewm(span=20).mean()
        ema_50 = df['Close'].ewm(span=50).mean()
        
        current_price = df['Close'].iloc[-1]
        ema20_current = ema_20.iloc[-1]
        ema50_current = ema_50.iloc[-1]
        
        # Déterminer la tendance
        is_bullish = ema20_current > ema50_current
        
        if direction == "BUY" and not is_bullish:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="trend_alignment",
                message="Entry BUY contre la tendance baissièrent (EMA20 < EMA50)",
                recommendation="En market baissier, privilégier les entries SELL"
            ))
        elif direction == "SELL" and is_bullish:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="trend_alignment",
                message="Entry SELL contre la tendance haussière (EMA20 > EMA50)",
                recommendation="En market haussier, privilégier les entries BUY"
            ))
        
        return issues
    
    def _validate_levels(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> List[ValidationIssue]:
        """Valide les niveaux techniques."""
        issues = []
        
        recent_low = df['Low'].tail(20).min()
        recent_high = df['High'].tail(20).max()
        
        if direction == "BUY":
            # Pour un BUY, le SL devrait être près d'un support
            if stop_loss < recent_low * 0.99:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="levels",
                    message="Stop Loss sous le support recent",
                    recommendation="Rapprocher le SL du support pour un meilleur RR"
                ))
        else:
            # Pour un SELL, le SL devrait être près d'une résistance
            if stop_loss > recent_high * 1.01:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="levels",
                    message="Stop Loss au-dessus de la résistance récente",
                    recommendation="Rapprocher le SL de la résistance pour un meilleur RR"
                ))
        
        return issues
    
    def _validate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        account_balance: float,
        risk_per_trade: float,
        position_size: float
    ) -> List[ValidationIssue]:
        """Valide la taille de position."""
        issues = []
        
        risk_amount = account_balance * (risk_per_trade / 100)
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit > 0:
            calculated_size = risk_amount / risk_per_unit
            
            # Vérifier si la taille est raisonnable (±20% de la taille calculée)
            if position_size > calculated_size * 1.2:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="position_size",
                    message="Taille de position supérieure au risque défini",
                    recommendation=f"Réduire la taille à {calculated_size:.4f} pour respecter le risque de {risk_per_trade}%"
                ))
            elif position_size < calculated_size * 0.8:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="position_size",
                    message="Taille de position inférieure au risque défini",
                    recommendation=f"Possible d'augmenter à {calculated_size:.4f}"
                ))
        
        return issues
    
    def _generate_improvements(
        self,
        issues: List[ValidationIssue],
        market_context: Dict[str, Any]
    ) -> List[str]:
        """Génère des suggestions d'amélioration."""
        improvements = []
        
        # Analyser les erreurs fréquentes
        categories = {}
        for issue in issues:
            if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.WARNING]:
                categories[issue.category] = categories.get(issue.category, 0) + 1
        
        # Suggérer des améliorations basées sur les problèmes courants
        if 'risk_reward' in categories:
            improvements.append("Améliorer le ratio Risk/Reward (minimum 1:1.5, idéalement 1:2 ou plus)")
        
        if 'trend_alignment' in categories:
            improvements.append("Trader dans la direction de la tendance (EMA20 vs EMA50)")
        
        if 'levels' in categories:
            improvements.append("Placer le Stop Loss près des supports/résistances naturels")
        
        if 'prices' in categories:
            improvements.append("Vérifier la cohérence des prix: Entry, SL, et TP doivent être alignés correctement")
        
        return improvements
    
    def _generate_summary(
        self,
        score: float,
        is_valid: bool,
        issues: List[ValidationIssue]
    ) -> Génère un résumé de la validation."""
        if score >= 80:
            quality = "excellent"
        elif score >= 60:
            quality = "bon"
        elif score >= 40:
            quality = "moyen"
        else:
            quality = "faible"
        
        error_count = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)
        
        summary = f"Trade {quality} (score: {score:.0f}/100)"
        if error_count > 0:
            summary += f" - {error_count} erreur(s) à corriger"
        if warning_count > 0:
            summary += f" - {warning_count} avertissement(s)"
        
        """Génère un résumé de la validation."""
    """Exemple d'utilisation du validateur."""
    validator = TradeValidator()
    
    # Exemple: Trade BUY sur EURUSD
    result = validator.validate_trade(
        symbol="EURUSD",
        direction="BUY",
        entry_price=1.0850,
        stop_loss=1.0820,
        take_profit=1.0910,
        account_balance=10000,
        risk_per_trade=2.0
    )
    
    print(f"\n{'='*60}")
    print(f"VALIDATION DE TRADE")
    print(f"{'='*60}")
    print(f"Score: {result.score}/100")
    print(f"Valid: {result.is_valid}")
    print(f"Résumé: {result.summary}")
    print(f"\nProblèmes détectés:")
    for issue in result.issues:
        print(f"  [{issue.severity.value.upper()}] {issue.category}: {issue.message}")
        print(f"    → {issue.recommendation}")
    
    if result.improvements:
        print(f"\nAméliorations suggérées:")
        for imp in result.improvements:
            print(f"  • {imp}")


if __name__ == "__main__":
    validate_trade_example()
