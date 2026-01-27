"""
Module de validation des données de marché.
Détecte les anomalies, données corrompues et incohérences.

Vérifications:
1. Cohérence OHLC (High >= Low, etc.)
2. Données aberrantes (spikes)
3. Gaps de données
4. Volumes anormaux
5. Validation croisée multi-sources
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import sys
import os




class ValidationLevel(Enum):
    """Niveau de sévérité des problèmes."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Représente un problème de validation."""
    level: ValidationLevel
    field: str
    message: str
    index: Optional[Any] = None
    value: Optional[Any] = None
    suggested_fix: Optional[str] = None


@dataclass
class ValidationResult:
    """Résultat de la validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    statistics: Dict = field(default_factory=dict)
    cleaned_data: Optional[pd.DataFrame] = None
    
    @property
    def has_errors(self) -> bool:
        return any(i.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL] for i in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        return any(i.level == ValidationLevel.WARNING for i in self.issues)
    
    def summary(self) -> str:
        """Résumé textuel des problèmes."""
        if not self.issues:
            return "✅ Données valides, aucun problème détecté"
        
        counts = {}
        for issue in self.issues:
            counts[issue.level.value] = counts.get(issue.level.value, 0) + 1
        
        parts = [f"{v} {k}" for k, v in counts.items()]
        return f"⚠️ Problèmes détectés: {', '.join(parts)}"


class DataValidator:
    """
    Validateur de données OHLCV.
    
    Détecte et peut corriger automatiquement les problèmes courants.
    """
    
    def __init__(
        self,
        max_price_change_pct: float = 20.0,  # Max 20% de variation
        max_gap_pct: float = 5.0,  # Max 5% de gap
        min_volume: int = 0,
        max_consecutive_same: int = 10,  # Max 10 bougies identiques
        outlier_std_threshold: float = 5.0  # 5 écarts-types
    ):
        self.max_price_change_pct = max_price_change_pct
        self.max_gap_pct = max_gap_pct
        self.min_volume = min_volume
        self.max_consecutive_same = max_consecutive_same
        self.outlier_std_threshold = outlier_std_threshold
    
    def validate(
        self,
        df: pd.DataFrame,
        auto_fix: bool = False
    ) -> ValidationResult:
        """
        Valide un DataFrame OHLCV.
        
        Args:
            df: DataFrame avec colonnes Open, High, Low, Close, Volume
            auto_fix: Tenter de corriger automatiquement les problèmes
        
        Returns:
            ValidationResult
        """
        issues = []
        
        if df is None or df.empty:
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    level=ValidationLevel.CRITICAL,
                    field="data",
                    message="DataFrame vide ou None"
                )]
            )
        
        # Vérifier les colonnes requises
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    level=ValidationLevel.CRITICAL,
                    field="columns",
                    message=f"Colonnes manquantes: {missing_cols}"
                )]
            )
        
        # Copie pour les corrections
        df_clean = df.copy() if auto_fix else df
        
        # 1. Vérifier cohérence OHLC
        ohlc_issues = self._validate_ohlc_consistency(df)
        issues.extend(ohlc_issues)
        
        # 2. Détecter les valeurs nulles
        null_issues = self._validate_nulls(df)
        issues.extend(null_issues)
        
        # 3. Détecter les spikes (variations extrêmes)
        spike_issues = self._validate_price_spikes(df)
        issues.extend(spike_issues)
        
        # 4. Détecter les gaps de données
        gap_issues = self._validate_gaps(df)
        issues.extend(gap_issues)
        
        # 5. Valider les volumes
        if 'Volume' in df.columns:
            volume_issues = self._validate_volume(df)
            issues.extend(volume_issues)
        
        # 6. Détecter les données dupliquées
        dup_issues = self._validate_duplicates(df)
        issues.extend(dup_issues)
        
        # 7. Détecter les bougies identiques consécutives
        same_issues = self._validate_consecutive_same(df)
        issues.extend(same_issues)
        
        # 8. Valider l'ordre temporel
        time_issues = self._validate_time_order(df)
        issues.extend(time_issues)
        
        # Correction automatique si demandée
        if auto_fix and issues:
            df_clean = self._auto_fix(df, issues)
        
        # Statistiques
        stats = self._calculate_statistics(df)
        
        # Déterminer validité
        is_valid = not any(
            i.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]
            for i in issues
        )
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            statistics=stats,
            cleaned_data=df_clean if auto_fix else None
        )
    
    def _validate_ohlc_consistency(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Vérifie que High >= Low, High >= Open/Close, Low <= Open/Close."""
        issues = []
        
        # High doit être >= Low
        invalid_hl = df[df['High'] < df['Low']]
        if len(invalid_hl) > 0:
            for idx in invalid_hl.index[:5]:  # Max 5 exemples
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    field="High/Low",
                    message=f"High ({df.loc[idx, 'High']}) < Low ({df.loc[idx, 'Low']})",
                    index=idx,
                    suggested_fix="Inverser High et Low"
                ))
        
        # High doit être >= Open et Close
        invalid_h = df[(df['High'] < df['Open']) | (df['High'] < df['Close'])]
        if len(invalid_h) > 0:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field="High",
                message=f"{len(invalid_h)} lignes où High < Open ou Close",
                suggested_fix="Ajuster High = max(Open, High, Close)"
            ))
        
        # Low doit être <= Open et Close
        invalid_l = df[(df['Low'] > df['Open']) | (df['Low'] > df['Close'])]
        if len(invalid_l) > 0:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field="Low",
                message=f"{len(invalid_l)} lignes où Low > Open ou Close",
                suggested_fix="Ajuster Low = min(Open, Low, Close)"
            ))
        
        # Prix négatifs
        for col in ['Open', 'High', 'Low', 'Close']:
            negative = df[df[col] <= 0]
            if len(negative) > 0:
                issues.append(ValidationIssue(
                    level=ValidationLevel.CRITICAL,
                    field=col,
                    message=f"{len(negative)} valeurs <= 0",
                    suggested_fix="Supprimer ou interpoler"
                ))
        
        return issues
    
    def _validate_nulls(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Détecte les valeurs nulles."""
        issues = []
        
        for col in ['Open', 'High', 'Low', 'Close']:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                pct = null_count / len(df) * 100
                level = ValidationLevel.CRITICAL if pct > 5 else ValidationLevel.WARNING
                issues.append(ValidationIssue(
                    level=level,
                    field=col,
                    message=f"{null_count} valeurs nulles ({pct:.1f}%)",
                    suggested_fix="Interpoler ou forward-fill"
                ))
        
        return issues
    
    def _validate_price_spikes(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Détecte les variations de prix anormales."""
        issues = []
        
        # Variation en pourcentage
        pct_change = df['Close'].pct_change().abs() * 100
        spikes = pct_change[pct_change > self.max_price_change_pct]
        
        if len(spikes) > 0:
            for idx in spikes.index[:5]:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    field="Close",
                    message=f"Spike de {pct_change[idx]:.1f}% détecté",
                    index=idx,
                    value=df.loc[idx, 'Close'],
                    suggested_fix="Vérifier si spike réel ou donnée corrompue"
                ))
        
        # Outliers basés sur Z-score
        mean = df['Close'].mean()
        std = df['Close'].std()
        if std > 0:
            zscore = (df['Close'] - mean) / std
            outliers = df[zscore.abs() > self.outlier_std_threshold]
            if len(outliers) > 0:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    field="Close",
                    message=f"{len(outliers)} outliers détectés (> {self.outlier_std_threshold}σ)",
                    suggested_fix="Vérifier manuellement ces valeurs"
                ))
        
        return issues
    
    def _validate_gaps(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Détecte les gaps de données (temps et prix)."""
        issues = []
        
        if not isinstance(df.index, pd.DatetimeIndex):
            return issues
        
        # Gaps temporels
        time_diff = df.index.to_series().diff()
        if len(time_diff) > 1:
            median_diff = time_diff.median()
            large_gaps = time_diff[time_diff > median_diff * 3]
            
            if len(large_gaps) > 0:
                issues.append(ValidationIssue(
                    level=ValidationLevel.INFO,
                    field="index",
                    message=f"{len(large_gaps)} gaps temporels détectés",
                    suggested_fix="Normal pour weekends/jours fériés"
                ))
        
        # Gaps de prix (gap open vs previous close)
        gap_pct = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)).abs() * 100
        large_price_gaps = gap_pct[gap_pct > self.max_gap_pct]
        
        if len(large_price_gaps) > 0:
            issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                field="Open",
                message=f"{len(large_price_gaps)} gaps de prix > {self.max_gap_pct}%",
                suggested_fix="Normal pour gaps overnight"
            ))
        
        return issues
    
    def _validate_volume(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Valide les données de volume."""
        issues = []
        
        # Volume nul
        zero_vol = df[df['Volume'] == 0]
        if len(zero_vol) > 0:
            pct = len(zero_vol) / len(df) * 100
            if pct > 10:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    field="Volume",
                    message=f"{len(zero_vol)} lignes sans volume ({pct:.1f}%)",
                    suggested_fix="Possible si marché fermé"
                ))
        
        # Volume négatif
        neg_vol = df[df['Volume'] < 0]
        if len(neg_vol) > 0:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field="Volume",
                message=f"{len(neg_vol)} volumes négatifs",
                suggested_fix="Remplacer par 0 ou valeur absolue"
            ))
        
        return issues
    
    def _validate_duplicates(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Détecte les lignes dupliquées."""
        issues = []
        
        # Index dupliqués
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field="index",
                message=f"{dup_count} index dupliqués",
                suggested_fix="Garder la dernière occurrence"
            ))
        
        # Lignes complètement dupliquées
        dup_rows = df.duplicated()
        if dup_rows.any():
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field="data",
                message=f"{dup_rows.sum()} lignes dupliquées",
                suggested_fix="Supprimer les duplicats"
            ))
        
        return issues
    
    def _validate_consecutive_same(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Détecte les bougies identiques consécutives (données gelées)."""
        issues = []
        
        # Compter les Close identiques consécutifs
        same_as_prev = (df['Close'] == df['Close'].shift(1))
        
        # Trouver les séquences
        max_consecutive = 0
        current = 0
        
        for is_same in same_as_prev:
            if is_same:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        
        if max_consecutive >= self.max_consecutive_same:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field="Close",
                message=f"{max_consecutive} bougies identiques consécutives (prix gelé?)",
                suggested_fix="Vérifier si données réelles ou feed gelé"
            ))
        
        return issues
    
    def _validate_time_order(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Vérifie que l'index est ordonné chronologiquement."""
        issues = []
        
        if isinstance(df.index, pd.DatetimeIndex):
            if not df.index.is_monotonic_increasing:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    field="index",
                    message="Index non ordonné chronologiquement",
                    suggested_fix="Trier par index"
                ))
        
        return issues
    
    def _auto_fix(
        self,
        df: pd.DataFrame,
        issues: List[ValidationIssue]
    ) -> pd.DataFrame:
        """Corrige automatiquement les problèmes réparables."""
        df_fixed = df.copy()
        
        # Trier par index
        if not df_fixed.index.is_monotonic_increasing:
            df_fixed = df_fixed.sort_index()
        
        # Supprimer duplicats d'index
        if df_fixed.index.duplicated().any():
            df_fixed = df_fixed[~df_fixed.index.duplicated(keep='last')]
        
        # Corriger High/Low inversés
        mask = df_fixed['High'] < df_fixed['Low']
        if mask.any():
            df_fixed.loc[mask, ['High', 'Low']] = df_fixed.loc[mask, ['Low', 'High']].values
        
        # Ajuster High = max(O, H, C)
        df_fixed['High'] = df_fixed[['Open', 'High', 'Close']].max(axis=1)
        
        # Ajuster Low = min(O, L, C)
        df_fixed['Low'] = df_fixed[['Open', 'Low', 'Close']].min(axis=1)
        
        # Interpoler les nulls
        for col in ['Open', 'High', 'Low', 'Close']:
            if df_fixed[col].isnull().any():
                df_fixed[col] = df_fixed[col].interpolate(method='linear')
                df_fixed[col] = df_fixed[col].ffill().bfill()
        
        # Volume négatif -> 0
        if 'Volume' in df_fixed.columns:
            df_fixed.loc[df_fixed['Volume'] < 0, 'Volume'] = 0
        
        return df_fixed
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calcule des statistiques sur les données."""
        stats = {
            'rows': len(df),
            'date_range': {
                'start': str(df.index.min()) if len(df) > 0 else None,
                'end': str(df.index.max()) if len(df) > 0 else None
            },
            'price_range': {
                'min': float(df['Low'].min()) if len(df) > 0 else None,
                'max': float(df['High'].max()) if len(df) > 0 else None,
                'current': float(df['Close'].iloc[-1]) if len(df) > 0 else None
            },
            'nulls': {col: int(df[col].isnull().sum()) for col in df.columns},
            'quality_score': 100  # À calculer
        }
        
        # Score de qualité
        total_issues = sum(stats['nulls'].values())
        if len(df) > 0:
            quality = max(0, 100 - (total_issues / len(df) * 100))
            stats['quality_score'] = round(quality, 1)
        
        return stats


class CrossSourceValidator:
    """
    Valide les données en comparant plusieurs sources.
    
    Détecte les écarts significatifs entre sources pour identifier la donnée fiable.
    """
    
    def __init__(self, max_price_diff_pct: float = 1.0):
        self.max_price_diff_pct = max_price_diff_pct
    
    def compare_sources(
        self,
        data_sources: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Compare les données de plusieurs sources.
        
        Args:
            data_sources: Dict {source_name: dataframe}
        
        Returns:
            Dict avec comparaison et source recommandée
        """
        if len(data_sources) < 2:
            return {
                'status': 'insufficient_sources',
                'message': 'Au moins 2 sources requises pour comparaison'
            }
        
        source_names = list(data_sources.keys())
        comparisons = []
        
        for i, name1 in enumerate(source_names):
            for name2 in source_names[i + 1:]:
                df1 = data_sources[name1]
                df2 = data_sources[name2]
                
                comparison = self._compare_two_sources(name1, df1, name2, df2)
                comparisons.append(comparison)
        
        # Déterminer la source la plus fiable
        reliability_scores = {name: 0 for name in source_names}
        
        for comp in comparisons:
            if comp['is_consistent']:
                reliability_scores[comp['source1']] += 1
                reliability_scores[comp['source2']] += 1
        
        best_source = max(reliability_scores, key=reliability_scores.get)
        
        return {
            'status': 'completed',
            'comparisons': comparisons,
            'reliability_scores': reliability_scores,
            'recommended_source': best_source,
            'all_consistent': all(c['is_consistent'] for c in comparisons)
        }
    
    def _compare_two_sources(
        self,
        name1: str,
        df1: pd.DataFrame,
        name2: str,
        df2: pd.DataFrame
    ) -> Dict:
        """Compare deux sources de données."""
        # Aligner sur l'index commun
        common_idx = df1.index.intersection(df2.index)
        
        if len(common_idx) == 0:
            return {
                'source1': name1,
                'source2': name2,
                'is_consistent': False,
                'error': 'Aucun index commun'
            }
        
        df1_aligned = df1.loc[common_idx]
        df2_aligned = df2.loc[common_idx]
        
        # Calculer les différences
        close_diff_pct = ((df1_aligned['Close'] - df2_aligned['Close']) / df1_aligned['Close']).abs() * 100
        
        avg_diff = close_diff_pct.mean()
        max_diff = close_diff_pct.max()
        
        is_consistent = max_diff < self.max_price_diff_pct
        
        return {
            'source1': name1,
            'source2': name2,
            'common_rows': len(common_idx),
            'avg_diff_pct': round(avg_diff, 4),
            'max_diff_pct': round(max_diff, 4),
            'is_consistent': is_consistent,
            'recommendation': 'OK' if is_consistent else f'Écart max {max_diff:.2f}% - vérifier'
        }


if __name__ == "__main__":
    print("=" * 60)
    print("TEST DATA VALIDATOR")
    print("=" * 60)
    
    # Créer des données de test avec quelques problèmes
    np.random.seed(42)
    n = 100
    
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    df = pd.DataFrame({
        'Open': close + np.random.randn(n) * 0.2,
        'High': close + abs(np.random.randn(n)) * 0.5,
        'Low': close - abs(np.random.randn(n)) * 0.5,
        'Close': close,
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    # Ajouter quelques problèmes
    df.loc[df.index[10], 'High'] = df.loc[df.index[10], 'Low'] - 1  # High < Low
    df.loc[df.index[20], 'Close'] = np.nan  # Null
    df.loc[df.index[30], 'Volume'] = -100  # Volume négatif
    df.loc[df.index[40], 'Close'] = 200  # Spike
    
    # Valider
    validator = DataValidator()
    result = validator.validate(df, auto_fix=True)
    
    print(f"\n{result.summary()}")
    print(f"\nDonnées valides: {result.is_valid}")
    print(f"Score qualité: {result.statistics.get('quality_score', 'N/A')}%")
    
    print("\n--- Problèmes détectés ---")
    for issue in result.issues:
        print(f"  [{issue.level.value}] {issue.field}: {issue.message}")
    
    if result.cleaned_data is not None:
        print(f"\n✅ Données corrigées disponibles ({len(result.cleaned_data)} lignes)")
