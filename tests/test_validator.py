"""
Tests unitaires pour le module de validation des données.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.validator import (
    DataValidator,
    ValidationResult,
    ValidationLevel,
    CrossSourceValidator
)


class TestDataValidator:
    """Tests pour DataValidator."""
    
    @pytest.fixture
    def valid_df(self):
        """DataFrame valide pour les tests."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2024-01-01', periods=n, freq='1h')
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        
        return pd.DataFrame({
            'Open': close + np.random.randn(n) * 0.2,
            'High': close + abs(np.random.randn(n)) * 0.5 + 0.1,
            'Low': close - abs(np.random.randn(n)) * 0.5 - 0.1,
            'Close': close,
            'Volume': np.random.randint(1000, 10000, n)
        }, index=dates)
    
    @pytest.fixture
    def validator(self):
        """Instance de DataValidator."""
        return DataValidator()
    
    def test_valid_data_passes(self, validator, valid_df):
        """Données valides doivent passer sans erreurs."""
        result = validator.validate(valid_df)
        
        assert result.is_valid
        assert not result.has_errors
    
    def test_empty_dataframe_fails(self, validator):
        """DataFrame vide doit échouer."""
        result = validator.validate(pd.DataFrame())
        
        assert not result.is_valid
        assert len(result.issues) > 0
        assert result.issues[0].level == ValidationLevel.CRITICAL
    
    def test_missing_columns_fails(self, validator):
        """Colonnes manquantes doivent échouer."""
        df = pd.DataFrame({'Close': [1, 2, 3]})
        result = validator.validate(df)
        
        assert not result.is_valid
        assert any('manquantes' in i.message for i in result.issues)
    
    def test_high_lower_than_low_detected(self, validator, valid_df):
        """High < Low doit être détecté."""
        df = valid_df.copy()
        df.loc[df.index[10], 'High'] = df.loc[df.index[10], 'Low'] - 1
        
        result = validator.validate(df)
        
        assert any('High' in i.field and 'Low' in i.message for i in result.issues)
    
    def test_null_values_detected(self, validator, valid_df):
        """Valeurs nulles doivent être détectées."""
        df = valid_df.copy()
        df.loc[df.index[20], 'Close'] = np.nan
        
        result = validator.validate(df)
        
        assert any('nulles' in i.message for i in result.issues)
    
    def test_negative_volume_detected(self, validator, valid_df):
        """Volumes négatifs doivent être détectés."""
        df = valid_df.copy()
        df.loc[df.index[30], 'Volume'] = -100
        
        result = validator.validate(df)
        
        assert any('négatif' in i.message for i in result.issues)
    
    def test_duplicate_index_detected(self, validator, valid_df):
        """Index dupliqués doivent être détectés."""
        df = valid_df.copy()
        df = pd.concat([df, df.iloc[[0]]])  # Dupliquer la première ligne
        
        result = validator.validate(df)
        
        assert any('dupliqué' in i.message for i in result.issues)
    
    def test_auto_fix_corrects_high_low(self, validator, valid_df):
        """Auto-fix doit corriger High/Low inversés."""
        df = valid_df.copy()
        original_high = df.loc[df.index[10], 'Low']
        original_low = df.loc[df.index[10], 'High']
        df.loc[df.index[10], 'High'] = original_low
        df.loc[df.index[10], 'Low'] = original_high
        
        result = validator.validate(df, auto_fix=True)
        
        assert result.cleaned_data is not None
        # Après fix, High >= Low
        assert result.cleaned_data.loc[df.index[10], 'High'] >= result.cleaned_data.loc[df.index[10], 'Low']
    
    def test_auto_fix_interpolates_nulls(self, validator, valid_df):
        """Auto-fix doit interpoler les valeurs nulles."""
        df = valid_df.copy()
        df.loc[df.index[50], 'Close'] = np.nan
        
        result = validator.validate(df, auto_fix=True)
        
        assert result.cleaned_data is not None
        assert not result.cleaned_data['Close'].isnull().any()
    
    def test_statistics_calculated(self, validator, valid_df):
        """Les statistiques doivent être calculées."""
        result = validator.validate(valid_df)
        
        assert 'rows' in result.statistics
        assert 'date_range' in result.statistics
        assert 'quality_score' in result.statistics
        assert result.statistics['rows'] == len(valid_df)
    
    def test_price_spike_detected(self, validator, valid_df):
        """Spike de prix doit être détecté."""
        df = valid_df.copy()
        df.loc[df.index[40], 'Close'] = df['Close'].mean() * 2  # Double du prix moyen
        
        result = validator.validate(df)
        
        # Devrait détecter comme spike ou outlier
        assert any('Spike' in i.message or 'outlier' in i.message for i in result.issues)


class TestCrossSourceValidator:
    """Tests pour CrossSourceValidator."""
    
    @pytest.fixture
    def validator(self):
        return CrossSourceValidator(max_price_diff_pct=1.0)
    
    @pytest.fixture
    def consistent_sources(self):
        """Deux sources avec données cohérentes."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2024-01-01', periods=n, freq='1h')
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        
        df1 = pd.DataFrame({'Close': close}, index=dates)
        df2 = pd.DataFrame({'Close': close * 1.001}, index=dates)  # 0.1% diff
        
        return {'source1': df1, 'source2': df2}
    
    def test_consistent_sources_pass(self, validator, consistent_sources):
        """Sources cohérentes doivent être validées."""
        result = validator.compare_sources(consistent_sources)
        
        assert result['status'] == 'completed'
        assert result['all_consistent']
    
    def test_single_source_insufficient(self, validator):
        """Une seule source est insuffisante."""
        result = validator.compare_sources({'source1': pd.DataFrame()})
        
        assert result['status'] == 'insufficient_sources'
    
    def test_inconsistent_sources_detected(self, validator):
        """Sources incohérentes doivent être détectées."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2024-01-01', periods=n, freq='1h')
        
        df1 = pd.DataFrame({'Close': [100] * n}, index=dates)
        df2 = pd.DataFrame({'Close': [110] * n}, index=dates)  # 10% diff
        
        result = validator.compare_sources({'source1': df1, 'source2': df2})
        
        assert result['status'] == 'completed'
        assert not result['all_consistent']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
