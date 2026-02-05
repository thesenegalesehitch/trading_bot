"""
Tests pour le module ICT Full Setup Detector.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestKillZoneAnalyzer:
    """Tests pour KillZoneAnalyzer."""
    
    def test_london_killzone(self):
        """Test détection London Killzone."""
        from quantum.domain.analysis.ict_full_setup import KillZoneAnalyzer
        
        # 9h UTC
        dt = datetime(2024, 1, 15, 9, 0, 0)
        assert KillZoneAnalyzer.get_current_killzone(dt) == "LONDON"
        
        # 10h30 UTC
        dt = datetime(2024, 1, 15, 10, 30, 0)
        assert KillZoneAnalyzer.get_current_killzone(dt) == "LONDON"
    
    def test_ny_killzone(self):
        """Test détection NY Killzone."""
        from quantum.domain.analysis.ict_full_setup import KillZoneAnalyzer
        
        # 14h UTC
        dt = datetime(2024, 1, 15, 14, 0, 0)
        assert KillZoneAnalyzer.get_current_killzone(dt) == "NY"
        
        # 15h45 UTC
        dt = datetime(2024, 1, 15, 15, 45, 0)
        assert KillZoneAnalyzer.get_current_killzone(dt) == "NY"
    
    def test_outside_killzone(self):
        """Test hors killzone."""
        from quantum.domain.analysis.ict_full_setup import KillZoneAnalyzer
        
        # 12h UTC (entre London et NY)
        dt = datetime(2024, 1, 15, 12, 0, 0)
        assert KillZoneAnalyzer.get_current_killzone(dt) is None
        
        # 17h UTC (après NY)
        dt = datetime(2024, 1, 15, 17, 0, 0)
        assert KillZoneAnalyzer.get_current_killzone(dt) is None
    
    def test_is_in_killzone(self):
        """Test is_in_killzone."""
        from quantum.domain.analysis.ict_full_setup import KillZoneAnalyzer
        
        dt_in = datetime(2024, 1, 15, 9, 0, 0)
        dt_out = datetime(2024, 1, 15, 12, 0, 0)
        
        assert KillZoneAnalyzer.is_in_killzone(dt_in) is True
        assert KillZoneAnalyzer.is_in_killzone(dt_out) is False


class TestVolumeSpikeDetector:
    """Tests pour VolumeSpikeDetector."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        n = 50
        
        self.df = pd.DataFrame({
            'Open': 100 + np.random.randn(n) * 0.5,
            'High': 100 + np.random.randn(n) * 0.5 + 1,
            'Low': 100 + np.random.randn(n) * 0.5 - 1,
            'Close': 100 + np.random.randn(n) * 0.5,
            'Volume': np.random.randint(1000, 5000, n)
        }, index=pd.date_range('2024-01-01', periods=n, freq='15min'))
    
    def test_avg_volume_calculation(self):
        """Test calcul volume moyen."""
        from quantum.domain.analysis.ict_full_setup import VolumeSpikeDetector
        
        detector = VolumeSpikeDetector(lookback=10)
        avg = detector.calculate_avg_volume(self.df)
        
        assert avg > 0
        assert isinstance(avg, (int, float))
    
    def test_no_volume_spike(self):
        """Test absence de volume spike."""
        from quantum.domain.analysis.ict_full_setup import VolumeSpikeDetector
        
        detector = VolumeSpikeDetector(lookback=10, spike_multiplier=1.5)
        is_spike, ratio = detector.is_volume_spike(self.df)
        
        # La fonction retourne un tuple (bool, float)
        assert isinstance(is_spike, (bool, np.bool_)) or isinstance(is_spike, np.bool_)
        assert isinstance(ratio, (int, float))
        assert ratio >= 0
        # Pas de spike car le volume n'est pas assez élevé
        assert is_spike == False or ratio < 1.5
    
    def test_volume_spike_detection(self):
        """Test détection volume spike."""
        from quantum.domain.analysis.ict_full_setup import VolumeSpikeDetector
        
        # Créer un volume spike
        self.df.iloc[-1, self.df.columns.get_loc('Volume')] = 15000
        
        detector = VolumeSpikeDetector(lookback=10, spike_multiplier=1.5)
        is_spike, ratio = detector.is_volume_spike(self.df)
        
        # Le spike doit être détecté
        assert is_spike == True
        assert ratio >= 1.5
    
    def test_volume_score(self):
        """Test score de volume."""
        from quantum.domain.analysis.ict_full_setup import VolumeSpikeDetector
        
        detector = VolumeSpikeDetector(lookback=10)
        score = detector.get_volume_score(self.df)
        
        assert 0 <= score <= 1


class TestLiquidityDetector:
    """Tests pour LiquidityDetector."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        n = 100
        
        self.df = pd.DataFrame({
            'Open': 100 + np.cumsum(np.random.randn(n) * 0.3),
            'High': 100 + np.cumsum(np.random.randn(n) * 0.3) + 1,
            'Low': 100 + np.cumsum(np.random.randn(n) * 0.3) - 1,
            'Close': 100 + np.cumsum(np.random.randn(n) * 0.3),
            'Volume': np.random.randint(1000, 5000, n)
        }, index=pd.date_range('2024-01-01', periods=n, freq='15min'))
    
    def test_session_levels(self):
        """Test niveaux de session."""
        from quantum.domain.analysis.ict_full_setup import LiquidityDetector
        
        detector = LiquidityDetector(session_hours=24)
        pdh, pdl, hod, lod, session_start = detector.get_session_levels(self.df)
        
        assert pdh >= lod  # PDH >= LOD (logique)
        assert pdl <= hod  # PDL <= HOD
        assert isinstance(session_start, datetime)
    
    def test_sweep_detection(self):
        """Test détection de sweeps."""
        from quantum.domain.analysis.ict_full_setup import LiquidityDetector
        
        detector = LiquidityDetector(session_hours=24)
        pdh, pdl, hod, lod, _ = detector.get_session_levels(self.df)
        
        sweeps = detector.detect_sweeps(self.df, pdh, pdl, hod, lod)
        
        assert isinstance(sweeps, list)
        # Peut être vide si pas de sweep dans les données


class TestICTFullSetupDetector:
    """Tests pour ICTFullSetupDetector."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        n = 200
        
        # Créer un mouvement avec structure
        base = np.cumsum(np.random.randn(n) * 0.5)
        
        self.df = pd.DataFrame({
            'Open': 100 + base + np.random.randn(n) * 0.2,
            'High': 100 + base + np.random.randn(n) * 0.3 + 0.5,
            'Low': 100 + base + np.random.randn(n) * 0.3 - 0.5,
            'Close': 100 + base + np.random.randn(n) * 0.2,
            'Volume': np.random.randint(1000, 10000, n)
        }, index=pd.date_range('2024-01-15', periods=n, freq='15min'))
    
    def test_detector_initialization(self):
        """Test initialisation du detector."""
        from quantum.domain.analysis.ict_full_setup import ICTFullSetupDetector
        
        detector = ICTFullSetupDetector(
            session_hours=24,
            min_rr=2.0,
            volume_spike_multiplier=1.5
        )
        
        assert detector.session_hours == 24
        assert detector.min_rr == 2.0
    
    def test_detect_full_setup_empty_outside_killzone(self):
        """Test que pas de signal hors killzone."""
        from quantum.domain.analysis.ict_full_setup import ICTFullSetupDetector
        from unittest.mock import patch
        
        detector = ICTFullSetupDetector(min_rr=2.0)
        
        # Mock KillZoneAnalyzer pour simuler hors killzone
        with patch('quantum.domain.analysis.ict_full_setup.KillZoneAnalyzer.get_current_killzone') as mock_killzone:
            mock_killzone.return_value = None
            
            trades = detector.detect_full_setup(self.df, "BTCUSDT", "15m")
            
            assert trades == []
    
    def test_detect_full_setup_in_killzone(self):
        """Test détection en killzone."""
        from quantum.domain.analysis.ict_full_setup import ICTFullSetupDetector
        from unittest.mock import patch
        
        detector = ICTFullSetupDetector(min_rr=2.0)
        
        # Mock KillZoneAnalyzer pour simuler en killzone
        with patch('quantum.domain.analysis.ict_full_setup.KillZoneAnalyzer.get_current_killzone') as mock_killzone:
            mock_killzone.return_value = "LONDON"
            
            trades = detector.detect_full_setup(self.df, "BTCUSDT", "15m")
            
            assert isinstance(trades, list)


class TestICTAlertFormatter:
    """Tests pour ICTAlertFormatter."""
    
    def test_discord_embed_format(self):
        """Test format Discord embed."""
        from quantum.domain.analysis.ict_full_setup import ICTAlertFormatter, FullSetupTrade
        
        # Mock trade data
        trade_data = {
            "setup_id": "TEST_123",
            "symbol": "BTCUSDT",
            "direction": "BUY",
            "killzone": "LONDON",
            "entry": 50000.0,
            "stop_loss": 49500.0,
            "take_profits": [51000.0, 52000.0, 53000.0],
            "risk_reward": 2.5,
            "confidence": 85.0,
            "volume_spike": True,
            "detected_at": "2024-01-15T10:00:00",
            "timeframe": "15m",
            "sequence": {
                "swept_level": "LOD",
                "htf_fvg_tap": "H4 FVG",
                "mss_type": "BULLISH"
            }
        }
        
        embed = ICTAlertFormatter.format_discord_embed(trade_data)
        
        assert "title" in embed
        assert "color" in embed
        assert "fields" in embed
        assert "BUY" in embed["title"]
    
    def test_telegram_format(self):
        """Test format Telegram."""
        from quantum.domain.analysis.ict_full_setup import ICTAlertFormatter
        
        trade_data = {
            "symbol": "BTCUSDT",
            "direction": "SELL",
            "entry": 50000.0,
            "stop_loss": 50500.0,
            "take_profits": [49000.0, 48000.0, 47000.0],
            "risk_reward": 2.0,
            "confidence": 75.0,
            "volume_spike": True,
            "killzone": "NY",
            "timeframe": "1h",
            "setup_id": "TEST_456",
            "detected_at": datetime.now().isoformat(),  # Ajout detected_at
            "sequence": {
                "swept_level": "HOD",
                "htf_fvg_tap": "H4 FVG",
                "mss_type": "BEARISH"
            }
        }
        
        message = ICTAlertFormatter.format_telegram_message(trade_data)
        
        assert isinstance(message, str)
        assert "ICT FULL SETUP" in message
        assert "SELL" in message
        assert "BTCUSDT" in message


class TestDetectICTFullSetup:
    """Tests pour la fonction utilitaire detect_ict_full_setup."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        n = 100
        
        self.df = pd.DataFrame({
            'Open': 100 + np.random.randn(n) * 0.5,
            'High': 100 + np.random.randn(n) * 0.5 + 1,
            'Low': 100 + np.random.randn(n) * 0.5 - 1,
            'Close': 100 + np.random.randn(n) * 0.5,
            'Volume': np.random.randint(1000, 5000, n)
        }, index=pd.date_range('2024-01-01', periods=n, freq='15min'))
    
    def test_utility_function(self):
        """Test fonction utilitaire."""
        from quantum.domain.analysis.ict_full_setup import detect_ict_full_setup
        
        result = detect_ict_full_setup(self.df, "EURUSD", "15m", min_rr=2.0)
        
        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
