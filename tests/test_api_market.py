"""
Tests pour le routeur Market Data (Klines, Indicateurs).
"""

import pytest
import pandas as pd
from fastapi.testclient import TestClient

from quantum.infrastructure.api.main import app
from quantum.infrastructure.api.core.deps import get_current_user
from quantum.infrastructure.db.models import User

# ---- Mocks pour éviter les requêtes réseau (yfinance) ----
def mock_get_current_user():
    return User(id=1, email="test@quantum.com", full_name="Test User", is_active=True)

app.dependency_overrides[get_current_user] = mock_get_current_user

class MockDataDownloader:
    def get_data(self, symbol, period="1mo", interval="1h"):
        if symbol == "INVALID":
            return pd.DataFrame()
            
        # Simuler un DataFrame OHLCV
        dates = pd.date_range("2026-01-01", periods=10, freq="1h")
        data = {
            'Open': [100]*10,
            'High': [102]*10,
            'Low': [98]*10,
            'Close': [101]*10,
            'Volume': [1000]*10
        }
        return pd.DataFrame(data, index=dates)

class MockFeatureEngine:
    def create_all_features(self, df):
        df_feats = df.copy()
        df_feats['RSI'] = [50] * 10
        df_feats['MACD'] = [0.1] * 10
        return df_feats

# Appliquer le monkeypatch au module market avant son import
import quantum.infrastructure.api.routers.market as market_module
market_module.downloader = MockDataDownloader()
market_module.feature_engine = MockFeatureEngine()

client = TestClient(app)

def test_get_klines_success():
    response = client.get("/api/v1/market/klines/BTC-USD?interval=1h")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 10
    
    first_candle = data[0]
    assert "timestamp" in first_candle
    assert first_candle["open"] == 100.0
    assert first_candle["close"] == 101.0

def test_get_klines_not_found():
    response = client.get("/api/v1/market/klines/INVALID?interval=1h")
    assert response.status_code == 404
    assert "introuvables" in response.json()["detail"].lower()

def test_get_indicators_success():
    response = client.get("/api/v1/market/indicators/BTC-USD?interval=1d")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "RSI" in data
    assert "MACD" in data
    assert data["RSI"] == 50.0

def test_get_indicators_not_found():
    response = client.get("/api/v1/market/indicators/INVALID?interval=1d")
    assert response.status_code == 404
    assert "pas de données" in response.json()["detail"].lower()
