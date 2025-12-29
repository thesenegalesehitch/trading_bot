"""
FastAPI application for Quantum Trading System.

Provides REST API endpoints for signals, scanning, risk management, ML predictions,
market correlations, sentiment analysis, and real-time subscriptions.
"""

import sys
import os
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import config
from ml.service import ml_service
from db.cache import get_cache_manager
from data.data_sources import get_historical_data
from reporting.scan_coordinator import ScanCoordinator
from api.auth import get_current_active_user, get_admin_user, check_api_key, rate_limit_request

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Création de l'application FastAPI
app = FastAPI(
    title="Quantum Trading System API",
    description="Enterprise-grade trading platform with AI-powered signals and risk management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À configurer selon les besoins en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sécurité
security = HTTPBearer()

# Rate limiting basique (à améliorer avec Redis en production)
request_counts = {}

# Modèles Pydantic
class SignalRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g., 'EURUSD=X')")
    features: Optional[Dict] = Field(None, description="Optional custom features")

class ScanRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of symbols to scan")
    timeframe: str = Field("1h", description="Timeframe for analysis")

class RiskAnalysisRequest(BaseModel):
    portfolio: Dict = Field(..., description="Portfolio composition")
    confidence_level: float = Field(0.95, description="VaR confidence level")

class MLPredictRequest(BaseModel):
    features: Dict = Field(..., description="Features for ML prediction")

class CorrelationRequest(BaseModel):
    symbols: List[str] = Field(..., description="Symbols for correlation analysis")
    window: int = Field(252, description="Rolling window in days")

class SentimentRequest(BaseModel):
    symbol: str = Field(..., description="Symbol for sentiment analysis")
    sources: List[str] = Field(["news", "twitter"], description="Sentiment sources")

# Dépendances d'authentification et rate limiting
def rate_limit(request: Request):
    """Rate limiting avancé avec Redis."""
    client_ip = request.client.host if request.client else "unknown"
    endpoint = request.url.path

    if not rate_limit_request(client_ip, endpoint):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

# Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Quantum Trading System API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/signals/{symbol}")
async def get_signals(
    symbol: str,
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """Get trading signals for a specific symbol."""
    rate_limit(request)

    try:
        # Récupérer les données récentes
        data = get_historical_data(symbol, period="2d", interval="1h")
        if data.empty:
            raise HTTPException(status_code=404, detail="Symbol data not found")

        # Extraire les dernières features
        latest_data = data.tail(1)
        features = {
            'zscore': latest_data['zscore'].iloc[-1] if 'zscore' in latest_data else 0,
            'hurst': latest_data['hurst'].iloc[-1] if 'hurst' in latest_data else 0.5,
            'rsi': latest_data['rsi'].iloc[-1] if 'rsi' in latest_data else 50,
            'macd': latest_data['macd'].iloc[-1] if 'macd' in latest_data else 0,
            'atr': latest_data['atr'].iloc[-1] if 'atr' in latest_data else 0.01,
            'volume_ratio': latest_data['volume_ratio'].iloc[-1] if 'volume_ratio' in latest_data else 1.0
        }

        # Prédiction ML
        signal = ml_service.predict_signal(features)

        # Ajouter métadonnées
        signal.update({
            "symbol": symbol,
            "data_points": len(data),
            "last_update": latest_data.index[-1].isoformat()
        })

        return signal

    except Exception as e:
        logger.error(f"Error getting signals for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/scan")
async def scan_symbols(
    scan_request: ScanRequest,
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """Multi-asset signal scanning."""
    rate_limit(request)

    try:
        # Utiliser le ScanCoordinator existant
        scanner = ScanCoordinator()
        results = scanner.scan_symbols(scan_request.symbols)

        return {
            "scan_results": results,
            "symbols_scanned": len(scan_request.symbols),
            "timeframe": scan_request.timeframe,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error scanning symbols: {e}")
        raise HTTPException(status_code=500, detail="Scan failed")

@app.post("/api/v1/risk/portfolio")
async def analyze_portfolio_risk(
    risk_request: RiskAnalysisRequest,
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """Portfolio risk analysis with VaR calculations."""
    rate_limit(request)

    try:
        # Placeholder pour l'analyse de risque
        # À implémenter avec RiskEngine du PRD
        return {
            "portfolio_value": sum(risk_request.portfolio.values()),
            "var_95": 0.05,  # 5% VaR
            "var_99": 0.08,  # 8% VaR
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.12,
            "confidence_level": risk_request.confidence_level,
            "timestamp": datetime.now().isoformat(),
            "note": "Risk analysis implementation in progress"
        }

    except Exception as e:
        logger.error(f"Error analyzing portfolio risk: {e}")
        raise HTTPException(status_code=500, detail="Risk analysis failed")

@app.post("/api/v1/ml/predict")
async def ml_predict(
    ml_request: MLPredictRequest,
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """Direct ML prediction with custom features."""
    rate_limit(request)

    try:
        prediction = ml_service.predict_signal(ml_request.features)
        return prediction

    except Exception as e:
        logger.error(f"Error in ML prediction: {e}")
        raise HTTPException(status_code=500, detail="ML prediction failed")

@app.post("/api/v1/market/correlation")
async def get_market_correlations(
    correlation_request: CorrelationRequest,
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """Calculate cross-market correlations."""
    rate_limit(request)

    try:
        # Vérifier le cache
        cache = get_cache_manager()
        cache_key = f"correlations:{','.join(correlation_request.symbols)}:{correlation_request.window}"

        cached_result = cache.get_correlations()
        if cached_result:
            return {
                "correlations": cached_result,
                "cached": True,
                "timestamp": datetime.now().isoformat()
            }

        # Calcul des corrélations (placeholder)
        # À implémenter avec InterMarketAnalyzer du PRD
        correlations = {}
        for i, symbol1 in enumerate(correlation_request.symbols):
            correlations[symbol1] = {}
            for symbol2 in correlation_request.symbols[i+1:]:
                # Calcul de corrélation simplifié
                correlations[symbol1][symbol2] = 0.5  # Placeholder

        # Cacher le résultat
        cache.set_correlations(correlations)

        return {
            "correlations": correlations,
            "symbols": correlation_request.symbols,
            "window": correlation_request.window,
            "cached": False,
            "timestamp": datetime.now().isoformat(),
            "note": "Correlation analysis implementation in progress"
        }

    except Exception as e:
        logger.error(f"Error calculating correlations: {e}")
        raise HTTPException(status_code=500, detail="Correlation calculation failed")

@app.post("/api/v1/sentiment/news")
async def get_sentiment_analysis(
    sentiment_request: SentimentRequest,
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """News and social sentiment analysis."""
    rate_limit(request)

    try:
        # Placeholder pour l'analyse de sentiment
        # À implémenter avec SentimentAnalyzer du PRD
        return {
            "symbol": sentiment_request.symbol,
            "sentiment_score": 0.2,  # -1 to +1 scale
            "sources": sentiment_request.sources,
            "confidence": 0.75,
            "timestamp": datetime.now().isoformat(),
            "note": "Sentiment analysis implementation in progress"
        }

    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail="Sentiment analysis failed")

@app.get("/api/v1/realtime/subscribe/{symbol}")
async def subscribe_realtime(
    symbol: str,
    request: Request,
    api_key: str = Depends(get_api_key)
):
    """WebSocket subscription info for real-time data."""
    rate_limit(request)

    # Retourner les informations de connexion WebSocket
    # En production, retourner l'URL WebSocket et les credentials
    return {
        "symbol": symbol,
        "websocket_url": f"ws://localhost:8000/ws/{symbol}",
        "status": "subscription_info",
        "note": "Real-time WebSocket implementation in progress"
    }

@app.get("/api/v1/health")
async def health_check():
    """Detailed health check."""
    cache = get_cache_manager()
    cache_stats = cache.get_stats()

    return {
        "status": "healthy",
        "services": {
            "ml_service": "online" if ml_service.models else "offline",
            "cache": cache_stats["status"],
            "database": "unknown"  # À implémenter avec vérification DB
        },
        "timestamp": datetime.now().isoformat()
    }

# Gestionnaire d'erreurs global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )