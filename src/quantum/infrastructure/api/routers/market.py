"""
Routeur de données de marché (klines, ticker).
"""

from typing import Any, List, Dict
from fastapi import APIRouter, Depends, Query, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import asyncio
import json
import pandas as pd

from quantum.infrastructure.api.core.deps import get_current_user
from quantum.infrastructure.db.models import User
from quantum.domain.data.downloader import DataDownloader
from quantum.domain.data.feature_engine import FeatureEngine

router = APIRouter()
downloader = DataDownloader()
feature_engine = FeatureEngine()

# Active connections for WebSocket streaming
active_connections: List[WebSocket] = []

# Model
class KlineData(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

@router.get("/klines/{symbol}", response_model=List[KlineData])
async def get_klines(
    symbol: str, 
    interval: str = Query("1h", description="1m, 5m, 15m, 1h, 1d"),
    end_time: str = Query(None, description="ISO format date pour simulation/replay"),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Récupère les données historiques OHLCV d'un symbole."""
    try:
        # Période par défaut pour l'API
        period = "7d" if interval in ["1m", "5m", "15m"] else "1mo"
        
        df = downloader.get_data(symbol, period=period, interval=interval)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Données introuvables pour {symbol}")
            
        # Filtre pour le replay/simulation
        if end_time:
            try:
                import pandas as pd
                cutoff = pd.to_datetime(end_time).tz_localize(df.index.tz) if df.index.tz else pd.to_datetime(end_time)
                df = df[df.index <= cutoff]
            except Exception as e:
                logger.warning(f"Erreur format end_time: {e}")
            
        # Convertir en liste de dictionnaires
        df_reset = df.reset_index()
        # Gérer le nom de l'index textuel 'Date' vs 'Datetime'
        time_col = 'Datetime' if 'Datetime' in df_reset.columns else 'Date'
        if time_col not in df_reset.columns and 'index' in df_reset.columns:
            time_col = 'index'
            
        results = []
        for _, row in df_reset.iterrows():
            results.append(KlineData(
                timestamp=str(row[time_col]),
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=float(row['Volume'])
            ))
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/indicators/{symbol}")
async def get_indicators(
    symbol: str,
    interval: str = Query("1d", description="Intervalle"),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Récupère les indicateurs techniques calculés (RSI, MACD...)."""
    df = downloader.get_data(symbol, period="1y", interval=interval)
    if df.empty:
        raise HTTPException(status_code=404, detail="Pas de données")
        
    # Applique le FeatureEngine métier !
    df_features = feature_engine.create_all_features(df)
    
    # Retourne la dernière ligne d'indicateurs
    last_row = df_features.iloc[-1].fillna(0).to_dict()
    # Filtre pour garder les colonnes pertinentes
    indicators = {k: float(v) for k, v in last_row.items() if isinstance(v, (int, float))}
    return indicators

@router.websocket("/stream/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    """
    Endpoint WebSocket pour le streaming de prix en temps réel.
    Pour l'instant, simule un flux de ticks aléatoires autour du dernier prix connu.
    """
    await websocket.accept()
    active_connections.append(websocket)
    try:
        # Récupérer le dernier prix (simulation)
        df = downloader.get_data(symbol, period="1d", interval="1m")
        last_price = float(df['Close'].iloc[-1]) if not df.empty else 100.0
        
        while True:
            # Simulation d'un tick (±0.1%)
            import random
            variation = last_price * random.uniform(-0.001, 0.001)
            last_price += variation
            
            data = {
                "symbol": symbol,
                "price": round(last_price, 2),
                "timestamp": pd.Timestamp.now(tz='UTC').isoformat()
            }
            
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(1)  # Tick chaque seconde
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)
