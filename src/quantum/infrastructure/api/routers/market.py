"""
Routeur de données de marché (klines, ticker).
"""

from typing import Any, List, Dict
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel

from quantum.infrastructure.api.core.deps import get_current_user
from quantum.infrastructure.db.models import User
from quantum.domain.data.downloader import DataDownloader
from quantum.domain.data.feature_engine import FeatureEngine

router = APIRouter()
downloader = DataDownloader()
feature_engine = FeatureEngine()

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
    current_user: User = Depends(get_current_user)
) -> Any:
    """Récupère les données historiques OHLCV d'un symbole."""
    try:
        # Période par défaut pour l'API
        period = "7d" if interval in ["1m", "5m", "15m"] else "1mo"
        
        df = downloader.get_data(symbol, period=period, interval=interval)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Données introuvables pour {symbol}")
            
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
