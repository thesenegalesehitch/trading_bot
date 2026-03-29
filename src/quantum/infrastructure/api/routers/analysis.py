"""
Routeur d'analyse (ICT, SMC, Wyckoff). Connecte l'API aux moteurs métier puissants.
"""

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Query

from quantum.infrastructure.api.core.deps import get_current_user
from quantum.infrastructure.db.models import User

from quantum.domain.data.downloader import DataDownloader
from quantum.domain.analysis.ict_full_setup import ICTFullSetupDetector
from quantum.domain.analysis.smc import SmartMoneyConceptsAnalyzer
from quantum.domain.analysis.wyckoff import WyckoffAnalyzer

router = APIRouter()
downloader = DataDownloader()

# Constructeurs sans dépendances lourdes
ict_detector = ICTFullSetupDetector()
smc_analyzer = SmartMoneyConceptsAnalyzer()
wyckoff_analyzer = WyckoffAnalyzer()

@router.get("/ict/{symbol}")
async def get_ict_setups(
    symbol: str,
    timeframe: str = Query("15m", description="Timeframe d'analyse"),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Analyse ICT Full Setup (Sweep, FVG, MSS...)."""
    df = downloader.get_data(symbol, period="1mo", interval=timeframe)
    if df.empty:
        raise HTTPException(status_code=404, detail="Données insuffisantes")
        
    trades = ict_detector.detect_full_setup(df, symbol, timeframe)
    
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "setup_count": len(trades),
        "setups": [
            {
                "direction": t.direction,
                "confidence": t.confidence,
                "entry": t.ifvg_entry.entry_price,
                "stop_loss": t.ifvg_entry.stop_loss,
                "target": t.ifvg_entry.target_1,
                "risk_reward": t.ifvg_entry.risk_reward,
                "killzone": t.killzone,
            } for t in trades
        ]
    }

@router.get("/smc/{symbol}")
async def get_smc_analysis(
    symbol: str, timeframe: str = Query("1h"), current_user: User = Depends(get_current_user)
) -> Any:
    """Analyse Smart Money Concepts (FVG, Order Blocks)."""
    df = downloader.get_data(symbol, period="6mo", interval=timeframe)
    if df.empty:
        raise HTTPException(status_code=404)
        
    analysis = smc_analyzer.analyze(df)
    return analysis

@router.get("/wyckoff/{symbol}")
async def get_wyckoff_analysis(
    symbol: str, timeframe: str = Query("1d"), current_user: User = Depends(get_current_user)
) -> Any:
    """Analyse de Phase Wyckoff (Accumulation/Distribution)."""
    df = downloader.get_data(symbol, period="2y", interval=timeframe)
    if df.empty:
        raise HTTPException(status_code=404)
        
    result = wyckoff_analyzer.analyze(df)
    return {
        "phase": result.phase.value,
        "events": [
            {"event": e.event_type.value, "index": int(e.index), "price": float(e.price)} 
            for e in result.events[-5:]  # Retourne les 5 derniers évènements
        ],
        "signals": [s.name for s in result.signals]
    }
