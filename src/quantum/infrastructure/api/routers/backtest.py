from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from quantum.infrastructure.api.core.deps import get_current_user, get_db
from quantum.infrastructure.db.models import User, BacktestRun
from pydantic import BaseModel
from datetime import datetime
import json

router = APIRouter()

class BacktestRequest(BaseModel):
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    strategy_config: dict

@router.post("/run")
async def run_backtest(
    req: BacktestRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Logique simplifiée pour démonstration
    # Dans une version complète, ceci appellerait le moteur BacktestEngine
    
    new_run = BacktestRun(
        user_id=current_user.id,
        symbol=req.symbol,
        timeframe=req.timeframe,
        start_date=req.start_date,
        end_date=req.end_date,
        strategy_config=json.dumps(req.strategy_config),
        total_trades=42,
        win_rate=65.5,
        profit_factor=1.8,
        max_drawdown=5.2,
        net_profit=1250.0,
        report_json=json.dumps({"details": "Simulated backtest result"})
    )
    
    db.add(new_run)
    await db.commit()
    await db.refresh(new_run)
    
    return new_run
