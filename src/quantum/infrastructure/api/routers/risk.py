from typing import Any
from fastapi import APIRouter, Depends
import numpy as np
from pydantic import BaseModel

router = APIRouter()

class PositionSizeRequest(BaseModel):
    capital: float
    risk_percent: float
    entry_price: float
    stop_loss: float

class KellyRequest(BaseModel):
    win_rate: float
    win_loss_ratio: float

@router.post("/position-size")
async def calculate_position_size(req: PositionSizeRequest):
    risk_amount = req.capital * (req.risk_percent / 100)
    risk_per_unit = abs(req.entry_price - req.stop_loss)
    if risk_per_unit == 0: return {"size": 0}
    size = risk_amount / risk_per_unit
    return {
        "quantity": round(size, 4),
        "risk_amount": round(risk_amount, 2),
        "notional_value": round(size * req.entry_price, 2)
    }

@router.post("/kelly")
async def calculate_kelly(req: KellyRequest):
    # Kelly % = W - [(1 - W) / R]
    # W = Win rate, R = Win/Loss ratio
    if req.win_loss_ratio == 0: return {"kelly_percent": 0}
    kelly = req.win_rate - ((1 - req.win_rate) / req.win_loss_ratio)
    return {
        "kelly_percent": round(max(0, kelly) * 100, 2),
        "suggested_risk": round(max(0, kelly / 2) * 100, 2) # Half-Kelly safe
    }

@router.get("/var-info")
async def get_var_info():
    return {
        "description": "La Value at Risk (VaR) mesure la perte potentielle maximale sur un horizon de temps donné avec un niveau de confiance spécifique.",
        "levels": [0.95, 0.99],
        "methods": ["Historique", "Paramétrique", "Monte Carlo"]
    }
