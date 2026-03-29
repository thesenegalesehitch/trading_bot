"""
Routeur de gestion du risque et calcul du VaR (Value at Risk).
"""

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Body

from quantum.infrastructure.api.core.deps import get_current_user
from quantum.infrastructure.db.models import User
from quantum.domain.risk.var_calculator import VaRCalculator

router = APIRouter()
var_calculator = VaRCalculator()

@router.post("/var")
async def calculate_portfolio_var(
    portfolio: Dict[str, float] = Body(
        ..., 
        example={"EURUSD=X": 10000.0, "BTC-USD": 5000.0},
        description="Dictionnaire des positions {Symbole: Montant en USD}"
    ),
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Calcule la Value at Risk (VaR) d'un portefeuille via 3 méthodes:
    Historique, Paramétrique, et Monte Carlo.
    """
    if not portfolio:
        raise HTTPException(status_code=400, detail="Le portefeuille est vide.")
        
    try:
        results = var_calculator.calculate_combined_var(portfolio)
        return {
            "portfolio_value": sum(portfolio.values()),
            "var_results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
