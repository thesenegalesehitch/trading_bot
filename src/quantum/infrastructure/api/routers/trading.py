"""
Routeur de gestion de trading et portefeuille.
"""

from typing import Any, List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from quantum.infrastructure.api.core.deps import get_current_user, get_db
from quantum.infrastructure.db.models import User, Account, Trade, Symbol
from quantum.domain.data.downloader import DataDownloader

router = APIRouter()
downloader = DataDownloader()

class TradeCreate(BaseModel):
    symbol: str
    side: str  # BUY, SELL
    quantity: float
    stop_loss: float = None
    take_profit: float = None

class TradeResponse(BaseModel):
    id: int
    symbol: str
    side: str
    quantity: float
    price: float
    status: str
    pnl: float = None

    class Config:
        from_attributes = True

async def get_symbol_id(db: AsyncSession, symbol_str: str) -> int:
    """Helper pour récupérer ou créer un symbole."""
    result = await db.execute(select(Symbol).filter(Symbol.symbol == symbol_str))
    symbol = result.scalar_one_or_none()
    if not symbol:
        symbol = Symbol(symbol=symbol_str)
        db.add(symbol)
        await db.commit()
        await db.refresh(symbol)
    return symbol.id

@router.post("/open", response_model=TradeResponse, status_code=status.HTTP_201_CREATED)
async def open_trade(
    trade_in: TradeCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Ouvrir une nouvelle position (Paper Trading par défaut sur l'API)."""
    # Récupérer le compte DEMO de l'utilisateur
    result = await db.execute(
        select(Account).filter(Account.user_id == current_user.id, Account.account_type == "DEMO")
    )
    account = result.scalar_one_or_none()
    if not account:
        raise HTTPException(status_code=404, detail="Compte de trading introuvable.")

    # Obtenir le prix actuel via yfinance
    df = downloader.get_data(trade_in.symbol, period="1d", interval="1m")
    if df.empty:
        raise HTTPException(status_code=400, detail="Impossible d'obtenir le prix actuel.")
    current_price = float(df.iloc[-1]['Close'])

    cost = current_price * trade_in.quantity
    if account.balance < cost and trade_in.side == "BUY":
        raise HTTPException(status_code=400, detail="Solde insuffisant.")

    # Mettre à jour le solde (simplifié sans marge/effet de levier)
    if trade_in.side == "BUY":
        account.balance -= cost

    symbol_id = await get_symbol_id(db, trade_in.symbol)

    new_trade = Trade(
        account_id=account.id,
        symbol_id=symbol_id,
        side=trade_in.side,
        quantity=trade_in.quantity,
        price=current_price,
        timestamp=datetime.utcnow(),
        stop_loss=trade_in.stop_loss,
        take_profit=trade_in.take_profit,
        status="OPEN"
    )
    db.add(new_trade)
    await db.commit()
    await db.refresh(new_trade)
    
    return {
        "id": new_trade.id,
        "symbol": trade_in.symbol,
        "side": new_trade.side,
        "quantity": new_trade.quantity,
        "price": new_trade.price,
        "status": new_trade.status,
    }


@router.post("/close/{trade_id}", response_model=TradeResponse)
async def close_trade(
    trade_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Fermer une position ouverte."""
    # Obtenir le trade
    result = await db.execute(
        select(Trade, Account, Symbol)
        .join(Account, Trade.account_id == Account.id)
        .join(Symbol, Trade.symbol_id == Symbol.id)
        .filter(Trade.id == trade_id, Account.user_id == current_user.id)
    )
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Trade introuvable.")
        
    trade, account, symbol = row
    if trade.status != "OPEN":
        raise HTTPException(status_code=400, detail="Ce trade est déjà fermé.")

    # Obtenir prix actuel
    df = downloader.get_data(symbol.symbol, period="1d", interval="1m")
    if df.empty:
        raise HTTPException(status_code=400, detail="Impossible d'obtenir le prix actuel.")
    current_price = float(df.iloc[-1]['Close'])

    # Calcul PnL (simplifié)
    if trade.side == "BUY":
        pnl = (current_price - trade.price) * trade.quantity
        account.balance += (trade.price * trade.quantity) + pnl  # Rembourse l'achat + PnL
    else:
        pnl = (trade.price - current_price) * trade.quantity
        account.balance += pnl  # Juste le profit/perte s'ajoute au solde démo

    trade.exit_price = current_price
    trade.exit_timestamp = datetime.utcnow()
    trade.pnl = pnl
    trade.pnl_percent = (pnl / (trade.price * trade.quantity)) * 100
    trade.status = "CLOSED"

    await db.commit()
    await db.refresh(trade)

    return {
        "id": trade.id,
        "symbol": symbol.symbol,
        "side": trade.side,
        "quantity": trade.quantity,
        "price": trade.price,
        "status": trade.status,
        "pnl": pnl
    }


@router.get("/positions", response_model=List[TradeResponse])
async def get_open_positions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Any:
    """Lister les positions ouvertes du compte démo."""
    result = await db.execute(
        select(Trade, Symbol)
        .join(Account, Trade.account_id == Account.id)
        .join(Symbol, Trade.symbol_id == Symbol.id)
        .filter(Account.user_id == current_user.id, Trade.status == "OPEN")
    )
    rows = result.all()
    
    positions = []
    for trade, symbol in rows:
        positions.append({
            "id": trade.id,
            "symbol": symbol.symbol,
            "side": trade.side,
            "quantity": trade.quantity,
            "price": trade.price,
            "status": trade.status,
        })
    return positions
