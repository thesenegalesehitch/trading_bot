from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from quantum.infrastructure.api.core.deps import get_current_user, get_db
from quantum.infrastructure.db.models import User, TradingJournal, Trade
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

router = APIRouter()

class JournalCreate(BaseModel):
    trade_id: Optional[int] = None
    mood: str
    strategy_name: Optional[str] = None
    notes: str
    lessons_learned: Optional[str] = None

@router.post("/")
async def create_journal_entry(
    entry: JournalCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    new_entry = TradingJournal(
        user_id=current_user.id,
        trade_id=entry.trade_id,
        mood=entry.mood,
        strategy_name=entry.strategy_name,
        notes=entry.notes,
        lessons_learned=entry.lessons_learned
    )
    db.add(new_entry)
    await db.commit()
    await db.refresh(new_entry)
    return new_entry

@router.get("/", response_model=List[dict])
async def get_my_journal(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(TradingJournal).filter(TradingJournal.user_id == current_user.id).order_by(TradingJournal.timestamp.desc())
    )
    return [
        {
            "id": j.id,
            "timestamp": j.timestamp,
            "mood": j.mood,
            "notes": j.notes,
            "strategy": j.strategy_name
        } for j in result.scalars().all()
    ]
