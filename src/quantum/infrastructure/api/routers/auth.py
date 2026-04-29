"""
Routeur d'authentification (Register, Login, Profil).
"""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel, EmailStr

from quantum.infrastructure.db.session import get_db
from quantum.infrastructure.db.models import User, Account, AccountHistory
from quantum.infrastructure.api.core import security
from quantum.infrastructure.api.core.deps import get_current_user

router = APIRouter()

# --- Schémas Pydantic locaux ---
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class UserResponse(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    is_active: bool
    is_superuser: bool

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_in: UserCreate, db: AsyncSession = Depends(get_db)) -> Any:
    """Créer un nouveau compte utilisateur avec un compte de trading démo."""
    # Vérifier si l'email existe
    result = await db.execute(select(User).filter(User.email == user_in.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=400,
            detail="Un utilisateur avec cet email existe déjà.",
        )
    
    # Créer l'utilisateur
    user = User(
        email=user_in.email,
        hashed_password=security.get_password_hash(user_in.password),
        full_name=user_in.full_name,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    # Créer le compte de trading démo par défaut ($1M)
    demo_account = Account(
        user_id=user.id,
        account_type="DEMO",
        balance=1_000_000.0,
        currency="USD"
    )
    db.add(demo_account)
    await db.commit()
    await db.refresh(demo_account)
    
    # Historique initial
    history = AccountHistory(account_id=demo_account.id, balance=demo_account.balance)
    db.add(history)
    await db.commit()
    
    return user


@router.post("/login", response_model=Token)
async def login(
    db: AsyncSession = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    """Connexion avec Email (username) et Mot de passe. Retourne le JWT."""
    result = await db.execute(select(User).filter(User.email == form_data.username))
    user = result.scalar_one_or_none()
    
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou mot de passe incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Utilisateur inactif")
    
    # Générer le token JWT
    access_token = security.create_access_token(subject=str(user.id))
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    current_user: User = Depends(get_current_user)
) -> Any:
    """Récupère les informations du profil de l'utilisateur connecté."""
    return current_user
