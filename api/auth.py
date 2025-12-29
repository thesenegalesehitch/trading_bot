"""
Authentication and authorization for Quantum Trading API.

Implements JWT-based authentication with role-based access control.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import config

# Configuration JWT
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "quantum-trading-secret-key-2025")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Contexte de hashage des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Sécurité HTTP Bearer
security = HTTPBearer()

# Modèles Pydantic
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    role: str = "user"

class UserInDB(User):
    hashed_password: str

# Utilisateurs de démonstration (en production, utiliser une base de données)
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Quantum Admin",
        "email": "admin@quantumtrading.com",
        "hashed_password": pwd_context.hash("admin123"),
        "disabled": False,
        "role": "admin"
    },
    "trader": {
        "username": "trader",
        "full_name": "Demo Trader",
        "email": "trader@quantumtrading.com",
        "hashed_password": pwd_context.hash("trader123"),
        "disabled": False,
        "role": "trader"
    }
}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifie un mot de passe."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash un mot de passe."""
    return pwd_context.hash(password)

def get_user(db, username: str) -> Optional[UserInDB]:
    """Récupère un utilisateur de la base de données."""
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(fake_db, username: str, password: str) -> Optional[User]:
    """Authentifie un utilisateur."""
    user = get_user(fake_db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return User(**user.dict())

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Crée un token d'accès JWT."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Récupère l'utilisateur actuel depuis le token JWT."""
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=role)
    except JWTError:
        raise credentials_exception

    user = get_user(fake_users_db, username)
    if user is None:
        raise credentials_exception
    return User(**user.dict())

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Vérifie que l'utilisateur est actif."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_admin_user(current_user: User = Depends(get_current_active_user)):
    """Vérifie que l'utilisateur a les droits admin."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Not enough permissions"
        )
    return current_user

def check_api_key(api_key: str) -> bool:
    """Vérifie une clé API simple."""
    return api_key == config.system.API_KEY

# Rate limiting avec Redis (si disponible)
try:
    from db.cache import get_cache_manager
    cache = get_cache_manager()

    def rate_limit_request(client_ip: str, endpoint: str, limit: int = 100, window: int = 60) -> bool:
        """Rate limiting avec Redis."""
        if not cache.client:
            return True  # Pas de limitation si Redis indisponible

        key = f"rate_limit:{client_ip}:{endpoint}"
        current = cache.client.get(key)

        if current is None:
            cache.client.setex(key, window, 1)
            return True

        current = int(current)
        if current >= limit:
            return False

        cache.client.incr(key)
        return True

except ImportError:
    def rate_limit_request(client_ip: str, endpoint: str, limit: int = 100, window: int = 60) -> bool:
        """Rate limiting basique sans Redis."""
        # Implémentation simplifiée
        return True