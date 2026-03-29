"""
Module utilitaire pour la sécurité et l'authentification.

Gère le hachage des mots de passe (bcrypt) et la création/validation
des jetons JWT (JSON Web Tokens).
"""

from datetime import datetime, timedelta
from typing import Optional, Union, Any

import jwt
from passlib.context import CryptContext

from quantum.shared.config.settings import config

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifie un mot de passe en clair par rapport à son hachage."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Génère un hachage bcrypt à partir d'un mot de passe en clair."""
    return pwd_context.hash(password)


def create_access_token(
    subject: Union[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """Crée un jeton d'accès JWT."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=config.auth.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode = {"exp": expire, "sub": str(subject), "type": "access"}
    encoded_jwt = jwt.encode(
        to_encode, config.auth.SECRET_KEY, algorithm=config.auth.ALGORITHM
    )
    return encoded_jwt


def create_refresh_token(
    subject: Union[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """Crée un jeton de rafraîchissement JWT."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            days=config.auth.REFRESH_TOKEN_EXPIRE_DAYS
        )
    
    to_encode = {"exp": expire, "sub": str(subject), "type": "refresh"}
    encoded_jwt = jwt.encode(
        to_encode, config.auth.SECRET_KEY, algorithm=config.auth.ALGORITHM
    )
    return encoded_jwt
