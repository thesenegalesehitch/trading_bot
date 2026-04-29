"""
Module utilitaire pour la sécurité et l'authentification.

Gère le hachage des mots de passe (bcrypt) et la création/validation
des jetons JWT (JSON Web Tokens).
"""

from datetime import datetime, timedelta
from typing import Optional, Union, Any

import jwt
import bcrypt
import hashlib

from quantum.shared.config.settings import config



def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifie un mot de passe en clair par rapport à son hachage."""
    try:
        # Pre-hash pour contourner la limite de 72 caractères de bcrypt
        pwd_hash = hashlib.sha256(plain_password.encode('utf-8')).hexdigest()
        return bcrypt.checkpw(pwd_hash.encode('utf-8'), hashed_password.encode('utf-8'))
    except ValueError:
        return False


def get_password_hash(password: str) -> str:
    """Génère un hachage bcrypt à partir d'un mot de passe en clair."""
    # Pre-hash pour contourner la limite de 72 caractères de bcrypt
    pwd_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_hash.encode('utf-8'), salt).decode('utf-8')


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
