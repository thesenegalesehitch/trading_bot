"""
Tests pour le routeur d'authentification API (Register, Login).
"""

import pytest
from fastapi.testclient import TestClient
from passlib.context import CryptContext

from quantum.infrastructure.api.main import app
from quantum.infrastructure.api.core.security import verify_password
from quantum.infrastructure.db.models import User, Account

# ---- Mocks pour la Base de données Async ----
class MockResult:
    def __init__(self, data=None):
        self.data = data
    def scalar_one_or_none(self):
        return self.data
    def first(self):
        return self.data

class MockSession:
    def __init__(self):
        self.added = []
        self.committed = False
    
    async def execute(self, query):
        # Simulation basique 
        # Compile la requete avec ses parametres pour verifier l'email
        q_str = str(query.compile(compile_kwargs={"literal_binds": True}))
        if "test@quantum.com" in q_str:
            # Email déjà pris
            return MockResult(User(id=1, email="test@quantum.com", hashed_password="hashed"))
        elif "new@quantum.com" in q_str:
            # Nouvel email (register ok) ou login erroné
            return MockResult(None)
        elif "user@quantum.com" in q_str:
            # Login successful
            from quantum.infrastructure.api.core.security import get_password_hash
            hashed = get_password_hash("password123")
            return MockResult(User(id=2, email="user@quantum.com", hashed_password=hashed, is_active=True))
        
        return MockResult(None)
    
    def add(self, obj):
        self.added.append(obj)
    
    async def commit(self):
        self.committed = True
    
    async def refresh(self, obj):
        if hasattr(obj, 'id') and obj.id is None:
            obj.id = 999
        if hasattr(obj, 'is_active') and obj.is_active is None:
            obj.is_active = True
        if hasattr(obj, 'is_superuser') and obj.is_superuser is None:
            obj.is_superuser = False

async def override_get_db():
    try:
        db = MockSession()
        yield db
    finally:
        pass

# Remplace la dépendance réelle
from quantum.infrastructure.db.session import get_db
app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

def test_register_new_user():
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "new@quantum.com", "password": "SecurePassword1!", "full_name": "New User"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "new@quantum.com"
    assert data["full_name"] == "New User"
    assert "id" in data
    assert data["is_active"] is True

def test_register_existing_user():
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "test@quantum.com", "password": "pass", "full_name": "Test"}
    )
    assert response.status_code == 400
    assert "existe déjà" in response.json()["detail"]

def test_login_success():
    # user@quantum.com / password123 est mocké comme correct
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "user@quantum.com", "password": "password123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_login_failure():
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "new@quantum.com", "password": "wrongpassword"}
    )
    assert response.status_code == 401
    assert "Email ou mot de passe incorrect" in response.json()["detail"]
