import sys
import os
import asyncio

# Ajouter src au PYTHONPATH
sys.path.append(os.path.join(os.getcwd(), 'src'))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from quantum.infrastructure.db.models import User, Account
from passlib.hash import bcrypt

def create_default_user():
    print("👤 Création de l'utilisateur par défaut...")
    
    engine = create_engine("sqlite:///quantum.db")
    Session = sessionmaker(bind=engine)
    session = Session()

    # Vérifier si l'utilisateur existe
    email = "trader@quantum.com"
    user = session.query(User).filter_by(email=email).first()
    
    if not user:
        hashed_pw = bcrypt.hash("password123")
        user = User(
            email=email,
            hashed_password=hashed_pw,
            full_name="Alexandre Trader",
            is_active=True
        )
        session.add(user)
        session.flush()
        
        # Créer le compte démo
        account = Account(
            user_id=user.id,
            account_type="DEMO",
            balance=1000000.0
        )
        session.add(account)
        session.commit()
        print(f"✅ Utilisateur créé : {email} / password123")
    else:
        print("Utilisateur déjà présent.")

if __name__ == "__main__":
    create_default_user()
