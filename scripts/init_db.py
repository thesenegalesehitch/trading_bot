import asyncio
import sys
import os

# Ajouter src au PYTHONPATH
sys.path.append(os.path.join(os.getcwd(), "src"))

from quantum.infrastructure.db.session import init_db, close_db

async def run_init():
    print("Initialisation de la base de données...")
    await init_db()
    print("Base de données initialisée avec succès.")
    await close_db()

if __name__ == "__main__":
    asyncio.run(run_init())
