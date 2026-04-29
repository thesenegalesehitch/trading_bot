"""
API Principale du système de trading quantique.
Point d'entrée de l'application FastAPI.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from quantum.infrastructure.db.session import init_db, close_db
from quantum.infrastructure.api.routers import auth, market, analysis, trading, risk, academy, backtest, journal

from quantum.shared.config.settings import config

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Démarrage
    logger.info("Démarrage du système: Initialisation DB...")
    await init_db()
    yield
    # Arrêt
    logger.info("Arrêt du système: Fermeture connexions DB...")
    await close_db()


app = FastAPI(
    title="Quantum Trading System API",
    description="API REST sécurisée pour la plateforme de trading quantitatif.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configuration CORS restrictive
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Uniquement le frontend local par défaut
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Inclusion des routeurs métier
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentification"])
app.include_router(market.router, prefix="/api/v1/market", tags=["Données Marché"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Analyse Technique"])
app.include_router(trading.router, prefix="/api/v1/trading", tags=["Trading Démo"])
app.include_router(risk.router, prefix="/api/v1/risk", tags=["Gestion du Risque"])
app.include_router(academy.router, prefix="/api/v1/academy", tags=["Académie"])
app.include_router(backtest.router, prefix="/api/v1/backtest", tags=["Backtesting"])
app.include_router(journal.router, prefix="/api/v1/journal", tags=["Journal de Trading"])

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Quantum Trading System API v2.0 - Plateforme Référence"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Erreur serveur globale: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False, 
            "message": "Une erreur interne est survenue. Veuillez contacter le support.",
            # En production, on ne renvoie pas 'detail' pour éviter les fuites d'info
            "detail": str(exc) if config.system.MODE == "dev" else None 
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("quantum.infrastructure.api.main:app", host="0.0.0.0", port=8000, reload=True)