"""
Module de gestion de la base de données — Sessions async et sync.

Fournit les sessions SQLAlchemy pour l'ensemble de l'application.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quantum.shared.config.settings import config
from quantum.infrastructure.db.models import Base

# Moteur async pour FastAPI
async_engine = create_async_engine(
    config.database.DATABASE_URL,
    pool_size=config.database.POOL_SIZE,
    max_overflow=config.database.MAX_OVERFLOW,
    pool_timeout=config.database.POOL_TIMEOUT,
    echo=config.database.ECHO_SQL,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Moteur sync pour scripts et CLI
sync_engine = create_engine(
    config.database.DATABASE_URL_SYNC,
    echo=config.database.ECHO_SQL,
)

SyncSessionLocal = sessionmaker(bind=sync_engine)


async def get_db() -> AsyncSession:
    """Dépendance FastAPI — fournit une session DB async.

    Usage::

        @router.get("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Crée toutes les tables dans la base de données."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Ferme le pool de connexions."""
    await async_engine.dispose()
