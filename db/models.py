"""
Database models for Quantum Trading System.

SQLAlchemy models for PostgreSQL database schema as per PRD.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class Symbol(Base):
    """Symbol master data table."""
    __tablename__ = 'symbols'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False)
    name = Column(String(100))
    asset_class = Column(String(20))  # forex, crypto, equity, commodity
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class MarketData(Base):
    """OHLCV market data table."""
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'))
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    interval = Column(String(10))  # 1m, 5m, 15m, 1h, 1d, etc.


class Signals(Base):
    """Trading signals table."""
    __tablename__ = 'signals'

    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'))
    timestamp = Column(DateTime, nullable=False)
    signal_type = Column(String(20))  # BUY, SELL, HOLD, etc.
    probability = Column(Float)
    model_version = Column(String(50))
    features = Column(Text)  # JSON string of features used


class Trades(Base):
    """Executed trades history table."""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'))
    signal_id = Column(Integer, ForeignKey('signals.id'))
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    entry_price = Column(Float)
    exit_price = Column(Float)
    quantity = Column(Float)
    pnl = Column(Float)
    status = Column(String(20))  # open, closed, cancelled


class RiskMetrics(Base):
    """Risk calculations history table."""
    __tablename__ = 'risk_metrics'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    portfolio_value = Column(Float)
    var_95 = Column(Float)
    var_99 = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)


class ML_Predictions(Base):
    """ML model predictions table."""
    __tablename__ = 'ml_predictions'

    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'))
    timestamp = Column(DateTime, nullable=False)
    model_name = Column(String(50))
    prediction = Column(Float)
    confidence = Column(Float)


class SentimentData(Base):
    """Sentiment analysis results table."""
    __tablename__ = 'sentiment_data'

    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'))
    timestamp = Column(DateTime, nullable=False)
    source = Column(String(50))  # news, twitter, reddit, etc.
    sentiment_score = Column(Float)  # -1 to +1 scale
    text = Column(Text)


# Database connection and session management would be added here
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker

# engine = create_engine(config.database.url)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()