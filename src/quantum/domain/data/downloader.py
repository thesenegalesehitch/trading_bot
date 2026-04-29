"""
Module de téléchargement des données historiques.
Utilise yfinance comme source principale avec support pour MT5.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yfinance as yf
from tqdm import tqdm
import os
import sys

# Sources de données alternatives
try:
    from polygon import RESTClient as PolygonClient
except ImportError:
    PolygonClient = None

try:
    import finnhub
except ImportError:
    finnhub = None

try:
    import requests
except ImportError:
    requests = None

try:
    from alpha_vantage.foreignexchange import ForeignExchange
except ImportError:
    ForeignExchange = None

# Ajouter le chemin parent pour les imports

from quantum.shared.config.settings import config


class DataDownloader:
    """
    Télécharge et gère les données historiques du marché.
    Supporte le backfilling massif sur plusieurs années.
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialise le téléchargeur de données.

        Args:
            cache_dir: Répertoire de cache pour les données téléchargées
        """
        self.cache_dir = cache_dir or config.system.DATA_DIR
        self.symbols = config.symbols.ACTIVE_SYMBOLS
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY') or config.data.ALPHA_VANTAGE_API_KEY
        self.polygon_key = os.getenv('POLYGON_API_KEY') or config.data.POLYGON_API_KEY
        self.finnhub_key = os.getenv('FINNHUB_API_KEY') or config.data.FINNHUB_API_KEY
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Crée le répertoire de cache s'il n'existe pas."""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    async def download_historical(
        self,
        symbol: str,
        years: int = None,
        interval: str = "1h"
    ) -> pd.DataFrame:
        """
        Télécharge les données historiques pour un symbole donné (Async).
        """
        years = years or config.data.HISTORICAL_YEARS
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        # Check cache first
        cached = self.load_from_cache(symbol, interval)
        if cached is not None:
            return cached

        # Sources logic...
        import asyncio
        
        # Wrap blocking yfinance in thread
        def _fetch():
            ticker = yf.Ticker(symbol)
            return ticker.history(start=start_date, end=end_date, interval=interval, auto_adjust=True)

        try:
            df = await asyncio.to_thread(_fetch)
            if not df.empty:
                df = self._clean_dataframe(df)
                cache_file = self._get_cache_path(symbol, interval)
                df.to_parquet(cache_file)
                return df
        except Exception as e:
            print(f"⚠️ Échec yfinance pour {symbol}: {e}")

        return pd.DataFrame()

    async def download_all_symbols(
        self,
        interval: str = "1h",
        years: int = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Télécharge les données pour tous les symboles en parallèle.
        """
        import asyncio
        tasks = [self.download_historical(s, years, interval) for s in self.symbols]
        results = await asyncio.gather(*tasks)
        
        return {s: df for s, df in zip(self.symbols, results) if not df.empty}
    
    def download_multiple_timeframes(
        self,
        symbol: str,
        timeframes: List[str] = None,
        years: int = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Télécharge les données pour plusieurs timeframes.
        
        Note: yfinance a des limitations sur les intervalles historiques:
        - 1m/2m/5m/15m: max 7 jours
        - 30m/1h: max 730 jours
        - 1d/1wk/1mo: illimité
        
        Args:
            symbol: Symbole à télécharger
            timeframes: Liste des timeframes
            years: Nombre d'années
        
        Returns:
            Dictionnaire {timeframe: DataFrame}
        """
        timeframes = timeframes or config.timeframes.TIMEFRAMES
        data = {}
        
        for tf in timeframes:
            # Adaptation des années selon le timeframe
            max_years = self._get_max_years_for_interval(tf)
            actual_years = min(years or config.data.HISTORICAL_YEARS, max_years)
            
            df = self.download_historical(symbol, actual_years, tf)
            if not df.empty:
                data[tf] = df
        
        return data
    
    def resample_to_timeframe(
        self,
        df: pd.DataFrame,
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample les données vers un timeframe cible.
        
        Args:
            df: DataFrame source
            target_timeframe: Timeframe cible
        
        Returns:
            DataFrame resamplé
        """
        resample_rule = config.timeframes.RESAMPLE_MAP.get(target_timeframe, target_timeframe)
        
        resampled = df.resample(resample_rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return resampled
    
    def load_from_cache(
        self,
        symbol: str,
        interval: str = "1h"
    ) -> Optional[pd.DataFrame]:
        """
        Charge les données depuis le cache si elles ne sont pas trop vieilles.
        
        Args:
            symbol: Symbole à charger
            interval: Intervalle des bougies
        
        Returns:
            DataFrame ou None si non trouvé ou expiré
        """
        cache_file = self._get_cache_path(symbol, interval)
        
        if os.path.exists(cache_file):
            # Vérifier l'âge du cache
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            expiry_hours = config.data.CACHE_EXPIRY_HOURS
            
            if datetime.now() - cache_time < timedelta(hours=expiry_hours):
                print(f"📦 Chargement depuis le cache: {symbol} ({interval})")
                return pd.read_parquet(cache_file)
            else:
                print(f"🔄 Cache expiré pour {symbol} ({interval}), re-téléchargement nécessaire")
        
        return None
    
    def get_data(
        self,
        symbol: str,
        interval: str = "1h",
        force_download: bool = False
    ) -> pd.DataFrame:
        """
        Récupère les données (cache ou téléchargement).
        
        Args:
            symbol: Symbole à récupérer
            interval: Intervalle des bougies
            force_download: Forcer le re-téléchargement
        
        Returns:
            DataFrame avec les données
        """
        if not force_download:
            cached = self.load_from_cache(symbol, interval)
            if cached is not None:
                return cached
        
        return self.download_historical(symbol, interval=interval)
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Valide l'intégrité des données téléchargées.
        
        Args:
            df: DataFrame à valider
        
        Returns:
            Dictionnaire avec les résultats de validation
        """
        validation = {
            "total_rows": len(df),
            "null_count": df.isnull().sum().sum(),
            "null_percent": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            "date_range": f"{df.index.min()} - {df.index.max()}",
            "duplicate_indices": df.index.duplicated().sum(),
            "price_anomalies": self._detect_price_anomalies(df),
            "is_valid": True
        }
        
        # Critères de validité
        if validation["null_percent"] > 5:
            validation["is_valid"] = False
            validation["issues"] = validation.get("issues", []) + ["Trop de valeurs nulles"]
        
        if validation["duplicate_indices"] > 0:
            validation["is_valid"] = False
            validation["issues"] = validation.get("issues", []) + ["Index dupliqués détectés"]
        
        return validation
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie et standardise le DataFrame."""
        # Renommer les colonnes si nécessaire
        df.columns = [col.capitalize() for col in df.columns]
        
        # Garder seulement OHLCV
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in required_cols if col in df.columns]
        df = df[available_cols].copy()
        
        # Supprimer les lignes avec des valeurs nulles
        df = df.dropna()
        
        # Supprimer les doublons d'index
        df = df[~df.index.duplicated(keep='first')]
        
        # Trier par date
        df = df.sort_index()
        
        return df
    
    def _get_cache_path(self, symbol: str, interval: str) -> str:
        """Génère le chemin du fichier cache."""
        safe_symbol = symbol.replace("=", "_").replace("/", "_")
        return os.path.join(self.cache_dir, f"{safe_symbol}_{interval}.parquet")
    
    def _get_max_years_for_interval(self, interval: str) -> float:
        """Retourne le maximum d'années disponibles pour un intervalle."""
        limits = {
            "1m": 0.16,  # ~60 jours
            "2m": 0.16,
            "5m": 0.16,
            "15m": 0.16,
            "30m": 2,
            "1h": 1.92,  # ~700 jours
            "4h": 2,
            "1d": 10,
            "1wk": 10,
            "1mo": 10
        }
        return limits.get(interval, 2)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Récupère le prix actuel d'un symbole.

        Args:
            symbol: Symbole (ex: "EURUSD=X")

        Returns:
            Prix actuel ou None si échec
        """
        try:
            ticker = yf.Ticker(symbol)
            # Récupérer les données récentes (dernière journée)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except Exception as e:
            print(f"❌ Erreur récupération prix actuel pour {symbol}: {e}")

        return None

    def _detect_price_anomalies(self, df: pd.DataFrame) -> int:
        """Détecte les anomalies de prix (variations extrêmes)."""
        if 'Close' not in df.columns:
            return 0

        returns = df['Close'].pct_change()
        # Variations > 10% en une bougie = potentielle anomalie
        anomalies = (returns.abs() > 0.10).sum()
        return int(anomalies)


if __name__ == "__main__":
    # Test du module
    downloader = DataDownloader()

    # Télécharger tous les symboles actifs
    print("📊 Test du téléchargement pour tous les symboles actifs...")
    all_data = downloader.download_all_symbols(years=2, interval="1h")

    for symbol, data in all_data.items():
        print(f"\n{symbol} shape: {data.shape}")
        if not data.empty:
            print(data.head(3))
            # Valider les données
            validation = downloader.validate_data(data)
            print(f"Validation: {validation}")
        else:
            print("❌ Aucune donnée reçue")
        print("-" * 50)
