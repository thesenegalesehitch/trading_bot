"""
SocialSentimentAnalyzer - Analyseur de domaine pour le sentiment Twitter/X.
"""

import logging
from typing import Dict, Optional
from quantum.shared.social.twitter_client import TwitterSentimentClient
from quantum.infrastructure.db.cache import get_cache

logger = logging.getLogger(__name__)

class SocialSentimentAnalyzer:
    """
    Analyse le sentiment social pour identifier les biais du marché.
    """
    
    def __init__(self):
        self.client = TwitterSentimentClient()
        self.cache = get_cache()

    def analyze_asset(self, symbol: str) -> Dict:
        """
        Génère une analyse de sentiment complète pour un actif.
        Consulte le cache Redis pour limiter les appels API Twitter.
        """
        cache_key = f"social_sentiment:{symbol}"
        
        # 1. Vérifier le cache
        cached = self.cache.get('social', symbol)
        if cached:
            return cached

        # 2. Appel au client
        social_data = self.client.get_asset_sentiment(symbol)
        
        # 3. Enrichissement (mapping pour le scorer)
        analysis = {
            'score': social_data['score'],
            'label': social_data['label'],
            'volume_signal': 'HIGH' if social_data['volume'] > 2000 else 'NORMAL',
            'is_bullish_bias': social_data['score'] > 60,
            'is_bearish_bias': social_data['score'] < 40,
            'source': social_data.get('source', 'Twitter/X')
        }
        
        # 4. Sauvegarder en cache (TTL 30 min car le sentiment social est moins volatil)
        self.cache.set('social', symbol, analysis, ttl=1800)
        
        return analysis

    def get_social_influence(self, symbol: str) -> float:
        """Retourne un modificateur de signal (-1.0 à 1.0)."""
        res = self.analyze_asset(symbol)
        score = res['score'] # 0-100
        # Normalisation vers -1.0 / 1.0
        return (score - 50) / 50
