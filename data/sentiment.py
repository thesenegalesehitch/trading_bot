"""
Module d'analyse de sentiment multi-sources.
Combine plusieurs indicateurs de sentiment pour enrichir l'analyse.

Sources:
- Fear & Greed Index (alternative.me)
- Reddit sentiment (gratuit)
- News sentiment (NewsAPI gratuit)
- Social sentiment scores
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import requests
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import config


@dataclass
class SentimentScore:
    """Score de sentiment avec métadonnées."""
    value: float  # -1 (extreme fear) à +1 (extreme greed)
    source: str
    timestamp: datetime
    confidence: float
    raw_data: Optional[Dict] = None


class FearGreedIndex:
    """
    Fear and Greed Index - Mesure le sentiment global du marché.
    Source: alternative.me (gratuit, pas de clé API)
    """
    
    def __init__(self):
        self.base_url = "https://api.alternative.me/fng/"
        self.last_fetch = None
        self.cached_data = None
        self.cache_duration = timedelta(hours=1)
    
    def fetch_current(self) -> Optional[SentimentScore]:
        """Récupère l'indice Fear & Greed actuel."""
        try:
            response = requests.get(self.base_url, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            if 'data' not in data or not data['data']:
                return None
            
            current = data['data'][0]
            value = int(current['value'])
            
            # Normaliser de 0-100 à -1 à +1
            normalized = (value - 50) / 50
            
            return SentimentScore(
                value=normalized,
                source="Fear & Greed Index",
                timestamp=datetime.now(),
                confidence=0.8,
                raw_data={
                    'value': value,
                    'classification': current.get('value_classification', ''),
                    'time_until_update': current.get('time_until_update', '')
                }
            )
            
        except Exception as e:
            print(f"⚠️ Erreur Fear & Greed Index: {e}")
            return None
    
    def fetch_historical(self, days: int = 30) -> pd.DataFrame:
        """Récupère l'historique Fear & Greed."""
        try:
            url = f"{self.base_url}?limit={days}"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                return pd.DataFrame()
            
            data = response.json()
            if 'data' not in data:
                return pd.DataFrame()
            
            records = []
            for item in data['data']:
                records.append({
                    'timestamp': datetime.fromtimestamp(int(item['timestamp'])),
                    'value': int(item['value']),
                    'classification': item.get('value_classification', '')
                })
            
            df = pd.DataFrame(records)
            if not df.empty:
                df = df.set_index('timestamp').sort_index()
                df['normalized'] = (df['value'] - 50) / 50
            
            return df
            
        except Exception as e:
            print(f"⚠️ Erreur Fear & Greed historique: {e}")
            return pd.DataFrame()
    
    def get_signal(self) -> Dict:
        """
        Génère un signal basé sur Fear & Greed.
        
        Logique contrarian:
        - Extreme Fear (<25) = Signal d'achat potentiel
        - Extreme Greed (>75) = Signal de vente potentiel
        """
        score = self.fetch_current()
        
        if score is None:
            return {"signal": "NEUTRAL", "reason": "Données non disponibles"}
        
        raw_value = score.raw_data['value']
        
        if raw_value <= 25:
            signal = "BUY"
            reason = f"Extreme Fear ({raw_value}) - Opportunité contrarian"
            strength = (25 - raw_value) / 25
        elif raw_value >= 75:
            signal = "SELL"
            reason = f"Extreme Greed ({raw_value}) - Prudence recommandée"
            strength = (raw_value - 75) / 25
        elif raw_value <= 40:
            signal = "LEAN_BUY"
            reason = f"Fear ({raw_value})"
            strength = (40 - raw_value) / 40
        elif raw_value >= 60:
            signal = "LEAN_SELL"
            reason = f"Greed ({raw_value})"
            strength = (raw_value - 60) / 40
        else:
            signal = "NEUTRAL"
            reason = f"Neutral zone ({raw_value})"
            strength = 0
        
        return {
            "signal": signal,
            "reason": reason,
            "strength": min(strength, 1.0),
            "value": raw_value,
            "classification": score.raw_data['classification']
        }


class NewsSentimentAnalyzer:
    """
    Analyse le sentiment des news via NewsAPI.
    Gratuit: 100 requêtes/jour
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('NEWSAPI_KEY', '')
        self.base_url = "https://newsapi.org/v2"
        self._available = bool(self.api_key)
    
    def is_available(self) -> bool:
        return self._available
    
    def fetch_headlines(
        self,
        keywords: List[str] = None,
        language: str = 'en'
    ) -> List[Dict]:
        """Récupère les headlines récentes."""
        if not self._available:
            return []
        
        keywords = keywords or ['forex', 'EUR USD', 'gold price', 'fed']
        
        headlines = []
        for keyword in keywords[:3]:  # Limiter les requêtes
            try:
                url = f"{self.base_url}/everything"
                params = {
                    'q': keyword,
                    'apiKey': self.api_key,
                    'language': language,
                    'sortBy': 'publishedAt',
                    'pageSize': 10
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code != 200:
                    continue
                
                data = response.json()
                
                for article in data.get('articles', []):
                    headlines.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'published_at': article.get('publishedAt', ''),
                        'url': article.get('url', '')
                    })
                    
            except Exception as e:
                print(f"⚠️ Erreur NewsAPI: {e}")
                continue
        
        return headlines
    
    def analyze_sentiment_basic(self, text: str) -> float:
        """
        Analyse de sentiment basique basée sur des mots-clés.
        Retourne un score entre -1 (négatif) et +1 (positif).
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        # Mots positifs pour le marché
        positive_words = [
            'rally', 'surge', 'gain', 'rise', 'bullish', 'growth', 'strong',
            'recovery', 'optimism', 'breakthrough', 'record high', 'outperform',
            'beat', 'exceed', 'support', 'momentum', 'break out', 'upgrade',
            'positive', 'confidence', 'expansion'
        ]
        
        # Mots négatifs pour le marché
        negative_words = [
            'crash', 'plunge', 'drop', 'fall', 'bearish', 'recession', 'weak',
            'crisis', 'fear', 'concern', 'decline', 'loss', 'tumble', 'sell-off',
            'miss', 'disappoint', 'resistance', 'breakdown', 'downgrade',
            'negative', 'uncertainty', 'contraction', 'inflation', 'default'
        ]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total
    
    def get_market_sentiment(self, symbol: str = "EUR/USD") -> Dict:
        """Analyse le sentiment du marché pour un symbole."""
        keywords = self._get_keywords_for_symbol(symbol)
        headlines = self.fetch_headlines(keywords)
        
        if not headlines:
            return {"sentiment": 0, "signal": "NEUTRAL", "articles": 0}
        
        sentiments = []
        for article in headlines:
            text = f"{article['title']} {article.get('description', '')}"
            score = self.analyze_sentiment_basic(text)
            sentiments.append(score)
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        
        if avg_sentiment > 0.3:
            signal = "BULLISH"
        elif avg_sentiment < -0.3:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"
        
        return {
            "sentiment": avg_sentiment,
            "signal": signal,
            "articles": len(headlines),
            "confidence": min(len(headlines) / 10, 1.0)
        }
    
    def _get_keywords_for_symbol(self, symbol: str) -> List[str]:
        """Génère les mots-clés de recherche pour un symbole."""
        mapping = {
            'EURUSD': ['EUR USD', 'euro dollar', 'ECB', 'eurozone'],
            'EURUSD=X': ['EUR USD', 'euro dollar', 'ECB', 'eurozone'],
            'GC=F': ['gold price', 'gold futures', 'precious metals'],
            'XAUUSD': ['gold price', 'XAU USD', 'gold trading'],
            'BTCUSDT': ['bitcoin', 'BTC', 'crypto market'],
            'BTCUSD': ['bitcoin', 'BTC', 'crypto market'],
        }
        
        return mapping.get(symbol.upper(), [symbol])


class SocialSentiment:
    """
    Analyse du sentiment sur les réseaux sociaux.
    Utilise des APIs gratuites quand disponibles.
    """
    
    def __init__(self):
        self.reddit_sentiment = RedditSentiment()
    
    def get_combined_sentiment(self, symbol: str) -> Dict:
        """Combine les sentiments de toutes les sources sociales."""
        results = {}
        
        # Reddit
        reddit_result = self.reddit_sentiment.get_sentiment(symbol)
        if reddit_result:
            results['reddit'] = reddit_result
        
        # Calculer score combiné
        if not results:
            return {"combined_score": 0, "signal": "NEUTRAL", "sources": 0}
        
        scores = [r.get('sentiment', 0) for r in results.values()]
        combined = np.mean(scores)
        
        if combined > 0.2:
            signal = "BULLISH"
        elif combined < -0.2:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"
        
        return {
            "combined_score": combined,
            "signal": signal,
            "sources": len(results),
            "details": results
        }


class RedditSentiment:
    """
    Analyse du sentiment Reddit via l'API publique (sans auth).
    Limité mais gratuit.
    """
    
    def __init__(self):
        self.subreddits = {
            'forex': ['Forex', 'ForexTrading'],
            'crypto': ['cryptocurrency', 'Bitcoin', 'CryptoMarkets'],
            'stocks': ['wallstreetbets', 'stocks', 'investing'],
            'gold': ['Gold', 'Silverbugs']
        }
    
    def get_sentiment(self, symbol: str) -> Optional[Dict]:
        """Récupère le sentiment Reddit pour un symbole."""
        category = self._categorize_symbol(symbol)
        subreddits = self.subreddits.get(category, ['Forex'])
        
        try:
            all_posts = []
            for subreddit in subreddits[:1]:  # Une seule requête
                url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=10"
                headers = {'User-Agent': 'QuantumTrading/1.0'}
                
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code != 200:
                    continue
                
                data = response.json()
                posts = data.get('data', {}).get('children', [])
                
                for post in posts:
                    post_data = post.get('data', {})
                    all_posts.append({
                        'title': post_data.get('title', ''),
                        'score': post_data.get('score', 0),
                        'upvote_ratio': post_data.get('upvote_ratio', 0.5),
                        'num_comments': post_data.get('num_comments', 0)
                    })
            
            if not all_posts:
                return None
            
            # Analyser le sentiment des titres
            sentiments = []
            for post in all_posts:
                text = post['title']
                score = self._basic_sentiment(text)
                # Pondérer par le nombre d'upvotes
                weight = np.log1p(post['score'])
                sentiments.append(score * weight)
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            return {
                "sentiment": np.clip(avg_sentiment / 5, -1, 1),
                "posts_analyzed": len(all_posts),
                "subreddits": subreddits[:1]
            }
            
        except Exception as e:
            print(f"⚠️ Erreur Reddit: {e}")
            return None
    
    def _categorize_symbol(self, symbol: str) -> str:
        """Catégorise un symbole."""
        symbol_upper = symbol.upper()
        if 'BTC' in symbol_upper or 'ETH' in symbol_upper or 'USDT' in symbol_upper:
            return 'crypto'
        elif 'GC=' in symbol_upper or 'XAU' in symbol_upper or 'GOLD' in symbol_upper:
            return 'gold'
        elif any(x in symbol_upper for x in ['EUR', 'USD', 'GBP', 'JPY', 'CHF']):
            return 'forex'
        return 'stocks'
    
    def _basic_sentiment(self, text: str) -> float:
        """Analyse de sentiment basique."""
        text_lower = text.lower()
        
        bullish = ['moon', 'buy', 'bullish', 'pump', 'gain', 'up', 'rocket', 'call', 'long']
        bearish = ['crash', 'sell', 'bearish', 'dump', 'loss', 'down', 'put', 'short', 'rekt']
        
        pos = sum(1 for w in bullish if w in text_lower)
        neg = sum(1 for w in bearish if w in text_lower)
        
        return pos - neg


class SentimentAggregator:
    """
    Agrégateur de sentiment multi-sources.
    Combine tous les indicateurs de sentiment en un score unique.
    """
    
    def __init__(self):
        self.fear_greed = FearGreedIndex()
        self.news = NewsSentimentAnalyzer()
        self.social = SocialSentiment()
        
        # Poids des sources
        self.weights = {
            'fear_greed': 0.40,
            'news': 0.35,
            'social': 0.25
        }
    
    def get_aggregated_sentiment(self, symbol: str = "EURUSD=X") -> Dict:
        """
        Calcule le sentiment agrégé à partir de toutes les sources.
        
        Returns:
            Dict avec score (-1 à +1), signal, et détails par source
        """
        results = {}
        scores = []
        weights = []
        
        # Fear & Greed Index
        fg_signal = self.fear_greed.get_signal()
        if fg_signal['signal'] != 'NEUTRAL':
            score_map = {
                'BUY': 0.8, 'LEAN_BUY': 0.4,
                'SELL': -0.8, 'LEAN_SELL': -0.4,
                'NEUTRAL': 0
            }
            fg_score = score_map.get(fg_signal['signal'], 0) * fg_signal.get('strength', 0.5)
            scores.append(fg_score)
            weights.append(self.weights['fear_greed'])
            results['fear_greed'] = {
                'score': fg_score,
                'raw_value': fg_signal.get('value'),
                'classification': fg_signal.get('classification')
            }
        
        # News sentiment
        if self.news.is_available():
            news_result = self.news.get_market_sentiment(symbol)
            if news_result['articles'] > 0:
                scores.append(news_result['sentiment'])
                weights.append(self.weights['news'] * news_result['confidence'])
                results['news'] = news_result
        
        # Social sentiment
        social_result = self.social.get_combined_sentiment(symbol)
        if social_result['sources'] > 0:
            scores.append(social_result['combined_score'])
            weights.append(self.weights['social'])
            results['social'] = social_result
        
        # Calculer le score agrégé
        if not scores:
            return {
                "aggregated_score": 0,
                "signal": "NEUTRAL",
                "confidence": 0,
                "sources": results
            }
        
        aggregated = np.average(scores, weights=weights)
        confidence = sum(weights) / sum(self.weights.values())
        
        # Déterminer le signal
        if aggregated > 0.3:
            signal = "BULLISH"
        elif aggregated < -0.3:
            signal = "BEARISH"
        elif aggregated > 0.1:
            signal = "LEAN_BULLISH"
        elif aggregated < -0.1:
            signal = "LEAN_BEARISH"
        else:
            signal = "NEUTRAL"
        
        return {
            "aggregated_score": round(aggregated, 3),
            "signal": signal,
            "confidence": round(confidence, 2),
            "sources": results
        }


if __name__ == "__main__":
    print("=" * 60)
    print("TEST ANALYSE DE SENTIMENT")
    print("=" * 60)
    
    # Test Fear & Greed
    print("\n--- Fear & Greed Index ---")
    fg = FearGreedIndex()
    signal = fg.get_signal()
    print(f"Signal: {signal}")
    
    # Test Aggregator
    print("\n--- Sentiment Agrégé ---")
    aggregator = SentimentAggregator()
    result = aggregator.get_aggregated_sentiment("EURUSD=X")
    print(f"Score: {result['aggregated_score']}")
    print(f"Signal: {result['signal']}")
    print(f"Confiance: {result['confidence']}")
    print(f"Sources: {list(result['sources'].keys())}")
