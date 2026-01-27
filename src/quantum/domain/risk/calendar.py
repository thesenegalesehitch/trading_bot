"""
Calendrier économique - Détection des événements à impact élevé.
Interdit le trading 30 min avant/après les annonces majeures.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
import sys
import os


from quantum.shared.config.settings import config


@dataclass
class EconomicEvent:
    """Représente un événement économique."""
    time: datetime
    currency: str
    impact: str  # "high", "medium", "low"
    event: str
    forecast: Optional[str] = None
    previous: Optional[str] = None


class EconomicCalendar:
    """
    Gère le calendrier économique et les blackouts.
    
    Utilise le scraping d'Investing.com ou un fichier cache.
    """
    
    def __init__(
        self,
        blackout_before: timedelta = None,
        blackout_after: timedelta = None
    ):
        self.blackout_before = blackout_before or config.risk.ECONOMIC_BLACKOUT_BEFORE
        self.blackout_after = blackout_after or config.risk.ECONOMIC_BLACKOUT_AFTER
        self.events: List[EconomicEvent] = []
        self.last_update: Optional[datetime] = None
        
        # Événements par défaut (haute importance)
        self._load_default_events()
    
    def _load_default_events(self):
        """Charge les événements récurrents importants."""
        # Ces événements sont récurrents chaque mois
        self.high_impact_events = [
            "Non-Farm Payrolls",
            "NFP",
            "Interest Rate Decision",
            "FOMC",
            "ECB Interest Rate",
            "CPI",
            "GDP",
            "Retail Sales",
            "PMI",
            "Unemployment Rate"
        ]
    
    def fetch_events(self, currencies: List[str] = None) -> bool:
        """
        Récupère les événements depuis Investing.com.
        
        Note: Le scraping peut échouer si le site change sa structure.
        """
        currencies = currencies or ["USD", "EUR"]
        
        try:
            url = "https://www.investing.com/economic-calendar/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                print(f"⚠️ Échec du fetch: {response.status_code}")
                return False
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parser les événements (structure simplifiée)
            # Note: La structure exacte peut varier
            event_rows = soup.find_all('tr', class_='js-event-item')
            
            for row in event_rows:
                try:
                    self._parse_event_row(row, currencies)
                except:
                    continue
            
            self.last_update = datetime.now()
            print(f"✅ {len(self.events)} événements chargés")
            return True
            
        except Exception as e:
            print(f"⚠️ Erreur lors du fetch: {e}")
            return False
    
    def _parse_event_row(self, row, currencies: List[str]):
        """Parse une ligne d'événement."""
        # Extraction simplifiée
        currency = row.get('data-currency', '')
        
        if currency not in currencies:
            return
        
        # Impact (nombre de bulls)
        impact_elem = row.find('td', class_='sentiment')
        if impact_elem:
            bulls = len(impact_elem.find_all('i', class_='grayFullBullishIcon'))
            if bulls >= 3:
                impact = "high"
            elif bulls >= 2:
                impact = "medium"
            else:
                impact = "low"
        else:
            return
        
        # Ignorer les événements à faible impact
        if impact != "high":
            return
        
        # Nom de l'événement
        event_elem = row.find('td', class_='event')
        event_name = event_elem.get_text(strip=True) if event_elem else "Unknown"
        
        # Heure (simplifiée - utilise l'heure actuelle comme placeholder)
        time = datetime.now()
        
        self.events.append(EconomicEvent(
            time=time,
            currency=currency,
            impact=impact,
            event=event_name
        ))
    
    def is_blackout_period(self, check_time: datetime = None) -> Dict:
        """
        Vérifie si on est dans une période de blackout.
        
        Args:
            check_time: Heure à vérifier (défaut: maintenant)
        
        Returns:
            Dict avec statut et événement concerné
        """
        check_time = check_time or datetime.now()
        
        for event in self.events:
            if event.impact != "high":
                continue
            
            blackout_start = event.time - self.blackout_before
            blackout_end = event.time + self.blackout_after
            
            if blackout_start <= check_time <= blackout_end:
                return {
                    "is_blackout": True,
                    "event": event.event,
                    "event_time": str(event.time),
                    "currency": event.currency,
                    "resume_at": str(blackout_end)
                }
        
        # Vérifier aussi les événements par nom
        if self._check_known_events(check_time):
            return {
                "is_blackout": True,
                "event": "Événement majeur potentiel",
                "reason": "Vérifier le calendrier manuellement"
            }
        
        return {"is_blackout": False}
    
    def _check_known_events(self, check_time: datetime) -> bool:
        """
        Vérifie les événements connus par leur horaire typique.
        
        Ex: NFP = 1er vendredi du mois à 13:30 UTC
        """
        # NFP (Non-Farm Payrolls)
        # Premier vendredi du mois, 13:30 UTC
        if check_time.weekday() == 4:  # Vendredi
            if check_time.day <= 7:  # Premier vendredi
                nfp_time = check_time.replace(hour=13, minute=30, second=0)
                if abs((check_time - nfp_time).total_seconds()) < 30 * 60:
                    return True
        
        # FOMC (8 fois par an, mercredi 18:00 UTC)
        # Simplifié: chaque mercredi à 18:00
        if check_time.weekday() == 2:  # Mercredi
            if check_time.hour == 18 or check_time.hour == 19:
                return True
        
        return False
    
    def can_trade(self) -> Dict:
        """
        Vérifie si le trading est autorisé maintenant.
        """
        blackout = self.is_blackout_period()
        
        if blackout["is_blackout"]:
            return {
                "allowed": False,
                "reason": f"Blackout: {blackout.get('event', 'Événement économique')}",
                "resume_at": blackout.get('resume_at')
            }
        
        return {"allowed": True}
    
    def get_upcoming_events(self, hours: int = 24) -> List[Dict]:
        """
        Retourne les événements à venir.
        """
        now = datetime.now()
        end_time = now + timedelta(hours=hours)
        
        upcoming = []
        for event in self.events:
            if now <= event.time <= end_time and event.impact == "high":
                upcoming.append({
                    "time": str(event.time),
                    "currency": event.currency,
                    "event": event.event,
                    "impact": event.impact
                })
        
        return upcoming
    
    def add_manual_event(
        self,
        event_time: datetime,
        event_name: str,
        currency: str = "USD"
    ):
        """
        Ajoute manuellement un événement.
        """
        self.events.append(EconomicEvent(
            time=event_time,
            currency=currency,
            impact="high",
            event=event_name
        ))


if __name__ == "__main__":
    # Test
    calendar = EconomicCalendar()
    
    print("=== Test Calendrier Économique ===\n")
    
    # Ajouter un événement test
    test_time = datetime.now() + timedelta(minutes=15)
    calendar.add_manual_event(test_time, "Test NFP", "USD")
    
    print("Événements ajoutés:")
    for e in calendar.events:
        print(f"  {e.time}: {e.event} ({e.currency})")
    
    # Vérifier blackout
    print("\n=== Vérification Blackout ===")
    
    # Maintenant (pas de blackout)
    result = calendar.can_trade()
    print(f"Maintenant: {result}")
    
    # À l'heure de l'événement (blackout)
    result = calendar.is_blackout_period(test_time)
    print(f"À {test_time}: {result}")
