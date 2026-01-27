"""
Event Bus - Bus événementiel asynchrone.

Ce module fournit un système de pub/sub pour la communication
découplée entre les composants du module Web3.
"""

import asyncio
import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import weakref

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Priorité des événements."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    IMMEDIATE = 4


@dataclass
class Event:
    """
    Événement du bus.
    
    DESCRIPTION:
    ============
    Structure représentant un événement transitant
    sur le bus événementiel.
    """
    event_type: str
    data: Dict[str, Any]
    source: str
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None  # Pour tracer les événements liés
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'event_type': self.event_type,
            'data': self.data,
            'source': self.source,
            'priority': self.priority.name,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id,
        }


@dataclass
class Subscription:
    """Représente une souscription à un type d'événement."""
    subscriber_id: str
    event_types: Set[str]
    callback: Callable[[Event], None]
    filter_func: Optional[Callable[[Event], bool]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    event_count: int = 0


class EventBus:
    """
    Bus événementiel asynchrone pour le module Web3.
    
    DESCRIPTION:
    ============
    Système de publication/souscription permettant aux
    composants de communiquer de manière découplée.
    
    FONCTIONNALITÉS:
    ================
    - Publication asynchrone d'événements
    - Souscription par type d'événement
    - Filtrage personnalisé
    - Gestion des priorités
    - Historique des événements
    
    USAGE:
    ======
    ```python
    bus = EventBus()
    await bus.start()
    
    # Souscrire
    def on_whale(event: Event):
        print(f"Whale: {event.data}")
    
    bus.subscribe("whale_alert", on_whale)
    
    # Publier
    await bus.publish(Event(
        event_type="whale_alert",
        data={"amount": 1000},
        source="mempool"
    ))
    ```
    
    ARCHITECTURE:
    =============
    Le bus utilise une queue prioritaire pour garantir
    que les événements IMMEDIATE sont traités en premier.
    """
    
    def __init__(self, max_queue_size: int = 10000):
        """
        Initialise le bus événementiel.
        
        Args:
            max_queue_size: Taille maximale de la queue
        """
        self._subscriptions: Dict[str, List[Subscription]] = {}
        self._all_subscriptions: List[Subscription] = []  # Pour les subscribers globaux
        
        # Queues par priorité
        self._queues: Dict[EventPriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=max_queue_size)
            for priority in EventPriority
        }
        
        # Historique
        self._event_history: List[Event] = []
        self._max_history_size = 1000
        
        # État
        self._running = False
        self._processor_tasks: List[asyncio.Task] = []
        
        # Métriques
        self._metrics = {
            'events_published': 0,
            'events_delivered': 0,
            'events_filtered': 0,
            'delivery_errors': 0,
        }
        
        # Compteur de subscribers
        self._subscriber_counter = 0
        
        logger.info("EventBus initialisé")
    
    async def start(self):
        """
        Démarre le bus événementiel.
        
        Lance les processeurs pour chaque niveau de priorité.
        """
        if self._running:
            return
        
        self._running = True
        
        # Lancer un processeur par priorité
        for priority in EventPriority:
            task = asyncio.create_task(self._process_queue(priority))
            self._processor_tasks.append(task)
        
        logger.info("EventBus démarré")
    
    async def stop(self):
        """Arrête le bus événementiel."""
        self._running = False
        
        for task in self._processor_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._processor_tasks.clear()
        logger.info("EventBus arrêté")
    
    def subscribe(
        self,
        event_types: Any,
        callback: Callable[[Event], None],
        filter_func: Optional[Callable[[Event], bool]] = None
    ) -> str:
        """
        Souscrit à un ou plusieurs types d'événements.
        
        Args:
            event_types: Type(s) d'événement (str ou List[str])
            callback: Fonction appelée pour chaque événement
            filter_func: Fonction de filtrage optionnelle
            
        Returns:
            ID de la souscription
        """
        # Normaliser event_types
        if isinstance(event_types, str):
            types_set = {event_types}
        elif event_types is None or event_types == "*":
            types_set = set()  # Tous les événements
        else:
            types_set = set(event_types)
        
        # Générer un ID
        self._subscriber_counter += 1
        subscriber_id = f"sub_{self._subscriber_counter}"
        
        subscription = Subscription(
            subscriber_id=subscriber_id,
            event_types=types_set,
            callback=callback,
            filter_func=filter_func,
        )
        
        # Enregistrer
        if not types_set:
            # Souscription globale
            self._all_subscriptions.append(subscription)
        else:
            for event_type in types_set:
                if event_type not in self._subscriptions:
                    self._subscriptions[event_type] = []
                self._subscriptions[event_type].append(subscription)
        
        logger.debug(f"Souscription créée: {subscriber_id} pour {types_set or 'tous'}")
        return subscriber_id
    
    def unsubscribe(self, subscriber_id: str) -> bool:
        """
        Se désabonne.
        
        Args:
            subscriber_id: ID de la souscription
            
        Returns:
            True si désabonnement réussi
        """
        # Chercher dans les souscriptions typées
        for event_type, subs in self._subscriptions.items():
            for sub in subs[:]:
                if sub.subscriber_id == subscriber_id:
                    subs.remove(sub)
                    logger.debug(f"Désabonné: {subscriber_id}")
                    return True
        
        # Chercher dans les souscriptions globales
        for sub in self._all_subscriptions[:]:
            if sub.subscriber_id == subscriber_id:
                self._all_subscriptions.remove(sub)
                return True
        
        return False
    
    async def publish(self, event: Event):
        """
        Publie un événement.
        
        Args:
            event: Événement à publier
        """
        try:
            self._metrics['events_published'] += 1
            
            # Ajouter à la queue appropriée
            queue = self._queues[event.priority]
            
            await asyncio.wait_for(
                queue.put(event),
                timeout=1.0
            )
            
            # Ajouter à l'historique
            self._add_to_history(event)
            
        except asyncio.TimeoutError:
            logger.warning(f"Queue {event.priority.name} pleine, événement abandonné")
        except Exception as e:
            logger.error(f"Erreur publication événement: {e}")
    
    def publish_sync(self, event: Event):
        """
        Publie un événement de manière synchrone.
        
        Args:
            event: Événement à publier
        """
        try:
            self._queues[event.priority].put_nowait(event)
            self._metrics['events_published'] += 1
            self._add_to_history(event)
        except asyncio.QueueFull:
            logger.warning("Queue pleine")
    
    async def _process_queue(self, priority: EventPriority):
        """
        Processeur pour une queue de priorité.
        
        Args:
            priority: Priorité à traiter
        """
        queue = self._queues[priority]
        
        while self._running:
            try:
                # Pour IMMEDIATE, pas de timeout
                if priority == EventPriority.IMMEDIATE:
                    event = await asyncio.wait_for(queue.get(), timeout=0.1)
                else:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                
                await self._deliver_event(event)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur processeur {priority.name}: {e}")
    
    async def _deliver_event(self, event: Event):
        """
        Délivre un événement aux subscribers.
        
        Args:
            event: Événement à délivrer
        """
        # Collecte des subscribers concernés
        subscribers: List[Subscription] = []
        
        # Subscribers spécifiques au type
        if event.event_type in self._subscriptions:
            subscribers.extend(self._subscriptions[event.event_type])
        
        # Subscribers globaux
        subscribers.extend(self._all_subscriptions)
        
        for sub in subscribers:
            try:
                # Appliquer le filtre si présent
                if sub.filter_func and not sub.filter_func(event):
                    self._metrics['events_filtered'] += 1
                    continue
                
                # Appeler le callback
                if asyncio.iscoroutinefunction(sub.callback):
                    await sub.callback(event)
                else:
                    sub.callback(event)
                
                sub.event_count += 1
                self._metrics['events_delivered'] += 1
                
            except Exception as e:
                logger.error(f"Erreur livraison à {sub.subscriber_id}: {e}")
                self._metrics['delivery_errors'] += 1
    
    def _add_to_history(self, event: Event):
        """Ajoute un événement à l'historique."""
        self._event_history.append(event)
        
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size:]
    
    def get_history(
        self,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100
    ) -> List[Event]:
        """
        Récupère l'historique des événements.
        
        Args:
            event_type: Filtrer par type
            source: Filtrer par source
            limit: Nombre maximum
            
        Returns:
            Liste des événements
        """
        events = self._event_history[-limit:]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if source:
            events = [e for e in events if e.source == source]
        
        return events
    
    def get_subscriptions(self) -> Dict[str, int]:
        """
        Retourne le nombre de souscriptions par type.
        
        Returns:
            Dictionnaire type -> count
        """
        result = {
            event_type: len(subs)
            for event_type, subs in self._subscriptions.items()
        }
        result['*'] = len(self._all_subscriptions)
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques du bus.
        
        Returns:
            Dictionnaire des métriques
        """
        queue_sizes = {
            priority.name: self._queues[priority].qsize()
            for priority in EventPriority
        }
        
        return {
            **self._metrics,
            'queue_sizes': queue_sizes,
            'subscription_count': sum(len(s) for s in self._subscriptions.values()) + len(self._all_subscriptions),
            'history_size': len(self._event_history),
            'running': self._running,
        }


# Instance globale
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Retourne l'instance globale du bus événementiel."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
