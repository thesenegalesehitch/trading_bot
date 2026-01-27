"""
Système de hooks pour l'intégration avec le système principal.
"""

from web3_innovation.hooks.signal_dispatcher import SignalDispatcher, IntegrationMode
from web3_innovation.hooks.event_bus import EventBus

__all__ = ["SignalDispatcher", "IntegrationMode", "EventBus"]
