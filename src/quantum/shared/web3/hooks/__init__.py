"""
Système de hooks pour l'intégration avec le système principal.
"""

from quantum.shared.web3.hooks.signal_dispatcher import SignalDispatcher, IntegrationMode
from quantum.shared.web3.hooks.event_bus import EventBus

__all__ = ["SignalDispatcher", "IntegrationMode", "EventBus"]
