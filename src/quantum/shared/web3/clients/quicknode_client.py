"""
Client WebSocket QuickNode multi-chain.

Ce module gère les connexions WebSocket vers les endpoints QuickNode
pour Ethereum, Solana et autres chaînes supportées.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
except ImportError:
    websockets = None  # type: ignore
    WebSocketClientProtocol = None  # type: ignore

try:
    from web3 import Web3
    from web3.providers import WebsocketProvider
except ImportError:
    Web3 = None  # type: ignore
    WebsocketProvider = None  # type: ignore

from quantum.shared.web3.settings import web3_config, Chain

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """État de la connexion WebSocket."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class SubscriptionInfo:
    """Information sur une souscription active."""
    subscription_id: str
    chain: Chain
    subscription_type: str  # 'pending_transactions', 'new_heads', etc.
    callback: Callable[[Dict], None]
    subscribed_at: datetime = field(default_factory=datetime.utcnow)
    message_count: int = 0


class QuickNodeClient:
    """
    Client WebSocket multi-chain pour QuickNode.
    
    DESCRIPTION:
    ============
    Gère les connexions WebSocket persistantes vers les endpoints
    QuickNode pour Ethereum et Solana, avec support pour les
    souscriptions temps réel.
    
    INNOVATION:
    ===========
    Connexion simultanée à Ethereum et Solana mempools pour 
    détecter les patterns cross-chain en temps réel. Utilise
    un système de callbacks asynchrones pour traitement non-bloquant.
    
    FONCTIONNALITÉS:
    ================
    - Connexion multi-chain (ETH, SOL, Polygon)
    - Auto-reconnexion avec backoff exponentiel
    - Souscription aux pending transactions (mempool)
    - Souscription aux nouveaux blocs
    - Gestion des heartbeats
    
    USAGE:
    ======
    ```python
    client = QuickNodeClient()
    await client.connect_ethereum()
    await client.subscribe_pending_transactions(my_callback)
    ```
    
    RISQUE ASSOCIÉ:
    ===============
    - Latence réseau peut causer des faux positifs
    - Déconnexions fréquentes en période de haute activité
    - Rate limiting sur endpoints QuickNode (selon plan)
    """
    
    def __init__(self):
        """
        Initialise le client QuickNode.
        
        DESCRIPTION:
        ============
        Configure les endpoints et initialise l'état interne
        des connexions.
        """
        self.config = web3_config.quicknode
        
        # État des connexions
        self._connections: Dict[Chain, Any] = {}
        self._connection_states: Dict[Chain, ConnectionState] = {
            chain: ConnectionState.DISCONNECTED for chain in Chain
        }
        
        # Web3 instances (pour ETH-like chains)
        self._web3_instances: Dict[Chain, Any] = {}
        
        # Souscriptions actives
        self._subscriptions: Dict[str, SubscriptionInfo] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        
        # Tasks d'écoute
        self._listener_tasks: Dict[Chain, asyncio.Task] = {}
        self._heartbeat_tasks: Dict[Chain, asyncio.Task] = {}
        
        # Reconnexion
        self._reconnect_attempts: Dict[Chain, int] = {chain: 0 for chain in Chain}
        
        logger.info("QuickNodeClient initialisé")
    
    @property
    def is_connected(self) -> Dict[Chain, bool]:
        """Retourne l'état de connexion de chaque chaîne."""
        return {
            chain: state == ConnectionState.CONNECTED
            for chain, state in self._connection_states.items()
        }
    
    async def connect_ethereum(self) -> bool:
        """
        Établit la connexion WebSocket vers Ethereum.
        
        DESCRIPTION:
        ============
        Se connecte à l'endpoint QuickNode Ethereum et initialise
        une instance Web3 pour les appels RPC.
        
        INNOVATION:
        ===========
        Utilise web3.py pour les appels synchrones et websockets
        raw pour les souscriptions streaming haute performance.
        
        RISQUE:
        =======
        - Timeout possible si endpoint surchargé
        - Certains endpoints QuickNode ne supportent pas eth_subscribe
        
        Returns:
            True si connexion réussie, False sinon
        """
        chain = Chain.ETHEREUM
        
        if self._connection_states[chain] == ConnectionState.CONNECTED:
            logger.info("Déjà connecté à Ethereum")
            return True
        
        self._connection_states[chain] = ConnectionState.CONNECTING
        
        try:
            endpoint = self.config.ETH_WSS_ENDPOINT
            
            if not endpoint:
                logger.error("QUICKNODE_ETH_ENDPOINT non configuré")
                self._connection_states[chain] = ConnectionState.ERROR
                return False
            
            # Connexion WebSocket raw pour streaming
            if websockets:
                ws = await asyncio.wait_for(
                    websockets.connect(
                        endpoint,
                        ping_interval=self.config.HEARTBEAT_INTERVAL_SECONDS,
                        ping_timeout=10,
                    ),
                    timeout=self.config.CONNECTION_TIMEOUT_SECONDS
                )
                self._connections[chain] = ws
            
            # Instance Web3 pour appels RPC
            if Web3 and WebsocketProvider:
                http_endpoint = self.config.ETH_HTTP_ENDPOINT
                self._web3_instances[chain] = Web3(Web3.HTTPProvider(http_endpoint))
            
            self._connection_states[chain] = ConnectionState.CONNECTED
            self._reconnect_attempts[chain] = 0
            
            # Démarrer les tâches d'écoute
            self._listener_tasks[chain] = asyncio.create_task(
                self._listen_ethereum()
            )
            
            logger.info(f"✅ Connecté à Ethereum via QuickNode")
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout connexion Ethereum")
            self._connection_states[chain] = ConnectionState.ERROR
            return False
            
        except Exception as e:
            logger.error(f"Erreur connexion Ethereum: {e}")
            self._connection_states[chain] = ConnectionState.ERROR
            return False
    
    async def connect_solana(self) -> bool:
        """
        Établit la connexion WebSocket vers Solana.
        
        DESCRIPTION:
        ============
        Se connecte à l'endpoint QuickNode Solana pour le streaming
        des transactions et programmes.
        
        INNOVATION:
        ===========
        Support des souscriptions spécifiques Solana comme
        logsSubscribe et programSubscribe pour les DEX (Jupiter, Raydium).
        
        RISQUE:
        =======
        - Les endpoints publics Solana sont souvent surchargés
        - QuickNode recommended pour production
        
        Returns:
            True si connexion réussie, False sinon
        """
        chain = Chain.SOLANA
        
        if self._connection_states[chain] == ConnectionState.CONNECTED:
            logger.info("Déjà connecté à Solana")
            return True
        
        self._connection_states[chain] = ConnectionState.CONNECTING
        
        try:
            endpoint = self.config.SOL_WSS_ENDPOINT
            
            if not endpoint:
                logger.error("QUICKNODE_SOL_ENDPOINT non configuré")
                self._connection_states[chain] = ConnectionState.ERROR
                return False
            
            # Connexion WebSocket
            if websockets:
                ws = await asyncio.wait_for(
                    websockets.connect(
                        endpoint,
                        ping_interval=self.config.HEARTBEAT_INTERVAL_SECONDS,
                        ping_timeout=10,
                    ),
                    timeout=self.config.CONNECTION_TIMEOUT_SECONDS
                )
                self._connections[chain] = ws
            
            self._connection_states[chain] = ConnectionState.CONNECTED
            self._reconnect_attempts[chain] = 0
            
            # Démarrer les tâches d'écoute
            self._listener_tasks[chain] = asyncio.create_task(
                self._listen_solana()
            )
            
            logger.info(f"✅ Connecté à Solana via QuickNode")
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout connexion Solana")
            self._connection_states[chain] = ConnectionState.ERROR
            return False
            
        except Exception as e:
            logger.error(f"Erreur connexion Solana: {e}")
            self._connection_states[chain] = ConnectionState.ERROR
            return False
    
    async def subscribe_pending_transactions(
        self,
        callback: Callable[[Dict], None],
        chain: Chain = Chain.ETHEREUM
    ) -> Optional[str]:
        """
        Souscrit aux transactions pendantes (mempool).
        
        DESCRIPTION:
        ============
        Active le streaming des transactions dans la mempool
        avant leur inclusion dans un bloc.
        
        INNOVATION:
        ===========
        Permet de voir les transactions 10-30 secondes avant
        qu'elles ne soient confirmées et visibles sur les CEX.
        
        Args:
            callback: Fonction appelée pour chaque nouvelle transaction
            chain: Chaîne cible (ETHEREUM par défaut)
        
        Returns:
            ID de souscription ou None si échec
            
        RISQUE:
        =======
        - Volume très élevé sur mainnet (milliers de TX/seconde)
        - Nécessite filtrage côté client pour éviter saturation
        """
        if self._connection_states[chain] != ConnectionState.CONNECTED:
            logger.error(f"Non connecté à {chain.value}")
            return None
        
        try:
            ws = self._connections.get(chain)
            if not ws:
                return None
            
            if chain == Chain.ETHEREUM:
                # eth_subscribe pour pending transactions
                subscription_request = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_subscribe",
                    "params": ["newPendingTransactions"]
                }
                
                await ws.send(json.dumps(subscription_request))
                response = await ws.recv()
                result = json.loads(response)
                
                if "result" in result:
                    sub_id = result["result"]
                    
                    # Enregistrer la souscription
                    self._subscriptions[sub_id] = SubscriptionInfo(
                        subscription_id=sub_id,
                        chain=chain,
                        subscription_type="pending_transactions",
                        callback=callback
                    )
                    
                    # Enregistrer le callback
                    if sub_id not in self._callbacks:
                        self._callbacks[sub_id] = []
                    self._callbacks[sub_id].append(callback)
                    
                    logger.info(f"✅ Souscrit aux pending TX Ethereum: {sub_id}")
                    return sub_id
                else:
                    logger.error(f"Erreur souscription: {result}")
                    return None
                    
            elif chain == Chain.SOLANA:
                # logsSubscribe pour Solana (plus pertinent que TX raw)
                subscription_request = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "logsSubscribe",
                    "params": [
                        {"mentions": ["JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB"]},  # Jupiter
                        {"commitment": "processed"}
                    ]
                }
                
                await ws.send(json.dumps(subscription_request))
                response = await ws.recv()
                result = json.loads(response)
                
                if "result" in result:
                    sub_id = str(result["result"])
                    
                    self._subscriptions[sub_id] = SubscriptionInfo(
                        subscription_id=sub_id,
                        chain=chain,
                        subscription_type="logs",
                        callback=callback
                    )
                    
                    if sub_id not in self._callbacks:
                        self._callbacks[sub_id] = []
                    self._callbacks[sub_id].append(callback)
                    
                    logger.info(f"✅ Souscrit aux logs Solana: {sub_id}")
                    return sub_id
                    
            return None
            
        except Exception as e:
            logger.error(f"Erreur souscription pending TX: {e}")
            return None
    
    async def subscribe_new_blocks(
        self,
        callback: Callable[[Dict], None],
        chain: Chain = Chain.ETHEREUM
    ) -> Optional[str]:
        """
        Souscrit aux nouveaux blocs.
        
        DESCRIPTION:
        ============
        Notification à chaque nouveau bloc validé sur la chaîne.
        
        Args:
            callback: Fonction appelée pour chaque nouveau bloc
            chain: Chaîne cible
            
        Returns:
            ID de souscription ou None si échec
        """
        if self._connection_states[chain] != ConnectionState.CONNECTED:
            logger.error(f"Non connecté à {chain.value}")
            return None
        
        try:
            ws = self._connections.get(chain)
            if not ws:
                return None
            
            if chain == Chain.ETHEREUM:
                subscription_request = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "eth_subscribe",
                    "params": ["newHeads"]
                }
                
                await ws.send(json.dumps(subscription_request))
                response = await ws.recv()
                result = json.loads(response)
                
                if "result" in result:
                    sub_id = result["result"]
                    
                    self._subscriptions[sub_id] = SubscriptionInfo(
                        subscription_id=sub_id,
                        chain=chain,
                        subscription_type="new_blocks",
                        callback=callback
                    )
                    
                    if sub_id not in self._callbacks:
                        self._callbacks[sub_id] = []
                    self._callbacks[sub_id].append(callback)
                    
                    logger.info(f"✅ Souscrit aux nouveaux blocs Ethereum: {sub_id}")
                    return sub_id
                    
            return None
            
        except Exception as e:
            logger.error(f"Erreur souscription blocs: {e}")
            return None
    
    async def get_transaction(self, tx_hash: str, chain: Chain = Chain.ETHEREUM) -> Optional[Dict]:
        """
        Récupère les détails d'une transaction.
        
        Args:
            tx_hash: Hash de la transaction
            chain: Chaîne cible
            
        Returns:
            Détails de la transaction ou None
        """
        try:
            if chain == Chain.ETHEREUM:
                w3 = self._web3_instances.get(chain)
                if w3:
                    tx = w3.eth.get_transaction(tx_hash)
                    return dict(tx) if tx else None
            return None
        except Exception as e:
            logger.error(f"Erreur récupération TX {tx_hash}: {e}")
            return None
    
    async def _listen_ethereum(self):
        """Boucle d'écoute pour Ethereum."""
        ws = self._connections.get(Chain.ETHEREUM)
        
        while ws and self._connection_states[Chain.ETHEREUM] == ConnectionState.CONNECTED:
            try:
                message = await ws.recv()
                data = json.loads(message)
                
                # Traiter les notifications de souscription
                if "method" in data and data["method"] == "eth_subscription":
                    sub_id = data.get("params", {}).get("subscription")
                    result = data.get("params", {}).get("result")
                    
                    if sub_id and sub_id in self._callbacks:
                        # Mettre à jour le compteur
                        if sub_id in self._subscriptions:
                            self._subscriptions[sub_id].message_count += 1
                        
                        # Appeler les callbacks
                        for callback in self._callbacks[sub_id]:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(result)
                                else:
                                    callback(result)
                            except Exception as e:
                                logger.error(f"Erreur callback: {e}")
                                
            except Exception as e:
                logger.error(f"Erreur écoute Ethereum: {e}")
                await self._handle_reconnection(Chain.ETHEREUM)
                break
    
    async def _listen_solana(self):
        """Boucle d'écoute pour Solana."""
        ws = self._connections.get(Chain.SOLANA)
        
        while ws and self._connection_states[Chain.SOLANA] == ConnectionState.CONNECTED:
            try:
                message = await ws.recv()
                data = json.loads(message)
                
                # Traiter les notifications
                if "method" in data and "Notification" in data["method"]:
                    sub_id = str(data.get("params", {}).get("subscription", ""))
                    result = data.get("params", {}).get("result")
                    
                    if sub_id and sub_id in self._callbacks:
                        if sub_id in self._subscriptions:
                            self._subscriptions[sub_id].message_count += 1
                        
                        for callback in self._callbacks[sub_id]:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(result)
                                else:
                                    callback(result)
                            except Exception as e:
                                logger.error(f"Erreur callback Solana: {e}")
                                
            except Exception as e:
                logger.error(f"Erreur écoute Solana: {e}")
                await self._handle_reconnection(Chain.SOLANA)
                break
    
    async def _handle_reconnection(self, chain: Chain):
        """Gère la reconnexion automatique."""
        self._connection_states[chain] = ConnectionState.RECONNECTING
        
        max_attempts = self.config.MAX_RECONNECT_ATTEMPTS
        delay = self.config.RECONNECT_DELAY_SECONDS
        
        while self._reconnect_attempts[chain] < max_attempts:
            self._reconnect_attempts[chain] += 1
            attempt = self._reconnect_attempts[chain]
            
            logger.info(f"Tentative reconnexion {chain.value} ({attempt}/{max_attempts})...")
            
            # Backoff exponentiel
            await asyncio.sleep(delay * (2 ** (attempt - 1)))
            
            success = False
            if chain == Chain.ETHEREUM:
                success = await self.connect_ethereum()
            elif chain == Chain.SOLANA:
                success = await self.connect_solana()
            
            if success:
                logger.info(f"✅ Reconnecté à {chain.value}")
                # Ré-établir les souscriptions
                await self._restore_subscriptions(chain)
                return
        
        logger.error(f"❌ Échec reconnexion {chain.value} après {max_attempts} tentatives")
        self._connection_states[chain] = ConnectionState.ERROR
    
    async def _restore_subscriptions(self, chain: Chain):
        """Restaure les souscriptions après reconnexion."""
        for sub_id, info in list(self._subscriptions.items()):
            if info.chain == chain:
                logger.info(f"Restauration souscription {info.subscription_type}...")
                
                if info.subscription_type == "pending_transactions":
                    new_sub_id = await self.subscribe_pending_transactions(
                        info.callback, chain
                    )
                    if new_sub_id:
                        # Mettre à jour les références
                        del self._subscriptions[sub_id]
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Se désabonne d'une souscription.
        
        Args:
            subscription_id: ID de la souscription
            
        Returns:
            True si désabonnement réussi
        """
        if subscription_id not in self._subscriptions:
            return False
        
        info = self._subscriptions[subscription_id]
        ws = self._connections.get(info.chain)
        
        if not ws:
            return False
        
        try:
            if info.chain == Chain.ETHEREUM:
                request = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_unsubscribe",
                    "params": [subscription_id]
                }
                await ws.send(json.dumps(request))
            
            del self._subscriptions[subscription_id]
            if subscription_id in self._callbacks:
                del self._callbacks[subscription_id]
            
            logger.info(f"Désabonné: {subscription_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur désabonnement: {e}")
            return False
    
    async def disconnect(self, chain: Optional[Chain] = None):
        """
        Ferme les connexions WebSocket.
        
        Args:
            chain: Chaîne spécifique ou None pour toutes
        """
        chains = [chain] if chain else list(Chain)
        
        for c in chains:
            if c in self._connections:
                try:
                    # Annuler les tâches
                    if c in self._listener_tasks:
                        self._listener_tasks[c].cancel()
                    if c in self._heartbeat_tasks:
                        self._heartbeat_tasks[c].cancel()
                    
                    # Fermer la connexion
                    ws = self._connections[c]
                    await ws.close()
                    
                    del self._connections[c]
                    self._connection_states[c] = ConnectionState.DISCONNECTED
                    
                    logger.info(f"Déconnecté de {c.value}")
                    
                except Exception as e:
                    logger.error(f"Erreur déconnexion {c.value}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Retourne le statut de toutes les connexions.
        
        Returns:
            Dictionnaire avec l'état de chaque connexion
        """
        return {
            'connections': {
                chain.value: {
                    'state': self._connection_states[chain].value,
                    'reconnect_attempts': self._reconnect_attempts[chain],
                }
                for chain in Chain
            },
            'subscriptions': {
                sub_id: {
                    'chain': info.chain.value,
                    'type': info.subscription_type,
                    'message_count': info.message_count,
                    'subscribed_at': info.subscribed_at.isoformat(),
                }
                for sub_id, info in self._subscriptions.items()
            },
            'total_subscriptions': len(self._subscriptions),
        }


# Instance globale
_quicknode_client: Optional[QuickNodeClient] = None


def get_quicknode_client() -> QuickNodeClient:
    """Retourne l'instance globale du client QuickNode."""
    global _quicknode_client
    if _quicknode_client is None:
        _quicknode_client = QuickNodeClient()
    return _quicknode_client
