import json
import logging
import os
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any

class AuditLogger:
    """
    Système de journalisation d'audit transactionnel.
    Enregistre toutes les actions de trading dans un fichier JSONL avec un hash
    pour garantir l'intégrité de la séquence (Append-Only).
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.audit_file = os.path.join(log_dir, "audit_trail.jsonl")
        self._ensure_dir()
        self._last_hash = self._get_last_hash()
        
    def _ensure_dir(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
    def _get_last_hash(self) -> str:
        """Récupère le hash de la dernière entrée pour chaîner les logs."""
        if not os.path.exists(self.audit_file):
            return "GENESIS_HASH_00000000000000000000"
            
        try:
            with open(self.audit_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_entry = json.loads(lines[-1])
                    return last_entry.get("hash", "ERROR_NO_HASH")
        except Exception:
            return "ERROR_READING_HASH"
        return "GENESIS_HASH_00000000000000000000"
        
    def log_trade_execution(self, symbol: str, side: str, execution_type: str, details: Dict[str, Any]):
        """
        Enregistre une exécution de trade de manière immuable.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        payload = {
            "timestamp": timestamp,
            "event_type": "TRADE_EXECUTION",
            "symbol": symbol,
            "side": side,
            "execution_type": execution_type,  # 'LIVE' ou 'SIMULATED'
            "details": details,
            "previous_hash": self._last_hash
        }
        
        # Calcul du hash de l'entrée actuelle
        payload_str = json.dumps(payload, sort_keys=True)
        current_hash = hashlib.sha256(payload_str.encode('utf-8')).hexdigest()
        
        # Ajouter le hash au payload
        payload["hash"] = current_hash
        
        # Écriture Append-Only
        try:
            with open(self.audit_file, 'a') as f:
                f.write(json.dumps(payload) + "\n")
            self._last_hash = current_hash
        except Exception as e:
            logging.error(f"Erreur CRITIQUE d'audit log: {e}")
            # Ne pas crasher le système pour un log, mais alerter
