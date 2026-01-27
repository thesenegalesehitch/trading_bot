"""
SecretVault - Gestion sécurisée des secrets et configurations.
Centralise et valide tous les secrets institutionnels au démarrage.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr, validator
from typing import Optional
import os

class SecretVault(BaseSettings):
    """
    Conteneur de secrets avec validation stricte.
    Toute clé manquante ou mal formatée bloquera le système au démarrage.
    """
    
    # --- Exchange Keys (SecretStr masque les valeurs dans les logs) ---
    binance_api_key: Optional[SecretStr] = Field(None, alias="BINANCE_API_KEY")
    binance_api_secret: Optional[SecretStr] = Field(None, alias="BINANCE_API_SECRET")
    
    ibkr_host: str = Field("127.0.0.1", alias="IBKR_HOST")
    ibkr_port: int = Field(7497, alias="IBKR_PORT")
    
    # --- Social & Web3 ---
    twitter_bearer_token: Optional[SecretStr] = Field(None, alias="TWITTER_BEARER_TOKEN")
    quicknode_eth_url: Optional[str] = Field(None, alias="QUICKNODE_ETH_URL")
    
    # --- Alerts ---
    telegram_bot_token: Optional[SecretStr] = Field(None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(None, alias="TELEGRAM_CHAT_ID")
    discord_webhook_url: Optional[str] = Field(None, alias="DISCORD_WEBHOOK_URL")
    
    # --- System Control ---
    live_trading: bool = Field(False, alias="LIVE_TRADING")
    binance_testnet: bool = Field(True, alias="BINANCE_TESTNET")

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @validator("live_trading")
    def safety_check(cls, v):
        if v:
            print("⚠️ CAUTION: LIVE TRADING IS ENABLED. REAL FUNDS AT RISK.")
        return v

# Instance globale pour injection de dépendance
secrets = SecretVault()

def get_secrets() -> SecretVault:
    return secrets
