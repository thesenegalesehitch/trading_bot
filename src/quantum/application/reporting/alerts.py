"""
Module de système d'alertes multi-canal.
Envoie des notifications via Telegram, Discord, Email et Webhooks.

Fonctionnalités:
- Alertes de signaux de trading
- Notifications de risque
- Résumés journaliers
- Alertes personnalisables
"""

import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import os
import sys


from quantum.shared.config.settings import config


class AlertLevel(Enum):
    """Niveaux d'alerte."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    CRITICAL = "critical"
    SIGNAL = "signal"


@dataclass
class Alert:
    """Représente une alerte."""
    title: str
    message: str
    level: AlertLevel
    timestamp: datetime
    data: Optional[Dict] = None


class TelegramNotifier:
    """
    Notifications via Telegram Bot.
    
    Prérequis:
    1. Créer un bot via @BotFather
    2. Obtenir le token
    3. Obtenir le chat_id (envoyer un message au bot puis utiliser l'API getUpdates)
    """
    
    def __init__(
        self,
        token: str = None,
        chat_id: str = None
    ):
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID', '')
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self._available = bool(self.token and self.chat_id)
    
    def is_available(self) -> bool:
        return self._available
    
    def send(self, alert: Alert) -> bool:
        """Envoie une alerte via Telegram."""
        if not self._available:
            return False
        
        # Emoji selon le niveau
        emoji_map = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.SUCCESS: "✅",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.SIGNAL: "📊"
        }
        emoji = emoji_map.get(alert.level, "📌")
        
        # Formater le message
        message = f"{emoji} *{alert.title}*\n\n{alert.message}"
        
        if alert.data:
            message += "\n\n*Détails:*\n"
            for key, value in alert.data.items():
                message += f"• {key}: `{value}`\n"
        
        message += f"\n_{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}_"
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            print(f"❌ Erreur Telegram: {e}")
            return False
    
    def send_signal(
        self,
        symbol: str,
        signal: str,
        price: float,
        confidence: float,
        stop_loss: float = None,
        take_profit: float = None
    ) -> bool:
        """Envoie un signal de trading formaté."""
        direction_emoji = "🟢" if signal in ["BUY", "STRONG_BUY"] else "🔴" if signal in ["SELL", "STRONG_SELL"] else "⚪"
        
        message = f"{direction_emoji} Signal {signal}\n\n"
        message += f"📈 Symbole: {symbol}\n"
        message += f"💰 Prix: {price}\n"
        message += f"🎯 Confiance: {confidence}%\n"
        
        if stop_loss:
            message += f"🛑 Stop Loss: {stop_loss}\n"
        if take_profit:
            message += f"✨ Take Profit: {take_profit}\n"
        
        alert = Alert(
            title=f"Signal {symbol}",
            message=message,
            level=AlertLevel.SIGNAL,
            timestamp=datetime.now()
        )
        
        return self.send(alert)


class DiscordNotifier:
    """
    Notifications via Discord Webhook.
    
    Prérequis:
    1. Créer un webhook dans les paramètres du serveur
    2. Copier l'URL du webhook
    """
    
    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL', '')
        self._available = bool(self.webhook_url)
    
    def is_available(self) -> bool:
        return self._available
    
    def send(self, alert: Alert) -> bool:
        """Envoie une alerte via Discord."""
        if not self._available:
            return False
        
        # Couleur selon le niveau
        color_map = {
            AlertLevel.INFO: 0x3498DB,      # Bleu
            AlertLevel.SUCCESS: 0x2ECC71,   # Vert
            AlertLevel.WARNING: 0xF39C12,   # Orange
            AlertLevel.CRITICAL: 0xE74C3C,  # Rouge
            AlertLevel.SIGNAL: 0x9B59B6     # Violet
        }
        color = color_map.get(alert.level, 0x95A5A6)
        
        # Construire l'embed
        embed = {
            "title": alert.title,
            "description": alert.message,
            "color": color,
            "timestamp": alert.timestamp.isoformat(),
            "footer": {"text": "Quantum Trading System"}
        }
        
        if alert.data:
            embed["fields"] = [
                {"name": str(k), "value": str(v), "inline": True}
                for k, v in alert.data.items()
            ]
        
        payload = {"embeds": [embed]}
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            return response.status_code in [200, 204]
            
        except Exception as e:
            print(f"❌ Erreur Discord: {e}")
            return False
    
    def send_signal(
        self,
        symbol: str,
        signal: str,
        price: float,
        confidence: float,
        analysis: Dict = None
    ) -> bool:
        """Envoie un signal de trading avec embed rich."""
        color = 0x2ECC71 if "BUY" in signal else 0xE74C3C if "SELL" in signal else 0x95A5A6
        
        embed = {
            "title": f"📊 Signal de Trading: {symbol}",
            "color": color,
            "fields": [
                {"name": "Signal", "value": signal, "inline": True},
                {"name": "Prix", "value": str(price), "inline": True},
                {"name": "Confiance", "value": f"{confidence}%", "inline": True}
            ],
            "timestamp": datetime.now().isoformat(),
            "footer": {"text": "Quantum Trading System"}
        }
        
        if analysis:
            for key, value in list(analysis.items())[:6]:
                embed["fields"].append({
                    "name": str(key).replace("_", " ").title(),
                    "value": str(value),
                    "inline": True
                })
        
        try:
            response = requests.post(
                self.webhook_url,
                json={"embeds": [embed]},
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            return response.status_code in [200, 204]
            
        except Exception as e:
            print(f"❌ Erreur Discord: {e}")
            return False


class EmailNotifier:
    """
    Notifications par email via SMTP.
    
    Supporte Gmail, Outlook et serveurs SMTP personnalisés.
    """
    
    def __init__(
        self,
        smtp_server: str = None,
        smtp_port: int = None,
        username: str = None,
        password: str = None,
        from_email: str = None,
        to_email: str = None
    ):
        self.smtp_server = smtp_server or os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = smtp_port or int(os.getenv('SMTP_PORT', '587'))
        self.username = username or os.getenv('SMTP_USERNAME', '')
        self.password = password or os.getenv('SMTP_PASSWORD', '')
        self.from_email = from_email or os.getenv('EMAIL_FROM', '')
        self.to_email = to_email or os.getenv('EMAIL_TO', '')
        
        self._available = bool(self.username and self.password and self.to_email)
    
    def is_available(self) -> bool:
        return self._available
    
    def send(self, alert: Alert) -> bool:
        """Envoie une alerte par email."""
        if not self._available:
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[QTS] {alert.level.value.upper()}: {alert.title}"
            msg['From'] = self.from_email
            msg['To'] = self.to_email
            
            # Version texte
            text = f"{alert.title}\n\n{alert.message}"
            if alert.data:
                text += "\n\nDétails:\n"
                for k, v in alert.data.items():
                    text += f"- {k}: {v}\n"
            
            # Version HTML
            html = f"""
            <html>
            <body style="font-family: Arial, sans-serif; padding: 20px;">
                <h2 style="color: #333;">{alert.title}</h2>
                <p style="font-size: 14px; line-height: 1.6;">{alert.message.replace(chr(10), '<br>')}</p>
            """
            
            if alert.data:
                html += '<table style="border-collapse: collapse; margin-top: 20px;">'
                for k, v in alert.data.items():
                    html += f"""
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd; background: #f5f5f5;"><strong>{k}</strong></td>
                        <td style="padding: 8px; border: 1px solid #ddd;">{v}</td>
                    </tr>
                    """
                html += '</table>'
            
            html += f"""
                <p style="color: #888; font-size: 12px; margin-top: 20px;">
                    Envoyé le {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(text, 'plain'))
            msg.attach(MIMEText(html, 'html'))
            
            # Envoyer
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur Email: {e}")
            return False


class WebhookNotifier:
    """
    Notifications via Webhook personnalisé.
    
    Envoie des requêtes POST JSON à une URL configurée.
    """
    
    def __init__(self, webhook_url: str = None, headers: Dict = None):
        self.webhook_url = webhook_url or os.getenv('CUSTOM_WEBHOOK_URL', '')
        self.headers = headers or {'Content-Type': 'application/json'}
        self._available = bool(self.webhook_url)
    
    def is_available(self) -> bool:
        return self._available
    
    def send(self, alert: Alert) -> bool:
        """Envoie une alerte via webhook."""
        if not self._available:
            return False
        
        payload = {
            'title': alert.title,
            'message': alert.message,
            'level': alert.level.value,
            'timestamp': alert.timestamp.isoformat(),
            'data': alert.data or {}
        }
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            return response.status_code in [200, 201, 204]
            
        except Exception as e:
            print(f"❌ Erreur Webhook: {e}")
            return False


class AlertManager:
    """
    Gestionnaire centralisé des alertes.
    
    Gère l'envoi vers tous les canaux configurés.
    """
    
    def __init__(self):
        self.notifiers = {}
        self._initialize_notifiers()
        self.alert_history: List[Alert] = []
    
    def _initialize_notifiers(self):
        """Initialise tous les notifiers disponibles."""
        telegram = TelegramNotifier()
        if telegram.is_available():
            self.notifiers['telegram'] = telegram
            print("✅ Telegram configuré")
        
        discord = DiscordNotifier()
        if discord.is_available():
            self.notifiers['discord'] = discord
            print("✅ Discord configuré")
        
        email = EmailNotifier()
        if email.is_available():
            self.notifiers['email'] = email
            print("✅ Email configuré")
        
        webhook = WebhookNotifier()
        if webhook.is_available():
            self.notifiers['webhook'] = webhook
            print("✅ Webhook personnalisé configuré")
        
        if not self.notifiers:
            print("⚠️ Aucun canal de notification configuré")
    
    def send_alert(
        self,
        title: str,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        data: Dict = None,
        channels: List[str] = None
    ) -> Dict[str, bool]:
        """
        Envoie une alerte sur les canaux spécifiés.
        
        Args:
            title: Titre de l'alerte
            message: Message
            level: Niveau d'alerte
            data: Données additionnelles
            channels: Canaux à utiliser (tous si None)
        
        Returns:
            Dict avec statut d'envoi par canal
        """
        alert = Alert(
            title=title,
            message=message,
            level=level,
            timestamp=datetime.now(),
            data=data
        )
        
        self.alert_history.append(alert)
        
        results = {}
        target_channels = channels or list(self.notifiers.keys())
        
        for channel in target_channels:
            if channel in self.notifiers:
                results[channel] = self.notifiers[channel].send(alert)
            else:
                results[channel] = False
        
        return results
    
    def send_signal(
        self,
        symbol: str,
        signal: str,
        price: float,
        confidence: float,
        stop_loss: float = None,
        take_profit: float = None,
        analysis: Dict = None
    ) -> Dict[str, bool]:
        """Envoie un signal de trading sur tous les canaux."""
        results = {}
        
        for name, notifier in self.notifiers.items():
            if hasattr(notifier, 'send_signal'):
                if name == 'telegram':
                    results[name] = notifier.send_signal(
                        symbol, signal, price, confidence, stop_loss, take_profit
                    )
                elif name == 'discord':
                    results[name] = notifier.send_signal(
                        symbol, signal, price, confidence, analysis
                    )
            else:
                alert = Alert(
                    title=f"Signal {symbol}",
                    message=f"Signal: {signal}\nPrix: {price}\nConfiance: {confidence}%",
                    level=AlertLevel.SIGNAL,
                    timestamp=datetime.now(),
                    data={"stop_loss": stop_loss, "take_profit": take_profit}
                )
                results[name] = notifier.send(alert)
        
        return results
    
    def send_risk_alert(
        self,
        risk_type: str,
        current_value: float,
        threshold: float,
        action_required: str
    ) -> Dict[str, bool]:
        """Envoie une alerte de risque."""
        return self.send_alert(
            title=f"⚠️ Alerte Risque: {risk_type}",
            message=f"Valeur actuelle: {current_value}\nSeuil: {threshold}\n\n{action_required}",
            level=AlertLevel.WARNING,
            data={
                "type": risk_type,
                "current": current_value,
                "threshold": threshold
            }
        )
    
    def send_daily_summary(
        self,
        date: str,
        pnl: float,
        trades: int,
        win_rate: float,
        best_trade: str,
        worst_trade: str
    ) -> Dict[str, bool]:
        """Envoie un résumé journalier."""
        emoji = "📈" if pnl > 0 else "📉"
        
        return self.send_alert(
            title=f"{emoji} Résumé du {date}",
            message=f"Performance de la journée",
            level=AlertLevel.INFO,
            data={
                "P&L": f"${pnl:+.2f}",
                "Trades": trades,
                "Win Rate": f"{win_rate:.1f}%",
                "Meilleur": best_trade,
                "Pire": worst_trade
            }
        )
    
    def get_history(self, limit: int = 50) -> List[Dict]:
        """Retourne l'historique des alertes."""
        return [
            {
                "title": a.title,
                "message": a.message[:100],
                "level": a.level.value,
                "timestamp": a.timestamp.isoformat()
            }
            for a in self.alert_history[-limit:]
        ]


if __name__ == "__main__":
    print("=" * 60)
    print("TEST SYSTÈME D'ALERTES")
    print("=" * 60)
    
    # Initialiser le manager
    manager = AlertManager()
    
    print(f"\nCanaux disponibles: {list(manager.notifiers.keys())}")
    
    # Test d'alerte (sans envoyer réellement si pas configuré)
    print("\n--- Test d'alerte ---")
    
    alert = Alert(
        title="Test du Système",
        message="Ceci est un test du système d'alertes Quantum Trading.",
        level=AlertLevel.INFO,
        timestamp=datetime.now(),
        data={
            "version": "1.0",
            "mode": "test"
        }
    )
    
    print(f"Alerte créée: {alert.title}")
    print(f"Niveau: {alert.level.value}")
    
    # Test signal (affichage seulement)
    print("\n--- Exemple de Signal ---")
    print("Signal: BUY EURUSD")
    print("Prix: 1.0850")
    print("Confiance: 85%")
    print("Stop Loss: 1.0820")
    print("Take Profit: 1.0910")
    
    # Historique
    print("\n--- Configuration des Alertes ---")
    print("Variables d'environnement requises:")
    print("  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
    print("  DISCORD_WEBHOOK_URL")
    print("  SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD")
    print("  EMAIL_FROM, EMAIL_TO")
    print("  CUSTOM_WEBHOOK_URL")
