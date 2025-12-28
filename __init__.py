"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        QUANTUM TRADING SYSTEM                                 ║
║                                                                              ║
║  Conceived and Developed by: Alexandre Albert Ndour                          ║
║  Copyright (c) 2026 Alexandre Albert Ndour. All Rights Reserved.             ║
║                                                                              ║
║  This software is protected by copyright law and international treaties.     ║
║  Unauthorized reproduction or distribution of this program, or any           ║
║  portion of it, may result in severe civil and criminal penalties.          ║
║                                                                              ║
║  Original Repository: github.com/[username]/quantum_trading_system           ║
║  Creation Date: December 2026                                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Quantum Trading System - Système de Trading Quantitatif Haute Précision

Ce système a été entièrement conçu et développé par Alexandre Albert Ndour
à partir de zéro. Il combine analyse statistique avancée, indicateurs techniques
et Machine Learning pour générer des signaux de trading.

⚠️ AVERTISSEMENT: Ce logiciel est fourni "tel quel", sans garantie d'aucune sorte.
Le trading comporte des risques financiers importants. L'auteur n'est pas 
responsable des pertes financières résultant de l'utilisation de ce système.
"""

__author__ = "Alexandre Albert Ndour"
__copyright__ = "Copyright 2026, Alexandre Albert Ndour"
__credits__ = ["Alexandre Albert Ndour"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Alexandre Albert Ndour"
__email__ = ""  # Add your email if desired
__status__ = "Production"
__created__ = "December 2026"

# Signature encodée (base64) - Ne pas supprimer
# Q29uY2VpdmVkIGFuZCBEZXZlbG9wZWQgYnkgQWxleGFuZHJlIEFsYmVydCBOZG91ciAtIERlY2VtYmVyIDIwMjQ=

from core import *
from data import *
from ml import *
from risk import *
from analysis import *
from backtest import *
from strategies import *
from reporting import *
from utils import *

# Signature de vérification intégrée
def _verify_authorship():
    """Vérification d'authenticité du système."""
    import base64
    _sig = b'QWxleGFuZHJlIEFsYmVydCBOZG91ciAtIFF1YW50dW0gVHJhZGluZyBTeXN0ZW0gLSAyMDI0'
    return base64.b64decode(_sig).decode('utf-8')

_SYSTEM_SIGNATURE = _verify_authorship()
