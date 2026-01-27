"""
ScanCoordinator - Coordonne le scan multi-actifs en parallèle.
"""
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from quantum.shared.config.settings import config
from quantum.shared.utils.logger import get_logger

class ScanCoordinator:
    def __init__(self, system):
        self.system = system
        self.logger = get_logger("quantum.scanner")

    def scan_all_symbols(self) -> Dict[str, Dict]:
        symbols = config.symbols.ACTIVE_SYMBOLS
        results = {}
        
        self.logger.info("Démarrage scan parallèle", count=len(symbols))
        
        def _scan_one(symbol):
            try:
                analysis = self.system.analyze_symbol(symbol)
                return symbol, {"analysis": analysis}
            except Exception as e:
                return symbol, {"error": str(e)}

        with ThreadPoolExecutor(max_workers=min(len(symbols), 10)) as executor:
            future_to_symbol = {executor.submit(_scan_one, s): s for s in symbols}
            for future in tqdm(as_completed(future_to_symbol), total=len(symbols), desc="Scanning"):
                symbol, result = future.result()
                results[symbol] = result
                
        return results