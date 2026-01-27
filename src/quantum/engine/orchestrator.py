"""
Quantum Trading System - Orchestrateur Principal
===============================================
Author: Alexandre Albert Ndour
"""

import argparse
import sys
import os
import asyncio
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm

from quantum.shared.config.settings import config
from quantum.shared.utils.logger import get_logger
from quantum.domain.core.scorer import MultiCriteriaScorer
from quantum.shared.web3.engine import Web3IntelligenceEngine
from quantum.shared.web3.hooks.signal_dispatcher import IntegrationMode
from quantum.application.reporting.interface import TradingInterface
from quantum.application.reporting.scan_coordinator import ScanCoordinator
from quantum.domain.data.downloader import DataDownloader
from quantum.domain.data.kalman_filter import KalmanFilter
from quantum.domain.data.feature_engine import FeatureEngine
from quantum.domain.core.cointegration import CointegrationAnalyzer
from quantum.domain.core.hurst import HurstExponent
from quantum.domain.core.zscore import BollingerZScore
from quantum.domain.analysis.multi_tf import MultiTimeframeAnalyzer
from quantum.domain.analysis.smc import SmartMoneyConceptsAnalyzer
from quantum.domain.analysis.ichimoku import IchimokuAnalyzer
from quantum.domain.analysis.divergences import DivergenceDetector
from quantum.domain.analysis.wyckoff import WyckoffAnalyzer
from quantum.domain.ml.trainer import ModelTrainer
from quantum.domain.risk.manager import RiskManager
from quantum.domain.risk.circuit_breaker import CircuitBreaker
from quantum.domain.risk.calendar import EconomicCalendar
from quantum.application.backtest.engine import BacktestEngine


from quantum.domain.analysis.social_sentiment import SocialSentimentAnalyzer
from quantum.application.execution.service import ExecutionManager


class QuantumTradingSystem:
    def __init__(self):
        self.logger = get_logger("quantum.engine")
        self.logger.info("Initializing Quantum Trading System v3...")

        # Data
        self.downloader = DataDownloader()
        self.kalman = KalmanFilter()
        self.feature_engine = FeatureEngine()
        
        # Analysis
        self.coint_analyzer = CointegrationAnalyzer()
        self.hurst_calc = HurstExponent()
        self.zscore_calc = BollingerZScore()
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.smc_analyzer = SmartMoneyConceptsAnalyzer()
        self.ichimoku = IchimokuAnalyzer()
        self.divergence_detector = DivergenceDetector()
        self.wyckoff_analyzer = WyckoffAnalyzer()
        self.social_analyzer = SocialSentimentAnalyzer()
        
        # ML & Scorer
        self.ml_trainer = ModelTrainer()
        self.scorer = MultiCriteriaScorer()
        
        # Web3 Engine
        self.web3_engine = Web3IntelligenceEngine(mode=IntegrationMode.ENRICHMENT)
        
        # Risk, Interface & Execution
        self.risk_manager = RiskManager()
        self.circuit_breaker = CircuitBreaker()
        self.execution_manager = ExecutionManager(self.circuit_breaker)
        self.calendar = EconomicCalendar()
        self.interface = TradingInterface()
        self.scan_coordinator = ScanCoordinator(self)
        self.backtest_engine = BacktestEngine()
        
        self.data = {}
        self.logger.info("System Ready", status="ok")

    async def start(self):
        await self.web3_engine.start()

    async def stop(self):
        await self.web3_engine.stop()

    def load_data(self, symbol: str, force_download: bool = False):
        self.logger.info("Loading data", symbol=symbol)
        df = self.downloader.get_data(symbol, interval="1h", force_download=force_download)
        if df.empty: return df
        df = self.kalman.filter_dataframe(df, columns=['Close'])
        df = self.feature_engine.create_all_features(df)
        self.data[symbol] = df
        return df

    async def analyze_symbol(self, symbol: str):
        if symbol not in self.data: self.load_data(symbol)
        df = self.data.get(symbol)
        if df is None or df.empty: return {"error": "no data"}
        
        # 1. On-Chain & Social Intelligence
        onchain = self.web3_engine.get_current_analysis()
        social = self.social_analyzer.analyze_asset(symbol)
        
        # 2. Tech & Wyckoff
        wyckoff_result = self.wyckoff_analyzer.analyze(df)
        
        # 3. ML Inference
        ml_results = {'probability': 0.5}
        try:
            model_path = os.path.join(config.system.MODEL_DIR, f"{symbol}_model.pkl")
            if not self.ml_trainer.classifier.is_trained and os.path.exists(model_path):
                self.ml_trainer.load_model(model_path)
            
            if self.ml_trainer.classifier.is_trained:
                features_df = self.ml_trainer.preparer.prepare_features(df.tail(10))
                proba = self.ml_trainer.classifier.predict_proba(features_df.tail(1))
                ml_results['probability'] = float(proba[0])
        except Exception as e:
            self.logger.debug("Inférence ML non disponible", symbol=symbol, error=str(e))

        analysis = {
            'symbol': symbol,
            'price': df['Close'].iloc[-1],
            'onchain_data': onchain,
            'social_data': social,
            'ml_results': ml_results,
            'hurst_value': self.hurst_calc.calculate(df['Close']),
            'zscore_data': self.zscore_calc.get_current_status(df['Close']),
            'ichimoku_data': self.ichimoku.get_signal(df),
            'smc_data': self.smc_analyzer.analyze(df)['current_analysis'],
            'divergence_data': self.divergence_detector.get_divergence_signal(df),
            'wyckoff_phase': wyckoff_result.phase.value
        }
        
        # 4. Score Unifié
        score = self._compute_unified_score(analysis)
        analysis['combined_signal'] = score.direction
        analysis['confidence'] = score.confidence * 100
        
        # 5. Tentative d'Auto-Exécution (Live Trading)
        if analysis['confidence'] >= 85 and score.direction in ["BUY", "SELL"]:
            await self.execution_manager.execute_signal(
                symbol, score.direction, analysis['confidence'], analysis['price']
            )
            
        return analysis

    def _compute_unified_score(self, analysis):
        # Data mapping for scorer
        tech = {
            'kumo_position': analysis['ichimoku_data'].get('kumo_position'), 
            'divergence': analysis['divergence_data'].get('signal'),
            'wyckoff_phase': analysis['wyckoff_phase']
        }
        ml = analysis.get('ml_results', {})
        stat = {
            'zscore': analysis['zscore_data'].get('zscore'), 
            'hurst': analysis['hurst_value']
        }
        social = analysis.get('social_data')
        risk = {'circuit_breaker_active': self.circuit_breaker.is_active()}
        
        return self.scorer.calculate_score(tech, ml, analysis['onchain_data'], stat, social, risk)

    def scan_all_symbols(self):
        self.logger.info("Scanning Market...")
        results = self.scan_coordinator.scan_all_symbols()
        self.interface.print_scan_report(results)
        return results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="analyze")
    parser.add_argument("--symbol", default="EURUSD=X")
    args = parser.parse_args()
    
    system = QuantumTradingSystem()
    try:
        await system.start()
        
        if args.mode == "scan":
            system.scan_all_symbols()
        else:
            res = system.analyze_symbol(args.symbol)
            print(f"ANALYSIS RESULT: {res['combined_signal']} (Confidence: {res['confidence']:.1f}%)")
            
    finally:
        await system.stop()

if __name__ == "__main__":
    asyncio.run(main())
