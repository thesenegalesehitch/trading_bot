# Innovation 2: What If Simulator Module
# Allows replaying any trading scenario
# "What if j'avais bought à 1.0850 le 15 Mars?"

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

@dataclass
class WhatIfScenario:
    """Scenario to simulate"""
    symbol: str
    entry_price: float
    entry_date: datetime
    side: str  # 'BUY' or 'SELL'
    # Trade parameters
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None  # Percentage
    # Simulation settings
    simulation_end_date: Optional[datetime] = None
    
@dataclass
class SimulationPoint:
    """A single point in the simulation"""
    timestamp: datetime
    price: float
    high: float
    low: float
    stop_loss_hit: bool = False
    take_profit_hit: bool = False
    trailing_stop_hit: bool = False
    
@dataclass
class WhatIfResult:
    """Result of what-if simulation"""
    scenario: WhatIfScenario
    # Results
    result: str  # 'WIN', 'LOSS', 'STOP_LOSS', 'TAKE_PROFIT', 'OPEN', 'BREAKEVEN'
    pips_gained: float
    percentage_gain: float
    max_profit: float
    max_profit_timestamp: Optional[datetime]
    max_loss: float
    max_loss_timestamp: Optional[datetime]
    # Exit details
    exit_price: Optional[float]
    exit_reason: str
    exit_timestamp: Optional[datetime]
    # Analysis
    verdict: str
    recommendations: List[str]
    # Detailed timeline
    timeline: List[SimulationPoint] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class WhatIfSimulator:
    """
    Simulates any trading scenario to analyze what would have happened.
    
    Mission: "Permettre de replay n'importe quel scénario"
    
    This agent retrieves minute-by-minute data from the specified date,
    simulates various exit strategies, and generates a verdict.
    """
    
    def __init__(self, data_provider=None):
        """
        Initialize the simulator.
        
        Args:
            data_provider: Optional data provider for historical prices
        """
        self.data_provider = data_provider
        self.simulation_cache: Dict[str, WhatIfResult] = {}
    
    def simulate(self, scenario: WhatIfScenario) -> WhatIfResult:
        """
        Main entry point to simulate a what-if scenario.
        
        Args:
            scenario: The scenario to simulate
            
        Returns:
            WhatIfResult with detailed simulation results
        """
        # Get historical data for the simulation period
        data = self._get_historical_data(scenario)
        
        if not data:
            return self._create_error_result(scenario, "Données historiques non disponibles")
        
        # Run simulation
        timeline = self._run_simulation(scenario, data)
        
        # Calculate results
        result = self._calculate_results(scenario, timeline)
        
        # Generate verdict and recommendations
        result.verdict = self._generate_verdict(result)
        result.recommendations = self._generate_recommendations(result)
        
        # Cache result
        cache_key = f"{scenario.symbol}_{scenario.entry_date.isoformat()}"
        self.simulation_cache[cache_key] = result
        
        return result
    
    def _get_historical_data(
        self, 
        scenario: WhatIfScenario
    ) -> List[Dict[str, Any]]:
        """Get historical price data for the simulation period"""
        
        # If data provider is available, use it
        if self.data_provider:
            end_date = scenario.simulation_end_date or datetime.now()
            return self.data_provider.get_historical_data(
                symbol=scenario.symbol,
                start_date=scenario.entry_date,
                end_date=end_date,
                interval='1h'  # Hourly data for simulation
            )
        
        # Otherwise, return empty list (would need real implementation)
        return []
    
    def _run_simulation(
        self, 
        scenario: WhatIfScenario,
        data: List[Dict[str, Any]]
    ) -> List[SimulationPoint]:
        """Run the simulation on historical data"""
        
        timeline = []
        
        # Track best prices
        best_entry = scenario.entry_price
        
        if scenario.side == 'BUY':
            # For BUY: track max profit and loss
            peak_price = scenario.entry_price
            trough_price = scenario.entry_price
            
            for point in data:
                timestamp = point.get('timestamp')
                price = point.get('close', 0)
                high = point.get('high', 0)
                low = point.get('low', 0)
                
                # Update peak/trough
                if price > peak_price:
                    peak_price = price
                if price < trough_price:
                    trough_price = price
                
                sim_point = SimulationPoint(
                    timestamp=timestamp,
                    price=price,
                    high=high,
                    low=low
                )
                
                # Check stop loss
                if scenario.stop_loss and low <= scenario.stop_loss:
                    sim_point.stop_loss_hit = True
                    sim_point.take_profit_hit = False
                    sim_point.trailing_stop_hit = False
                
                # Check take profit
                elif scenario.take_profit and high >= scenario.take_profit:
                    sim_point.stop_loss_hit = False
                    sim_point.take_profit_hit = True
                    sim_point.trailing_stop_hit = False
                
                # Check trailing stop
                elif scenario.trailing_stop and peak_price:
                    trailing_stop_level = peak_price * (1 - scenario.trailing_stop / 100)
                    if price <= trailing_stop_level:
                        sim_point.stop_loss_hit = False
                        sim_point.take_profit_hit = False
                        sim_point.trailing_stop_hit = True
                
                timeline.append(sim_point)
        
        else:  # SELL
            # For SELL: opposite logic
            peak_price = scenario.entry_price
            trough_price = scenario.entry_price
            
            for point in data:
                timestamp = point.get('timestamp')
                price = point.get('close', 0)
                high = point.get('high', 0)
                low = point.get('low', 0)
                
                # Update peak/trough
                if price > peak_price:
                    peak_price = price
                if price < trough_price:
                    trough_price = price
                
                sim_point = SimulationPoint(
                    timestamp=timestamp,
                    price=price,
                    high=high,
                    low=low
                )
                
                # Check stop loss (for SELL, stop loss is above entry)
                if scenario.stop_loss and high >= scenario.stop_loss:
                    sim_point.stop_loss_hit = True
                
                # Check take profit (for SELL, TP is below entry)
                elif scenario.take_profit and low <= scenario.take_profit:
                    sim_point.take_profit_hit = True
                
                # Check trailing stop
                elif scenario.trailing_stop and trough_price:
                    trailing_stop_level = trough_price * (1 + scenario.trailing_stop / 100)
                    if price >= trailing_stop_level:
                        sim_point.trailing_stop_hit = True
                
                timeline.append(sim_point)
        
        return timeline
    
    def _calculate_results(
        self, 
        scenario: WhatIfScenario,
        timeline: List[SimulationPoint]
    ) -> WhatIfResult:
        """Calculate simulation results"""
        
        if not timeline:
            return self._create_error_result(scenario, "Pas de données pour la simulation")
        
        entry_price = scenario.entry_price
        
        # Initialize tracking
        max_profit = 0.0
        max_profit_timestamp = None
        max_loss = 0.0
        max_loss_timestamp = None
        exit_price = None
        exit_reason = "OPEN"
        exit_timestamp = None
        
        # Track for trailing stop
        peak_price = entry_price if scenario.side == 'BUY' else entry_price
        trailing_stop_level = None
        
        for point in timeline:
            price = point.price
            
            if scenario.side == 'BUY':
                profit = ((price - entry_price) / entry_price) * 100
                if profit > max_profit:
                    max_profit = profit
                    max_profit_timestamp = point.timestamp
                    # Update trailing stop
                    if scenario.trailing_stop:
                        peak_price = price
                        trailing_stop_level = peak_price * (1 - scenario.trailing_stop / 100)
                
                if profit < max_loss:
                    max_loss = profit
                    max_loss_timestamp = point.timestamp
                
                # Check exits
                if point.stop_loss_hit:
                    exit_price = scenario.stop_loss
                    exit_reason = "STOP_LOSS"
                    exit_timestamp = point.timestamp
                    break
                elif point.take_profit_hit:
                    exit_price = scenario.take_profit
                    exit_reason = "TAKE_PROFIT"
                    exit_timestamp = point.timestamp
                    break
                elif point.trailing_stop_hit:
                    exit_price = trailing_stop_level
                    exit_reason = "TRAILING_STOP"
                    exit_timestamp = point.timestamp
                    break
                    
            else:  # SELL
                profit = ((entry_price - price) / entry_price) * 100
                if profit > max_profit:
                    max_profit = profit
                    max_profit_timestamp = point.timestamp
                    if scenario.trailing_stop:
                        peak_price = price
                        trailing_stop_level = peak_price * (1 + scenario.trailing_stop / 100)
                
                if profit < max_loss:
                    max_loss = profit
                    max_loss_timestamp = point.timestamp
                
                if point.stop_loss_hit:
                    exit_price = scenario.stop_loss
                    exit_reason = "STOP_LOSS"
                    exit_timestamp = point.timestamp
                    break
                elif point.take_profit_hit:
                    exit_price = scenario.take_profit
                    exit_reason = "TAKE_PROFIT"
                    exit_timestamp = point.timestamp
                    break
                elif point.trailing_stop_hit:
                    exit_price = trailing_stop_level
                    exit_reason = "TRAILING_STOP"
                    exit_timestamp = point.timestamp
                    break
        
        # If no exit triggered, use last price
        if exit_price is None:
            last_point = timeline[-1]
            exit_price = last_point.price
            exit_reason = "OPEN"
            exit_timestamp = last_point.timestamp
            if scenario.side == 'BUY':
                max_profit = ((exit_price - entry_price) / entry_price) * 100
            else:
                max_profit = ((entry_price - exit_price) / entry_price) * 100
            max_loss = max_profit  # For open position, current is both max profit and loss
        
        # Calculate final pips
        if scenario.side == 'BUY':
            pips = (exit_price - entry_price) * 10000  # For forex
            percentage = ((exit_price - entry_price) / entry_price) * 100
        else:
            pips = (entry_price - exit_price) * 10000
            percentage = ((entry_price - exit_price) / entry_price) * 100
        
        # Determine result type
        if exit_reason == "STOP_LOSS":
            result_type = "LOSS"
        elif exit_reason == "TAKE_PROFIT":
            result_type = "WIN"
        elif percentage > 0:
            result_type = "WIN"
        elif percentage < 0:
            result_type = "LOSS"
        else:
            result_type = "BREAKEVEN"
        
        return WhatIfResult(
            scenario=scenario,
            result=result_type,
            pips_gained=pips,
            percentage_gain=percentage,
            max_profit=max_profit,
            max_profit_timestamp=max_profit_timestamp,
            max_loss=max_loss,
            max_loss_timestamp=max_loss_timestamp,
            exit_price=exit_price,
            exit_reason=exit_reason,
            exit_timestamp=exit_timestamp,
            verdict="",  # Will be filled later
            recommendations=[],  # Will be filled later
            timeline=timeline
        )
    
    def _generate_verdict(self, result: WhatIfResult) -> str:
        """Generate a verdict for the simulation"""
        
        scenario = result.scenario
        
        if result.result == "WIN":
            if result.exit_reason == "TAKE_PROFIT":
                return f"Trade excellent! TP hit +{result.pips_gained:.0f} pips (puis le prix a chuté)"
            else:
                return f"Trade winner +{result.pips_gained:.0f} pips"
        elif result.result == "LOSS":
            if result.exit_reason == "STOP_LOSS":
                return f"Stop Loss hit -{abs(result.pips_gained):.0f} pips. Entry trop tôt ou SL trop serré."
            else:
                return f"Trade perdant -{abs(result.pips_gained):.0f} pips"
        else:
            return "Trade en cours / Breakeven"
    
    def _generate_recommendations(self, result: WhatIfResult) -> List[str]:
        """Generate recommendations based on simulation"""
        
        recommendations = []
        
        scenario = result.scenario
        entry_price = scenario.entry_price
        
        # Analyze the trade
        if result.max_profit > 5:  # More than 5%
            recommendations.append(f"Prochain fois: take profit partial à {result.max_profit/2:.1f}%")
        
        if result.max_loss < -3:
            recommendations.append("Stop loss trop serré - augmente à minimum 3%")
        
        # Suggest better entry
        if result.result == "LOSS" and scenario.side == "BUY":
            better_entry = entry_price * 0.98  # 2% lower
            recommendations.append(f"Meilleure entrée possible: {better_entry:.5f}")
        elif result.result == "LOSS" and scenario.side == "SELL":
            better_entry = entry_price * 1.02  # 2% higher
            recommendations.append(f"Meilleure entrée possible: {better_entry:.5f}")
        
        # Risk/Reward suggestions
        if scenario.take_profit and scenario.stop_loss:
            rr_ratio = abs(scenario.take_profit - entry_price) / abs(entry_price - scenario.stop_loss)
            if rr_ratio < 2:
                recommendations.append(f"Risk/Reward ratio faible ({rr_ratio:.1f}). Vise au moins 1:2")
            else:
                recommendations.append(f"Bon Risk/Reward ratio: 1:{rr_ratio:.1f}")
        
        return recommendations
    
    def _create_error_result(
        self, 
        scenario: WhatIfScenario, 
        error: str
    ) -> WhatIfResult:
        """Create an error result"""
        return WhatIfResult(
            scenario=scenario,
            result="ERROR",
            pips_gained=0,
            percentage_gain=0,
            max_profit=0,
            max_profit_timestamp=None,
            max_loss=0,
            max_loss_timestamp=None,
            exit_price=None,
            exit_reason=error,
            exit_timestamp=None,
            verdict=f"Erreur: {error}",
            recommendations=["Vérifiez la disponibilité des données historiques"]
        )
    
    def to_dict(self, result: WhatIfResult) -> Dict[str, Any]:
        """Convert result to dictionary"""
        
        return {
            "resultat": result.result,
            "pips_gagnés": f"{result.pips_gained:+.0f}",
            "pourcentage": f"{result.percentage_gain:+.2f}%",
            "max_profit": f"+{result.max_profit:.2f}%" if result.max_profit > 0 else f"{result.max_profit:.2f}%",
            "max_profit_timestamp": result.max_profit_timestamp.isoformat() if result.max_profit_timestamp else None,
            "max_loss": f"{result.max_loss:.2f}%",
            "exit_price": result.exit_price,
            "exit_reason": result.exit_reason,
            "verdict": result.verdict,
            "recommendations": result.recommendations
        }
    
    def to_json(self, result: WhatIfResult) -> str:
        """Convert result to JSON string"""
        return json.dumps(self.to_dict(result), indent=2, ensure_ascii=False)
    
    def format_report(self, result: WhatIfResult) -> str:
        """Format a detailed report"""
        
        scenario = result.scenario
        
        report = f"""
════════════════════════════════════════
    WHAT-IF SIMULATOR REPORT
════════════════════════════════════════

📊 SCÉNARIO:
• Symbol: {scenario.symbol}
• Entry: {scenario.entry_price} @ {scenario.entry_date.strftime('%Y-%m-%d %H:%M')}
• Direction: {scenario.side}
• SL: {scenario.stop_loss or 'Non défini'}
• TP: {scenario.take_profit or 'Non défini'}

════════════════════════════════════════

📈 RÉSULTAT: {result.result} {result.percentage_gain:+.2f}%

• Pips: {result.pips_gained:+.0f}
• Profit max: {result.max_profit:+.2f}% @ {result.max_profit_timestamp.strftime('%H:%M') if result.max_profit_timestamp else 'N/A'}
• Perte max: {result.max_loss:.2f}% @ {result.max_loss_timestamp.strftime('%H:%M') if result.max_loss_timestamp else 'N/A'}
• Exit: {result.exit_reason} @ {result.exit_price}

════════════════════════════════════════

⚖️ VERDICT: {result.verdict}

📋 RECOMMENDATIONS:
"""
        for rec in result.recommendations:
            report += f"• {rec}\n"
        
        report += "════════════════════════════════════════\n"
        
        return report
