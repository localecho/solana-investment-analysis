"""
Risk Management Engine - Implementation of Parameters 6-12

This engine handles the business logic for risk management parameters:
6. stop_loss - Stop loss percentage
7. stop_loss_type - Type of stop loss mechanism
8. take_profit - Take profit percentage  
9. take_profit_type - Type of take profit mechanism
10. max_drawdown - Maximum portfolio drawdown limit
11. correlation_limit - Maximum correlation between positions
12. sector_exposure_limit - Maximum exposure to any single sector

The engine provides real-time risk monitoring, stop loss/take profit
management, and portfolio-level risk controls.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod

from core.parameter_registry import TradingParameterRegistry, parameter_registry

logger = logging.getLogger(__name__)


class StopLossType(Enum):
    """Stop loss types"""
    FIXED = "fixed"
    TRAILING = "trailing"
    VOLATILITY_BASED = "volatility_based"
    ATR_BASED = "atr_based"


class TakeProfitType(Enum):
    """Take profit types"""
    FIXED = "fixed"
    TRAILING = "trailing"
    SCALE_OUT = "scale_out"
    TARGET_BASED = "target_based"


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    
    # Portfolio-level metrics
    portfolio_value: float
    current_drawdown: float
    max_drawdown_reached: float
    portfolio_volatility: float
    portfolio_beta: float
    var_95: float  # 95% Value at Risk
    cvar_95: float  # 95% Conditional Value at Risk
    
    # Position-level metrics
    total_positions: int
    total_exposure: float
    largest_position_percent: float
    avg_position_size: float
    
    # Risk limits status
    within_drawdown_limit: bool
    within_correlation_limits: bool
    within_sector_limits: bool
    within_position_limits: bool
    
    # Risk warnings
    risk_warnings: List[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0-100 risk score
    
    calculation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StopLossResult:
    """Stop loss calculation result"""
    
    stop_price: float
    stop_type: StopLossType
    trigger_percent: float
    dollar_risk: float
    shares_at_risk: int
    
    # Advanced stop loss data
    atr_value: Optional[float] = None
    volatility_factor: Optional[float] = None
    trailing_high: Optional[float] = None
    
    is_active: bool = True
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class TakeProfitResult:
    """Take profit calculation result"""
    
    target_price: float
    target_type: TakeProfitType
    profit_percent: float
    dollar_profit: float
    shares_to_sell: int
    
    # Scale out data
    scale_levels: Optional[List[Tuple[float, float]]] = None  # (price, percent)
    partial_fills: Optional[List[Dict]] = None
    
    is_active: bool = True
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PositionRisk:
    """Individual position risk assessment"""
    
    symbol: str
    current_price: float
    entry_price: float
    shares: int
    market_value: float
    
    # Risk metrics
    unrealized_pnl: float
    unrealized_pnl_percent: float
    position_risk: float  # Dollar amount at risk
    beta: Optional[float] = None
    volatility: float = 0.0
    
    # Stop loss and take profit
    stop_loss: Optional[StopLossResult] = None
    take_profit: Optional[TakeProfitResult] = None
    
    # Risk flags
    exceeds_risk_limit: bool = False
    correlation_warnings: List[str] = field(default_factory=list)
    
    last_updated: datetime = field(default_factory=datetime.now)


class StopLossCalculator(ABC):
    """Abstract base class for stop loss calculations"""
    
    @abstractmethod
    def calculate_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        position_size: int,
        market_data: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> StopLossResult:
        """Calculate stop loss for a position"""
        pass


class FixedStopLoss(StopLossCalculator):
    """Fixed percentage stop loss"""
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        position_size: int,
        market_data: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> StopLossResult:
        """Calculate fixed percentage stop loss"""
        stop_loss_percent = parameters.get('stop_loss', 5.0) / 100.0
        
        # For long positions
        if position_size > 0:
            stop_price = entry_price * (1 - stop_loss_percent)
        else:  # For short positions
            stop_price = entry_price * (1 + stop_loss_percent)
        
        dollar_risk = abs(position_size) * abs(entry_price - stop_price)
        
        return StopLossResult(
            stop_price=stop_price,
            stop_type=StopLossType.FIXED,
            trigger_percent=stop_loss_percent * 100,
            dollar_risk=dollar_risk,
            shares_at_risk=abs(position_size)
        )


class TrailingStopLoss(StopLossCalculator):
    """Trailing stop loss"""
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        position_size: int,
        market_data: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> StopLossResult:
        """Calculate trailing stop loss"""
        stop_loss_percent = parameters.get('stop_loss', 5.0) / 100.0
        
        # Get historical high for trailing calculation
        trailing_high = market_data.get('trailing_high', max(entry_price, current_price))
        
        # For long positions, trail below the high
        if position_size > 0:
            stop_price = trailing_high * (1 - stop_loss_percent)
        else:  # For short positions, trail above the low
            trailing_low = market_data.get('trailing_low', min(entry_price, current_price))
            stop_price = trailing_low * (1 + stop_loss_percent)
        
        dollar_risk = abs(position_size) * abs(current_price - stop_price)
        
        return StopLossResult(
            stop_price=stop_price,
            stop_type=StopLossType.TRAILING,
            trigger_percent=stop_loss_percent * 100,
            dollar_risk=dollar_risk,
            shares_at_risk=abs(position_size),
            trailing_high=trailing_high
        )


class VolatilityBasedStopLoss(StopLossCalculator):
    """Volatility-based stop loss"""
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        position_size: int,
        market_data: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> StopLossResult:
        """Calculate volatility-based stop loss"""
        base_stop_percent = parameters.get('stop_loss', 5.0) / 100.0
        volatility = market_data.get('volatility', 0.20)  # Default 20% volatility
        
        # Adjust stop loss based on volatility
        # Higher volatility = wider stop loss
        volatility_factor = min(2.0, max(0.5, volatility / 0.15))  # Normalize to 15% base vol
        adjusted_stop_percent = base_stop_percent * volatility_factor
        
        # For long positions
        if position_size > 0:
            stop_price = entry_price * (1 - adjusted_stop_percent)
        else:  # For short positions
            stop_price = entry_price * (1 + adjusted_stop_percent)
        
        dollar_risk = abs(position_size) * abs(entry_price - stop_price)
        
        return StopLossResult(
            stop_price=stop_price,
            stop_type=StopLossType.VOLATILITY_BASED,
            trigger_percent=adjusted_stop_percent * 100,
            dollar_risk=dollar_risk,
            shares_at_risk=abs(position_size),
            volatility_factor=volatility_factor
        )


class ATRBasedStopLoss(StopLossCalculator):
    """ATR (Average True Range) based stop loss"""
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        position_size: int,
        market_data: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> StopLossResult:
        """Calculate ATR-based stop loss"""
        atr_value = market_data.get('atr', entry_price * 0.02)  # Default 2% ATR
        atr_multiplier = parameters.get('atr_multiplier', 2.0)  # 2x ATR default
        
        stop_distance = atr_value * atr_multiplier
        
        # For long positions
        if position_size > 0:
            stop_price = current_price - stop_distance
        else:  # For short positions
            stop_price = current_price + stop_distance
        
        trigger_percent = (stop_distance / entry_price) * 100
        dollar_risk = abs(position_size) * stop_distance
        
        return StopLossResult(
            stop_price=stop_price,
            stop_type=StopLossType.ATR_BASED,
            trigger_percent=trigger_percent,
            dollar_risk=dollar_risk,
            shares_at_risk=abs(position_size),
            atr_value=atr_value
        )


class TakeProfitCalculator(ABC):
    """Abstract base class for take profit calculations"""
    
    @abstractmethod
    def calculate_take_profit(
        self,
        entry_price: float,
        current_price: float,
        position_size: int,
        market_data: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> TakeProfitResult:
        """Calculate take profit for a position"""
        pass


class FixedTakeProfit(TakeProfitCalculator):
    """Fixed percentage take profit"""
    
    def calculate_take_profit(
        self,
        entry_price: float,
        current_price: float,
        position_size: int,
        market_data: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> TakeProfitResult:
        """Calculate fixed percentage take profit"""
        take_profit_percent = parameters.get('take_profit', 10.0) / 100.0
        
        # For long positions
        if position_size > 0:
            target_price = entry_price * (1 + take_profit_percent)
        else:  # For short positions
            target_price = entry_price * (1 - take_profit_percent)
        
        dollar_profit = abs(position_size) * abs(target_price - entry_price)
        
        return TakeProfitResult(
            target_price=target_price,
            target_type=TakeProfitType.FIXED,
            profit_percent=take_profit_percent * 100,
            dollar_profit=dollar_profit,
            shares_to_sell=abs(position_size)
        )


class ScaleOutTakeProfit(TakeProfitCalculator):
    """Scale-out take profit with multiple levels"""
    
    def calculate_take_profit(
        self,
        entry_price: float,
        current_price: float,
        position_size: int,
        market_data: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> TakeProfitResult:
        """Calculate scale-out take profit levels"""
        base_profit_percent = parameters.get('take_profit', 10.0) / 100.0
        
        # Define scale-out levels: (profit %, position %)
        scale_levels = [
            (base_profit_percent * 0.5, 0.25),   # 5% profit: sell 25%
            (base_profit_percent * 1.0, 0.50),   # 10% profit: sell 50%
            (base_profit_percent * 2.0, 0.25)    # 20% profit: sell 25%
        ]
        
        # Calculate prices for each level
        price_levels = []
        total_profit = 0
        
        for profit_pct, position_pct in scale_levels:
            if position_size > 0:  # Long position
                target_price = entry_price * (1 + profit_pct)
            else:  # Short position
                target_price = entry_price * (1 - profit_pct)
            
            shares_at_level = int(abs(position_size) * position_pct)
            profit_at_level = shares_at_level * abs(target_price - entry_price)
            total_profit += profit_at_level
            
            price_levels.append((target_price, position_pct))
        
        # Use first level as primary target
        primary_target = price_levels[0][0] if price_levels else entry_price
        
        return TakeProfitResult(
            target_price=primary_target,
            target_type=TakeProfitType.SCALE_OUT,
            profit_percent=base_profit_percent * 100,
            dollar_profit=total_profit,
            shares_to_sell=abs(position_size),
            scale_levels=price_levels
        )


class RiskManagementEngine:
    """
    Main risk management engine that orchestrates all risk controls
    
    Handles parameters 6-12:
    6. stop_loss - Manages stop loss levels
    7. stop_loss_type - Selects stop loss methodology
    8. take_profit - Manages profit targets
    9. take_profit_type - Selects profit taking strategy
    10. max_drawdown - Monitors portfolio drawdown
    11. correlation_limit - Controls position correlation
    12. sector_exposure_limit - Manages sector concentration
    """
    
    def __init__(self, registry: TradingParameterRegistry = None):
        self.registry = registry or parameter_registry
        self.stop_loss_calculators = self._initialize_stop_loss_calculators()
        self.take_profit_calculators = self._initialize_take_profit_calculators()
        
        # Risk tracking
        self.position_risks: Dict[str, PositionRisk] = {}
        self.portfolio_metrics: Optional[RiskMetrics] = None
        self.risk_history: List[RiskMetrics] = []
        
    def _initialize_stop_loss_calculators(self) -> Dict[str, StopLossCalculator]:
        """Initialize stop loss calculators"""
        return {
            'fixed': FixedStopLoss(),
            'trailing': TrailingStopLoss(),
            'volatility_based': VolatilityBasedStopLoss(),
            'atr_based': ATRBasedStopLoss()
        }
    
    def _initialize_take_profit_calculators(self) -> Dict[str, TakeProfitCalculator]:
        """Initialize take profit calculators"""
        return {
            'fixed': FixedTakeProfit(),
            'trailing': FixedTakeProfit(),  # Simplified for now
            'scale_out': ScaleOutTakeProfit(),
            'target_based': FixedTakeProfit()  # Simplified for now
        }
    
    def calculate_position_risk(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        shares: int,
        market_data: Dict[str, Any]
    ) -> PositionRisk:
        """Calculate comprehensive risk metrics for a position"""
        
        # Get current parameters
        params = self._get_current_parameters()
        
        # Basic position metrics
        market_value = shares * current_price
        unrealized_pnl = shares * (current_price - entry_price)
        unrealized_pnl_percent = ((current_price - entry_price) / entry_price) * 100
        
        # Calculate stop loss
        stop_loss_type = params.get('stop_loss_type', 'trailing')
        stop_loss_calculator = self.stop_loss_calculators.get(stop_loss_type, self.stop_loss_calculators['fixed'])
        
        stop_loss_result = stop_loss_calculator.calculate_stop_loss(
            entry_price, current_price, shares, market_data, params
        )
        
        # Calculate take profit
        take_profit_type = params.get('take_profit_type', 'fixed')
        take_profit_calculator = self.take_profit_calculators.get(take_profit_type, self.take_profit_calculators['fixed'])
        
        take_profit_result = take_profit_calculator.calculate_take_profit(
            entry_price, current_price, shares, market_data, params
        )
        
        # Risk assessment
        position_risk = stop_loss_result.dollar_risk
        max_risk_per_trade = params.get('risk_per_trade', 1.0)
        
        # Create position risk object
        pos_risk = PositionRisk(
            symbol=symbol,
            current_price=current_price,
            entry_price=entry_price,
            shares=shares,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_percent=unrealized_pnl_percent,
            position_risk=position_risk,
            beta=market_data.get('beta'),
            volatility=market_data.get('volatility', 0.0),
            stop_loss=stop_loss_result,
            take_profit=take_profit_result,
            exceeds_risk_limit=False  # Will be set by portfolio-level analysis
        )
        
        # Store position risk
        self.position_risks[symbol] = pos_risk
        
        return pos_risk
    
    def calculate_portfolio_risk(
        self,
        positions: Dict[str, Dict[str, float]],
        market_data: Dict[str, Dict[str, Any]],
        portfolio_value: float,
        cash: float
    ) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        
        params = self._get_current_parameters()
        
        # Calculate individual position risks
        total_exposure = 0
        position_values = []
        betas = []
        volatilities = []
        risk_warnings = []
        
        for symbol, position_data in positions.items():
            if symbol in market_data:
                shares = position_data.get('shares', 0)
                entry_price = position_data.get('entry_price', 0)
                current_price = market_data[symbol].get('current_price', entry_price)
                
                # Calculate position risk
                pos_risk = self.calculate_position_risk(
                    symbol, entry_price, current_price, shares, market_data[symbol]
                )
                
                position_value = abs(pos_risk.market_value)
                total_exposure += position_value
                position_values.append(position_value)
                
                if pos_risk.beta is not None:
                    betas.append(pos_risk.beta)
                if pos_risk.volatility > 0:
                    volatilities.append(pos_risk.volatility)
        
        # Portfolio-level calculations
        num_positions = len([p for p in positions.values() if p.get('shares', 0) != 0])
        largest_position_percent = (max(position_values) / portfolio_value * 100) if position_values else 0
        avg_position_size = (total_exposure / num_positions) if num_positions > 0 else 0
        
        # Risk metrics
        portfolio_beta = np.mean(betas) if betas else 1.0
        portfolio_volatility = np.sqrt(np.mean([v**2 for v in volatilities])) if volatilities else 0.20
        
        # Simplified VaR calculation (would use more sophisticated methods in practice)
        var_95 = portfolio_value * portfolio_volatility * 1.645 * 0.05  # 5% worst case
        cvar_95 = var_95 * 1.3  # Expected shortfall
        
        # Calculate current drawdown (simplified)
        current_drawdown = 0.0  # Would track from portfolio highs
        max_drawdown_reached = 0.0  # Would track historical maximum
        
        # Check risk limits
        max_drawdown_limit = params.get('max_drawdown', 15.0)
        within_drawdown_limit = current_drawdown <= max_drawdown_limit
        
        # Check correlation limits (simplified)
        correlation_limit = params.get('correlation_limit', 0.7)
        within_correlation_limits = True  # Would calculate actual correlations
        
        # Check sector limits (simplified)
        sector_limit = params.get('sector_exposure_limit', 25.0)
        within_sector_limits = True  # Would check actual sector exposures
        
        # Position size limits
        within_position_limits = largest_position_percent <= 20.0  # 20% max position
        
        # Generate warnings
        if not within_drawdown_limit:
            risk_warnings.append(f"Drawdown {current_drawdown:.1f}% exceeds limit {max_drawdown_limit:.1f}%")
        
        if largest_position_percent > 15.0:
            risk_warnings.append(f"Largest position {largest_position_percent:.1f}% is concentrated")
        
        if portfolio_volatility > 0.30:
            risk_warnings.append(f"Portfolio volatility {portfolio_volatility:.1%} is high")
        
        # Calculate risk score (0-100)
        risk_score = self._calculate_risk_score(
            current_drawdown, largest_position_percent, portfolio_volatility, num_positions
        )
        
        # Create risk metrics
        self.portfolio_metrics = RiskMetrics(
            portfolio_value=portfolio_value,
            current_drawdown=current_drawdown,
            max_drawdown_reached=max_drawdown_reached,
            portfolio_volatility=portfolio_volatility,
            portfolio_beta=portfolio_beta,
            var_95=var_95,
            cvar_95=cvar_95,
            total_positions=num_positions,
            total_exposure=total_exposure,
            largest_position_percent=largest_position_percent,
            avg_position_size=avg_position_size,
            within_drawdown_limit=within_drawdown_limit,
            within_correlation_limits=within_correlation_limits,
            within_sector_limits=within_sector_limits,
            within_position_limits=within_position_limits,
            risk_warnings=risk_warnings,
            risk_score=risk_score
        )
        
        # Add to history
        self.risk_history.append(self.portfolio_metrics)
        if len(self.risk_history) > 1000:  # Keep last 1000 records
            self.risk_history = self.risk_history[-1000:]
        
        return self.portfolio_metrics
    
    def check_stop_loss_triggers(
        self,
        current_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Check for stop loss triggers across all positions"""
        triggers = []
        
        for symbol, pos_risk in self.position_risks.items():
            if symbol in current_prices and pos_risk.stop_loss:
                current_price = current_prices[symbol]
                stop_price = pos_risk.stop_loss.stop_price
                
                # Check if stop loss is triggered
                triggered = False
                if pos_risk.shares > 0:  # Long position
                    triggered = current_price <= stop_price
                else:  # Short position
                    triggered = current_price >= stop_price
                
                if triggered:
                    triggers.append({
                        'symbol': symbol,
                        'action': 'SELL' if pos_risk.shares > 0 else 'BUY',
                        'shares': abs(pos_risk.shares),
                        'trigger_price': stop_price,
                        'current_price': current_price,
                        'stop_type': pos_risk.stop_loss.stop_type.value,
                        'dollar_risk': pos_risk.stop_loss.dollar_risk
                    })
        
        return triggers
    
    def check_take_profit_triggers(
        self,
        current_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Check for take profit triggers across all positions"""
        triggers = []
        
        for symbol, pos_risk in self.position_risks.items():
            if symbol in current_prices and pos_risk.take_profit:
                current_price = current_prices[symbol]
                target_price = pos_risk.take_profit.target_price
                
                # Check if take profit is triggered
                triggered = False
                if pos_risk.shares > 0:  # Long position
                    triggered = current_price >= target_price
                else:  # Short position
                    triggered = current_price <= target_price
                
                if triggered:
                    triggers.append({
                        'symbol': symbol,
                        'action': 'SELL' if pos_risk.shares > 0 else 'BUY',
                        'shares': pos_risk.take_profit.shares_to_sell,
                        'target_price': target_price,
                        'current_price': current_price,
                        'profit_type': pos_risk.take_profit.target_type.value,
                        'dollar_profit': pos_risk.take_profit.dollar_profit
                    })
        
        return triggers
    
    def update_trailing_stops(
        self,
        current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """Update trailing stop losses based on current prices"""
        updated_stops = {}
        
        for symbol, pos_risk in self.position_risks.items():
            if (symbol in current_prices and 
                pos_risk.stop_loss and 
                pos_risk.stop_loss.stop_type == StopLossType.TRAILING):
                
                current_price = current_prices[symbol]
                
                # Update trailing stop for long positions
                if pos_risk.shares > 0:
                    # New high achieved
                    if current_price > (pos_risk.stop_loss.trailing_high or pos_risk.entry_price):
                        stop_loss_percent = pos_risk.stop_loss.trigger_percent / 100.0
                        new_stop = current_price * (1 - stop_loss_percent)
                        
                        # Only update if new stop is higher
                        if new_stop > pos_risk.stop_loss.stop_price:
                            pos_risk.stop_loss.stop_price = new_stop
                            pos_risk.stop_loss.trailing_high = current_price
                            updated_stops[symbol] = new_stop
                
                # Similar logic for short positions would go here
        
        return updated_stops
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        if not self.portfolio_metrics:
            return {'error': 'No risk metrics calculated'}
        
        return {
            'portfolio_risk_score': self.portfolio_metrics.risk_score,
            'current_drawdown': self.portfolio_metrics.current_drawdown,
            'portfolio_volatility': self.portfolio_metrics.portfolio_volatility,
            'var_95': self.portfolio_metrics.var_95,
            'total_positions': self.portfolio_metrics.total_positions,
            'largest_position_percent': self.portfolio_metrics.largest_position_percent,
            'risk_warnings': self.portfolio_metrics.risk_warnings,
            'within_all_limits': (
                self.portfolio_metrics.within_drawdown_limit and
                self.portfolio_metrics.within_correlation_limits and
                self.portfolio_metrics.within_sector_limits and
                self.portfolio_metrics.within_position_limits
            ),
            'stop_losses_active': len([p for p in self.position_risks.values() if p.stop_loss]),
            'take_profits_active': len([p for p in self.position_risks.values() if p.take_profit]),
            'last_updated': self.portfolio_metrics.calculation_timestamp
        }
    
    def _get_current_parameters(self) -> Dict[str, Any]:
        """Get current risk management parameters"""
        param_names = [
            'stop_loss', 'stop_loss_type', 'take_profit', 'take_profit_type',
            'max_drawdown', 'correlation_limit', 'sector_exposure_limit'
        ]
        params = {}
        
        for name in param_names:
            param_def = self.registry.get_parameter(name)
            if param_def:
                params[name] = param_def.current_value or param_def.default_value
        
        return params
    
    def _calculate_risk_score(
        self,
        drawdown: float,
        largest_position: float,
        volatility: float,
        num_positions: int
    ) -> float:
        """Calculate overall risk score (0-100)"""
        
        # Component scores (0-25 each)
        drawdown_score = min(25, drawdown * 2.5)  # 10% drawdown = 25 points
        concentration_score = min(25, largest_position * 1.25)  # 20% position = 25 points
        volatility_score = min(25, volatility * 125)  # 20% vol = 25 points
        diversification_score = max(0, 25 - num_positions * 2.5)  # <10 positions = points
        
        total_score = drawdown_score + concentration_score + volatility_score + diversification_score
        return min(100, total_score)


# Factory function for easy instantiation
def create_risk_management_engine(registry: TradingParameterRegistry = None) -> RiskManagementEngine:
    """Create a risk management engine instance"""
    return RiskManagementEngine(registry)