"""
Position Sizing Engine - Implementation of Parameters 1-5

This engine handles the business logic for position sizing parameters:
1. method - Position sizing methodology
2. value - Position size value (percentage of portfolio)  
3. max_position_size - Maximum position size limit
4. risk_per_trade - Maximum risk per individual trade
5. volatility_adjustment - Adjust position size based on volatility

The engine provides real-time position sizing calculations based on current
market conditions, portfolio state, and configured parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

from core.parameter_registry import TradingParameterRegistry, parameter_registry

logger = logging.getLogger(__name__)


@dataclass
class PositionSizingResult:
    """Result of position sizing calculation"""
    
    # Core results
    position_size: float  # Final position size
    position_size_percent: float  # Position size as % of portfolio
    shares_to_trade: int  # Number of shares to trade
    dollar_amount: float  # Dollar amount to invest
    
    # Risk metrics
    risk_amount: float  # Dollar amount at risk
    risk_percent: float  # Risk as % of portfolio
    risk_per_trade_actual: float  # Actual risk per trade
    
    # Sizing components
    base_size: float  # Size before adjustments
    volatility_adjustment_factor: float  # Volatility adjustment multiplier
    liquidity_adjustment_factor: float  # Liquidity adjustment multiplier
    final_adjustment_factor: float  # Combined adjustment factor
    
    # Validation flags
    within_max_position: bool  # Position within max limit
    within_risk_limit: bool  # Risk within limit
    sufficient_liquidity: bool  # Sufficient liquidity available
    
    # Metadata
    sizing_method: str  # Method used for sizing
    calculation_timestamp: datetime  # When calculation was performed
    warnings: List[str]  # Any warnings generated
    
    def is_valid(self) -> bool:
        """Check if position sizing result is valid"""
        return (
            self.within_max_position and 
            self.within_risk_limit and 
            self.sufficient_liquidity and
            self.position_size > 0
        )


@dataclass
class MarketData:
    """Market data for position sizing calculations"""
    
    symbol: str
    current_price: float
    volatility: float  # Annualized volatility
    avg_daily_volume: float  # Average daily volume
    bid_ask_spread: float  # Bid-ask spread percentage
    market_cap: Optional[float] = None
    beta: Optional[float] = None
    
    # Price data for volatility calculation
    price_history: Optional[pd.Series] = None
    volume_history: Optional[pd.Series] = None


@dataclass
class PortfolioState:
    """Current portfolio state for position sizing"""
    
    total_value: float  # Total portfolio value
    cash_available: float  # Available cash
    positions: Dict[str, Dict[str, float]]  # Current positions
    
    # Risk metrics
    portfolio_volatility: float  # Portfolio volatility
    current_drawdown: float  # Current drawdown
    correlation_matrix: Optional[pd.DataFrame] = None
    
    # Sector exposures
    sector_exposures: Optional[Dict[str, float]] = None


class PositionSizingMethod(ABC):
    """Abstract base class for position sizing methods"""
    
    @abstractmethod
    def calculate_size(
        self, 
        portfolio_value: float,
        market_data: MarketData,
        parameters: Dict[str, Any]
    ) -> float:
        """Calculate position size using this method"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get method name"""
        pass


class FixedPositionSizing(PositionSizingMethod):
    """Fixed dollar amount position sizing"""
    
    def calculate_size(
        self, 
        portfolio_value: float,
        market_data: MarketData,
        parameters: Dict[str, Any]
    ) -> float:
        """Calculate fixed position size"""
        fixed_amount = parameters.get('value', 10000)  # Default $10K
        return min(fixed_amount, portfolio_value * 0.1)  # Cap at 10% of portfolio
    
    def get_name(self) -> str:
        return "fixed"


class PercentagePositionSizing(PositionSizingMethod):
    """Percentage of portfolio position sizing"""
    
    def calculate_size(
        self, 
        portfolio_value: float,
        market_data: MarketData,
        parameters: Dict[str, Any]
    ) -> float:
        """Calculate percentage-based position size"""
        percentage = parameters.get('value', 5.0) / 100.0
        return portfolio_value * percentage
    
    def get_name(self) -> str:
        return "percentage"


class KellyPositionSizing(PositionSizingMethod):
    """Kelly Criterion position sizing"""
    
    def calculate_size(
        self, 
        portfolio_value: float,
        market_data: MarketData,
        parameters: Dict[str, Any]
    ) -> float:
        """Calculate Kelly Criterion position size"""
        # Simplified Kelly: f = (bp - q) / b
        # Where: b = odds, p = win probability, q = loss probability
        
        # Use default values if signal data not available
        win_probability = parameters.get('win_prob', 0.55)  # 55% win rate
        avg_win = parameters.get('avg_win', 0.08)  # 8% average win
        avg_loss = parameters.get('avg_loss', 0.04)  # 4% average loss
        
        # Kelly fraction
        kelly_fraction = (win_probability * avg_win - (1 - win_probability) * avg_loss) / avg_win
        
        # Apply safety factor (typically 0.25 to 0.5 of full Kelly)
        safety_factor = parameters.get('kelly_safety', 0.25)
        kelly_fraction *= safety_factor
        
        # Cap at maximum position size
        max_position = parameters.get('value', 10.0) / 100.0
        kelly_fraction = min(kelly_fraction, max_position)
        
        return max(0, portfolio_value * kelly_fraction)
    
    def get_name(self) -> str:
        return "kelly"


class VolatilityAdjustedPositionSizing(PositionSizingMethod):
    """Volatility-adjusted position sizing"""
    
    def calculate_size(
        self, 
        portfolio_value: float,
        market_data: MarketData,
        parameters: Dict[str, Any]
    ) -> float:
        """Calculate volatility-adjusted position size"""
        base_percentage = parameters.get('value', 5.0) / 100.0
        target_volatility = parameters.get('target_vol', 0.15)  # 15% target vol
        
        # Adjust based on asset volatility
        if market_data.volatility > 0:
            vol_adjustment = target_volatility / market_data.volatility
            # Cap adjustment between 0.2x and 2.0x
            vol_adjustment = max(0.2, min(2.0, vol_adjustment))
        else:
            vol_adjustment = 1.0
        
        adjusted_percentage = base_percentage * vol_adjustment
        return portfolio_value * adjusted_percentage
    
    def get_name(self) -> str:
        return "volatility_adjusted"


class RiskParityPositionSizing(PositionSizingMethod):
    """Risk parity position sizing"""
    
    def calculate_size(
        self, 
        portfolio_value: float,
        market_data: MarketData,
        parameters: Dict[str, Any]
    ) -> float:
        """Calculate risk parity position size"""
        # Size inversely proportional to volatility
        if market_data.volatility <= 0:
            return 0
        
        # Target risk contribution
        target_risk_contrib = parameters.get('value', 5.0) / 100.0
        
        # Position size = (target risk contribution * portfolio value) / volatility
        position_size = (target_risk_contrib * portfolio_value) / market_data.volatility
        
        # Apply reasonable bounds
        max_position = portfolio_value * 0.2  # Max 20%
        return min(position_size, max_position)
    
    def get_name(self) -> str:
        return "risk_parity"


class PositionSizingEngine:
    """
    Main position sizing engine that orchestrates all sizing calculations
    
    Handles parameters 1-5:
    1. method - Selects appropriate sizing method
    2. value - Provides base sizing value
    3. max_position_size - Enforces position limits
    4. risk_per_trade - Manages trade-level risk
    5. volatility_adjustment - Applies volatility adjustments
    """
    
    def __init__(self, registry: TradingParameterRegistry = None):
        self.registry = registry or parameter_registry
        self.sizing_methods = self._initialize_sizing_methods()
        
    def _initialize_sizing_methods(self) -> Dict[str, PositionSizingMethod]:
        """Initialize available position sizing methods"""
        return {
            'fixed': FixedPositionSizing(),
            'percentage': PercentagePositionSizing(),
            'kelly': KellyPositionSizing(),
            'volatility_adjusted': VolatilityAdjustedPositionSizing(),
            'risk_parity': RiskParityPositionSizing()
        }
    
    def calculate_position_size(
        self,
        symbol: str,
        market_data: MarketData,
        portfolio_state: PortfolioState,
        signal_strength: float = 1.0,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> PositionSizingResult:
        """
        Calculate optimal position size for a given symbol
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            portfolio_state: Current portfolio state
            signal_strength: Signal strength (0.0 to 1.0)
            additional_params: Additional parameters override
            
        Returns:
            PositionSizingResult with complete sizing analysis
        """
        try:
            # Get current parameters
            params = self._get_current_parameters()
            if additional_params:
                params.update(additional_params)
            
            # Step 1: Calculate base position size using selected method
            base_size = self._calculate_base_size(
                market_data, portfolio_state, params, signal_strength
            )
            
            # Step 2: Apply volatility adjustment if enabled
            vol_adjustment_factor = self._calculate_volatility_adjustment(
                market_data, params
            )
            
            # Step 3: Apply liquidity adjustment
            liquidity_adjustment_factor = self._calculate_liquidity_adjustment(
                market_data, params
            )
            
            # Step 4: Calculate final position size
            final_adjustment_factor = vol_adjustment_factor * liquidity_adjustment_factor
            adjusted_size = base_size * final_adjustment_factor
            
            # Step 5: Apply position size limits
            final_size, within_limits = self._apply_position_limits(
                adjusted_size, portfolio_state, params
            )
            
            # Step 6: Calculate shares and dollar amounts
            shares_to_trade = int(final_size / market_data.current_price)
            dollar_amount = shares_to_trade * market_data.current_price
            
            # Step 7: Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                dollar_amount, shares_to_trade, market_data, portfolio_state, params
            )
            
            # Step 8: Perform validation checks
            validation_results = self._validate_position(
                dollar_amount, risk_metrics, market_data, portfolio_state, params
            )
            
            # Step 9: Generate warnings
            warnings = self._generate_warnings(
                final_size, dollar_amount, risk_metrics, validation_results, params
            )
            
            return PositionSizingResult(
                # Core results
                position_size=final_size,
                position_size_percent=(final_size / portfolio_state.total_value) * 100,
                shares_to_trade=shares_to_trade,
                dollar_amount=dollar_amount,
                
                # Risk metrics
                risk_amount=risk_metrics['risk_amount'],
                risk_percent=risk_metrics['risk_percent'],
                risk_per_trade_actual=risk_metrics['risk_per_trade_actual'],
                
                # Sizing components
                base_size=base_size,
                volatility_adjustment_factor=vol_adjustment_factor,
                liquidity_adjustment_factor=liquidity_adjustment_factor,
                final_adjustment_factor=final_adjustment_factor,
                
                # Validation flags
                within_max_position=validation_results['within_max_position'],
                within_risk_limit=validation_results['within_risk_limit'], 
                sufficient_liquidity=validation_results['sufficient_liquidity'],
                
                # Metadata
                sizing_method=params['method'],
                calculation_timestamp=datetime.now(),
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Position sizing calculation failed for {symbol}: {str(e)}")
            raise
    
    def _get_current_parameters(self) -> Dict[str, Any]:
        """Get current position sizing parameters"""
        param_names = ['method', 'value', 'max_position_size', 'risk_per_trade', 'volatility_adjustment']
        params = {}
        
        for name in param_names:
            param_def = self.registry.get_parameter(name)
            if param_def:
                params[name] = param_def.current_value or param_def.default_value
        
        return params
    
    def _calculate_base_size(
        self,
        market_data: MarketData,
        portfolio_state: PortfolioState,
        params: Dict[str, Any],
        signal_strength: float
    ) -> float:
        """Calculate base position size using selected method"""
        method_name = params.get('method', 'percentage')
        
        if method_name not in self.sizing_methods:
            logger.warning(f"Unknown sizing method: {method_name}, using percentage")
            method_name = 'percentage'
        
        sizing_method = self.sizing_methods[method_name]
        base_size = sizing_method.calculate_size(
            portfolio_state.total_value, market_data, params
        )
        
        # Apply signal strength scaling
        base_size *= signal_strength
        
        return base_size
    
    def _calculate_volatility_adjustment(
        self, 
        market_data: MarketData, 
        params: Dict[str, Any]
    ) -> float:
        """Calculate volatility adjustment factor"""
        if not params.get('volatility_adjustment', True):
            return 1.0
        
        if market_data.volatility <= 0:
            return 1.0
        
        # Target volatility for normalization (15% default)
        target_vol = 0.15
        
        # Adjustment factor: inverse relationship with volatility
        adjustment = target_vol / market_data.volatility
        
        # Apply reasonable bounds: 0.25x to 2.0x
        adjustment = max(0.25, min(2.0, adjustment))
        
        return adjustment
    
    def _calculate_liquidity_adjustment(
        self,
        market_data: MarketData,
        params: Dict[str, Any]
    ) -> float:
        """Calculate liquidity adjustment factor"""
        # Reduce position size for illiquid stocks
        min_daily_volume = params.get('minimum_volume', 100000)
        
        if market_data.avg_daily_volume < min_daily_volume:
            # Reduce size proportionally to liquidity shortfall
            liquidity_ratio = market_data.avg_daily_volume / min_daily_volume
            return max(0.1, liquidity_ratio)  # Minimum 10% of original size
        
        return 1.0
    
    def _apply_position_limits(
        self,
        position_size: float,
        portfolio_state: PortfolioState,
        params: Dict[str, Any]
    ) -> Tuple[float, bool]:
        """Apply position size limits"""
        max_position_percent = params.get('max_position_size', 10.0) / 100.0
        max_position_dollar = portfolio_state.total_value * max_position_percent
        
        if position_size <= max_position_dollar:
            return position_size, True
        else:
            return max_position_dollar, False
    
    def _calculate_risk_metrics(
        self,
        dollar_amount: float,
        shares: int,
        market_data: MarketData,
        portfolio_state: PortfolioState,
        params: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate risk metrics for the position"""
        # Simple risk calculation: position size * expected loss
        # In practice, this would use stop loss, volatility, etc.
        
        risk_per_trade_param = params.get('risk_per_trade', 1.0) / 100.0
        
        # Risk amount: could be based on stop loss distance or volatility
        # For now, use a simple percentage of position
        risk_amount = dollar_amount * risk_per_trade_param
        risk_percent = (risk_amount / portfolio_state.total_value) * 100
        
        return {
            'risk_amount': risk_amount,
            'risk_percent': risk_percent,
            'risk_per_trade_actual': risk_per_trade_param * 100
        }
    
    def _validate_position(
        self,
        dollar_amount: float,
        risk_metrics: Dict[str, float],
        market_data: MarketData,
        portfolio_state: PortfolioState,
        params: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Validate position against all constraints"""
        max_position_percent = params.get('max_position_size', 10.0)
        max_risk_percent = params.get('risk_per_trade', 1.0)
        min_volume = params.get('minimum_volume', 100000)
        
        position_percent = (dollar_amount / portfolio_state.total_value) * 100
        
        return {
            'within_max_position': position_percent <= max_position_percent,
            'within_risk_limit': risk_metrics['risk_percent'] <= max_risk_percent,
            'sufficient_liquidity': market_data.avg_daily_volume >= min_volume
        }
    
    def _generate_warnings(
        self,
        position_size: float,
        dollar_amount: float,
        risk_metrics: Dict[str, float],
        validation_results: Dict[str, bool],
        params: Dict[str, Any]
    ) -> List[str]:
        """Generate warnings for the position sizing result"""
        warnings = []
        
        if not validation_results['within_max_position']:
            warnings.append("Position size exceeds maximum limit")
        
        if not validation_results['within_risk_limit']:
            warnings.append("Risk per trade exceeds limit")
        
        if not validation_results['sufficient_liquidity']:
            warnings.append("Insufficient liquidity for position size")
        
        if position_size <= 0:
            warnings.append("Position size calculated as zero or negative")
        
        if dollar_amount < 1000:  # Minimum trade size
            warnings.append("Position size below minimum trade threshold")
        
        return warnings
    
    def calculate_portfolio_allocation(
        self,
        symbols: List[str],
        market_data_dict: Dict[str, MarketData],
        portfolio_state: PortfolioState,
        signal_strengths: Dict[str, float]
    ) -> Dict[str, PositionSizingResult]:
        """Calculate position sizes for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            if symbol in market_data_dict:
                signal_strength = signal_strengths.get(symbol, 1.0)
                result = self.calculate_position_size(
                    symbol, 
                    market_data_dict[symbol],
                    portfolio_state,
                    signal_strength
                )
                results[symbol] = result
            else:
                logger.warning(f"No market data available for {symbol}")
        
        return results
    
    def optimize_portfolio_sizing(
        self,
        target_symbols: List[str],
        market_data_dict: Dict[str, MarketData],
        portfolio_state: PortfolioState,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, PositionSizingResult]:
        """Optimize position sizes across multiple symbols"""
        # This is a simplified version - full optimization would use
        # mathematical optimization techniques
        
        constraints = constraints or {}
        max_total_allocation = constraints.get('max_total_allocation', 0.8)  # 80% max
        
        # Calculate individual sizes
        individual_results = {}
        total_allocation = 0
        
        for symbol in target_symbols:
            if symbol in market_data_dict:
                result = self.calculate_position_size(
                    symbol,
                    market_data_dict[symbol],
                    portfolio_state
                )
                individual_results[symbol] = result
                total_allocation += result.position_size_percent / 100
        
        # Scale down if total allocation exceeds limit
        if total_allocation > max_total_allocation:
            scale_factor = max_total_allocation / total_allocation
            
            for symbol, result in individual_results.items():
                # Create scaled result
                scaled_size = result.position_size * scale_factor
                scaled_shares = int(scaled_size / market_data_dict[symbol].current_price)
                scaled_dollar = scaled_shares * market_data_dict[symbol].current_price
                
                # Update result with scaled values
                result.position_size = scaled_size
                result.position_size_percent *= scale_factor
                result.shares_to_trade = scaled_shares
                result.dollar_amount = scaled_dollar
                result.warnings.append(f"Position scaled by {scale_factor:.2f} for portfolio constraints")
        
        return individual_results
    
    def get_sizing_summary(self) -> Dict[str, Any]:
        """Get summary of current position sizing configuration"""
        params = self._get_current_parameters()
        
        return {
            'method': params.get('method'),
            'base_size_percent': params.get('value'),
            'max_position_percent': params.get('max_position_size'),
            'max_risk_percent': params.get('risk_per_trade'),
            'volatility_adjustment_enabled': params.get('volatility_adjustment'),
            'available_methods': list(self.sizing_methods.keys()),
            'last_updated': datetime.now()
        }


# Factory function for easy instantiation
def create_position_sizing_engine(registry: TradingParameterRegistry = None) -> PositionSizingEngine:
    """Create a position sizing engine instance"""
    return PositionSizingEngine(registry)