"""
Execution Context Engine - Implementation of Parameters 13-18

This engine handles the business logic for execution context parameters:
13. timeframe - Trading timeframe
14. order_type - Default order type
15. slippage_assumption - Expected slippage percentage
16. commission_model - Commission calculation model
17. market_hours_only - Trade only during market hours
18. minimum_volume - Minimum daily volume requirement

The engine provides order execution optimization, timing analysis,
cost modeling, and market microstructure considerations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
import logging
from abc import ABC, abstractmethod
import pytz

from core.parameter_registry import TradingParameterRegistry, parameter_registry

logger = logging.getLogger(__name__)


class TimeFrame(Enum):
    """Trading timeframes"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"


class CommissionModel(Enum):
    """Commission models"""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    TIERED = "tiered"
    CUSTOM = "custom"


@dataclass
class MarketHours:
    """Market hours definition"""
    
    open_time: time
    close_time: time
    timezone: str = "US/Eastern"
    
    # Pre/post market hours
    pre_market_open: Optional[time] = None
    post_market_close: Optional[time] = None
    
    # Market holidays
    holidays: List[datetime] = field(default_factory=list)
    
    def is_market_open(self, timestamp: datetime) -> bool:
        """Check if market is open at given timestamp"""
        tz = pytz.timezone(self.timezone)
        local_time = timestamp.astimezone(tz).time()
        
        # Check if it's a holiday
        date_only = timestamp.date()
        if any(holiday.date() == date_only for holiday in self.holidays):
            return False
        
        # Check if it's a weekend
        if timestamp.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Check if within market hours
        return self.open_time <= local_time <= self.close_time


@dataclass
class ExecutionCost:
    """Comprehensive execution cost breakdown"""
    
    # Commission costs
    commission: float
    commission_type: str
    
    # Market impact costs
    slippage: float
    slippage_percent: float
    market_impact: float
    
    # Timing costs
    timing_cost: float
    opportunity_cost: float
    
    # Total costs
    total_cost: float
    total_cost_bps: float  # Basis points
    
    # Cost components breakdown
    fixed_costs: float
    variable_costs: float
    
    calculation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OrderExecutionPlan:
    """Execution plan for an order"""
    
    symbol: str
    order_type: OrderType
    shares: int
    target_price: Optional[float]
    
    # Execution strategy
    execution_strategy: str
    time_horizon: timedelta
    max_participation_rate: float  # % of volume
    
    # Timing
    start_time: datetime
    end_time: datetime
    market_hours_only: bool
    
    # Cost estimates
    estimated_costs: ExecutionCost
    
    # Risk controls
    price_limit: Optional[float]
    time_limit: Optional[datetime]
    
    # Progress tracking
    filled_shares: int = 0
    avg_fill_price: float = 0.0
    remaining_shares: int = 0
    
    is_active: bool = True
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class VolumeProfile:
    """Volume analysis for execution timing"""
    
    symbol: str
    avg_daily_volume: float
    volume_by_hour: Dict[int, float]  # Hour -> average volume
    volume_percentiles: Dict[str, float]  # P10, P25, P50, P75, P90
    
    # VWAP data
    vwap: float
    vwap_std: float
    
    # Liquidity metrics
    bid_ask_spread: float
    market_depth: float
    
    last_updated: datetime = field(default_factory=datetime.now)


class CommissionCalculator(ABC):
    """Abstract base class for commission calculations"""
    
    @abstractmethod
    def calculate_commission(
        self,
        shares: int,
        price: float,
        order_value: float,
        parameters: Dict[str, Any]
    ) -> float:
        """Calculate commission for an order"""
        pass


class FixedCommissionCalculator(CommissionCalculator):
    """Fixed commission per trade"""
    
    def calculate_commission(
        self,
        shares: int,
        price: float,
        order_value: float,
        parameters: Dict[str, Any]
    ) -> float:
        """Calculate fixed commission"""
        return parameters.get('fixed_commission', 1.0)


class PercentageCommissionCalculator(CommissionCalculator):
    """Percentage-based commission"""
    
    def calculate_commission(
        self,
        shares: int,
        price: float,
        order_value: float,
        parameters: Dict[str, Any]
    ) -> float:
        """Calculate percentage-based commission"""
        commission_rate = parameters.get('commission_rate', 0.005)  # 0.5% default
        return order_value * commission_rate


class TieredCommissionCalculator(CommissionCalculator):
    """Tiered commission structure"""
    
    def calculate_commission(
        self,
        shares: int,
        price: float,
        order_value: float,
        parameters: Dict[str, Any]
    ) -> float:
        """Calculate tiered commission"""
        # Example tiered structure
        if order_value <= 10000:
            return min(order_value * 0.01, 50)  # 1% up to $50
        elif order_value <= 100000:
            return 50 + (order_value - 10000) * 0.005  # $50 + 0.5%
        else:
            return 50 + 450 + (order_value - 100000) * 0.003  # $500 + 0.3%


class SlippageCalculator:
    """Calculate expected slippage based on order characteristics"""
    
    @staticmethod
    def calculate_linear_slippage(
        order_size: int,
        avg_daily_volume: float,
        participation_rate: float,
        base_slippage: float
    ) -> float:
        """Calculate linear slippage model"""
        
        # Participation rate impact
        participation_impact = participation_rate * 100  # Convert to bps
        
        # Size impact
        size_ratio = order_size / avg_daily_volume
        size_impact = size_ratio * 50  # 50 bps per 100% of daily volume
        
        # Total slippage
        total_slippage = base_slippage + participation_impact + size_impact
        
        return max(0, total_slippage)
    
    @staticmethod
    def calculate_square_root_slippage(
        order_size: int,
        avg_daily_volume: float,
        volatility: float,
        base_slippage: float
    ) -> float:
        """Calculate square root slippage model (more realistic)"""
        
        # Market impact = volatility * sqrt(order_size / daily_volume)
        size_ratio = order_size / avg_daily_volume
        market_impact = volatility * np.sqrt(size_ratio) * 100  # Convert to bps
        
        return base_slippage + market_impact


class ExecutionContextEngine:
    """
    Main execution context engine that manages order execution parameters
    
    Handles parameters 13-18:
    13. timeframe - Manages execution timing
    14. order_type - Selects optimal order types
    15. slippage_assumption - Models execution costs
    16. commission_model - Calculates trading costs
    17. market_hours_only - Enforces trading windows
    18. minimum_volume - Ensures liquidity requirements
    """
    
    def __init__(self, registry: TradingParameterRegistry = None):
        self.registry = registry or parameter_registry
        self.commission_calculators = self._initialize_commission_calculators()
        self.market_hours = self._initialize_market_hours()
        
        # Execution tracking
        self.active_orders: Dict[str, OrderExecutionPlan] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.volume_profiles: Dict[str, VolumeProfile] = {}
        
    def _initialize_commission_calculators(self) -> Dict[str, CommissionCalculator]:
        """Initialize commission calculators"""
        return {
            'fixed': FixedCommissionCalculator(),
            'percentage': PercentageCommissionCalculator(),
            'tiered': TieredCommissionCalculator(),
            'custom': PercentageCommissionCalculator()  # Default to percentage
        }
    
    def _initialize_market_hours(self) -> MarketHours:
        """Initialize default market hours (US stock market)"""
        return MarketHours(
            open_time=time(9, 30),  # 9:30 AM
            close_time=time(16, 0),  # 4:00 PM
            timezone="US/Eastern",
            pre_market_open=time(4, 0),  # 4:00 AM
            post_market_close=time(20, 0)  # 8:00 PM
        )
    
    def create_execution_plan(
        self,
        symbol: str,
        shares: int,
        target_price: Optional[float] = None,
        market_data: Optional[Dict[str, Any]] = None,
        urgency: str = "normal"  # low, normal, high, urgent
    ) -> OrderExecutionPlan:
        """
        Create optimal execution plan for an order
        
        Args:
            symbol: Trading symbol
            shares: Number of shares to trade
            target_price: Target execution price
            market_data: Current market data
            urgency: Execution urgency level
            
        Returns:
            OrderExecutionPlan with optimized execution strategy
        """
        
        # Get current parameters
        params = self._get_current_parameters()
        market_data = market_data or {}
        
        # Determine optimal order type
        order_type = self._select_order_type(shares, target_price, market_data, params, urgency)
        
        # Calculate execution horizon
        time_horizon = self._calculate_execution_horizon(shares, market_data, params, urgency)
        
        # Determine timing
        start_time = datetime.now()
        end_time = start_time + time_horizon
        
        # Check market hours constraint
        market_hours_only = params.get('market_hours_only', True)
        
        # Calculate participation rate
        participation_rate = self._calculate_participation_rate(shares, market_data, urgency)
        
        # Estimate execution costs
        estimated_costs = self.calculate_execution_costs(
            symbol, shares, target_price or market_data.get('current_price', 100),
            market_data, params
        )
        
        # Create execution plan
        plan = OrderExecutionPlan(
            symbol=symbol,
            order_type=order_type,
            shares=shares,
            target_price=target_price,
            execution_strategy=self._get_execution_strategy(urgency),
            time_horizon=time_horizon,
            max_participation_rate=participation_rate,
            start_time=start_time,
            end_time=end_time,
            market_hours_only=market_hours_only,
            estimated_costs=estimated_costs,
            price_limit=self._calculate_price_limit(target_price, market_data, params),
            time_limit=end_time,
            remaining_shares=shares
        )
        
        # Store active order
        order_id = f"{symbol}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        self.active_orders[order_id] = plan
        
        return plan
    
    def calculate_execution_costs(
        self,
        symbol: str,
        shares: int,
        price: float,
        market_data: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> ExecutionCost:
        """Calculate comprehensive execution costs"""
        
        params = parameters or self._get_current_parameters()
        order_value = shares * price
        
        # Calculate commission
        commission_model = params.get('commission_model', 'percentage')
        commission_calculator = self.commission_calculators.get(
            commission_model, self.commission_calculators['percentage']
        )
        commission = commission_calculator.calculate_commission(shares, price, order_value, params)
        
        # Calculate slippage
        base_slippage = params.get('slippage_assumption', 0.1) / 100.0  # Convert to decimal
        avg_volume = market_data.get('avg_daily_volume', 100000)
        volatility = market_data.get('volatility', 0.20)
        
        # Use square root model for slippage
        slippage_bps = SlippageCalculator.calculate_square_root_slippage(
            shares, avg_volume, volatility, base_slippage * 10000  # Convert to bps
        )
        slippage = (slippage_bps / 10000) * order_value
        
        # Market impact (additional to slippage)
        size_ratio = shares / avg_volume
        market_impact = order_value * volatility * np.sqrt(size_ratio) * 0.1
        
        # Timing costs (opportunity cost of delayed execution)
        timing_cost = order_value * 0.0001  # 1 bps default
        opportunity_cost = 0.0  # Would be calculated based on alpha decay
        
        # Total costs
        total_cost = commission + slippage + market_impact + timing_cost + opportunity_cost
        total_cost_bps = (total_cost / order_value) * 10000
        
        return ExecutionCost(
            commission=commission,
            commission_type=commission_model,
            slippage=slippage,
            slippage_percent=(slippage / order_value) * 100,
            market_impact=market_impact,
            timing_cost=timing_cost,
            opportunity_cost=opportunity_cost,
            total_cost=total_cost,
            total_cost_bps=total_cost_bps,
            fixed_costs=commission,
            variable_costs=slippage + market_impact
        )
    
    def analyze_volume_profile(
        self,
        symbol: str,
        historical_data: pd.DataFrame
    ) -> VolumeProfile:
        """Analyze volume patterns for optimal execution timing"""
        
        if 'volume' not in historical_data.columns:
            raise ValueError("Historical data must include 'volume' column")
        
        # Calculate volume statistics
        avg_daily_volume = historical_data['volume'].mean()
        
        # Volume by hour (if intraday data available)
        volume_by_hour = {}
        if hasattr(historical_data.index, 'hour'):
            volume_by_hour = historical_data.groupby(historical_data.index.hour)['volume'].mean().to_dict()
        
        # Volume percentiles
        volume_percentiles = {
            'P10': historical_data['volume'].quantile(0.10),
            'P25': historical_data['volume'].quantile(0.25),
            'P50': historical_data['volume'].quantile(0.50),
            'P75': historical_data['volume'].quantile(0.75),
            'P90': historical_data['volume'].quantile(0.90)
        }
        
        # VWAP calculation (if price data available)
        vwap = 0.0
        vwap_std = 0.0
        if 'close' in historical_data.columns:
            typical_price = (historical_data['high'] + historical_data['low'] + historical_data['close']) / 3
            vwap = (typical_price * historical_data['volume']).sum() / historical_data['volume'].sum()
            vwap_std = ((typical_price - vwap) ** 2 * historical_data['volume']).sum() / historical_data['volume'].sum()
            vwap_std = np.sqrt(vwap_std)
        
        # Liquidity metrics (simplified)
        bid_ask_spread = 0.001  # 0.1% default
        market_depth = avg_daily_volume * 0.1  # 10% of daily volume
        
        profile = VolumeProfile(
            symbol=symbol,
            avg_daily_volume=avg_daily_volume,
            volume_by_hour=volume_by_hour,
            volume_percentiles=volume_percentiles,
            vwap=vwap,
            vwap_std=vwap_std,
            bid_ask_spread=bid_ask_spread,
            market_depth=market_depth
        )
        
        # Store profile
        self.volume_profiles[symbol] = profile
        
        return profile
    
    def optimize_execution_timing(
        self,
        symbol: str,
        shares: int,
        time_horizon: timedelta
    ) -> List[Dict[str, Any]]:
        """Optimize execution timing based on volume patterns"""
        
        if symbol not in self.volume_profiles:
            # Use default timing if no profile available
            return [{'time': datetime.now(), 'shares': shares, 'reason': 'No volume profile available'}]
        
        profile = self.volume_profiles[symbol]
        
        # Create execution schedule based on volume patterns
        schedule = []
        remaining_shares = shares
        
        # If we have hourly volume data, use it
        if profile.volume_by_hour:
            total_volume_weight = sum(profile.volume_by_hour.values())
            
            for hour, volume_weight in profile.volume_by_hour.items():
                if remaining_shares <= 0:
                    break
                
                # Allocate shares proportionally to volume
                volume_ratio = volume_weight / total_volume_weight
                shares_this_hour = int(shares * volume_ratio)
                shares_this_hour = min(shares_this_hour, remaining_shares)
                
                if shares_this_hour > 0:
                    execution_time = datetime.now().replace(hour=hour, minute=0, second=0)
                    schedule.append({
                        'time': execution_time,
                        'shares': shares_this_hour,
                        'reason': f'High volume hour ({volume_weight:.0f} avg volume)'
                    })
                    remaining_shares -= shares_this_hour
        
        # If no schedule created or shares remaining, use simple time-based splitting
        if not schedule or remaining_shares > 0:
            num_slices = min(10, max(1, int(time_horizon.total_seconds() / 3600)))  # Max 10 slices, 1 per hour
            shares_per_slice = remaining_shares // num_slices
            
            for i in range(num_slices):
                if shares_per_slice > 0:
                    execution_time = datetime.now() + timedelta(hours=i)
                    schedule.append({
                        'time': execution_time,
                        'shares': shares_per_slice,
                        'reason': 'Time-weighted execution'
                    })
        
        return schedule
    
    def check_liquidity_requirements(
        self,
        symbol: str,
        shares: int,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if order meets liquidity requirements"""
        
        params = self._get_current_parameters()
        min_volume = params.get('minimum_volume', 100000)
        
        avg_daily_volume = market_data.get('avg_daily_volume', 0)
        current_volume = market_data.get('current_volume', 0)
        
        # Calculate order size as percentage of daily volume
        order_pct_of_volume = (shares / avg_daily_volume * 100) if avg_daily_volume > 0 else 100
        
        # Liquidity checks
        meets_min_volume = avg_daily_volume >= min_volume
        reasonable_size = order_pct_of_volume <= 20  # Max 20% of daily volume
        sufficient_current_volume = current_volume >= (shares * 0.1)  # 10% of order in current volume
        
        return {
            'meets_requirements': meets_min_volume and reasonable_size,
            'avg_daily_volume': avg_daily_volume,
            'minimum_required': min_volume,
            'order_percent_of_volume': order_pct_of_volume,
            'current_volume': current_volume,
            'warnings': self._generate_liquidity_warnings(
                meets_min_volume, reasonable_size, sufficient_current_volume, order_pct_of_volume
            )
        }
    
    def is_market_open(self, timestamp: Optional[datetime] = None) -> bool:
        """Check if market is currently open"""
        timestamp = timestamp or datetime.now()
        return self.market_hours.is_market_open(timestamp)
    
    def get_next_market_open(self, from_time: Optional[datetime] = None) -> datetime:
        """Get next market open time"""
        from_time = from_time or datetime.now()
        
        # Simple logic - assumes next business day if market closed
        next_day = from_time.date()
        while True:
            next_day += timedelta(days=1)
            # Skip weekends
            if next_day.weekday() < 5:
                next_open = datetime.combine(next_day, self.market_hours.open_time)
                return pytz.timezone(self.market_hours.timezone).localize(next_open)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution context and active orders"""
        
        active_order_count = len(self.active_orders)
        total_active_value = sum(
            order.shares * (order.target_price or 100) 
            for order in self.active_orders.values()
        )
        
        return {
            'market_open': self.is_market_open(),
            'active_orders': active_order_count,
            'total_active_value': total_active_value,
            'execution_parameters': self._get_current_parameters(),
            'market_hours': {
                'open': self.market_hours.open_time.strftime('%H:%M'),
                'close': self.market_hours.close_time.strftime('%H:%M'),
                'timezone': self.market_hours.timezone
            },
            'volume_profiles_cached': len(self.volume_profiles),
            'last_updated': datetime.now()
        }
    
    def _get_current_parameters(self) -> Dict[str, Any]:
        """Get current execution context parameters"""
        param_names = [
            'timeframe', 'order_type', 'slippage_assumption',
            'commission_model', 'market_hours_only', 'minimum_volume'
        ]
        params = {}
        
        for name in param_names:
            param_def = self.registry.get_parameter(name)
            if param_def:
                params[name] = param_def.current_value or param_def.default_value
        
        return params
    
    def _select_order_type(
        self,
        shares: int,
        target_price: Optional[float],
        market_data: Dict[str, Any],
        params: Dict[str, Any],
        urgency: str
    ) -> OrderType:
        """Select optimal order type based on conditions"""
        
        default_order_type = params.get('order_type', 'limit')
        
        # Override based on urgency
        if urgency == 'urgent':
            return OrderType.MARKET
        elif urgency == 'low':
            return OrderType.LIMIT
        
        # Use default if configured properly
        try:
            return OrderType(default_order_type)
        except ValueError:
            return OrderType.LIMIT
    
    def _calculate_execution_horizon(
        self,
        shares: int,
        market_data: Dict[str, Any],
        params: Dict[str, Any],
        urgency: str
    ) -> timedelta:
        """Calculate optimal execution time horizon"""
        
        # Base horizon based on order size
        avg_volume = market_data.get('avg_daily_volume', 100000)
        size_ratio = shares / avg_volume
        
        if size_ratio <= 0.05:  # Small order (5% of daily volume)
            base_horizon = timedelta(minutes=30)
        elif size_ratio <= 0.20:  # Medium order (20% of daily volume)
            base_horizon = timedelta(hours=2)
        else:  # Large order
            base_horizon = timedelta(hours=8)
        
        # Adjust based on urgency
        urgency_multipliers = {
            'urgent': 0.1,
            'high': 0.5,
            'normal': 1.0,
            'low': 2.0
        }
        
        multiplier = urgency_multipliers.get(urgency, 1.0)
        return base_horizon * multiplier
    
    def _calculate_participation_rate(
        self,
        shares: int,
        market_data: Dict[str, Any],
        urgency: str
    ) -> float:
        """Calculate maximum participation rate in volume"""
        
        # Base participation rates by urgency
        base_rates = {
            'urgent': 0.30,    # 30% of volume
            'high': 0.20,      # 20% of volume
            'normal': 0.10,    # 10% of volume
            'low': 0.05        # 5% of volume
        }
        
        return base_rates.get(urgency, 0.10)
    
    def _get_execution_strategy(self, urgency: str) -> str:
        """Get execution strategy name based on urgency"""
        
        strategies = {
            'urgent': 'Aggressive TWAP',
            'high': 'TWAP',
            'normal': 'VWAP',
            'low': 'Opportunistic'
        }
        
        return strategies.get(urgency, 'VWAP')
    
    def _calculate_price_limit(
        self,
        target_price: Optional[float],
        market_data: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Optional[float]:
        """Calculate price limit for order protection"""
        
        if not target_price:
            return None
        
        # Apply slippage tolerance as price limit
        slippage_tolerance = params.get('slippage_assumption', 0.1) / 100.0
        
        # For buy orders, limit is target + tolerance
        # For sell orders, limit is target - tolerance
        # This is simplified - in practice would depend on order side
        
        return target_price * (1 + slippage_tolerance)
    
    def _generate_liquidity_warnings(
        self,
        meets_min_volume: bool,
        reasonable_size: bool,
        sufficient_current_volume: bool,
        order_pct: float
    ) -> List[str]:
        """Generate liquidity warnings"""
        
        warnings = []
        
        if not meets_min_volume:
            warnings.append("Average daily volume below minimum requirement")
        
        if not reasonable_size:
            warnings.append(f"Order size ({order_pct:.1f}%) exceeds recommended 20% of daily volume")
        
        if not sufficient_current_volume:
            warnings.append("Insufficient current volume for immediate execution")
        
        if order_pct > 50:
            warnings.append("Order size extremely large relative to daily volume - consider breaking up")
        
        return warnings


# Factory function for easy instantiation
def create_execution_context_engine(registry: TradingParameterRegistry = None) -> ExecutionContextEngine:
    """Create an execution context engine instance"""
    return ExecutionContextEngine(registry)