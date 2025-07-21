"""
Trading Parameter Registry - Central Configuration Manager

This module provides a comprehensive registry for all 44+ trading parameters,
organized by functional categories with validation and type checking.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ParameterCategory(Enum):
    """Trading parameter categories"""
    POSITION_SIZING = "position_sizing"
    RISK_MANAGEMENT = "risk_management"
    EXECUTION_CONTEXT = "execution_context"
    DATA_REQUIREMENTS = "data_requirements"
    UNIVERSE_SELECTION = "universe_selection"
    ALPHA_GENERATION = "alpha_generation"
    PORTFOLIO_CONSTRUCTION = "portfolio_construction"
    ADVANCED_FEATURES = "advanced_features"


class ParameterType(Enum):
    """Parameter data types"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    ENUM = "enum"
    LIST = "list"
    DICT = "dict"


@dataclass
class ParameterDefinition:
    """Complete parameter definition with validation rules"""
    
    # Basic properties
    name: str
    category: ParameterCategory
    parameter_type: ParameterType
    description: str
    
    # Validation rules
    required: bool = True
    default_value: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    validation_function: Optional[Callable] = None
    
    # Metadata
    unit: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    affects: List[str] = field(default_factory=list)
    risk_level: str = "medium"  # low, medium, high, critical
    
    # Runtime properties
    current_value: Any = None
    last_updated: Optional[datetime] = None
    update_frequency: str = "on_change"  # on_change, daily, intraday, real_time


class TradingParameterRegistry:
    """
    Central registry for all trading parameters with validation and management.
    
    Manages 44+ parameters across 8 categories:
    - Position Sizing (5 parameters)
    - Risk Management (7 parameters) 
    - Execution Context (6 parameters)
    - Data Requirements (5 parameters)
    - Universe Selection (6 parameters)
    - Alpha Generation (5 parameters)
    - Portfolio Construction (5 parameters)
    - Advanced Features (5 parameters)
    """
    
    def __init__(self):
        self.parameters: Dict[str, ParameterDefinition] = {}
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize all 44+ trading parameters"""
        
        # POSITION SIZING PARAMETERS (1-5)
        self._add_position_sizing_parameters()
        
        # RISK MANAGEMENT PARAMETERS (6-12)
        self._add_risk_management_parameters()
        
        # EXECUTION CONTEXT PARAMETERS (13-18)
        self._add_execution_context_parameters()
        
        # DATA REQUIREMENTS PARAMETERS (19-23)
        self._add_data_requirements_parameters()
        
        # UNIVERSE SELECTION PARAMETERS (24-29)
        self._add_universe_selection_parameters()
        
        # ALPHA GENERATION PARAMETERS (30-34)
        self._add_alpha_generation_parameters()
        
        # PORTFOLIO CONSTRUCTION PARAMETERS (35-39)
        self._add_portfolio_construction_parameters()
        
        # ADVANCED FEATURES PARAMETERS (40-44)
        self._add_advanced_features_parameters()
        
    def _add_position_sizing_parameters(self):
        """Position Sizing Parameters (1-5)"""
        
        self.parameters["method"] = ParameterDefinition(
            name="method",
            category=ParameterCategory.POSITION_SIZING,
            parameter_type=ParameterType.ENUM,
            description="Position sizing methodology",
            allowed_values=["fixed", "percentage", "kelly", "volatility_adjusted", "risk_parity"],
            default_value="percentage",
            risk_level="high"
        )
        
        self.parameters["value"] = ParameterDefinition(
            name="value",
            category=ParameterCategory.POSITION_SIZING,
            parameter_type=ParameterType.PERCENTAGE,
            description="Position size value (percentage of portfolio)",
            min_value=0.01,
            max_value=100.0,
            default_value=5.0,
            unit="%",
            dependencies=["method"],
            risk_level="critical"
        )
        
        self.parameters["max_position_size"] = ParameterDefinition(
            name="max_position_size",
            category=ParameterCategory.POSITION_SIZING,
            parameter_type=ParameterType.PERCENTAGE,
            description="Maximum position size limit",
            min_value=0.01,
            max_value=100.0,
            default_value=10.0,
            unit="%",
            risk_level="critical"
        )
        
        self.parameters["risk_per_trade"] = ParameterDefinition(
            name="risk_per_trade",
            category=ParameterCategory.POSITION_SIZING,
            parameter_type=ParameterType.PERCENTAGE,
            description="Maximum risk per individual trade",
            min_value=0.01,
            max_value=10.0,
            default_value=1.0,
            unit="%",
            risk_level="critical"
        )
        
        self.parameters["volatility_adjustment"] = ParameterDefinition(
            name="volatility_adjustment",
            category=ParameterCategory.POSITION_SIZING,
            parameter_type=ParameterType.BOOLEAN,
            description="Adjust position size based on volatility",
            default_value=True,
            affects=["value"],
            risk_level="medium"
        )
    
    def _add_risk_management_parameters(self):
        """Risk Management Parameters (6-12)"""
        
        self.parameters["stop_loss"] = ParameterDefinition(
            name="stop_loss",
            category=ParameterCategory.RISK_MANAGEMENT,
            parameter_type=ParameterType.PERCENTAGE,
            description="Stop loss percentage",
            min_value=0.1,
            max_value=50.0,
            default_value=5.0,
            unit="%",
            risk_level="critical"
        )
        
        self.parameters["stop_loss_type"] = ParameterDefinition(
            name="stop_loss_type",
            category=ParameterCategory.RISK_MANAGEMENT,
            parameter_type=ParameterType.ENUM,
            description="Type of stop loss mechanism",
            allowed_values=["fixed", "trailing", "volatility_based", "atr_based"],
            default_value="trailing",
            dependencies=["stop_loss"],
            risk_level="high"
        )
        
        self.parameters["take_profit"] = ParameterDefinition(
            name="take_profit",
            category=ParameterCategory.RISK_MANAGEMENT,
            parameter_type=ParameterType.PERCENTAGE,
            description="Take profit percentage",
            min_value=0.1,
            max_value=100.0,
            default_value=10.0,
            unit="%",
            risk_level="medium"
        )
        
        self.parameters["take_profit_type"] = ParameterDefinition(
            name="take_profit_type",
            category=ParameterCategory.RISK_MANAGEMENT,
            parameter_type=ParameterType.ENUM,
            description="Type of take profit mechanism",
            allowed_values=["fixed", "trailing", "scale_out", "target_based"],
            default_value="fixed",
            dependencies=["take_profit"],
            risk_level="medium"
        )
        
        self.parameters["max_drawdown"] = ParameterDefinition(
            name="max_drawdown",
            category=ParameterCategory.RISK_MANAGEMENT,
            parameter_type=ParameterType.PERCENTAGE,
            description="Maximum portfolio drawdown limit",
            min_value=1.0,
            max_value=50.0,
            default_value=15.0,
            unit="%",
            risk_level="critical"
        )
        
        self.parameters["correlation_limit"] = ParameterDefinition(
            name="correlation_limit",
            category=ParameterCategory.RISK_MANAGEMENT,
            parameter_type=ParameterType.FLOAT,
            description="Maximum correlation between positions",
            min_value=0.0,
            max_value=1.0,
            default_value=0.7,
            risk_level="high"
        )
        
        self.parameters["sector_exposure_limit"] = ParameterDefinition(
            name="sector_exposure_limit",
            category=ParameterCategory.RISK_MANAGEMENT,
            parameter_type=ParameterType.PERCENTAGE,
            description="Maximum exposure to any single sector",
            min_value=1.0,
            max_value=100.0,
            default_value=25.0,
            unit="%",
            risk_level="high"
        )
    
    def _add_execution_context_parameters(self):
        """Execution Context Parameters (13-18)"""
        
        self.parameters["timeframe"] = ParameterDefinition(
            name="timeframe",
            category=ParameterCategory.EXECUTION_CONTEXT,
            parameter_type=ParameterType.ENUM,
            description="Trading timeframe",
            allowed_values=["1m", "5m", "15m", "1h", "4h", "1d", "1w"],
            default_value="1d",
            risk_level="medium"
        )
        
        self.parameters["order_type"] = ParameterDefinition(
            name="order_type",
            category=ParameterCategory.EXECUTION_CONTEXT,
            parameter_type=ParameterType.ENUM,
            description="Default order type",
            allowed_values=["market", "limit", "stop", "stop_limit", "iceberg"],
            default_value="limit",
            risk_level="medium"
        )
        
        self.parameters["slippage_assumption"] = ParameterDefinition(
            name="slippage_assumption",
            category=ParameterCategory.EXECUTION_CONTEXT,
            parameter_type=ParameterType.PERCENTAGE,
            description="Expected slippage percentage",
            min_value=0.0,
            max_value=5.0,
            default_value=0.1,
            unit="%",
            risk_level="medium"
        )
        
        self.parameters["commission_model"] = ParameterDefinition(
            name="commission_model",
            category=ParameterCategory.EXECUTION_CONTEXT,
            parameter_type=ParameterType.ENUM,
            description="Commission calculation model",
            allowed_values=["fixed", "percentage", "tiered", "custom"],
            default_value="percentage",
            risk_level="low"
        )
        
        self.parameters["market_hours_only"] = ParameterDefinition(
            name="market_hours_only",
            category=ParameterCategory.EXECUTION_CONTEXT,
            parameter_type=ParameterType.BOOLEAN,
            description="Trade only during market hours",
            default_value=True,
            risk_level="medium"
        )
        
        self.parameters["minimum_volume"] = ParameterDefinition(
            name="minimum_volume",
            category=ParameterCategory.EXECUTION_CONTEXT,
            parameter_type=ParameterType.INTEGER,
            description="Minimum daily volume requirement",
            min_value=1000,
            default_value=100000,
            unit="shares",
            risk_level="medium"
        )
    
    def _add_data_requirements_parameters(self):
        """Data Requirements Parameters (19-23)"""
        
        self.parameters["lookback_period"] = ParameterDefinition(
            name="lookback_period",
            category=ParameterCategory.DATA_REQUIREMENTS,
            parameter_type=ParameterType.INTEGER,
            description="Historical data lookback period",
            min_value=1,
            max_value=1000,
            default_value=252,
            unit="periods",
            risk_level="medium"
        )
        
        self.parameters["data_frequency"] = ParameterDefinition(
            name="data_frequency",
            category=ParameterCategory.DATA_REQUIREMENTS,
            parameter_type=ParameterType.ENUM,
            description="Data update frequency",
            allowed_values=["real_time", "minute", "hourly", "daily", "weekly"],
            default_value="daily",
            risk_level="low"
        )
        
        self.parameters["data_quality_filter"] = ParameterDefinition(
            name="data_quality_filter",
            category=ParameterCategory.DATA_REQUIREMENTS,
            parameter_type=ParameterType.BOOLEAN,
            description="Apply data quality filtering",
            default_value=True,
            risk_level="medium"
        )
        
        self.parameters["benchmark"] = ParameterDefinition(
            name="benchmark",
            category=ParameterCategory.DATA_REQUIREMENTS,
            parameter_type=ParameterType.STRING,
            description="Benchmark for performance comparison",
            default_value="SPY",
            risk_level="low"
        )
        
        self.parameters["market_regime_filter"] = ParameterDefinition(
            name="market_regime_filter",
            category=ParameterCategory.DATA_REQUIREMENTS,
            parameter_type=ParameterType.BOOLEAN,
            description="Apply market regime filtering",
            default_value=False,
            risk_level="medium"
        )
    
    def _add_universe_selection_parameters(self):
        """Universe Selection Parameters (24-29)"""
        
        self.parameters["selection_method"] = ParameterDefinition(
            name="selection_method",
            category=ParameterCategory.UNIVERSE_SELECTION,
            parameter_type=ParameterType.ENUM,
            description="Universe selection methodology",
            allowed_values=["top_n", "percentile", "custom_filter", "sector_based"],
            default_value="top_n",
            risk_level="medium"
        )
        
        self.parameters["universe_size"] = ParameterDefinition(
            name="universe_size",
            category=ParameterCategory.UNIVERSE_SELECTION,
            parameter_type=ParameterType.INTEGER,
            description="Number of assets in universe",
            min_value=1,
            max_value=5000,
            default_value=100,
            dependencies=["selection_method"],
            risk_level="medium"
        )
        
        self.parameters["liquidity_filter"] = ParameterDefinition(
            name="liquidity_filter",
            category=ParameterCategory.UNIVERSE_SELECTION,
            parameter_type=ParameterType.CURRENCY,
            description="Minimum daily liquidity requirement",
            min_value=1000,
            default_value=1000000,
            unit="USD",
            risk_level="high"
        )
        
        self.parameters["market_cap_filter"] = ParameterDefinition(
            name="market_cap_filter",
            category=ParameterCategory.UNIVERSE_SELECTION,
            parameter_type=ParameterType.CURRENCY,
            description="Minimum market capitalization",
            min_value=1000000,
            default_value=1000000000,
            unit="USD",
            risk_level="medium"
        )
        
        self.parameters["sector_filter"] = ParameterDefinition(
            name="sector_filter",
            category=ParameterCategory.UNIVERSE_SELECTION,
            parameter_type=ParameterType.LIST,
            description="Allowed sectors",
            default_value=[],
            risk_level="medium"
        )
        
        self.parameters["rebalance_frequency"] = ParameterDefinition(
            name="rebalance_frequency",
            category=ParameterCategory.UNIVERSE_SELECTION,
            parameter_type=ParameterType.ENUM,
            description="Universe rebalancing frequency",
            allowed_values=["daily", "weekly", "monthly", "quarterly"],
            default_value="monthly",
            risk_level="medium"
        )
    
    def _add_alpha_generation_parameters(self):
        """Alpha Generation Parameters (30-34)"""
        
        self.parameters["signal_confidence"] = ParameterDefinition(
            name="signal_confidence",
            category=ParameterCategory.ALPHA_GENERATION,
            parameter_type=ParameterType.FLOAT,
            description="Minimum signal confidence threshold",
            min_value=0.0,
            max_value=1.0,
            default_value=0.6,
            risk_level="high"
        )
        
        self.parameters["signal_magnitude"] = ParameterDefinition(
            name="signal_magnitude",
            category=ParameterCategory.ALPHA_GENERATION,
            parameter_type=ParameterType.FLOAT,
            description="Minimum signal magnitude threshold",
            min_value=0.0,
            max_value=10.0,
            default_value=1.0,
            risk_level="medium"
        )
        
        self.parameters["signal_direction"] = ParameterDefinition(
            name="signal_direction",
            category=ParameterCategory.ALPHA_GENERATION,
            parameter_type=ParameterType.ENUM,
            description="Allowed signal directions",
            allowed_values=["long_only", "short_only", "long_short"],
            default_value="long_short",
            risk_level="high"
        )
        
        self.parameters["signal_persistence"] = ParameterDefinition(
            name="signal_persistence",
            category=ParameterCategory.ALPHA_GENERATION,
            parameter_type=ParameterType.INTEGER,
            description="Minimum signal persistence periods",
            min_value=1,
            max_value=100,
            default_value=3,
            unit="periods",
            risk_level="medium"
        )
        
        self.parameters["multi_factor_model"] = ParameterDefinition(
            name="multi_factor_model",
            category=ParameterCategory.ALPHA_GENERATION,
            parameter_type=ParameterType.BOOLEAN,
            description="Use multi-factor model",
            default_value=True,
            risk_level="medium"
        )
    
    def _add_portfolio_construction_parameters(self):
        """Portfolio Construction Parameters (35-39)"""
        
        self.parameters["weighting_method"] = ParameterDefinition(
            name="weighting_method",
            category=ParameterCategory.PORTFOLIO_CONSTRUCTION,
            parameter_type=ParameterType.ENUM,
            description="Portfolio weighting methodology",
            allowed_values=["equal_weight", "market_cap", "risk_parity", "signal_weight", "optimization"],
            default_value="signal_weight",
            risk_level="high"
        )
        
        self.parameters["target_allocation"] = ParameterDefinition(
            name="target_allocation",
            category=ParameterCategory.PORTFOLIO_CONSTRUCTION,
            parameter_type=ParameterType.DICT,
            description="Target asset allocation",
            default_value={},
            risk_level="medium"
        )
        
        self.parameters["rebalancing_threshold"] = ParameterDefinition(
            name="rebalancing_threshold",
            category=ParameterCategory.PORTFOLIO_CONSTRUCTION,
            parameter_type=ParameterType.PERCENTAGE,
            description="Rebalancing threshold",
            min_value=0.1,
            max_value=50.0,
            default_value=5.0,
            unit="%",
            risk_level="medium"
        )
        
        self.parameters["transaction_cost_model"] = ParameterDefinition(
            name="transaction_cost_model",
            category=ParameterCategory.PORTFOLIO_CONSTRUCTION,
            parameter_type=ParameterType.ENUM,
            description="Transaction cost model",
            allowed_values=["linear", "square_root", "market_impact", "custom"],
            default_value="linear",
            risk_level="medium"
        )
        
        self.parameters["leverage_limit"] = ParameterDefinition(
            name="leverage_limit",
            category=ParameterCategory.PORTFOLIO_CONSTRUCTION,
            parameter_type=ParameterType.FLOAT,
            description="Maximum leverage ratio",
            min_value=1.0,
            max_value=10.0,
            default_value=1.0,
            risk_level="critical"
        )
    
    def _add_advanced_features_parameters(self):
        """Advanced Features Parameters (40-44)"""
        
        self.parameters["regime_detection"] = ParameterDefinition(
            name="regime_detection",
            category=ParameterCategory.ADVANCED_FEATURES,
            parameter_type=ParameterType.BOOLEAN,
            description="Enable market regime detection",
            default_value=False,
            risk_level="medium"
        )
        
        self.parameters["alternative_data"] = ParameterDefinition(
            name="alternative_data",
            category=ParameterCategory.ADVANCED_FEATURES,
            parameter_type=ParameterType.BOOLEAN,
            description="Use alternative data sources",
            default_value=False,
            risk_level="medium"
        )
        
        self.parameters["machine_learning"] = ParameterDefinition(
            name="machine_learning",
            category=ParameterCategory.ADVANCED_FEATURES,
            parameter_type=ParameterType.BOOLEAN,
            description="Enable machine learning models",
            default_value=False,
            risk_level="high"
        )
        
        self.parameters["stress_testing"] = ParameterDefinition(
            name="stress_testing",
            category=ParameterCategory.ADVANCED_FEATURES,
            parameter_type=ParameterType.BOOLEAN,
            description="Enable stress testing",
            default_value=True,
            risk_level="medium"
        )
        
        self.parameters["attribution_analysis"] = ParameterDefinition(
            name="attribution_analysis",
            category=ParameterCategory.ADVANCED_FEATURES,
            parameter_type=ParameterType.BOOLEAN,
            description="Enable attribution analysis",
            default_value=True,
            risk_level="low"
        )
    
    def get_parameter(self, name: str) -> Optional[ParameterDefinition]:
        """Get parameter definition by name"""
        return self.parameters.get(name)
    
    def get_parameters_by_category(self, category: ParameterCategory) -> Dict[str, ParameterDefinition]:
        """Get all parameters in a category"""
        return {
            name: param for name, param in self.parameters.items()
            if param.category == category
        }
    
    def get_all_parameters(self) -> Dict[str, ParameterDefinition]:
        """Get all parameters"""
        return self.parameters.copy()
    
    def get_parameter_count(self) -> int:
        """Get total parameter count"""
        return len(self.parameters)
    
    def get_category_summary(self) -> Dict[str, int]:
        """Get parameter count by category"""
        summary = {}
        for category in ParameterCategory:
            summary[category.value] = len(self.get_parameters_by_category(category))
        return summary
    
    def validate_parameter_value(self, name: str, value: Any) -> tuple[bool, str]:
        """Validate a parameter value"""
        param = self.get_parameter(name)
        if not param:
            return False, f"Parameter '{name}' not found"
        
        # Type validation
        if param.parameter_type == ParameterType.INTEGER and not isinstance(value, int):
            return False, f"Parameter '{name}' must be an integer"
        
        if param.parameter_type == ParameterType.FLOAT and not isinstance(value, (int, float)):
            return False, f"Parameter '{name}' must be a number"
        
        if param.parameter_type == ParameterType.BOOLEAN and not isinstance(value, bool):
            return False, f"Parameter '{name}' must be a boolean"
        
        if param.parameter_type == ParameterType.STRING and not isinstance(value, str):
            return False, f"Parameter '{name}' must be a string"
        
        # Range validation
        if param.min_value is not None and value < param.min_value:
            return False, f"Parameter '{name}' must be >= {param.min_value}"
        
        if param.max_value is not None and value > param.max_value:
            return False, f"Parameter '{name}' must be <= {param.max_value}"
        
        # Allowed values validation
        if param.allowed_values and value not in param.allowed_values:
            return False, f"Parameter '{name}' must be one of {param.allowed_values}"
        
        # Custom validation
        if param.validation_function:
            try:
                if not param.validation_function(value):
                    return False, f"Parameter '{name}' failed custom validation"
            except Exception as e:
                return False, f"Parameter '{name}' validation error: {str(e)}"
        
        return True, "Valid"
    
    def update_parameter_value(self, name: str, value: Any) -> bool:
        """Update parameter value with validation"""
        is_valid, message = self.validate_parameter_value(name, value)
        if is_valid:
            param = self.get_parameter(name)
            param.current_value = value
            param.last_updated = datetime.now()
            logger.info(f"Updated parameter '{name}' to {value}")
            return True
        else:
            logger.error(f"Failed to update parameter '{name}': {message}")
            return False
    
    def export_configuration(self) -> Dict:
        """Export current parameter configuration"""
        config = {}
        for name, param in self.parameters.items():
            config[name] = {
                "category": param.category.value,
                "type": param.parameter_type.value,
                "current_value": param.current_value,
                "default_value": param.default_value,
                "last_updated": param.last_updated.isoformat() if param.last_updated else None
            }
        return config
    
    def import_configuration(self, config: Dict) -> bool:
        """Import parameter configuration"""
        try:
            for name, param_config in config.items():
                if name in self.parameters:
                    value = param_config.get("current_value")
                    if value is not None:
                        self.update_parameter_value(name, value)
            return True
        except Exception as e:
            logger.error(f"Failed to import configuration: {str(e)}")
            return False


# Global parameter registry instance
parameter_registry = TradingParameterRegistry()