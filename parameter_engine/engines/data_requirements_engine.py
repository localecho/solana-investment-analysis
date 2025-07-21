"""
Data Requirements Engine - Implementation of Parameters 19-23

This engine handles the business logic for data requirements parameters:
19. lookback_period - Historical data lookback period
20. data_frequency - Data update frequency
21. data_quality_filter - Apply data quality filtering
22. benchmark - Benchmark for performance comparison
23. market_regime_filter - Apply market regime filtering

The engine provides data pipeline management, quality control,
benchmark tracking, and regime detection capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod
import warnings

from core.parameter_registry import TradingParameterRegistry, parameter_registry

logger = logging.getLogger(__name__)


class DataFrequency(Enum):
    """Data frequency options"""
    REAL_TIME = "real_time"
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"


class MarketRegime(Enum):
    """Market regime types"""
    BULL_MARKET = "bull"
    BEAR_MARKET = "bear"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"
    TRENDING = "trending"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"
    RECOVERY = "recovery"


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""
    
    # Completeness metrics
    total_records: int
    missing_records: int
    completeness_ratio: float
    
    # Accuracy metrics
    outlier_count: int
    outlier_ratio: float
    price_consistency_score: float
    volume_consistency_score: float
    
    # Timeliness metrics
    latest_timestamp: datetime
    data_lag_minutes: float
    update_frequency_actual: str
    
    # Reliability metrics
    duplicate_records: int
    data_gaps: List[Tuple[datetime, datetime]]
    gap_duration_total: timedelta
    
    # Overall quality score (0-100)
    overall_quality_score: float
    
    # Issues and warnings
    quality_issues: List[str] = field(default_factory=list)
    quality_warnings: List[str] = field(default_factory=list)
    
    assessment_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkMetrics:
    """Benchmark comparison metrics"""
    
    benchmark_symbol: str
    benchmark_return: float
    strategy_return: float
    excess_return: float
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    information_ratio: float
    treynor_ratio: float
    jensen_alpha: float
    
    # Tracking metrics
    tracking_error: float
    correlation: float
    beta: float
    r_squared: float
    
    # Performance attribution
    selection_effect: float
    timing_effect: float
    interaction_effect: float
    
    calculation_period: Tuple[datetime, datetime]
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class RegimeDetectionResult:
    """Market regime detection result"""
    
    current_regime: MarketRegime
    regime_confidence: float  # 0-1
    regime_duration: timedelta
    regime_start_date: datetime
    
    # Regime characteristics
    volatility_level: float
    trend_strength: float
    momentum_score: float
    
    # Regime history
    previous_regime: Optional[MarketRegime] = None
    regime_transition_date: Optional[datetime] = None
    
    # Regime probabilities
    regime_probabilities: Dict[str, float] = field(default_factory=dict)
    
    # Indicators used
    indicators_used: List[str] = field(default_factory=list)
    
    detection_timestamp: datetime = field(default_factory=datetime.now)


class DataQualityFilter(ABC):
    """Abstract base class for data quality filters"""
    
    @abstractmethod
    def apply_filter(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityMetrics]:
        """Apply quality filter to data and return metrics"""
        pass


class BasicDataQualityFilter(DataQualityFilter):
    """Basic data quality filtering"""
    
    def apply_filter(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityMetrics]:
        """Apply basic quality checks and filtering"""
        
        original_count = len(data)
        filtered_data = data.copy()
        issues = []
        warnings = []
        
        # Remove duplicates
        duplicates_before = len(filtered_data)
        filtered_data = filtered_data.drop_duplicates()
        duplicate_count = duplicates_before - len(filtered_data)
        
        if duplicate_count > 0:
            warnings.append(f"Removed {duplicate_count} duplicate records")
        
        # Check for missing values
        missing_data = filtered_data.isnull().sum()
        total_missing = missing_data.sum()
        
        if total_missing > 0:
            # Fill or remove missing values
            if 'close' in filtered_data.columns:
                filtered_data['close'].fillna(method='ffill', inplace=True)
            if 'volume' in filtered_data.columns:
                filtered_data['volume'].fillna(0, inplace=True)
            
            warnings.append(f"Handled {total_missing} missing values")
        
        # Check for price outliers (using IQR method)
        outlier_count = 0
        if 'close' in filtered_data.columns:
            Q1 = filtered_data['close'].quantile(0.25)
            Q3 = filtered_data['close'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = (filtered_data['close'] < lower_bound) | (filtered_data['close'] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                # Cap outliers instead of removing
                filtered_data.loc[filtered_data['close'] < lower_bound, 'close'] = lower_bound
                filtered_data.loc[filtered_data['close'] > upper_bound, 'close'] = upper_bound
                warnings.append(f"Capped {outlier_count} price outliers")
        
        # Check for volume outliers
        if 'volume' in filtered_data.columns:
            # Remove negative volumes
            negative_volume = (filtered_data['volume'] < 0).sum()
            if negative_volume > 0:
                filtered_data = filtered_data[filtered_data['volume'] >= 0]
                issues.append(f"Removed {negative_volume} records with negative volume")
        
        # Calculate quality metrics
        completeness_ratio = len(filtered_data) / original_count if original_count > 0 else 0
        outlier_ratio = outlier_count / original_count if original_count > 0 else 0
        
        # Data consistency checks
        price_consistency = self._check_price_consistency(filtered_data)
        volume_consistency = self._check_volume_consistency(filtered_data)
        
        # Check data gaps
        data_gaps = self._identify_data_gaps(filtered_data)
        gap_duration = sum([(end - start) for start, end in data_gaps], timedelta())
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            completeness_ratio, outlier_ratio, price_consistency, volume_consistency, len(data_gaps)
        )
        
        # Latest data info
        latest_timestamp = filtered_data.index.max() if not filtered_data.empty else datetime.min
        data_lag = (datetime.now() - latest_timestamp).total_seconds() / 60 if latest_timestamp != datetime.min else float('inf')
        
        metrics = DataQualityMetrics(
            total_records=len(filtered_data),
            missing_records=original_count - len(filtered_data),
            completeness_ratio=completeness_ratio,
            outlier_count=outlier_count,
            outlier_ratio=outlier_ratio,
            price_consistency_score=price_consistency,
            volume_consistency_score=volume_consistency,
            latest_timestamp=latest_timestamp,
            data_lag_minutes=data_lag,
            update_frequency_actual="unknown",
            duplicate_records=duplicate_count,
            data_gaps=data_gaps,
            gap_duration_total=gap_duration,
            overall_quality_score=quality_score,
            quality_issues=issues,
            quality_warnings=warnings
        )
        
        return filtered_data, metrics
    
    def _check_price_consistency(self, data: pd.DataFrame) -> float:
        """Check price data consistency"""
        if 'close' not in data.columns or len(data) < 2:
            return 100.0
        
        # Check for reasonable price changes (less than 50% in one period)
        price_changes = data['close'].pct_change().abs()
        extreme_changes = (price_changes > 0.5).sum()
        consistency_score = max(0, 100 - (extreme_changes / len(data) * 100))
        
        return consistency_score
    
    def _check_volume_consistency(self, data: pd.DataFrame) -> float:
        """Check volume data consistency"""
        if 'volume' not in data.columns or len(data) < 2:
            return 100.0
        
        # Check for zero volume days
        zero_volume_days = (data['volume'] == 0).sum()
        consistency_score = max(0, 100 - (zero_volume_days / len(data) * 100))
        
        return consistency_score
    
    def _identify_data_gaps(self, data: pd.DataFrame) -> List[Tuple[datetime, datetime]]:
        """Identify gaps in time series data"""
        if data.empty or not isinstance(data.index, pd.DatetimeIndex):
            return []
        
        # Assume daily data for gap detection
        expected_freq = pd.Timedelta(days=1)
        gaps = []
        
        for i in range(1, len(data)):
            time_diff = data.index[i] - data.index[i-1]
            if time_diff > expected_freq * 3:  # Gap of more than 3 days (accounting for weekends)
                gaps.append((data.index[i-1], data.index[i]))
        
        return gaps
    
    def _calculate_quality_score(
        self, 
        completeness: float, 
        outlier_ratio: float, 
        price_consistency: float, 
        volume_consistency: float, 
        gap_count: int
    ) -> float:
        """Calculate overall quality score"""
        
        # Weight components
        completeness_weight = 0.3
        outlier_weight = 0.2
        price_weight = 0.3
        volume_weight = 0.1
        gap_weight = 0.1
        
        # Calculate component scores
        completeness_score = completeness * 100
        outlier_score = max(0, 100 - outlier_ratio * 100)
        gap_score = max(0, 100 - gap_count * 10)  # Deduct 10 points per gap
        
        # Weighted average
        total_score = (
            completeness_score * completeness_weight +
            outlier_score * outlier_weight +
            price_consistency * price_weight +
            volume_consistency * volume_weight +
            gap_score * gap_weight
        )
        
        return min(100, max(0, total_score))


class RegimeDetector(ABC):
    """Abstract base class for market regime detection"""
    
    @abstractmethod
    def detect_regime(self, data: pd.DataFrame) -> RegimeDetectionResult:
        """Detect current market regime"""
        pass


class VolatilityRegimeDetector(RegimeDetector):
    """Volatility-based regime detection"""
    
    def detect_regime(self, data: pd.DataFrame) -> RegimeDetectionResult:
        """Detect regime based on volatility patterns"""
        
        if 'close' not in data.columns or len(data) < 20:
            return self._default_regime_result()
        
        # Calculate rolling volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
        current_vol = volatility.iloc[-1]
        
        # Calculate percentiles for regime classification
        vol_25 = volatility.quantile(0.25)
        vol_75 = volatility.quantile(0.75)
        
        # Determine volatility regime
        if current_vol > vol_75:
            vol_regime = MarketRegime.HIGH_VOLATILITY
            confidence = min(1.0, (current_vol - vol_75) / (volatility.max() - vol_75))
        elif current_vol < vol_25:
            vol_regime = MarketRegime.LOW_VOLATILITY
            confidence = min(1.0, (vol_25 - current_vol) / (vol_25 - volatility.min()))
        else:
            vol_regime = MarketRegime.SIDEWAYS
            confidence = 0.5
        
        # Calculate trend component
        trend_window = 50
        if len(data) >= trend_window:
            price_start = data['close'].iloc[-trend_window]
            price_end = data['close'].iloc[-1]
            trend_strength = (price_end - price_start) / price_start
        else:
            trend_strength = 0.0
        
        # Determine overall regime
        if abs(trend_strength) > 0.2:  # Strong trend
            if trend_strength > 0:
                current_regime = MarketRegime.BULL_MARKET
            else:
                current_regime = MarketRegime.BEAR_MARKET
        else:
            current_regime = vol_regime
        
        # Estimate regime duration (simplified)
        regime_duration = timedelta(days=30)  # Default 30 days
        regime_start = data.index[-1] - regime_duration
        
        return RegimeDetectionResult(
            current_regime=current_regime,
            regime_confidence=confidence,
            regime_duration=regime_duration,
            regime_start_date=regime_start,
            volatility_level=current_vol,
            trend_strength=abs(trend_strength),
            momentum_score=trend_strength,
            indicators_used=['volatility', 'trend']
        )
    
    def _default_regime_result(self) -> RegimeDetectionResult:
        """Return default regime when insufficient data"""
        return RegimeDetectionResult(
            current_regime=MarketRegime.SIDEWAYS,
            regime_confidence=0.5,
            regime_duration=timedelta(days=30),
            regime_start_date=datetime.now() - timedelta(days=30),
            volatility_level=0.2,
            trend_strength=0.0,
            momentum_score=0.0,
            indicators_used=['default']
        )


class DataRequirementsEngine:
    """
    Main data requirements engine that manages data pipeline and quality
    
    Handles parameters 19-23:
    19. lookback_period - Manages historical data requirements
    20. data_frequency - Controls data update schedules
    21. data_quality_filter - Applies quality controls
    22. benchmark - Manages benchmark tracking
    23. market_regime_filter - Applies regime-aware filtering
    """
    
    def __init__(self, registry: TradingParameterRegistry = None):
        self.registry = registry or parameter_registry
        self.quality_filters = self._initialize_quality_filters()
        self.regime_detectors = self._initialize_regime_detectors()
        
        # Data tracking
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.quality_metrics: Dict[str, DataQualityMetrics] = {}
        self.benchmark_metrics: Dict[str, BenchmarkMetrics] = {}
        self.regime_history: List[RegimeDetectionResult] = []
        
    def _initialize_quality_filters(self) -> Dict[str, DataQualityFilter]:
        """Initialize data quality filters"""
        return {
            'basic': BasicDataQualityFilter(),
            'advanced': BasicDataQualityFilter()  # Would be more sophisticated in practice
        }
    
    def _initialize_regime_detectors(self) -> Dict[str, RegimeDetector]:
        """Initialize regime detectors"""
        return {
            'volatility': VolatilityRegimeDetector(),
            'trend': VolatilityRegimeDetector()  # Would be separate implementation
        }
    
    def validate_data_requirements(
        self,
        symbol: str,
        data: pd.DataFrame,
        strategy_requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate that data meets strategy requirements"""
        
        params = self._get_current_parameters()
        requirements = strategy_requirements or {}
        
        # Check lookback period requirement
        required_periods = params.get('lookback_period', 252)
        actual_periods = len(data)
        sufficient_history = actual_periods >= required_periods
        
        # Check data frequency requirement
        required_frequency = params.get('data_frequency', 'daily')
        actual_frequency = self._infer_data_frequency(data)
        frequency_match = self._check_frequency_compatibility(required_frequency, actual_frequency)
        
        # Check data completeness
        completeness_threshold = requirements.get('min_completeness', 0.95)
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        sufficient_completeness = (1 - missing_ratio) >= completeness_threshold
        
        # Check data recency
        max_lag_days = requirements.get('max_lag_days', 1)
        if not data.empty:
            latest_date = data.index.max()
            data_lag_days = (datetime.now() - latest_date).days
            data_recent = data_lag_days <= max_lag_days
        else:
            data_recent = False
            data_lag_days = float('inf')
        
        return {
            'meets_requirements': (
                sufficient_history and frequency_match and 
                sufficient_completeness and data_recent
            ),
            'history_check': {
                'required_periods': required_periods,
                'actual_periods': actual_periods,
                'sufficient': sufficient_history
            },
            'frequency_check': {
                'required': required_frequency,
                'actual': actual_frequency,
                'compatible': frequency_match
            },
            'completeness_check': {
                'threshold': completeness_threshold,
                'actual': 1 - missing_ratio,
                'sufficient': sufficient_completeness
            },
            'recency_check': {
                'max_lag_days': max_lag_days,
                'actual_lag_days': data_lag_days,
                'recent': data_recent
            }
        }
    
    def apply_data_quality_filter(
        self,
        symbol: str,
        data: pd.DataFrame,
        filter_type: str = "basic"
    ) -> Tuple[pd.DataFrame, DataQualityMetrics]:
        """Apply data quality filtering"""
        
        params = self._get_current_parameters()
        apply_filtering = params.get('data_quality_filter', True)
        
        if not apply_filtering:
            # Create basic metrics without filtering
            metrics = DataQualityMetrics(
                total_records=len(data),
                missing_records=0,
                completeness_ratio=1.0,
                outlier_count=0,
                outlier_ratio=0.0,
                price_consistency_score=100.0,
                volume_consistency_score=100.0,
                latest_timestamp=data.index.max() if not data.empty else datetime.min,
                data_lag_minutes=0.0,
                update_frequency_actual="unknown",
                duplicate_records=0,
                data_gaps=[],
                gap_duration_total=timedelta(),
                overall_quality_score=100.0
            )
            return data, metrics
        
        # Apply selected filter
        quality_filter = self.quality_filters.get(filter_type, self.quality_filters['basic'])
        filtered_data, metrics = quality_filter.apply_filter(data)
        
        # Store metrics
        self.quality_metrics[symbol] = metrics
        
        # Cache filtered data
        self.data_cache[symbol] = filtered_data
        
        return filtered_data, metrics
    
    def calculate_benchmark_metrics(
        self,
        strategy_returns: pd.Series,
        benchmark_symbol: Optional[str] = None,
        benchmark_returns: Optional[pd.Series] = None
    ) -> BenchmarkMetrics:
        """Calculate comprehensive benchmark comparison metrics"""
        
        params = self._get_current_parameters()
        benchmark_symbol = benchmark_symbol or params.get('benchmark', 'SPY')
        
        if benchmark_returns is None:
            # In practice, would fetch benchmark data
            # For now, generate synthetic benchmark returns
            benchmark_returns = pd.Series(
                np.random.normal(0.0008, 0.02, len(strategy_returns)),
                index=strategy_returns.index
            )
        
        # Align returns
        aligned_strategy, aligned_benchmark = strategy_returns.align(benchmark_returns, join='inner')
        
        if len(aligned_strategy) == 0:
            raise ValueError("No overlapping data between strategy and benchmark")
        
        # Calculate return metrics
        strategy_total_return = (1 + aligned_strategy).prod() - 1
        benchmark_total_return = (1 + aligned_benchmark).prod() - 1
        excess_return = strategy_total_return - benchmark_total_return
        
        # Risk-adjusted metrics
        strategy_vol = aligned_strategy.std() * np.sqrt(252)
        benchmark_vol = aligned_benchmark.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (strategy_total_return - risk_free_rate) / strategy_vol if strategy_vol > 0 else 0
        
        # Information ratio
        active_returns = aligned_strategy - aligned_benchmark
        tracking_error = active_returns.std() * np.sqrt(252)
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
        
        # Beta and correlation
        correlation = aligned_strategy.corr(aligned_benchmark)
        beta = aligned_strategy.cov(aligned_benchmark) / aligned_benchmark.var() if aligned_benchmark.var() > 0 else 1.0
        r_squared = correlation ** 2
        
        # Treynor ratio
        treynor_ratio = (strategy_total_return - risk_free_rate) / beta if beta != 0 else 0
        
        # Jensen's alpha
        jensen_alpha = strategy_total_return - (risk_free_rate + beta * (benchmark_total_return - risk_free_rate))
        
        # Performance attribution (simplified)
        selection_effect = excess_return * 0.7  # Simplified attribution
        timing_effect = excess_return * 0.2
        interaction_effect = excess_return * 0.1
        
        metrics = BenchmarkMetrics(
            benchmark_symbol=benchmark_symbol,
            benchmark_return=benchmark_total_return,
            strategy_return=strategy_total_return,
            excess_return=excess_return,
            sharpe_ratio=sharpe_ratio,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            jensen_alpha=jensen_alpha,
            tracking_error=tracking_error,
            correlation=correlation,
            beta=beta,
            r_squared=r_squared,
            selection_effect=selection_effect,
            timing_effect=timing_effect,
            interaction_effect=interaction_effect,
            calculation_period=(aligned_strategy.index.min(), aligned_strategy.index.max())
        )
        
        # Store metrics
        self.benchmark_metrics[benchmark_symbol] = metrics
        
        return metrics
    
    def detect_market_regime(
        self,
        data: pd.DataFrame,
        detector_type: str = "volatility"
    ) -> RegimeDetectionResult:
        """Detect current market regime"""
        
        params = self._get_current_parameters()
        apply_regime_filter = params.get('market_regime_filter', False)
        
        if not apply_regime_filter:
            # Return default regime if filtering disabled
            return RegimeDetectionResult(
                current_regime=MarketRegime.SIDEWAYS,
                regime_confidence=0.5,
                regime_duration=timedelta(days=30),
                regime_start_date=datetime.now() - timedelta(days=30),
                volatility_level=0.2,
                trend_strength=0.0,
                momentum_score=0.0,
                indicators_used=['disabled']
            )
        
        # Apply regime detection
        detector = self.regime_detectors.get(detector_type, self.regime_detectors['volatility'])
        regime_result = detector.detect_regime(data)
        
        # Store in history
        self.regime_history.append(regime_result)
        if len(self.regime_history) > 100:  # Keep last 100 results
            self.regime_history = self.regime_history[-100:]
        
        return regime_result
    
    def filter_data_by_regime(
        self,
        data: pd.DataFrame,
        target_regimes: List[MarketRegime]
    ) -> pd.DataFrame:
        """Filter data to include only specified market regimes"""
        
        if not self.regime_history:
            logger.warning("No regime history available for filtering")
            return data
        
        # Simple filtering - in practice would map dates to regimes
        # For now, return all data if any target regime is current
        current_regime = self.regime_history[-1].current_regime if self.regime_history else MarketRegime.SIDEWAYS
        
        if current_regime in target_regimes:
            return data
        else:
            # Return empty DataFrame if current regime not in targets
            return pd.DataFrame()
    
    def get_data_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive data pipeline status"""
        
        params = self._get_current_parameters()
        
        # Calculate average quality scores
        avg_quality_score = 0
        if self.quality_metrics:
            avg_quality_score = np.mean([m.overall_quality_score for m in self.quality_metrics.values()])
        
        # Count regime changes
        regime_changes = 0
        if len(self.regime_history) > 1:
            for i in range(1, len(self.regime_history)):
                if self.regime_history[i].current_regime != self.regime_history[i-1].current_regime:
                    regime_changes += 1
        
        return {
            'parameters': params,
            'cached_symbols': list(self.data_cache.keys()),
            'average_quality_score': avg_quality_score,
            'quality_filters_active': params.get('data_quality_filter', True),
            'regime_detection_active': params.get('market_regime_filter', False),
            'current_regime': self.regime_history[-1].current_regime.value if self.regime_history else 'unknown',
            'regime_changes_detected': regime_changes,
            'benchmark_symbol': params.get('benchmark', 'SPY'),
            'data_frequency': params.get('data_frequency', 'daily'),
            'lookback_period': params.get('lookback_period', 252),
            'last_updated': datetime.now()
        }
    
    def optimize_data_requirements(
        self,
        strategy_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize data requirements based on strategy characteristics"""
        
        # Extract strategy characteristics
        strategy_horizon = strategy_characteristics.get('holding_period_days', 30)
        strategy_frequency = strategy_characteristics.get('rebalance_frequency', 'monthly')
        alpha_decay = strategy_characteristics.get('alpha_decay_days', 5)
        
        # Optimize lookback period
        # Rule: 10x holding period, minimum 60 days, maximum 1000 days
        optimal_lookback = max(60, min(1000, strategy_horizon * 10))
        
        # Optimize data frequency
        if strategy_horizon <= 1:  # Intraday strategy
            optimal_frequency = 'minute'
        elif strategy_horizon <= 5:  # Short-term strategy
            optimal_frequency = 'hourly'
        else:  # Medium to long-term strategy
            optimal_frequency = 'daily'
        
        # Quality requirements based on strategy sensitivity
        quality_requirement = 'advanced' if alpha_decay <= 2 else 'basic'
        
        # Regime filtering recommendation
        regime_filtering = strategy_horizon > 30  # Only for longer-term strategies
        
        return {
            'optimal_lookback_period': optimal_lookback,
            'optimal_data_frequency': optimal_frequency,
            'quality_filter_level': quality_requirement,
            'enable_regime_filtering': regime_filtering,
            'reasoning': {
                'lookback': f"Based on {strategy_horizon}-day holding period",
                'frequency': f"Aligned with strategy rebalancing: {strategy_frequency}",
                'quality': f"Alpha decay of {alpha_decay} days requires {quality_requirement} filtering",
                'regime': f"Regime filtering {'recommended' if regime_filtering else 'not needed'} for this horizon"
            }
        }
    
    def _get_current_parameters(self) -> Dict[str, Any]:
        """Get current data requirements parameters"""
        param_names = [
            'lookback_period', 'data_frequency', 'data_quality_filter',
            'benchmark', 'market_regime_filter'
        ]
        params = {}
        
        for name in param_names:
            param_def = self.registry.get_parameter(name)
            if param_def:
                params[name] = param_def.current_value or param_def.default_value
        
        return params
    
    def _infer_data_frequency(self, data: pd.DataFrame) -> str:
        """Infer data frequency from DataFrame"""
        if data.empty or len(data) < 2:
            return 'unknown'
        
        # Calculate typical time difference
        time_diffs = data.index.to_series().diff().dropna()
        median_diff = time_diffs.median()
        
        if median_diff <= pd.Timedelta(minutes=1):
            return 'minute'
        elif median_diff <= pd.Timedelta(hours=1):
            return 'hourly'
        elif median_diff <= pd.Timedelta(days=1):
            return 'daily'
        elif median_diff <= pd.Timedelta(weeks=1):
            return 'weekly'
        else:
            return 'monthly'
    
    def _check_frequency_compatibility(self, required: str, actual: str) -> bool:
        """Check if actual frequency is compatible with required"""
        
        frequency_hierarchy = {
            'real_time': 0,
            'minute': 1,
            'hourly': 2, 
            'daily': 3,
            'weekly': 4,
            'monthly': 5
        }
        
        req_level = frequency_hierarchy.get(required, 3)
        actual_level = frequency_hierarchy.get(actual, 3)
        
        # Actual frequency should be same or higher resolution than required
        return actual_level <= req_level


# Factory function for easy instantiation
def create_data_requirements_engine(registry: TradingParameterRegistry = None) -> DataRequirementsEngine:
    """Create a data requirements engine instance"""
    return DataRequirementsEngine(registry)