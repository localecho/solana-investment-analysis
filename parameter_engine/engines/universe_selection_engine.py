"""
Universe Selection Engine - Implementation of Parameters 24-29

This engine handles the business logic for universe selection parameters:
24. universe_size - Maximum number of stocks in universe
25. sector_filter - Allowed sectors for trading  
26. market_cap_min - Minimum market capitalization
27. liquidity_filter - Minimum liquidity requirements
28. fundamental_criteria - Fundamental screening criteria
29. technical_criteria - Technical screening criteria

The engine provides stock screening, filtering, and universe construction
capabilities for systematic trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod

from core.parameter_registry import TradingParameterRegistry, parameter_registry

logger = logging.getLogger(__name__)


class SectorCategory(Enum):
    """Standard sector classifications"""
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCIALS = "financials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    INDUSTRIALS = "industrials"
    ENERGY = "energy"
    UTILITIES = "utilities"
    MATERIALS = "materials"
    REAL_ESTATE = "real_estate"
    TELECOMMUNICATIONS = "telecommunications"


class FundamentalCriteria(Enum):
    """Fundamental screening criteria"""
    PE_RATIO = "pe_ratio"
    PB_RATIO = "pb_ratio"
    ROE = "roe"
    ROA = "roa"
    DEBT_TO_EQUITY = "debt_to_equity"
    REVENUE_GROWTH = "revenue_growth"
    EARNINGS_GROWTH = "earnings_growth"
    DIVIDEND_YIELD = "dividend_yield"
    CURRENT_RATIO = "current_ratio"
    PROFIT_MARGIN = "profit_margin"


class TechnicalCriteria(Enum):
    """Technical screening criteria"""
    RSI = "rsi"
    MACD = "macd"
    MOVING_AVERAGE = "moving_average"
    BOLLINGER_BANDS = "bollinger_bands"
    VOLUME_TREND = "volume_trend"
    PRICE_MOMENTUM = "price_momentum"
    VOLATILITY = "volatility"
    ATR = "atr"
    STOCHASTIC = "stochastic"
    WILLIAMS_R = "williams_r"


@dataclass
class StockData:
    """Comprehensive stock data for screening"""
    
    symbol: str
    company_name: str
    sector: str
    market_cap: float
    
    # Price and volume data
    current_price: float
    avg_daily_volume: float
    price_history: Optional[pd.Series] = None
    volume_history: Optional[pd.Series] = None
    
    # Fundamental data
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    debt_to_equity: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    dividend_yield: Optional[float] = None
    current_ratio: Optional[float] = None
    profit_margin: Optional[float] = None
    
    # Technical indicators
    rsi: Optional[float] = None
    macd: Optional[float] = None
    ma_20: Optional[float] = None
    ma_50: Optional[float] = None
    ma_200: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    volatility: Optional[float] = None
    atr: Optional[float] = None
    
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ScreeningResult:
    """Result of stock screening process"""
    
    symbol: str
    included: bool
    score: float  # Overall screening score
    
    # Individual test results
    sector_pass: bool
    market_cap_pass: bool
    liquidity_pass: bool
    fundamental_pass: bool
    technical_pass: bool
    
    # Detailed scores
    fundamental_score: float
    technical_score: float
    
    # Reasons for inclusion/exclusion
    inclusion_reasons: List[str] = field(default_factory=list)
    exclusion_reasons: List[str] = field(default_factory=list)
    
    screening_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class UniverseComposition:
    """Final universe composition and statistics"""
    
    selected_symbols: List[str]
    total_candidates: int
    universe_size: int
    
    # Composition breakdown
    sector_breakdown: Dict[str, int]
    market_cap_distribution: Dict[str, int]  # small, mid, large cap
    
    # Quality metrics
    avg_liquidity: float
    avg_market_cap: float
    avg_fundamental_score: float
    avg_technical_score: float
    
    # Screening statistics
    pass_rates: Dict[str, float]  # Pass rate for each filter
    
    construction_timestamp: datetime = field(default_factory=datetime.now)


class StockScreener(ABC):
    """Abstract base class for stock screening"""
    
    @abstractmethod
    def screen_stocks(
        self, 
        stocks: List[StockData], 
        criteria: Dict[str, Any]
    ) -> List[ScreeningResult]:
        """Screen stocks based on criteria"""
        pass


class BasicStockScreener(StockScreener):
    """Basic stock screening implementation"""
    
    def screen_stocks(
        self, 
        stocks: List[StockData], 
        criteria: Dict[str, Any]
    ) -> List[ScreeningResult]:
        """Screen stocks using basic criteria"""
        
        results = []
        
        for stock in stocks:
            result = self._screen_individual_stock(stock, criteria)
            results.append(result)
        
        return results
    
    def _screen_individual_stock(
        self, 
        stock: StockData, 
        criteria: Dict[str, Any]
    ) -> ScreeningResult:
        """Screen individual stock against all criteria"""
        
        # Test each criteria
        sector_pass = self._test_sector_filter(stock, criteria)
        market_cap_pass = self._test_market_cap(stock, criteria)
        liquidity_pass = self._test_liquidity(stock, criteria)
        fundamental_pass, fundamental_score = self._test_fundamental_criteria(stock, criteria)
        technical_pass, technical_score = self._test_technical_criteria(stock, criteria)
        
        # Overall inclusion decision
        included = all([
            sector_pass, market_cap_pass, liquidity_pass, 
            fundamental_pass, technical_pass
        ])
        
        # Calculate overall score
        overall_score = (fundamental_score + technical_score) / 2
        
        # Collect reasons
        inclusion_reasons = []
        exclusion_reasons = []
        
        if sector_pass:
            inclusion_reasons.append(f"Sector '{stock.sector}' allowed")
        else:
            exclusion_reasons.append(f"Sector '{stock.sector}' filtered out")
        
        if market_cap_pass:
            inclusion_reasons.append(f"Market cap ${stock.market_cap/1e9:.1f}B meets minimum")
        else:
            exclusion_reasons.append(f"Market cap ${stock.market_cap/1e9:.1f}B below minimum")
        
        if liquidity_pass:
            inclusion_reasons.append(f"Daily volume {stock.avg_daily_volume:,.0f} sufficient")
        else:
            exclusion_reasons.append(f"Daily volume {stock.avg_daily_volume:,.0f} insufficient")
        
        if fundamental_pass:
            inclusion_reasons.append(f"Fundamental score {fundamental_score:.1f} passed")
        else:
            exclusion_reasons.append(f"Fundamental score {fundamental_score:.1f} failed")
        
        if technical_pass:
            inclusion_reasons.append(f"Technical score {technical_score:.1f} passed")
        else:
            exclusion_reasons.append(f"Technical score {technical_score:.1f} failed")
        
        return ScreeningResult(
            symbol=stock.symbol,
            included=included,
            score=overall_score,
            sector_pass=sector_pass,
            market_cap_pass=market_cap_pass,
            liquidity_pass=liquidity_pass,
            fundamental_pass=fundamental_pass,
            technical_pass=technical_pass,
            fundamental_score=fundamental_score,
            technical_score=technical_score,
            inclusion_reasons=inclusion_reasons,
            exclusion_reasons=exclusion_reasons
        )
    
    def _test_sector_filter(self, stock: StockData, criteria: Dict[str, Any]) -> bool:
        """Test sector filter criteria"""
        allowed_sectors = criteria.get('sector_filter', [])
        
        if not allowed_sectors or 'all' in allowed_sectors:
            return True
        
        return stock.sector.lower() in [s.lower() for s in allowed_sectors]
    
    def _test_market_cap(self, stock: StockData, criteria: Dict[str, Any]) -> bool:
        """Test market cap criteria"""
        min_market_cap = criteria.get('market_cap_min', 0)
        return stock.market_cap >= min_market_cap
    
    def _test_liquidity(self, stock: StockData, criteria: Dict[str, Any]) -> bool:
        """Test liquidity criteria"""
        min_volume = criteria.get('liquidity_filter', 0)
        return stock.avg_daily_volume >= min_volume
    
    def _test_fundamental_criteria(
        self, 
        stock: StockData, 
        criteria: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """Test fundamental criteria and return pass/fail + score"""
        
        fundamental_criteria = criteria.get('fundamental_criteria', {})
        
        if not fundamental_criteria:
            return True, 50.0  # Default neutral score
        
        scores = []
        
        # PE Ratio test
        if 'pe_ratio' in fundamental_criteria and stock.pe_ratio is not None:
            min_pe, max_pe = fundamental_criteria['pe_ratio']
            if min_pe <= stock.pe_ratio <= max_pe:
                scores.append(100)
            else:
                scores.append(0)
        
        # PB Ratio test
        if 'pb_ratio' in fundamental_criteria and stock.pb_ratio is not None:
            min_pb, max_pb = fundamental_criteria['pb_ratio']
            if min_pb <= stock.pb_ratio <= max_pb:
                scores.append(100)
            else:
                scores.append(0)
        
        # ROE test
        if 'roe' in fundamental_criteria and stock.roe is not None:
            min_roe = fundamental_criteria['roe']
            if stock.roe >= min_roe:
                scores.append(100)
            else:
                scores.append(0)
        
        # Revenue Growth test
        if 'revenue_growth' in fundamental_criteria and stock.revenue_growth is not None:
            min_growth = fundamental_criteria['revenue_growth']
            if stock.revenue_growth >= min_growth:
                scores.append(100)
            else:
                scores.append(0)
        
        # Calculate average score
        if scores:
            avg_score = np.mean(scores)
            passing = avg_score >= 60  # 60% threshold
        else:
            avg_score = 50.0
            passing = True  # No criteria means pass
        
        return passing, avg_score
    
    def _test_technical_criteria(
        self, 
        stock: StockData, 
        criteria: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """Test technical criteria and return pass/fail + score"""
        
        technical_criteria = criteria.get('technical_criteria', {})
        
        if not technical_criteria:
            return True, 50.0  # Default neutral score
        
        scores = []
        
        # RSI test
        if 'rsi' in technical_criteria and stock.rsi is not None:
            min_rsi, max_rsi = technical_criteria['rsi']
            if min_rsi <= stock.rsi <= max_rsi:
                scores.append(100)
            else:
                scores.append(0)
        
        # Moving Average test
        if 'moving_average' in technical_criteria:
            ma_criteria = technical_criteria['moving_average']
            above_ma20 = ma_criteria.get('above_ma20', False)
            above_ma50 = ma_criteria.get('above_ma50', False)
            
            ma_score = 0
            if above_ma20 and stock.ma_20 and stock.current_price > stock.ma_20:
                ma_score += 50
            if above_ma50 and stock.ma_50 and stock.current_price > stock.ma_50:
                ma_score += 50
            
            scores.append(ma_score)
        
        # Volume trend test
        if 'volume_trend' in technical_criteria:
            # Simplified: assume increasing volume is good
            scores.append(75)  # Default good score
        
        # Price momentum test
        if 'price_momentum' in technical_criteria:
            momentum_days = technical_criteria['price_momentum'].get('days', 20)
            min_momentum = technical_criteria['price_momentum'].get('min_return', 0.05)
            
            # Would calculate actual momentum from price history
            # For now, use simplified logic
            scores.append(60)  # Default moderate score
        
        # Calculate average score
        if scores:
            avg_score = np.mean(scores)
            passing = avg_score >= 60  # 60% threshold
        else:
            avg_score = 50.0
            passing = True  # No criteria means pass
        
        return passing, avg_score


class UniverseConstructor:
    """Constructs final universe from screening results"""
    
    def construct_universe(
        self, 
        screening_results: List[ScreeningResult],
        universe_size: int,
        diversification_constraints: Optional[Dict[str, Any]] = None
    ) -> UniverseComposition:
        """Construct final universe from screening results"""
        
        # Filter to passing stocks
        passed_stocks = [r for r in screening_results if r.included]
        
        # Sort by overall score
        passed_stocks.sort(key=lambda x: x.score, reverse=True)
        
        # Apply diversification constraints if specified
        if diversification_constraints:
            passed_stocks = self._apply_diversification_constraints(
                passed_stocks, diversification_constraints
            )
        
        # Select top N stocks
        final_selection = passed_stocks[:universe_size]
        selected_symbols = [r.symbol for r in final_selection]
        
        # Calculate composition statistics
        composition = self._calculate_composition_stats(
            final_selection, screening_results, universe_size
        )
        
        return composition
    
    def _apply_diversification_constraints(
        self,
        results: List[ScreeningResult],
        constraints: Dict[str, Any]
    ) -> List[ScreeningResult]:
        """Apply diversification constraints to selection"""
        
        # This is a simplified implementation
        # In practice, would implement sophisticated portfolio optimization
        
        max_per_sector = constraints.get('max_per_sector', float('inf'))
        
        if max_per_sector == float('inf'):
            return results
        
        # Group by sector and limit
        sector_counts = {}
        filtered_results = []
        
        for result in results:
            # Would need to get sector info from stock data
            # For now, use simplified logic
            sector = "unknown"  # Would get from stock data
            
            if sector_counts.get(sector, 0) < max_per_sector:
                filtered_results.append(result)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        return filtered_results
    
    def _calculate_composition_stats(
        self,
        final_selection: List[ScreeningResult],
        all_results: List[ScreeningResult],
        universe_size: int
    ) -> UniverseComposition:
        """Calculate universe composition statistics"""
        
        selected_symbols = [r.symbol for r in final_selection]
        
        # Calculate pass rates
        total_candidates = len(all_results)
        pass_rates = {
            'sector': sum(1 for r in all_results if r.sector_pass) / total_candidates,
            'market_cap': sum(1 for r in all_results if r.market_cap_pass) / total_candidates,
            'liquidity': sum(1 for r in all_results if r.liquidity_pass) / total_candidates,
            'fundamental': sum(1 for r in all_results if r.fundamental_pass) / total_candidates,
            'technical': sum(1 for r in all_results if r.technical_pass) / total_candidates,
            'overall': len(final_selection) / total_candidates if total_candidates > 0 else 0
        }
        
        # Calculate average scores
        if final_selection:
            avg_fundamental_score = np.mean([r.fundamental_score for r in final_selection])
            avg_technical_score = np.mean([r.technical_score for r in final_selection])
        else:
            avg_fundamental_score = 0
            avg_technical_score = 0
        
        return UniverseComposition(
            selected_symbols=selected_symbols,
            total_candidates=total_candidates,
            universe_size=len(final_selection),
            sector_breakdown={},  # Would populate with actual sector data
            market_cap_distribution={},  # Would populate with actual market cap data
            avg_liquidity=0,  # Would calculate from stock data
            avg_market_cap=0,  # Would calculate from stock data
            avg_fundamental_score=avg_fundamental_score,
            avg_technical_score=avg_technical_score,
            pass_rates=pass_rates
        )


class UniverseSelectionEngine:
    """
    Main universe selection engine that manages stock screening and universe construction
    
    Handles parameters 24-29:
    24. universe_size - Controls final universe size
    25. sector_filter - Manages sector inclusion/exclusion
    26. market_cap_min - Enforces market cap requirements
    27. liquidity_filter - Ensures liquidity standards
    28. fundamental_criteria - Applies fundamental screening
    29. technical_criteria - Applies technical screening
    """
    
    def __init__(self, registry: TradingParameterRegistry = None):
        self.registry = registry or parameter_registry
        self.screener = BasicStockScreener()
        self.constructor = UniverseConstructor()
        
        # Universe tracking
        self.current_universe: Optional[UniverseComposition] = None
        self.screening_history: List[Dict[str, Any]] = []
        
    def screen_and_select_universe(
        self,
        candidate_stocks: List[StockData],
        additional_constraints: Optional[Dict[str, Any]] = None
    ) -> UniverseComposition:
        """
        Screen candidate stocks and construct final universe
        
        Args:
            candidate_stocks: List of candidate stocks to screen
            additional_constraints: Additional constraints to apply
            
        Returns:
            UniverseComposition with final selected universe
        """
        
        # Get current parameters
        params = self._get_current_parameters()
        
        # Apply additional constraints
        if additional_constraints:
            params.update(additional_constraints)
        
        # Build screening criteria
        screening_criteria = self._build_screening_criteria(params)
        
        # Screen stocks
        screening_results = self.screener.screen_stocks(
            candidate_stocks, screening_criteria
        )
        
        # Construct universe
        universe_size = params.get('universe_size', 100)
        universe_composition = self.constructor.construct_universe(
            screening_results, universe_size
        )
        
        # Store results
        self.current_universe = universe_composition
        self._record_screening_event(screening_results, universe_composition, params)
        
        return universe_composition
    
    def get_universe_statistics(self) -> Dict[str, Any]:
        """Get comprehensive universe statistics"""
        
        if not self.current_universe:
            return {'error': 'No universe constructed'}
        
        params = self._get_current_parameters()
        
        return {
            'universe_size': self.current_universe.universe_size,
            'total_candidates': self.current_universe.total_candidates,
            'selection_rate': (
                self.current_universe.universe_size / 
                self.current_universe.total_candidates * 100
                if self.current_universe.total_candidates > 0 else 0
            ),
            'avg_fundamental_score': self.current_universe.avg_fundamental_score,
            'avg_technical_score': self.current_universe.avg_technical_score,
            'pass_rates': self.current_universe.pass_rates,
            'sector_breakdown': self.current_universe.sector_breakdown,
            'parameters_used': params,
            'construction_time': self.current_universe.construction_timestamp,
            'screening_events': len(self.screening_history)
        }
    
    def update_universe(
        self, 
        new_candidate_stocks: List[StockData]
    ) -> UniverseComposition:
        """Update universe with new candidate data"""
        
        logger.info("Updating universe with new candidate data")
        return self.screen_and_select_universe(new_candidate_stocks)
    
    def analyze_screening_efficiency(self) -> Dict[str, Any]:
        """Analyze the efficiency of current screening criteria"""
        
        if not self.screening_history:
            return {'error': 'No screening history available'}
        
        recent_events = self.screening_history[-10:]  # Last 10 events
        
        # Calculate efficiency metrics
        avg_candidates = np.mean([e['total_candidates'] for e in recent_events])
        avg_selected = np.mean([e['universe_size'] for e in recent_events])
        avg_selection_rate = avg_selected / avg_candidates * 100 if avg_candidates > 0 else 0
        
        # Calculate filter effectiveness
        filter_effectiveness = {}
        if self.current_universe and self.current_universe.pass_rates:
            for filter_name, pass_rate in self.current_universe.pass_rates.items():
                # Lower pass rate means more selective (more effective)
                effectiveness = (1 - pass_rate) * 100
                filter_effectiveness[filter_name] = effectiveness
        
        return {
            'avg_candidates_processed': avg_candidates,
            'avg_universe_size': avg_selected,
            'avg_selection_rate_percent': avg_selection_rate,
            'filter_effectiveness': filter_effectiveness,
            'total_screening_events': len(self.screening_history),
            'efficiency_score': self._calculate_efficiency_score()
        }
    
    def optimize_screening_parameters(
        self,
        target_universe_size: int,
        target_quality_score: float = 70.0
    ) -> Dict[str, Any]:
        """Suggest optimized screening parameters"""
        
        current_params = self._get_current_parameters()
        
        if not self.current_universe:
            return {
                'status': 'no_data',
                'message': 'No universe data available for optimization'
            }
        
        suggestions = {}
        
        # Analyze current vs target universe size
        current_size = self.current_universe.universe_size
        size_ratio = target_universe_size / current_size if current_size > 0 else 1
        
        if size_ratio > 1.2:  # Need more stocks
            suggestions['sector_filter'] = 'Consider expanding sector filter'
            suggestions['market_cap_min'] = 'Consider lowering minimum market cap'
            suggestions['liquidity_filter'] = 'Consider relaxing liquidity requirements'
        elif size_ratio < 0.8:  # Need fewer stocks
            suggestions['fundamental_criteria'] = 'Consider tightening fundamental criteria'
            suggestions['technical_criteria'] = 'Consider stricter technical requirements'
        
        # Analyze quality vs target
        current_quality = (
            self.current_universe.avg_fundamental_score + 
            self.current_universe.avg_technical_score
        ) / 2
        
        if current_quality < target_quality_score:
            suggestions['quality_improvement'] = 'Consider raising screening thresholds'
        
        return {
            'current_universe_size': current_size,
            'target_universe_size': target_universe_size,
            'current_quality_score': current_quality,
            'target_quality_score': target_quality_score,
            'optimization_suggestions': suggestions,
            'size_adjustment_needed': abs(size_ratio - 1.0) > 0.1
        }
    
    def _get_current_parameters(self) -> Dict[str, Any]:
        """Get current universe selection parameters"""
        param_names = [
            'universe_size', 'sector_filter', 'market_cap_min',
            'liquidity_filter', 'fundamental_criteria', 'technical_criteria'
        ]
        params = {}
        
        for name in param_names:
            param_def = self.registry.get_parameter(name)
            if param_def:
                params[name] = param_def.current_value or param_def.default_value
        
        return params
    
    def _build_screening_criteria(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive screening criteria from parameters"""
        
        criteria = {}
        
        # Sector filter
        sector_filter = params.get('sector_filter', [])
        if sector_filter and sector_filter != ['all']:
            criteria['sector_filter'] = sector_filter
        
        # Market cap minimum
        market_cap_min = params.get('market_cap_min', 0)
        if market_cap_min > 0:
            criteria['market_cap_min'] = market_cap_min
        
        # Liquidity filter
        liquidity_filter = params.get('liquidity_filter', 0)
        if liquidity_filter > 0:
            criteria['liquidity_filter'] = liquidity_filter
        
        # Fundamental criteria
        fundamental_criteria = params.get('fundamental_criteria', {})
        if fundamental_criteria:
            criteria['fundamental_criteria'] = fundamental_criteria
        
        # Technical criteria
        technical_criteria = params.get('technical_criteria', {})
        if technical_criteria:
            criteria['technical_criteria'] = technical_criteria
        
        return criteria
    
    def _record_screening_event(
        self,
        screening_results: List[ScreeningResult],
        universe_composition: UniverseComposition,
        parameters_used: Dict[str, Any]
    ):
        """Record screening event for history tracking"""
        
        event = {
            'timestamp': datetime.now(),
            'total_candidates': len(screening_results),
            'universe_size': universe_composition.universe_size,
            'avg_fundamental_score': universe_composition.avg_fundamental_score,
            'avg_technical_score': universe_composition.avg_technical_score,
            'pass_rates': universe_composition.pass_rates,
            'parameters_used': parameters_used.copy()
        }
        
        self.screening_history.append(event)
        
        # Keep last 100 events
        if len(self.screening_history) > 100:
            self.screening_history = self.screening_history[-100:]
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate overall screening efficiency score"""
        
        if not self.current_universe:
            return 0.0
        
        # Components of efficiency score
        selection_efficiency = min(100, (
            self.current_universe.universe_size / 
            max(1, self.current_universe.total_candidates) * 100
        ))
        
        quality_score = (
            self.current_universe.avg_fundamental_score + 
            self.current_universe.avg_technical_score
        ) / 2
        
        # Balanced efficiency score
        efficiency_score = (selection_efficiency * 0.3 + quality_score * 0.7)
        
        return min(100, max(0, efficiency_score))


# Factory function for easy instantiation
def create_universe_selection_engine(registry: TradingParameterRegistry = None) -> UniverseSelectionEngine:
    """Create a universe selection engine instance"""
    return UniverseSelectionEngine(registry)