"""
Alpha Generation Engine - Implementation of Parameters 30-34

This engine handles the business logic for alpha generation parameters:
30. signal_strength - Signal confidence threshold for trading
31. factor_models - Factor model selection for alpha generation
32. momentum_lookback - Momentum calculation period in days
33. mean_reversion_window - Mean reversion analysis window
34. technical_indicators - Technical indicator selection and weighting

The engine provides comprehensive alpha generation capabilities including
factor models, momentum strategies, mean reversion, and technical analysis.
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


class FactorModel(Enum):
    """Factor model types for alpha generation"""
    FAMA_FRENCH_3 = "fama_french_3"
    FAMA_FRENCH_5 = "fama_french_5"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    VALUE = "value"
    GROWTH = "growth"
    SIZE = "size"
    VOLATILITY = "volatility"
    CUSTOM = "custom"


class TechnicalIndicator(Enum):
    """Technical indicators for alpha generation"""
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER_BANDS = "bollinger_bands"
    MOVING_AVERAGE = "moving_average"
    STOCHASTIC = "stochastic"
    WILLIAMS_R = "williams_r"
    ATR = "atr"
    ADX = "adx"
    CCI = "cci"
    MOMENTUM = "momentum"
    ROC = "roc"
    VOLUME_OSCILLATOR = "volume_oscillator"


class SignalType(Enum):
    """Signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


@dataclass
class AlphaSignal:
    """Individual alpha signal"""
    
    symbol: str
    signal_type: SignalType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    
    # Signal components
    factor_score: float
    momentum_score: float
    mean_reversion_score: float
    technical_score: float
    
    # Signal metadata
    signal_source: str  # Which model/indicator generated this
    lookback_period: int
    expected_return: Optional[float] = None
    risk_score: Optional[float] = None
    
    # Timing
    signal_timestamp: datetime = field(default_factory=datetime.now)
    expiry_timestamp: Optional[datetime] = None
    
    # Supporting data
    supporting_factors: List[str] = field(default_factory=list)
    warning_flags: List[str] = field(default_factory=list)


@dataclass
class PortfolioAlpha:
    """Portfolio-level alpha analysis"""
    
    portfolio_alpha: float  # Annualized alpha
    information_ratio: float
    hit_rate: float  # Percentage of correct signals
    avg_signal_strength: float
    
    # Signal distribution
    signal_counts: Dict[str, int]  # Count by signal type
    sector_alpha: Dict[str, float]  # Alpha by sector
    
    # Performance attribution
    factor_contribution: Dict[str, float]
    momentum_contribution: float
    mean_reversion_contribution: float
    technical_contribution: float
    
    # Risk metrics
    alpha_volatility: float
    max_drawdown: float
    sharpe_ratio: float
    
    analysis_period: Tuple[datetime, datetime]
    calculation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FactorExposure:
    """Factor model exposure analysis"""
    
    symbol: str
    factor_loadings: Dict[str, float]  # Factor name -> loading
    total_factor_score: float
    
    # Individual factor scores
    value_score: Optional[float] = None
    growth_score: Optional[float] = None
    quality_score: Optional[float] = None
    momentum_score: Optional[float] = None
    size_score: Optional[float] = None
    volatility_score: Optional[float] = None
    
    # Model fit statistics
    r_squared: float = 0.0
    residual_volatility: float = 0.0
    
    last_updated: datetime = field(default_factory=datetime.now)


class AlphaModel(ABC):
    """Abstract base class for alpha generation models"""
    
    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        market_data: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> AlphaSignal:
        """Generate alpha signal for a symbol"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name"""
        pass


class FactorAlphaModel(AlphaModel):
    """Factor-based alpha generation model"""
    
    def generate_signal(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        market_data: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> AlphaSignal:
        """Generate signal based on factor models"""
        
        factor_models = parameters.get('factor_models', ['fama_french_3'])
        signal_threshold = parameters.get('signal_strength', 0.6)
        
        # Calculate factor exposures
        factor_exposure = self._calculate_factor_exposure(
            symbol, price_data, market_data, factor_models
        )
        
        # Generate factor-based signal
        factor_score = factor_exposure.total_factor_score
        
        # Determine signal type and strength
        if factor_score >= signal_threshold:
            signal_type = SignalType.BUY
            strength = min(1.0, factor_score)
        elif factor_score <= -signal_threshold:
            signal_type = SignalType.SELL
            strength = min(1.0, abs(factor_score))
        else:
            signal_type = SignalType.HOLD
            strength = 0.5
        
        # Calculate confidence based on model fit
        confidence = min(1.0, factor_exposure.r_squared + 0.3)
        
        return AlphaSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            factor_score=factor_score,
            momentum_score=0.0,  # Will be filled by momentum model
            mean_reversion_score=0.0,  # Will be filled by mean reversion model
            technical_score=0.0,  # Will be filled by technical model
            signal_source="factor_model",
            lookback_period=parameters.get('lookback_period', 252),
            supporting_factors=[f.name for f in factor_models if isinstance(f, FactorModel)]
        )
    
    def _calculate_factor_exposure(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        market_data: Dict[str, Any],
        factor_models: List[str]
    ) -> FactorExposure:
        """Calculate factor model exposures"""
        
        if 'close' not in price_data.columns:
            raise ValueError("Price data must contain 'close' column")
        
        # Calculate returns
        returns = price_data['close'].pct_change().dropna()
        
        # Initialize factor scores
        factor_scores = {}
        
        # Value factor (simplified P/E based)
        if 'value' in factor_models or 'fama_french_3' in factor_models:
            pe_ratio = market_data.get('pe_ratio', 15.0)
            # Lower P/E = higher value score
            value_score = max(-1.0, min(1.0, (20 - pe_ratio) / 10))
            factor_scores['value'] = value_score
        
        # Growth factor (simplified earnings growth based)
        if 'growth' in factor_models or 'fama_french_5' in factor_models:
            earnings_growth = market_data.get('earnings_growth', 0.05)
            # Higher growth = higher score
            growth_score = max(-1.0, min(1.0, earnings_growth * 10))
            factor_scores['growth'] = growth_score
        
        # Quality factor (simplified ROE based)
        if 'quality' in factor_models or 'fama_french_5' in factor_models:
            roe = market_data.get('roe', 0.1)
            # Higher ROE = higher quality score
            quality_score = max(-1.0, min(1.0, (roe - 0.1) * 5))
            factor_scores['quality'] = quality_score
        
        # Momentum factor
        if 'momentum' in factor_models or 'fama_french_3' in factor_models:
            if len(returns) >= 60:  # Need at least 60 days
                momentum_return = (price_data['close'].iloc[-1] / price_data['close'].iloc[-60] - 1)
                momentum_score = max(-1.0, min(1.0, momentum_return * 5))
                factor_scores['momentum'] = momentum_score
        
        # Size factor
        if 'size' in factor_models or 'fama_french_3' in factor_models:
            market_cap = market_data.get('market_cap', 1e9)
            # Smaller cap = higher size factor score (for small cap premium)
            size_score = max(-1.0, min(1.0, (5e9 - market_cap) / 5e9))
            factor_scores['size'] = size_score
        
        # Volatility factor
        if 'volatility' in factor_models:
            if len(returns) >= 30:
                volatility = returns.std() * np.sqrt(252)
                # Lower volatility = higher score (low vol anomaly)
                vol_score = max(-1.0, min(1.0, (0.25 - volatility) * 2))
                factor_scores['volatility'] = vol_score
        
        # Calculate total factor score (equal weight for now)
        if factor_scores:
            total_score = np.mean(list(factor_scores.values()))
        else:
            total_score = 0.0
        
        # Calculate model fit (simplified)
        r_squared = min(0.8, len(factor_scores) * 0.15)  # More factors = better fit
        residual_vol = 0.02  # Simplified
        
        return FactorExposure(
            symbol=symbol,
            factor_loadings=factor_scores,
            total_factor_score=total_score,
            value_score=factor_scores.get('value'),
            growth_score=factor_scores.get('growth'),
            quality_score=factor_scores.get('quality'),
            momentum_score=factor_scores.get('momentum'),
            size_score=factor_scores.get('size'),
            volatility_score=factor_scores.get('volatility'),
            r_squared=r_squared,
            residual_volatility=residual_vol
        )
    
    def get_model_name(self) -> str:
        return "factor_alpha_model"


class MomentumAlphaModel(AlphaModel):
    """Momentum-based alpha generation model"""
    
    def generate_signal(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        market_data: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> AlphaSignal:
        """Generate momentum-based signal"""
        
        lookback_period = parameters.get('momentum_lookback', 60)
        signal_threshold = parameters.get('signal_strength', 0.6)
        
        if len(price_data) < lookback_period:
            # Not enough data
            return self._create_neutral_signal(symbol, "insufficient_data")
        
        # Calculate momentum scores
        momentum_scores = self._calculate_momentum_scores(price_data, lookback_period)
        
        # Combined momentum score
        total_momentum_score = np.mean(list(momentum_scores.values()))
        
        # Generate signal
        if total_momentum_score >= signal_threshold:
            signal_type = SignalType.BUY
            strength = min(1.0, total_momentum_score)
        elif total_momentum_score <= -signal_threshold:
            signal_type = SignalType.SELL
            strength = min(1.0, abs(total_momentum_score))
        else:
            signal_type = SignalType.HOLD
            strength = 0.5
        
        # Calculate confidence based on consistency of momentum signals
        momentum_values = list(momentum_scores.values())
        consistency = 1.0 - np.std(momentum_values) if len(momentum_values) > 1 else 0.5
        confidence = max(0.3, min(1.0, consistency))
        
        return AlphaSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            factor_score=0.0,
            momentum_score=total_momentum_score,
            mean_reversion_score=0.0,
            technical_score=0.0,
            signal_source="momentum_model",
            lookback_period=lookback_period,
            supporting_factors=list(momentum_scores.keys())
        )
    
    def _calculate_momentum_scores(
        self, 
        price_data: pd.DataFrame, 
        lookback_period: int
    ) -> Dict[str, float]:
        """Calculate various momentum scores"""
        
        prices = price_data['close']
        scores = {}
        
        # Price momentum (return over lookback period)
        if len(prices) >= lookback_period:
            price_momentum = (prices.iloc[-1] / prices.iloc[-lookback_period] - 1)
            scores['price_momentum'] = max(-1.0, min(1.0, price_momentum * 2))
        
        # Risk-adjusted momentum (return / volatility)
        if len(prices) >= lookback_period:
            returns = prices.pct_change().dropna()
            if len(returns) >= 20:
                period_return = price_momentum if 'price_momentum' in locals() else 0
                period_vol = returns.tail(lookback_period).std() * np.sqrt(252)
                if period_vol > 0:
                    risk_adj_momentum = period_return / period_vol
                    scores['risk_adjusted_momentum'] = max(-1.0, min(1.0, risk_adj_momentum))
        
        # Relative strength (vs market - simplified)
        # Would compare to market index in practice
        if len(prices) >= 30:
            recent_return = (prices.iloc[-1] / prices.iloc[-30] - 1)
            market_return = 0.01  # Simplified 1% monthly market return
            relative_strength = recent_return - market_return
            scores['relative_strength'] = max(-1.0, min(1.0, relative_strength * 10))
        
        return scores
    
    def _create_neutral_signal(self, symbol: str, reason: str) -> AlphaSignal:
        """Create neutral signal with warning"""
        return AlphaSignal(
            symbol=symbol,
            signal_type=SignalType.HOLD,
            strength=0.5,
            confidence=0.3,
            factor_score=0.0,
            momentum_score=0.0,
            mean_reversion_score=0.0,
            technical_score=0.0,
            signal_source="momentum_model",
            lookback_period=0,
            warning_flags=[reason]
        )
    
    def get_model_name(self) -> str:
        return "momentum_alpha_model"


class MeanReversionAlphaModel(AlphaModel):
    """Mean reversion alpha generation model"""
    
    def generate_signal(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        market_data: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> AlphaSignal:
        """Generate mean reversion signal"""
        
        reversion_window = parameters.get('mean_reversion_window', 20)
        signal_threshold = parameters.get('signal_strength', 0.6)
        
        if len(price_data) < reversion_window * 2:
            return self._create_neutral_signal(symbol, "insufficient_data_mean_reversion")
        
        # Calculate mean reversion scores
        reversion_scores = self._calculate_mean_reversion_scores(
            price_data, reversion_window
        )
        
        # Combined mean reversion score
        total_reversion_score = np.mean(list(reversion_scores.values()))
        
        # Mean reversion signals are contrarian
        # Positive score means price is below mean (buy signal)
        # Negative score means price is above mean (sell signal)
        if total_reversion_score >= signal_threshold:
            signal_type = SignalType.BUY  # Price below mean, expect reversion up
            strength = min(1.0, total_reversion_score)
        elif total_reversion_score <= -signal_threshold:
            signal_type = SignalType.SELL  # Price above mean, expect reversion down
            strength = min(1.0, abs(total_reversion_score))
        else:
            signal_type = SignalType.HOLD
            strength = 0.5
        
        # Calculate confidence
        reversion_values = list(reversion_scores.values())
        consistency = 1.0 - np.std(reversion_values) if len(reversion_values) > 1 else 0.5
        confidence = max(0.3, min(1.0, consistency))
        
        return AlphaSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            factor_score=0.0,
            momentum_score=0.0,
            mean_reversion_score=total_reversion_score,
            technical_score=0.0,
            signal_source="mean_reversion_model",
            lookback_period=reversion_window,
            supporting_factors=list(reversion_scores.keys())
        )
    
    def _calculate_mean_reversion_scores(
        self, 
        price_data: pd.DataFrame, 
        window: int
    ) -> Dict[str, float]:
        """Calculate mean reversion scores"""
        
        prices = price_data['close']
        scores = {}
        
        # Simple moving average reversion
        if len(prices) >= window:
            sma = prices.rolling(window=window).mean()
            current_price = prices.iloc[-1]
            mean_price = sma.iloc[-1]
            
            if mean_price > 0:
                # Deviation from mean (positive = below mean)
                deviation = (mean_price - current_price) / mean_price
                scores['sma_reversion'] = max(-1.0, min(1.0, deviation * 2))
        
        # Bollinger Bands reversion
        if len(prices) >= window:
            sma = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            
            if len(sma) > 0 and len(std) > 0:
                upper_band = sma + (2 * std)
                lower_band = sma - (2 * std)
                current_price = prices.iloc[-1]
                
                # Position within Bollinger Bands
                if upper_band.iloc[-1] != lower_band.iloc[-1]:
                    band_position = (current_price - lower_band.iloc[-1]) / (
                        upper_band.iloc[-1] - lower_band.iloc[-1]
                    )
                    # Reversion score: low position = buy, high position = sell
                    reversion_score = 1.0 - 2 * band_position  # -1 to 1
                    scores['bollinger_reversion'] = max(-1.0, min(1.0, reversion_score))
        
        # Z-score reversion
        if len(prices) >= window:
            recent_prices = prices.tail(window)
            mean_price = recent_prices.mean()
            std_price = recent_prices.std()
            current_price = prices.iloc[-1]
            
            if std_price > 0:
                z_score = (current_price - mean_price) / std_price
                # Reversion score: negative z-score = buy, positive = sell
                reversion_score = -z_score / 2  # Scale to reasonable range
                scores['zscore_reversion'] = max(-1.0, min(1.0, reversion_score))
        
        return scores
    
    def _create_neutral_signal(self, symbol: str, reason: str) -> AlphaSignal:
        """Create neutral signal with warning"""
        return AlphaSignal(
            symbol=symbol,
            signal_type=SignalType.HOLD,
            strength=0.5,
            confidence=0.3,
            factor_score=0.0,
            momentum_score=0.0,
            mean_reversion_score=0.0,
            technical_score=0.0,
            signal_source="mean_reversion_model",
            lookback_period=0,
            warning_flags=[reason]
        )
    
    def get_model_name(self) -> str:
        return "mean_reversion_alpha_model"


class TechnicalAlphaModel(AlphaModel):
    """Technical indicator based alpha generation model"""
    
    def generate_signal(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        market_data: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> AlphaSignal:
        """Generate technical indicator based signal"""
        
        technical_indicators = parameters.get('technical_indicators', ['rsi', 'macd', 'moving_average'])
        signal_threshold = parameters.get('signal_strength', 0.6)
        
        if len(price_data) < 50:  # Need minimum data for technical indicators
            return self._create_neutral_signal(symbol, "insufficient_data_technical")
        
        # Calculate technical scores
        technical_scores = self._calculate_technical_scores(
            price_data, technical_indicators
        )
        
        # Combined technical score
        if technical_scores:
            total_technical_score = np.mean(list(technical_scores.values()))
        else:
            total_technical_score = 0.0
        
        # Generate signal
        if total_technical_score >= signal_threshold:
            signal_type = SignalType.BUY
            strength = min(1.0, total_technical_score)
        elif total_technical_score <= -signal_threshold:
            signal_type = SignalType.SELL
            strength = min(1.0, abs(total_technical_score))
        else:
            signal_type = SignalType.HOLD
            strength = 0.5
        
        # Calculate confidence
        if technical_scores:
            technical_values = list(technical_scores.values())
            consistency = 1.0 - np.std(technical_values) if len(technical_values) > 1 else 0.5
            confidence = max(0.3, min(1.0, consistency))
        else:
            confidence = 0.3
        
        return AlphaSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            factor_score=0.0,
            momentum_score=0.0,
            mean_reversion_score=0.0,
            technical_score=total_technical_score,
            signal_source="technical_model",
            lookback_period=50,
            supporting_factors=list(technical_scores.keys()) if technical_scores else []
        )
    
    def _calculate_technical_scores(
        self, 
        price_data: pd.DataFrame, 
        indicators: List[str]
    ) -> Dict[str, float]:
        """Calculate technical indicator scores"""
        
        scores = {}
        prices = price_data['close']
        
        # RSI Score
        if 'rsi' in indicators and len(prices) >= 14:
            rsi = self._calculate_rsi(prices, 14)
            if not np.isnan(rsi):
                # RSI transformation: oversold (30) = buy (+1), overbought (70) = sell (-1)
                if rsi <= 30:
                    rsi_score = (30 - rsi) / 30  # 0 to 1
                elif rsi >= 70:
                    rsi_score = -(rsi - 70) / 30  # 0 to -1
                else:
                    rsi_score = 0
                scores['rsi'] = max(-1.0, min(1.0, rsi_score))
        
        # MACD Score
        if 'macd' in indicators and len(prices) >= 26:
            macd_line, signal_line = self._calculate_macd(prices)
            if not (np.isnan(macd_line) or np.isnan(signal_line)):
                macd_diff = macd_line - signal_line
                # Normalize MACD difference
                macd_score = max(-1.0, min(1.0, macd_diff * 100))
                scores['macd'] = macd_score
        
        # Moving Average Score
        if 'moving_average' in indicators and len(prices) >= 50:
            ma_20 = prices.rolling(window=20).mean().iloc[-1]
            ma_50 = prices.rolling(window=50).mean().iloc[-1]
            current_price = prices.iloc[-1]
            
            # Price vs moving averages
            ma_score = 0
            if current_price > ma_20:
                ma_score += 0.5
            if current_price > ma_50:
                ma_score += 0.5
            if ma_20 > ma_50:  # Short MA above long MA
                ma_score += 0.5
            
            # Normalize to -1 to 1 range
            ma_score = (ma_score / 1.5) * 2 - 1  # 0-1.5 -> -1 to 1
            scores['moving_average'] = ma_score
        
        # Bollinger Bands Score
        if 'bollinger_bands' in indicators and len(prices) >= 20:
            sma_20 = prices.rolling(window=20).mean()
            std_20 = prices.rolling(window=20).std()
            
            if len(sma_20) > 0 and len(std_20) > 0:
                upper_band = sma_20.iloc[-1] + (2 * std_20.iloc[-1])
                lower_band = sma_20.iloc[-1] - (2 * std_20.iloc[-1])
                current_price = prices.iloc[-1]
                
                # Bollinger Bands position
                if upper_band != lower_band:
                    bb_position = (current_price - lower_band) / (upper_band - lower_band)
                    # Transform to signal: near lower band = buy, near upper band = sell
                    bb_score = (0.5 - bb_position) * 2  # 0-1 -> 1 to -1
                    scores['bollinger_bands'] = max(-1.0, min(1.0, bb_score))
        
        return scores
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return np.nan
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else np.nan
    
    def _calculate_macd(
        self, 
        prices: pd.Series, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Tuple[float, float]:
        """Calculate MACD indicator"""
        if len(prices) < slow_period:
            return np.nan, np.nan
        
        ema_fast = prices.ewm(span=fast_period).mean()
        ema_slow = prices.ewm(span=slow_period).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        
        return macd_line.iloc[-1], signal_line.iloc[-1]
    
    def _create_neutral_signal(self, symbol: str, reason: str) -> AlphaSignal:
        """Create neutral signal with warning"""
        return AlphaSignal(
            symbol=symbol,
            signal_type=SignalType.HOLD,
            strength=0.5,
            confidence=0.3,
            factor_score=0.0,
            momentum_score=0.0,
            mean_reversion_score=0.0,
            technical_score=0.0,
            signal_source="technical_model",
            lookback_period=0,
            warning_flags=[reason]
        )
    
    def get_model_name(self) -> str:
        return "technical_alpha_model"


class AlphaGenerationEngine:
    """
    Main alpha generation engine that orchestrates all alpha models
    
    Handles parameters 30-34:
    30. signal_strength - Controls signal confidence thresholds
    31. factor_models - Selects and weights factor models
    32. momentum_lookback - Sets momentum calculation periods
    33. mean_reversion_window - Controls mean reversion analysis
    34. technical_indicators - Manages technical indicator selection
    """
    
    def __init__(self, registry: TradingParameterRegistry = None):
        self.registry = registry or parameter_registry
        self.alpha_models = self._initialize_alpha_models()
        
        # Alpha tracking
        self.signal_history: List[AlphaSignal] = []
        self.portfolio_alpha: Optional[PortfolioAlpha] = None
        
    def _initialize_alpha_models(self) -> Dict[str, AlphaModel]:
        """Initialize alpha generation models"""
        return {
            'factor': FactorAlphaModel(),
            'momentum': MomentumAlphaModel(),
            'mean_reversion': MeanReversionAlphaModel(),
            'technical': TechnicalAlphaModel()
        }
    
    def generate_alpha_signal(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        market_data: Dict[str, Any],
        model_weights: Optional[Dict[str, float]] = None
    ) -> AlphaSignal:
        """
        Generate comprehensive alpha signal for a symbol
        
        Args:
            symbol: Trading symbol
            price_data: Historical price data
            market_data: Current market data and fundamentals
            model_weights: Optional weights for combining models
            
        Returns:
            Combined alpha signal
        """
        
        # Get current parameters
        params = self._get_current_parameters()
        
        # Default equal weights if not provided
        if model_weights is None:
            model_weights = {
                'factor': 0.3,
                'momentum': 0.25,
                'mean_reversion': 0.25,
                'technical': 0.2
            }
        
        # Generate signals from each model
        individual_signals = {}
        
        for model_name, model in self.alpha_models.items():
            try:
                signal = model.generate_signal(symbol, price_data, market_data, params)
                individual_signals[model_name] = signal
            except Exception as e:
                logger.warning(f"Error generating {model_name} signal for {symbol}: {e}")
                # Create neutral signal for failed model
                individual_signals[model_name] = self._create_neutral_signal(
                    symbol, f"{model_name}_error"
                )
        
        # Combine signals using weights
        combined_signal = self._combine_signals(
            individual_signals, model_weights, params
        )
        
        # Store signal in history
        self.signal_history.append(combined_signal)
        if len(self.signal_history) > 1000:  # Keep last 1000 signals
            self.signal_history = self.signal_history[-1000:]
        
        return combined_signal
    
    def generate_portfolio_signals(
        self,
        symbols: List[str],
        price_data_dict: Dict[str, pd.DataFrame],
        market_data_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, AlphaSignal]:
        """Generate alpha signals for multiple symbols"""
        
        signals = {}
        
        for symbol in symbols:
            if symbol in price_data_dict and symbol in market_data_dict:
                try:
                    signal = self.generate_alpha_signal(
                        symbol,
                        price_data_dict[symbol],
                        market_data_dict[symbol]
                    )
                    signals[symbol] = signal
                except Exception as e:
                    logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def calculate_portfolio_alpha(
        self,
        portfolio_signals: Dict[str, AlphaSignal],
        portfolio_returns: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None
    ) -> PortfolioAlpha:
        """Calculate portfolio-level alpha metrics"""
        
        if not portfolio_signals:
            raise ValueError("No portfolio signals provided")
        
        # Calculate signal statistics
        signal_counts = {}
        strengths = []
        confidences = []
        
        for signal in portfolio_signals.values():
            signal_type = signal.signal_type.value
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            strengths.append(signal.strength)
            confidences.append(signal.confidence)
        
        avg_signal_strength = np.mean(strengths) if strengths else 0
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Calculate portfolio alpha (simplified)
        if portfolio_returns is not None and benchmark_returns is not None:
            # Align returns
            aligned_portfolio, aligned_benchmark = portfolio_returns.align(
                benchmark_returns, join='inner'
            )
            
            if len(aligned_portfolio) > 0:
                portfolio_alpha_annual = (aligned_portfolio.mean() - aligned_benchmark.mean()) * 252
                information_ratio = (
                    (aligned_portfolio.mean() - aligned_benchmark.mean()) /
                    (aligned_portfolio - aligned_benchmark).std()
                ) * np.sqrt(252) if (aligned_portfolio - aligned_benchmark).std() > 0 else 0
                
                # Calculate other metrics
                alpha_volatility = aligned_portfolio.std() * np.sqrt(252)
                sharpe_ratio = (aligned_portfolio.mean() * 252) / alpha_volatility if alpha_volatility > 0 else 0
                
                # Hit rate (simplified - would need actual predictions vs outcomes)
                hit_rate = avg_confidence  # Simplified approximation
                
                # Max drawdown (simplified)
                cumulative_returns = (1 + aligned_portfolio).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdowns = (cumulative_returns - rolling_max) / rolling_max
                max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
            else:
                portfolio_alpha_annual = 0
                information_ratio = 0
                alpha_volatility = 0
                sharpe_ratio = 0
                hit_rate = 0
                max_drawdown = 0
        else:
            # Use signal-based estimates
            portfolio_alpha_annual = avg_signal_strength * 0.15  # Estimate 15% max alpha
            information_ratio = avg_confidence * 1.5  # Rough estimate
            alpha_volatility = 0.20  # Default estimate
            sharpe_ratio = portfolio_alpha_annual / alpha_volatility if alpha_volatility > 0 else 0
            hit_rate = avg_confidence
            max_drawdown = 0.1  # Default estimate
        
        # Performance attribution
        factor_contribution = np.mean([s.factor_score for s in portfolio_signals.values()])
        momentum_contribution = np.mean([s.momentum_score for s in portfolio_signals.values()])
        mean_reversion_contribution = np.mean([s.mean_reversion_score for s in portfolio_signals.values()])
        technical_contribution = np.mean([s.technical_score for s in portfolio_signals.values()])
        
        self.portfolio_alpha = PortfolioAlpha(
            portfolio_alpha=portfolio_alpha_annual,
            information_ratio=information_ratio,
            hit_rate=hit_rate,
            avg_signal_strength=avg_signal_strength,
            signal_counts=signal_counts,
            sector_alpha={},  # Would populate with sector breakdown
            factor_contribution={
                'factor': factor_contribution,
                'momentum': momentum_contribution,
                'mean_reversion': mean_reversion_contribution,
                'technical': technical_contribution
            },
            momentum_contribution=momentum_contribution,
            mean_reversion_contribution=mean_reversion_contribution,
            technical_contribution=technical_contribution,
            alpha_volatility=alpha_volatility,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            analysis_period=(datetime.now() - timedelta(days=252), datetime.now())
        )
        
        return self.portfolio_alpha
    
    def get_alpha_summary(self) -> Dict[str, Any]:
        """Get comprehensive alpha generation summary"""
        
        params = self._get_current_parameters()
        
        # Recent signal statistics
        recent_signals = self.signal_history[-100:] if self.signal_history else []
        
        if recent_signals:
            signal_type_counts = {}
            avg_strength = np.mean([s.strength for s in recent_signals])
            avg_confidence = np.mean([s.confidence for s in recent_signals])
            
            for signal in recent_signals:
                signal_type = signal.signal_type.value
                signal_type_counts[signal_type] = signal_type_counts.get(signal_type, 0) + 1
        else:
            signal_type_counts = {}
            avg_strength = 0
            avg_confidence = 0
        
        return {
            'parameters': params,
            'recent_signal_count': len(recent_signals),
            'avg_signal_strength': avg_strength,
            'avg_confidence': avg_confidence,
            'signal_distribution': signal_type_counts,
            'portfolio_alpha': (
                self.portfolio_alpha.portfolio_alpha if self.portfolio_alpha else 0
            ),
            'information_ratio': (
                self.portfolio_alpha.information_ratio if self.portfolio_alpha else 0
            ),
            'active_models': list(self.alpha_models.keys()),
            'total_signals_generated': len(self.signal_history),
            'last_updated': datetime.now()
        }
    
    def optimize_alpha_parameters(
        self,
        historical_performance: Dict[str, float],
        target_alpha: float = 0.10,
        target_information_ratio: float = 1.0
    ) -> Dict[str, Any]:
        """Suggest optimized alpha generation parameters"""
        
        current_params = self._get_current_parameters()
        suggestions = {}
        
        current_alpha = historical_performance.get('alpha', 0)
        current_ir = historical_performance.get('information_ratio', 0)
        
        # Alpha optimization suggestions
        if current_alpha < target_alpha:
            if current_params.get('signal_strength', 0.6) > 0.5:
                suggestions['signal_strength'] = 'Consider lowering signal threshold for more signals'
            
            suggestions['factor_models'] = 'Consider adding momentum or quality factors'
            suggestions['technical_indicators'] = 'Consider adding RSI or MACD indicators'
        
        # Information ratio optimization
        if current_ir < target_information_ratio:
            suggestions['mean_reversion_window'] = 'Consider optimizing mean reversion window'
            suggestions['momentum_lookback'] = 'Consider shorter momentum periods for higher turnover'
        
        return {
            'current_alpha': current_alpha,
            'target_alpha': target_alpha,
            'current_information_ratio': current_ir,
            'target_information_ratio': target_information_ratio,
            'parameter_suggestions': suggestions,
            'optimization_potential': abs(target_alpha - current_alpha) + abs(target_information_ratio - current_ir)
        }
    
    def _get_current_parameters(self) -> Dict[str, Any]:
        """Get current alpha generation parameters"""
        param_names = [
            'signal_strength', 'factor_models', 'momentum_lookback',
            'mean_reversion_window', 'technical_indicators'
        ]
        params = {}
        
        for name in param_names:
            param_def = self.registry.get_parameter(name)
            if param_def:
                params[name] = param_def.current_value or param_def.default_value
        
        return params
    
    def _combine_signals(
        self,
        individual_signals: Dict[str, AlphaSignal],
        model_weights: Dict[str, float],
        parameters: Dict[str, Any]
    ) -> AlphaSignal:
        """Combine individual model signals into final signal"""
        
        if not individual_signals:
            return self._create_neutral_signal("unknown", "no_signals")
        
        # Get a reference signal for basic info
        reference_signal = list(individual_signals.values())[0]
        
        # Calculate weighted scores
        weighted_strength = 0
        weighted_confidence = 0
        total_weight = 0
        
        component_scores = {
            'factor_score': 0,
            'momentum_score': 0,
            'mean_reversion_score': 0,
            'technical_score': 0
        }
        
        supporting_factors = []
        warning_flags = []
        
        for model_name, signal in individual_signals.items():
            weight = model_weights.get(model_name, 0)
            
            if weight > 0:
                # Convert signal type to numeric for combination
                signal_numeric = self._signal_type_to_numeric(signal.signal_type)
                
                weighted_strength += signal_numeric * signal.strength * weight
                weighted_confidence += signal.confidence * weight
                total_weight += weight
                
                # Component scores
                component_scores['factor_score'] += signal.factor_score * weight
                component_scores['momentum_score'] += signal.momentum_score * weight
                component_scores['mean_reversion_score'] += signal.mean_reversion_score * weight
                component_scores['technical_score'] += signal.technical_score * weight
                
                supporting_factors.extend(signal.supporting_factors)
                warning_flags.extend(signal.warning_flags)
        
        # Normalize by total weight
        if total_weight > 0:
            final_strength_numeric = weighted_strength / total_weight
            final_confidence = weighted_confidence / total_weight
            
            for key in component_scores:
                component_scores[key] /= total_weight
        else:
            final_strength_numeric = 0
            final_confidence = 0.3
        
        # Convert back to signal type and strength
        final_signal_type, final_strength = self._numeric_to_signal_type(final_strength_numeric)
        
        # Apply signal threshold
        signal_threshold = parameters.get('signal_strength', 0.6)
        if abs(final_strength_numeric) < signal_threshold:
            final_signal_type = SignalType.HOLD
            final_strength = 0.5
        
        return AlphaSignal(
            symbol=reference_signal.symbol,
            signal_type=final_signal_type,
            strength=final_strength,
            confidence=final_confidence,
            factor_score=component_scores['factor_score'],
            momentum_score=component_scores['momentum_score'],
            mean_reversion_score=component_scores['mean_reversion_score'],
            technical_score=component_scores['technical_score'],
            signal_source="combined_alpha_models",
            lookbook_period=max([s.lookback_period for s in individual_signals.values()]),
            supporting_factors=list(set(supporting_factors)),
            warning_flags=list(set(warning_flags))
        )
    
    def _signal_type_to_numeric(self, signal_type: SignalType) -> float:
        """Convert signal type to numeric value for combination"""
        mapping = {
            SignalType.STRONG_SELL: -1.0,
            SignalType.SELL: -0.7,
            SignalType.HOLD: 0.0,
            SignalType.BUY: 0.7,
            SignalType.STRONG_BUY: 1.0
        }
        return mapping.get(signal_type, 0.0)
    
    def _numeric_to_signal_type(self, numeric_value: float) -> Tuple[SignalType, float]:
        """Convert numeric value back to signal type and strength"""
        abs_value = abs(numeric_value)
        
        if numeric_value >= 0.8:
            return SignalType.STRONG_BUY, abs_value
        elif numeric_value >= 0.3:
            return SignalType.BUY, abs_value
        elif numeric_value <= -0.8:
            return SignalType.STRONG_SELL, abs_value
        elif numeric_value <= -0.3:
            return SignalType.SELL, abs_value
        else:
            return SignalType.HOLD, 0.5
    
    def _create_neutral_signal(self, symbol: str, reason: str) -> AlphaSignal:
        """Create neutral signal"""
        return AlphaSignal(
            symbol=symbol,
            signal_type=SignalType.HOLD,
            strength=0.5,
            confidence=0.3,
            factor_score=0.0,
            momentum_score=0.0,
            mean_reversion_score=0.0,
            technical_score=0.0,
            signal_source="neutral",
            lookback_period=0,
            warning_flags=[reason]
        )


# Factory function for easy instantiation
def create_alpha_generation_engine(registry: TradingParameterRegistry = None) -> AlphaGenerationEngine:
    """Create an alpha generation engine instance"""
    return AlphaGenerationEngine(registry)