"""
OpenAI Rate Limiter for SlackToDoc Production
Handles rate limiting, cost optimization, and request management for OpenAI API
"""

import os
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import json

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 60  # OpenAI default for GPT-4
    tokens_per_minute: int = 40000  # OpenAI default for GPT-4
    max_concurrent_requests: int = 10
    burst_allowance: float = 1.5  # Allow 50% burst
    cost_limit_per_hour: float = 10.0  # USD
    cost_limit_per_day: float = 100.0  # USD

class OpenAIRateLimiter:
    """Production-ready rate limiter for OpenAI API with cost tracking"""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        
        # Request tracking
        self.request_times = deque()
        self.token_usage = deque()
        self.concurrent_requests = 0
        self.request_lock = asyncio.Lock()
        
        # Cost tracking
        self.hourly_costs = defaultdict(float)
        self.daily_costs = defaultdict(float)
        
        # Model pricing (per 1K tokens)
        self.model_pricing = {
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-32k": {"input": 0.06, "output": 0.12},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004}
        }
        
        # Circuit breaker
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = None
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 300  # 5 minutes
        
        # Performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "total_tokens_used": 0,
            "total_cost": 0.0
        }
    
    async def acquire_request_slot(self, estimated_tokens: int = 1000, model: str = "gpt-4-turbo-preview") -> bool:
        """Acquire a slot for making an API request"""
        async with self.request_lock:
            try:
                # Check circuit breaker
                if not self._check_circuit_breaker():
                    logger.warning("Circuit breaker is open, rejecting request")
                    return False
                
                # Check concurrent request limit
                if self.concurrent_requests >= self.config.max_concurrent_requests:
                    logger.warning(f"Max concurrent requests ({self.config.max_concurrent_requests}) reached")
                    return False
                
                # Check rate limits
                if not self._check_rate_limits(estimated_tokens):
                    logger.warning("Rate limits exceeded")
                    return False
                
                # Check cost limits
                estimated_cost = self._estimate_request_cost(estimated_tokens, model)
                if not self._check_cost_limits(estimated_cost):
                    logger.warning(f"Cost limits exceeded. Estimated cost: ${estimated_cost:.4f}")
                    return False
                
                # Acquire slot
                self.concurrent_requests += 1
                current_time = time.time()
                self.request_times.append(current_time)
                
                logger.debug(f"Request slot acquired. Concurrent: {self.concurrent_requests}")
                return True
                
            except Exception as e:
                logger.error(f"Error acquiring request slot: {str(e)}")
                return False
    
    async def release_request_slot(self, success: bool, tokens_used: int = 0, 
                                 response_time: float = 0.0, model: str = "gpt-4-turbo-preview"):
        """Release a request slot and update metrics"""
        async with self.request_lock:
            try:
                self.concurrent_requests = max(0, self.concurrent_requests - 1)
                
                # Update performance stats
                self.performance_stats["total_requests"] += 1
                
                if success:
                    self.performance_stats["successful_requests"] += 1
                    self.performance_stats["total_tokens_used"] += tokens_used
                    
                    # Update response time (moving average)
                    total_requests = self.performance_stats["total_requests"]
                    current_avg = self.performance_stats["average_response_time"]
                    self.performance_stats["average_response_time"] = (
                        (current_avg * (total_requests - 1) + response_time) / total_requests
                    )
                    
                    # Track cost
                    cost = self._calculate_actual_cost(tokens_used, model)
                    self._track_cost(cost)
                    self.performance_stats["total_cost"] += cost
                    
                    # Reset circuit breaker on success
                    self.circuit_breaker_failures = 0
                    
                else:
                    self.performance_stats["failed_requests"] += 1
                    self._record_failure()
                
                # Track token usage
                if tokens_used > 0:
                    current_time = time.time()
                    self.token_usage.append((current_time, tokens_used))
                
                logger.debug(f"Request slot released. Success: {success}, Concurrent: {self.concurrent_requests}")
                
            except Exception as e:
                logger.error(f"Error releasing request slot: {str(e)}")
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows requests"""
        if self.circuit_breaker_failures < self.circuit_breaker_threshold:
            return True
        
        if self.circuit_breaker_last_failure is None:
            return True
        
        time_since_failure = time.time() - self.circuit_breaker_last_failure
        if time_since_failure > self.circuit_breaker_timeout:
            # Reset circuit breaker
            self.circuit_breaker_failures = 0
            self.circuit_breaker_last_failure = None
            logger.info("Circuit breaker reset")
            return True
        
        return False
    
    def _check_rate_limits(self, estimated_tokens: int) -> bool:
        """Check if request is within rate limits"""
        current_time = time.time()
        one_minute_ago = current_time - 60
        
        # Clean old entries
        while self.request_times and self.request_times[0] < one_minute_ago:
            self.request_times.popleft()
        
        while self.token_usage and self.token_usage[0][0] < one_minute_ago:
            self.token_usage.popleft()
        
        # Check request rate
        requests_in_last_minute = len(self.request_times)
        max_requests = int(self.config.requests_per_minute * self.config.burst_allowance)
        
        if requests_in_last_minute >= max_requests:
            logger.warning(f"Request rate limit exceeded: {requests_in_last_minute}/{max_requests}")
            return False
        
        # Check token rate
        tokens_in_last_minute = sum(usage[1] for usage in self.token_usage)
        max_tokens = int(self.config.tokens_per_minute * self.config.burst_allowance)
        
        if tokens_in_last_minute + estimated_tokens > max_tokens:
            logger.warning(f"Token rate limit exceeded: {tokens_in_last_minute + estimated_tokens}/{max_tokens}")
            return False
        
        return True
    
    def _check_cost_limits(self, estimated_cost: float) -> bool:
        """Check if request is within cost limits"""
        current_hour = datetime.now().hour
        current_date = datetime.now().date()
        
        # Check hourly limit
        if self.hourly_costs[current_hour] + estimated_cost > self.config.cost_limit_per_hour:
            logger.warning(f"Hourly cost limit exceeded: ${self.hourly_costs[current_hour] + estimated_cost:.4f}")
            return False
        
        # Check daily limit
        if self.daily_costs[current_date] + estimated_cost > self.config.cost_limit_per_day:
            logger.warning(f"Daily cost limit exceeded: ${self.daily_costs[current_date] + estimated_cost:.4f}")
            return False
        
        return True
    
    def _estimate_request_cost(self, estimated_tokens: int, model: str) -> float:
        """Estimate cost of a request"""
        if model not in self.model_pricing:
            model = "gpt-4-turbo-preview"  # Default fallback
        
        pricing = self.model_pricing[model]
        
        # Estimate input/output split (typically 80/20 for our use case)
        input_tokens = int(estimated_tokens * 0.8)
        output_tokens = int(estimated_tokens * 0.2)
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def _calculate_actual_cost(self, tokens_used: int, model: str) -> float:
        """Calculate actual cost of a completed request"""
        # For simplicity, we'll use the estimated cost method
        # In practice, you'd get actual token counts from the API response
        return self._estimate_request_cost(tokens_used, model)
    
    def _track_cost(self, cost: float):
        """Track cost by hour and day"""
        current_hour = datetime.now().hour
        current_date = datetime.now().date()
        
        self.hourly_costs[current_hour] += cost
        self.daily_costs[current_date] += cost
        
        # Clean old entries (keep last 24 hours and 30 days)
        current_time = datetime.now()
        
        # Clean hourly costs
        for hour in list(self.hourly_costs.keys()):
            if isinstance(hour, int) and abs(hour - current_hour) > 1:
                continue  # Keep current and adjacent hours
        
        # Clean daily costs
        for date in list(self.daily_costs.keys()):
            if (current_time.date() - date).days > 30:
                del self.daily_costs[date]
    
    def _record_failure(self):
        """Record a request failure for circuit breaker"""
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = time.time()
        
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            logger.warning(f"Circuit breaker opened after {self.circuit_breaker_failures} failures")
    
    async def wait_for_rate_limit(self, estimated_tokens: int = 1000) -> float:
        """Calculate how long to wait before making a request"""
        current_time = time.time()
        one_minute_ago = current_time - 60
        
        # Clean old entries
        while self.request_times and self.request_times[0] < one_minute_ago:
            self.request_times.popleft()
        
        while self.token_usage and self.token_usage[0][0] < one_minute_ago:
            self.token_usage.popleft()
        
        # Calculate wait times
        request_wait = 0.0
        token_wait = 0.0
        
        # Request rate wait
        if len(self.request_times) >= self.config.requests_per_minute:
            oldest_request = self.request_times[0]
            request_wait = 60 - (current_time - oldest_request)
        
        # Token rate wait
        tokens_in_minute = sum(usage[1] for usage in self.token_usage)
        if tokens_in_minute + estimated_tokens > self.config.tokens_per_minute:
            oldest_token_time = self.token_usage[0][0] if self.token_usage else current_time
            token_wait = 60 - (current_time - oldest_token_time)
        
        return max(request_wait, token_wait, 0)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current rate limiter statistics"""
        current_time = time.time()
        one_minute_ago = current_time - 60
        
        # Clean and count current usage
        recent_requests = sum(1 for t in self.request_times if t > one_minute_ago)
        recent_tokens = sum(usage[1] for usage in self.token_usage if usage[0] > one_minute_ago)
        
        return {
            "concurrent_requests": self.concurrent_requests,
            "requests_last_minute": recent_requests,
            "tokens_last_minute": recent_tokens,
            "requests_limit": self.config.requests_per_minute,
            "tokens_limit": self.config.tokens_per_minute,
            "circuit_breaker_failures": self.circuit_breaker_failures,
            "circuit_breaker_open": self.circuit_breaker_failures >= self.circuit_breaker_threshold,
            "hourly_cost": self.hourly_costs.get(datetime.now().hour, 0.0),
            "daily_cost": self.daily_costs.get(datetime.now().date(), 0.0),
            "performance_stats": self.performance_stats.copy()
        }
    
    def reset_costs(self):
        """Reset cost tracking (for testing or manual reset)"""
        self.hourly_costs.clear()
        self.daily_costs.clear()
        self.performance_stats["total_cost"] = 0.0
        logger.info("Cost tracking reset")
    
    def adjust_limits(self, requests_per_minute: Optional[int] = None, 
                     tokens_per_minute: Optional[int] = None,
                     cost_limit_per_hour: Optional[float] = None):
        """Dynamically adjust rate limits"""
        if requests_per_minute is not None:
            self.config.requests_per_minute = requests_per_minute
            logger.info(f"Request rate limit adjusted to {requests_per_minute}/min")
        
        if tokens_per_minute is not None:
            self.config.tokens_per_minute = tokens_per_minute
            logger.info(f"Token rate limit adjusted to {tokens_per_minute}/min")
        
        if cost_limit_per_hour is not None:
            self.config.cost_limit_per_hour = cost_limit_per_hour
            logger.info(f"Hourly cost limit adjusted to ${cost_limit_per_hour}")


# Global rate limiter instance
rate_limiter = OpenAIRateLimiter()