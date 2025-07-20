"""
Real-time Monitoring Dashboard for SlackToDoc
Provides comprehensive system health monitoring and metrics collection
"""

import os
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import psutil
import aioredis
import asyncpg

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    process_count: int
    uptime_seconds: float

@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    timestamp: float
    active_requests: int
    response_time_avg: float
    response_time_p95: float
    error_rate: float
    documents_created_total: int
    documents_created_hourly: int
    slack_events_processed: int
    notion_api_calls: int
    openai_requests: int
    openai_tokens_used: int
    openai_cost_hourly: float

@dataclass
class HealthStatus:
    """Overall system health status"""
    status: str  # "healthy", "degraded", "unhealthy"
    checks: Dict[str, bool]
    issues: List[str]
    last_check: float
    uptime_percent: float

class MonitoringDashboard:
    """Production monitoring dashboard with real-time metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.redis_client: Optional[aioredis.Redis] = None
        self.db_pool: Optional[asyncpg.Pool] = None
        
        # Metrics storage
        self.system_metrics_history = deque(maxlen=1440)  # 24 hours of minute data
        self.app_metrics_history = deque(maxlen=1440)
        self.response_times = deque(maxlen=1000)  # Last 1000 requests
        self.error_counts = defaultdict(int)
        
        # Health check configuration
        self.health_checks = {
            "database": self._check_database_health,
            "redis": self._check_redis_health,
            "slack_api": self._check_slack_api_health,
            "notion_api": self._check_notion_api_health,
            "openai_api": self._check_openai_api_health,
            "disk_space": self._check_disk_space,
            "memory": self._check_memory_usage,
            "cpu": self._check_cpu_usage
        }
        
        # SLA thresholds
        self.sla_thresholds = {
            "response_time_p95": 5.0,  # 5 seconds
            "error_rate": 0.01,  # 1%
            "uptime": 0.999,  # 99.9%
            "cpu_usage": 0.80,  # 80%
            "memory_usage": 0.85,  # 85%
            "disk_usage": 0.90  # 90%
        }
        
        # Alert state tracking
        self.alert_states = {}
        self.last_alert_times = {}
        
        # Performance counters
        self.performance_counters = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "documents_created": 0,
            "slack_events": 0,
            "notion_calls": 0,
            "openai_requests": 0,
            "openai_tokens": 0,
            "openai_cost": 0.0
        }
    
    async def initialize(self):
        """Initialize monitoring components"""
        try:
            # Initialize Redis connection
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_client = aioredis.from_url(redis_url)
            
            # Initialize database connection
            database_url = os.getenv("DATABASE_URL")
            if database_url:
                self.db_pool = await asyncpg.create_pool(database_url, min_size=1, max_size=5)
            
            logger.info("Monitoring dashboard initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {str(e)}")
            raise
    
    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            # Uptime
            uptime_seconds = time.time() - self.start_time
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_io=network_io,
                process_count=process_count,
                uptime_seconds=uptime_seconds
            )
            
            # Store in history
            self.system_metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}")
            raise
    
    async def collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics"""
        try:
            # Calculate response time metrics
            response_times = list(self.response_times)
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
            p95_response_time = self._calculate_percentile(response_times, 95) if response_times else 0.0
            
            # Calculate error rate
            total_requests = self.performance_counters["requests_total"]
            failed_requests = self.performance_counters["requests_failed"]
            error_rate = (failed_requests / total_requests) if total_requests > 0 else 0.0
            
            # Get hourly document creation count
            current_hour = datetime.now().hour
            hourly_docs = await self._get_hourly_document_count()
            
            metrics = ApplicationMetrics(
                timestamp=time.time(),
                active_requests=0,  # Would track from request middleware
                response_time_avg=avg_response_time,
                response_time_p95=p95_response_time,
                error_rate=error_rate,
                documents_created_total=self.performance_counters["documents_created"],
                documents_created_hourly=hourly_docs,
                slack_events_processed=self.performance_counters["slack_events"],
                notion_api_calls=self.performance_counters["notion_calls"],
                openai_requests=self.performance_counters["openai_requests"],
                openai_tokens_used=self.performance_counters["openai_tokens"],
                openai_cost_hourly=self.performance_counters["openai_cost"]
            )
            
            # Store in history
            self.app_metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {str(e)}")
            raise
    
    async def perform_health_checks(self) -> HealthStatus:
        """Perform comprehensive health checks"""
        try:
            checks = {}
            issues = []
            
            # Run all health checks
            for check_name, check_func in self.health_checks.items():
                try:
                    is_healthy = await check_func()
                    checks[check_name] = is_healthy
                    
                    if not is_healthy:
                        issues.append(f"{check_name} check failed")
                        
                except Exception as e:
                    checks[check_name] = False
                    issues.append(f"{check_name} check error: {str(e)}")
            
            # Determine overall status
            if all(checks.values()):
                status = "healthy"
            elif any(checks.values()):
                status = "degraded"
            else:
                status = "unhealthy"
            
            # Calculate uptime percentage
            uptime_percent = self._calculate_uptime_percentage()
            
            health_status = HealthStatus(
                status=status,
                checks=checks,
                issues=issues,
                last_check=time.time(),
                uptime_percent=uptime_percent
            )
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return HealthStatus(
                status="unhealthy",
                checks={},
                issues=[f"Health check system error: {str(e)}"],
                last_check=time.time(),
                uptime_percent=0.0
            )
    
    async def _check_database_health(self) -> bool:
        """Check database connectivity and performance"""
        try:
            if not self.db_pool:
                return False
            
            async with self.db_pool.acquire() as conn:
                # Simple query to test connectivity
                result = await conn.fetchval("SELECT 1")
                return result == 1
                
        except Exception as e:
            logger.warning(f"Database health check failed: {str(e)}")
            return False
    
    async def _check_redis_health(self) -> bool:
        """Check Redis connectivity and performance"""
        try:
            if not self.redis_client:
                return False
            
            # Ping Redis
            pong = await self.redis_client.ping()
            return pong is True
            
        except Exception as e:
            logger.warning(f"Redis health check failed: {str(e)}")
            return False
    
    async def _check_slack_api_health(self) -> bool:
        """Check Slack API connectivity"""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://slack.com/api/api.test",
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("ok", False)
                
                return False
                
        except Exception as e:
            logger.warning(f"Slack API health check failed: {str(e)}")
            return False
    
    async def _check_notion_api_health(self) -> bool:
        """Check Notion API connectivity"""
        try:
            import httpx
            
            # Simple connectivity test (doesn't require auth)
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.notion.com/v1/users/me",
                    headers={"Notion-Version": "2022-06-28"},
                    timeout=10.0
                )
                
                # Expect 401 (unauthorized) which means API is reachable
                return response.status_code in [200, 401]
                
        except Exception as e:
            logger.warning(f"Notion API health check failed: {str(e)}")
            return False
    
    async def _check_openai_api_health(self) -> bool:
        """Check OpenAI API connectivity"""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    timeout=10.0
                )
                
                # Expect 401 (unauthorized) which means API is reachable
                return response.status_code in [200, 401]
                
        except Exception as e:
            logger.warning(f"OpenAI API health check failed: {str(e)}")
            return False
    
    async def _check_disk_space(self) -> bool:
        """Check disk space availability"""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            return usage_percent < (self.sla_thresholds["disk_usage"] * 100)
            
        except Exception as e:
            logger.warning(f"Disk space check failed: {str(e)}")
            return False
    
    async def _check_memory_usage(self) -> bool:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent < (self.sla_thresholds["memory_usage"] * 100)
            
        except Exception as e:
            logger.warning(f"Memory usage check failed: {str(e)}")
            return False
    
    async def _check_cpu_usage(self) -> bool:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < (self.sla_thresholds["cpu_usage"] * 100)
            
        except Exception as e:
            logger.warning(f"CPU usage check failed: {str(e)}")
            return False
    
    def record_request(self, response_time: float, success: bool):
        """Record request metrics"""
        self.performance_counters["requests_total"] += 1
        
        if success:
            self.performance_counters["requests_successful"] += 1
        else:
            self.performance_counters["requests_failed"] += 1
        
        self.response_times.append(response_time)
    
    def record_document_creation(self):
        """Record document creation"""
        self.performance_counters["documents_created"] += 1
    
    def record_slack_event(self):
        """Record Slack event processing"""
        self.performance_counters["slack_events"] += 1
    
    def record_notion_api_call(self):
        """Record Notion API call"""
        self.performance_counters["notion_calls"] += 1
    
    def record_openai_request(self, tokens_used: int, cost: float):
        """Record OpenAI API usage"""
        self.performance_counters["openai_requests"] += 1
        self.performance_counters["openai_tokens"] += tokens_used
        self.performance_counters["openai_cost"] += cost
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        
        return sorted_values[index]
    
    def _calculate_uptime_percentage(self) -> float:
        """Calculate uptime percentage over last 24 hours"""
        try:
            # Simple calculation based on system metrics
            if len(self.system_metrics_history) < 10:
                return 100.0  # Not enough data yet
            
            # Count healthy vs total checks
            total_checks = len(self.system_metrics_history)
            healthy_checks = sum(1 for m in self.system_metrics_history 
                               if m.cpu_percent < 95 and m.memory_percent < 95)
            
            return (healthy_checks / total_checks) * 100 if total_checks > 0 else 100.0
            
        except Exception as e:
            logger.error(f"Uptime calculation failed: {str(e)}")
            return 0.0
    
    async def _get_hourly_document_count(self) -> int:
        """Get document creation count for current hour"""
        try:
            if not self.db_pool:
                return 0
            
            current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
            
            async with self.db_pool.acquire() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM documents WHERE created_at >= $1",
                    current_hour
                )
                return count or 0
                
        except Exception as e:
            logger.error(f"Failed to get hourly document count: {str(e)}")
            return 0
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            # Collect latest metrics
            system_metrics = await self.collect_system_metrics()
            app_metrics = await self.collect_application_metrics()
            health_status = await self.perform_health_checks()
            
            return {
                "timestamp": time.time(),
                "system_metrics": asdict(system_metrics),
                "application_metrics": asdict(app_metrics),
                "health_status": asdict(health_status),
                "sla_compliance": self._check_sla_compliance(system_metrics, app_metrics),
                "performance_counters": self.performance_counters.copy(),
                "uptime_seconds": time.time() - self.start_time
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {str(e)}")
            return {"error": str(e), "timestamp": time.time()}
    
    def _check_sla_compliance(self, system_metrics: SystemMetrics, 
                            app_metrics: ApplicationMetrics) -> Dict[str, bool]:
        """Check SLA compliance"""
        return {
            "response_time": app_metrics.response_time_p95 <= self.sla_thresholds["response_time_p95"],
            "error_rate": app_metrics.error_rate <= self.sla_thresholds["error_rate"],
            "uptime": True,  # Calculated separately
            "cpu_usage": system_metrics.cpu_percent <= (self.sla_thresholds["cpu_usage"] * 100),
            "memory_usage": system_metrics.memory_percent <= (self.sla_thresholds["memory_usage"] * 100),
            "disk_usage": system_metrics.disk_percent <= (self.sla_thresholds["disk_usage"] * 100)
        }
    
    async def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        try:
            dashboard_data = await self.get_dashboard_data()
            
            if format == "json":
                import json
                return json.dumps(dashboard_data, indent=2, default=str)
            elif format == "prometheus":
                return self._format_prometheus_metrics(dashboard_data)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")
            return f"# Export error: {str(e)}"
    
    def _format_prometheus_metrics(self, data: Dict[str, Any]) -> str:
        """Format metrics in Prometheus format"""
        lines = []
        
        # System metrics
        if "system_metrics" in data:
            sm = data["system_metrics"]
            lines.extend([
                f"# HELP system_cpu_percent CPU usage percentage",
                f"# TYPE system_cpu_percent gauge",
                f"system_cpu_percent {sm['cpu_percent']}",
                f"# HELP system_memory_percent Memory usage percentage",
                f"# TYPE system_memory_percent gauge", 
                f"system_memory_percent {sm['memory_percent']}",
                f"# HELP system_disk_percent Disk usage percentage",
                f"# TYPE system_disk_percent gauge",
                f"system_disk_percent {sm['disk_percent']}"
            ])
        
        # Application metrics
        if "application_metrics" in data:
            am = data["application_metrics"]
            lines.extend([
                f"# HELP app_response_time_avg Average response time",
                f"# TYPE app_response_time_avg gauge",
                f"app_response_time_avg {am['response_time_avg']}",
                f"# HELP app_error_rate Error rate",
                f"# TYPE app_error_rate gauge",
                f"app_error_rate {am['error_rate']}",
                f"# HELP app_documents_created_total Total documents created",
                f"# TYPE app_documents_created_total counter",
                f"app_documents_created_total {am['documents_created_total']}"
            ])
        
        return "\n".join(lines)


# Global monitoring dashboard instance
monitoring_dashboard = MonitoringDashboard()