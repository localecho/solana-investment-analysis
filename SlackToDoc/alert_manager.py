"""
Alert Manager for SlackToDoc Production
Handles alerting, escalation, and incident management with multiple notification channels
"""

import os
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import httpx

# Configure logging
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    source: str
    timestamp: float
    resolved_at: Optional[float] = None
    acknowledged_at: Optional[float] = None
    acknowledged_by: Optional[str] = None
    metadata: Dict[str, Any] = None
    escalation_level: int = 0
    suppressed_until: Optional[float] = None

@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    name: str
    type: str  # slack, email, pagerduty, webhook
    config: Dict[str, Any]
    enabled: bool = True
    severity_filter: List[AlertSeverity] = None

class AlertManager:
    """Production alert manager with escalation and multiple channels"""
    
    def __init__(self):
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Notification channels
        self.notification_channels: List[NotificationChannel] = []
        self._setup_default_channels()
        
        # Escalation rules
        self.escalation_rules = {
            AlertSeverity.INFO: {
                "initial_delay": 0,
                "escalation_delays": [],
                "max_escalations": 0
            },
            AlertSeverity.WARNING: {
                "initial_delay": 300,  # 5 minutes
                "escalation_delays": [900],  # 15 minutes
                "max_escalations": 1
            },
            AlertSeverity.CRITICAL: {
                "initial_delay": 60,  # 1 minute
                "escalation_delays": [300, 600],  # 5, 10 minutes
                "max_escalations": 2
            },
            AlertSeverity.EMERGENCY: {
                "initial_delay": 0,  # Immediate
                "escalation_delays": [120, 300],  # 2, 5 minutes
                "max_escalations": 2
            }
        }
        
        # Alert suppression rules
        self.suppression_rules = {
            "duplicate_window": 300,  # 5 minutes
            "rate_limit_window": 60,  # 1 minute
            "max_alerts_per_minute": 10
        }
        
        # Metrics
        self.metrics = {
            "alerts_sent": 0,
            "alerts_acknowledged": 0,
            "alerts_resolved": 0,
            "escalations_triggered": 0,
            "notification_failures": 0
        }
        
        # Background tasks
        self.escalation_tasks: Dict[str, asyncio.Task] = {}
        self.cleanup_task: Optional[asyncio.Task] = None
    
    def _setup_default_channels(self):
        """Setup default notification channels"""
        try:
            # Slack channel
            slack_webhook = os.getenv("SLACK_ALERT_WEBHOOK")
            if slack_webhook:
                self.notification_channels.append(
                    NotificationChannel(
                        name="slack_alerts",
                        type="slack",
                        config={"webhook_url": slack_webhook},
                        severity_filter=[AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
                    )
                )
            
            # PagerDuty integration
            pagerduty_key = os.getenv("PAGERDUTY_INTEGRATION_KEY")
            if pagerduty_key:
                self.notification_channels.append(
                    NotificationChannel(
                        name="pagerduty",
                        type="pagerduty",
                        config={"integration_key": pagerduty_key},
                        severity_filter=[AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
                    )
                )
            
            # Email notifications
            email_config = {
                "smtp_server": os.getenv("SMTP_SERVER"),
                "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                "username": os.getenv("SMTP_USERNAME"),
                "password": os.getenv("SMTP_PASSWORD"),
                "from_email": os.getenv("ALERT_FROM_EMAIL"),
                "to_emails": os.getenv("ALERT_TO_EMAILS", "").split(",")
            }
            
            if all([email_config["smtp_server"], email_config["username"], email_config["password"]]):
                self.notification_channels.append(
                    NotificationChannel(
                        name="email_alerts",
                        type="email",
                        config=email_config,
                        severity_filter=[AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
                    )
                )
            
            # Webhook notifications
            webhook_url = os.getenv("ALERT_WEBHOOK_URL")
            if webhook_url:
                self.notification_channels.append(
                    NotificationChannel(
                        name="webhook_alerts",
                        type="webhook",
                        config={"url": webhook_url},
                        severity_filter=[AlertSeverity.INFO, AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
                    )
                )
            
            logger.info(f"Configured {len(self.notification_channels)} notification channels")
            
        except Exception as e:
            logger.error(f"Failed to setup notification channels: {str(e)}")
    
    async def trigger_alert(self, title: str, description: str, severity: AlertSeverity, 
                          source: str, metadata: Dict[str, Any] = None) -> str:
        """Trigger a new alert"""
        try:
            # Generate alert ID
            alert_id = f"{source}_{int(time.time())}_{hash(title) % 10000}"
            
            # Check for duplicates
            if self._is_duplicate_alert(title, source):
                logger.info(f"Suppressing duplicate alert: {title}")
                return alert_id
            
            # Check rate limits
            if self._is_rate_limited():
                logger.warning("Alert rate limit exceeded, suppressing alert")
                return alert_id
            
            # Create alert
            alert = Alert(
                id=alert_id,
                title=title,
                description=description,
                severity=severity,
                status=AlertStatus.ACTIVE,
                source=source,
                timestamp=time.time(),
                metadata=metadata or {}
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Send notifications
            await self._send_notifications(alert)
            
            # Setup escalation if needed
            if severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                await self._setup_escalation(alert)
            
            # Update metrics
            self.metrics["alerts_sent"] += 1
            
            logger.info(f"Alert triggered: {alert_id} - {title} ({severity.value})")
            return alert_id
            
        except Exception as e:
            logger.error(f"Failed to trigger alert: {str(e)}")
            return ""
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id not in self.active_alerts:
                logger.warning(f"Alert not found: {alert_id}")
                return False
            
            alert = self.active_alerts[alert_id]
            
            if alert.status == AlertStatus.ACKNOWLEDGED:
                logger.info(f"Alert already acknowledged: {alert_id}")
                return True
            
            # Update alert
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = time.time()
            alert.acknowledged_by = acknowledged_by
            
            # Cancel escalation
            if alert_id in self.escalation_tasks:
                self.escalation_tasks[alert_id].cancel()
                del self.escalation_tasks[alert_id]
            
            # Send acknowledgment notifications
            await self._send_acknowledgment_notification(alert)
            
            # Update metrics
            self.metrics["alerts_acknowledged"] += 1
            
            logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {str(e)}")
            return False
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert"""
        try:
            if alert_id not in self.active_alerts:
                logger.warning(f"Alert not found: {alert_id}")
                return False
            
            alert = self.active_alerts[alert_id]
            
            # Update alert
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = time.time()
            
            # Cancel escalation
            if alert_id in self.escalation_tasks:
                self.escalation_tasks[alert_id].cancel()
                del self.escalation_tasks[alert_id]
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            # Send resolution notifications
            await self._send_resolution_notification(alert)
            
            # Update metrics
            self.metrics["alerts_resolved"] += 1
            
            logger.info(f"Alert resolved: {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve alert: {str(e)}")
            return False
    
    async def suppress_alert(self, alert_id: str, duration_minutes: int = 60) -> bool:
        """Temporarily suppress an alert"""
        try:
            if alert_id not in self.active_alerts:
                logger.warning(f"Alert not found: {alert_id}")
                return False
            
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            alert.suppressed_until = time.time() + (duration_minutes * 60)
            
            logger.info(f"Alert suppressed: {alert_id} for {duration_minutes} minutes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to suppress alert: {str(e)}")
            return False
    
    def _is_duplicate_alert(self, title: str, source: str) -> bool:
        """Check if alert is a duplicate within the suppression window"""
        current_time = time.time()
        window_start = current_time - self.suppression_rules["duplicate_window"]
        
        for alert in self.alert_history:
            if (alert.title == title and 
                alert.source == source and 
                alert.timestamp > window_start):
                return True
        
        return False
    
    def _is_rate_limited(self) -> bool:
        """Check if alert rate limit is exceeded"""
        current_time = time.time()
        window_start = current_time - self.suppression_rules["rate_limit_window"]
        
        recent_alerts = sum(1 for alert in self.alert_history 
                          if alert.timestamp > window_start)
        
        return recent_alerts >= self.suppression_rules["max_alerts_per_minute"]
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications to all configured channels"""
        tasks = []
        
        for channel in self.notification_channels:
            if not channel.enabled:
                continue
            
            # Check severity filter
            if (channel.severity_filter and 
                alert.severity not in channel.severity_filter):
                continue
            
            # Create notification task
            task = asyncio.create_task(
                self._send_to_channel(alert, channel)
            )
            tasks.append(task)
        
        # Wait for all notifications to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count failures
            failures = sum(1 for result in results if isinstance(result, Exception))
            if failures > 0:
                self.metrics["notification_failures"] += failures
                logger.warning(f"Failed to send {failures} notifications")
    
    async def _send_to_channel(self, alert: Alert, channel: NotificationChannel):
        """Send notification to a specific channel"""
        try:
            if channel.type == "slack":
                await self._send_slack_notification(alert, channel)
            elif channel.type == "pagerduty":
                await self._send_pagerduty_notification(alert, channel)
            elif channel.type == "email":
                await self._send_email_notification(alert, channel)
            elif channel.type == "webhook":
                await self._send_webhook_notification(alert, channel)
            else:
                logger.warning(f"Unknown channel type: {channel.type}")
                
        except Exception as e:
            logger.error(f"Failed to send notification to {channel.name}: {str(e)}")
            raise
    
    async def _send_slack_notification(self, alert: Alert, channel: NotificationChannel):
        """Send Slack notification"""
        webhook_url = channel.config["webhook_url"]
        
        # Determine color based on severity
        color_map = {
            AlertSeverity.INFO: "#36a64f",      # Green
            AlertSeverity.WARNING: "#ffcc00",   # Yellow
            AlertSeverity.CRITICAL: "#ff6600",  # Orange
            AlertSeverity.EMERGENCY: "#ff0000"  # Red
        }
        
        payload = {
            "attachments": [{
                "color": color_map.get(alert.severity, "#cccccc"),
                "title": f"{alert.severity.value.upper()}: {alert.title}",
                "text": alert.description,
                "fields": [
                    {"title": "Source", "value": alert.source, "short": True},
                    {"title": "Alert ID", "value": alert.id, "short": True},
                    {"title": "Time", "value": datetime.fromtimestamp(alert.timestamp).strftime("%Y-%m-%d %H:%M:%S"), "short": True}
                ],
                "footer": "SlackToDoc Monitoring",
                "ts": int(alert.timestamp)
            }]
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(webhook_url, json=payload, timeout=10.0)
            response.raise_for_status()
    
    async def _send_pagerduty_notification(self, alert: Alert, channel: NotificationChannel):
        """Send PagerDuty notification"""
        integration_key = channel.config["integration_key"]
        
        # Map severity to PagerDuty severity
        severity_map = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning", 
            AlertSeverity.CRITICAL: "error",
            AlertSeverity.EMERGENCY: "critical"
        }
        
        payload = {
            "routing_key": integration_key,
            "event_action": "trigger",
            "dedup_key": alert.id,
            "payload": {
                "summary": alert.title,
                "source": alert.source,
                "severity": severity_map.get(alert.severity, "error"),
                "component": "SlackToDoc",
                "group": "production",
                "class": "monitoring",
                "custom_details": {
                    "description": alert.description,
                    "alert_id": alert.id,
                    "metadata": alert.metadata
                }
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=10.0
            )
            response.raise_for_status()
    
    async def _send_email_notification(self, alert: Alert, channel: NotificationChannel):
        """Send email notification"""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        config = channel.config
        
        # Create email
        msg = MIMEMultipart()
        msg["From"] = config["from_email"]
        msg["To"] = ", ".join(config["to_emails"])
        msg["Subject"] = f"[{alert.severity.value.upper()}] SlackToDoc Alert: {alert.title}"
        
        body = f"""
SlackToDoc Production Alert

Alert ID: {alert.id}
Severity: {alert.severity.value.upper()}
Source: {alert.source}
Time: {datetime.fromtimestamp(alert.timestamp).strftime("%Y-%m-%d %H:%M:%S UTC")}

Description:
{alert.description}

Metadata:
{json.dumps(alert.metadata, indent=2) if alert.metadata else "None"}

---
This is an automated alert from SlackToDoc monitoring system.
        """
        
        msg.attach(MIMEText(body, "plain"))
        
        # Send email
        with smtplib.SMTP(config["smtp_server"], config["smtp_port"]) as server:
            server.starttls()
            server.login(config["username"], config["password"])
            server.send_message(msg)
    
    async def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel):
        """Send webhook notification"""
        webhook_url = channel.config["url"]
        
        payload = {
            "alert": asdict(alert),
            "timestamp": time.time(),
            "source": "slacktodoc_monitoring"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(webhook_url, json=payload, timeout=10.0)
            response.raise_for_status()
    
    async def _setup_escalation(self, alert: Alert):
        """Setup escalation for an alert"""
        rules = self.escalation_rules.get(alert.severity)
        if not rules or rules["max_escalations"] == 0:
            return
        
        # Create escalation task
        task = asyncio.create_task(self._escalation_worker(alert, rules))
        self.escalation_tasks[alert.id] = task
    
    async def _escalation_worker(self, alert: Alert, rules: Dict[str, Any]):
        """Worker that handles alert escalation"""
        try:
            # Initial delay
            if rules["initial_delay"] > 0:
                await asyncio.sleep(rules["initial_delay"])
            
            # Check if alert is still active
            if alert.id not in self.active_alerts or alert.status != AlertStatus.ACTIVE:
                return
            
            # Escalation loop
            for escalation_level, delay in enumerate(rules["escalation_delays"], 1):
                # Update escalation level
                alert.escalation_level = escalation_level
                
                # Send escalation notification
                await self._send_escalation_notification(alert)
                
                # Update metrics
                self.metrics["escalations_triggered"] += 1
                
                # Wait for next escalation
                if escalation_level < len(rules["escalation_delays"]):
                    await asyncio.sleep(delay)
                    
                    # Check if alert is still active
                    if alert.id not in self.active_alerts or alert.status != AlertStatus.ACTIVE:
                        return
                        
        except asyncio.CancelledError:
            logger.info(f"Escalation cancelled for alert: {alert.id}")
        except Exception as e:
            logger.error(f"Escalation error for alert {alert.id}: {str(e)}")
    
    async def _send_escalation_notification(self, alert: Alert):
        """Send escalation notification"""
        escalation_alert = Alert(
            id=f"{alert.id}_escalation_{alert.escalation_level}",
            title=f"ESCALATION {alert.escalation_level}: {alert.title}",
            description=f"Alert has been escalated to level {alert.escalation_level}.\n\nOriginal alert: {alert.description}",
            severity=alert.severity,
            status=AlertStatus.ACTIVE,
            source=f"{alert.source}_escalation",
            timestamp=time.time(),
            metadata=alert.metadata
        )
        
        await self._send_notifications(escalation_alert)
    
    async def _send_acknowledgment_notification(self, alert: Alert):
        """Send acknowledgment notification"""
        # Only send for critical and emergency alerts
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            ack_alert = Alert(
                id=f"{alert.id}_ack",
                title=f"ACKNOWLEDGED: {alert.title}",
                description=f"Alert has been acknowledged by {alert.acknowledged_by}",
                severity=AlertSeverity.INFO,
                status=AlertStatus.RESOLVED,
                source=f"{alert.source}_ack",
                timestamp=time.time()
            )
            
            await self._send_notifications(ack_alert)
    
    async def _send_resolution_notification(self, alert: Alert):
        """Send resolution notification"""
        # Only send for critical and emergency alerts
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            resolution_alert = Alert(
                id=f"{alert.id}_resolved",
                title=f"RESOLVED: {alert.title}",
                description=f"Alert has been resolved",
                severity=AlertSeverity.INFO,
                status=AlertStatus.RESOLVED,
                source=f"{alert.source}_resolved",
                timestamp=time.time()
            )
            
            await self._send_notifications(resolution_alert)
    
    async def start_background_tasks(self):
        """Start background maintenance tasks"""
        if not self.cleanup_task:
            self.cleanup_task = asyncio.create_task(self._cleanup_worker())
    
    async def stop_background_tasks(self):
        """Stop background tasks"""
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            self.cleanup_task = None
        
        # Cancel all escalation tasks
        for task in self.escalation_tasks.values():
            task.cancel()
        self.escalation_tasks.clear()
    
    async def _cleanup_worker(self):
        """Background worker for cleanup tasks"""
        while True:
            try:
                # Clean up old alert history (keep last 7 days)
                cutoff_time = time.time() - (7 * 24 * 60 * 60)
                self.alert_history = [alert for alert in self.alert_history 
                                    if alert.timestamp > cutoff_time]
                
                # Clean up resolved escalation tasks
                completed_tasks = [alert_id for alert_id, task in self.escalation_tasks.items() 
                                 if task.done()]
                for alert_id in completed_tasks:
                    del self.escalation_tasks[alert_id]
                
                # Check for suppressed alerts that should be unsuppressed
                current_time = time.time()
                for alert in self.active_alerts.values():
                    if (alert.status == AlertStatus.SUPPRESSED and 
                        alert.suppressed_until and 
                        current_time > alert.suppressed_until):
                        alert.status = AlertStatus.ACTIVE
                        alert.suppressed_until = None
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup worker error: {str(e)}")
                await asyncio.sleep(60)
    
    def get_alert_status(self) -> Dict[str, Any]:
        """Get current alert status"""
        return {
            "active_alerts": len(self.active_alerts),
            "critical_alerts": len([a for a in self.active_alerts.values() 
                                   if a.severity == AlertSeverity.CRITICAL]),
            "emergency_alerts": len([a for a in self.active_alerts.values() 
                                    if a.severity == AlertSeverity.EMERGENCY]),
            "acknowledged_alerts": len([a for a in self.active_alerts.values() 
                                       if a.status == AlertStatus.ACKNOWLEDGED]),
            "suppressed_alerts": len([a for a in self.active_alerts.values() 
                                     if a.status == AlertStatus.SUPPRESSED]),
            "metrics": self.metrics.copy(),
            "channels_configured": len(self.notification_channels),
            "escalations_running": len(self.escalation_tasks)
        }


# Global alert manager instance
alert_manager = AlertManager()