"""
SlackToDoc - Transform Slack conversations into organized Notion documentation

This FastAPI application provides the core backend for SlackToDoc, handling:
- Slack webhook events and bot interactions
- Message processing and content extraction
- Notion document creation and synchronization
- Configuration management and user authentication
"""

import os
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, validator
import structlog

# Slack SDK imports
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.signature import SignatureVerifier
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

# Notion client
from notion_client import AsyncClient as NotionClient

# OpenAI for content processing
import openai

# Configuration and security
import yaml
from jose import JWTError, jwt
import redis.asyncio as redis
from datetime import timedelta
import html

# Import Sprint 2 production components
from slack_handler import slack_handler
from notion_handler import notion_handler
from ai_fallback_handler import fallback_handler
from monitoring_dashboard import monitoring_dashboard
from alert_manager import alert_manager, AlertSeverity
from beta_signup_handler import beta_signup_handler
from openai_rate_limiter import rate_limiter

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Slack configuration
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

# Notion configuration
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Data Models
class SlackMessage(BaseModel):
    """Represents a processed Slack message"""
    user_id: str = Field(..., description="Slack user ID")
    channel_id: str = Field(..., description="Slack channel ID")
    thread_ts: Optional[str] = Field(None, description="Thread timestamp if part of thread")
    text: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    
    @validator('text')
    def validate_text_content(cls, v):
        if len(v) > 40000:  # Slack's max message length
            raise ValueError('Message too long')
        return html.escape(v)  # Prevent XSS


class ExtractedContent(BaseModel):
    """Represents content extracted from Slack conversation"""
    decisions: List[str] = Field(default_factory=list, description="Key decisions made")
    action_items: List[str] = Field(default_factory=list, description="Action items identified")
    key_insights: List[str] = Field(default_factory=list, description="Important insights")
    participants: List[str] = Field(default_factory=list, description="Conversation participants")
    topic: str = Field(..., description="Main conversation topic")
    summary: str = Field(..., description="Conversation summary")
    priority: int = Field(default=5, ge=1, le=10, description="Priority level 1-10")


class NotionDocument(BaseModel):
    """Represents a Notion document to be created"""
    title: str = Field(..., description="Document title")
    channel_name: str = Field(..., description="Source Slack channel")
    date_created: datetime = Field(default_factory=datetime.utcnow)
    participants: List[str] = Field(default_factory=list)
    summary: str = Field(..., description="Document summary")
    decisions: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    key_insights: List[str] = Field(default_factory=list)
    original_thread_url: str = Field(..., description="Link to original Slack thread")
    tags: List[str] = Field(default_factory=list)


class ConfigurationUpdate(BaseModel):
    """Configuration update request"""
    monitored_channels: Optional[List[str]] = None
    trigger_keywords: Optional[List[str]] = None
    notion_template: Optional[Dict[str, Any]] = None
    ai_model: Optional[str] = None


class SlackEventRequest(BaseModel):
    """Slack event webhook request"""
    token: str
    team_id: str
    api_app_id: str
    event: Dict[str, Any]
    type: str
    event_id: str
    event_time: int


# Global application state
class AppState:
    """Manages application-wide state and connections"""
    def __init__(self):
        self.slack_client: Optional[AsyncWebClient] = None
        self.notion_client: Optional[NotionClient] = None
        self.redis_client: Optional[redis.Redis] = None
        self.slack_app: Optional[AsyncApp] = None
        self.config: Dict[str, Any] = {}
        self.startup_time: datetime = datetime.utcnow()
        self.is_production_ready: bool = False
        
    async def initialize(self):
        """Initialize all external connections and production components"""
        try:
            logger.info("🚀 Initializing SlackToDoc Production System...")
            
            # Initialize Slack client
            if SLACK_BOT_TOKEN:
                self.slack_client = AsyncWebClient(token=SLACK_BOT_TOKEN)
                slack_handler.slack_client = self.slack_client
                logger.info("✅ Slack client initialized")
            
            # Initialize Notion client
            if NOTION_TOKEN:
                self.notion_client = NotionClient(auth=NOTION_TOKEN)
                notion_handler.notion_client = self.notion_client
                logger.info("✅ Notion client initialized")
            
            # Initialize Redis client
            self.redis_client = redis.from_url(REDIS_URL)
            await self.redis_client.ping()
            logger.info("✅ Redis client initialized")
            
            # Initialize OpenAI with rate limiting
            if OPENAI_API_KEY:
                openai.api_key = OPENAI_API_KEY
                logger.info("✅ OpenAI client initialized with rate limiting")
            
            # Initialize monitoring dashboard
            await monitoring_dashboard.initialize()
            logger.info("✅ Monitoring dashboard initialized")
            
            # Start alert manager background tasks
            await alert_manager.start_background_tasks()
            logger.info("✅ Alert manager started")
            
            # Load configuration
            await self.load_configuration()
            
            # Mark as production ready
            self.is_production_ready = True
            
            # Send startup alert
            await alert_manager.trigger_alert(
                "SlackToDoc System Started",
                f"SlackToDoc production system successfully started in {ENVIRONMENT} environment",
                AlertSeverity.INFO,
                "system_startup"
            )
            
            logger.info("🎉 SlackToDoc Production System Ready!")
            
        except Exception as e:
            logger.error("Failed to initialize application state", error=str(e))
            # Send critical alert
            await alert_manager.trigger_alert(
                "SlackToDoc Startup Failed",
                f"Critical error during system initialization: {str(e)}",
                AlertSeverity.EMERGENCY,
                "system_startup"
            )
            raise
    
    async def cleanup(self):
        """Cleanup connections and production components"""
        try:
            logger.info("🧹 Shutting down SlackToDoc Production System...")
            
            # Stop alert manager background tasks
            await alert_manager.stop_background_tasks()
            
            # Send shutdown alert
            await alert_manager.trigger_alert(
                "SlackToDoc System Shutdown",
                "SlackToDoc production system is shutting down gracefully",
                AlertSeverity.INFO,
                "system_shutdown"
            )
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("✅ SlackToDoc Production System shutdown complete")
            
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))
    
    async def load_configuration(self):
        """Load configuration from file and environment"""
        try:
            # Load default configuration
            default_config = {
                "slack": {
                    "monitored_channels": [],
                    "trigger_keywords": ["decision", "action item", "important", "todo"]
                },
                "notion": {
                    "workspace_id": os.getenv("NOTION_WORKSPACE_ID"),
                    "database_name": "Team Knowledge Base"
                },
                "processing": {
                    "min_message_count": 3,
                    "max_thread_age_hours": 24,
                    "ai_model": "gpt-4"
                },
                "security": {
                    "rate_limit_per_minute": 60,
                    "max_message_length": 40000
                }
            }
            
            # Try to load from config file
            config_path = "config.yaml"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    default_config.update(file_config)
            
            self.config = default_config
            logger.info("Configuration loaded", config_keys=list(self.config.keys()))
            
        except Exception as e:
            logger.error("Failed to load configuration", error=str(e))
            self.config = {}


# Global app state
app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    await app_state.initialize()
    logger.info("SlackToDoc application started")
    yield
    # Shutdown
    await app_state.cleanup()
    logger.info("SlackToDoc application stopped")


# Create FastAPI application
app = FastAPI(
    title="SlackToDoc API",
    description="Transform Slack conversations into organized Notion documentation",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ENVIRONMENT == "development" else ["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if ENVIRONMENT == "development" else ["your-domain.com"]
)

# Security
security = HTTPBearer()


# Authentication utilities
def create_jwt_token(data: Dict[str, Any]) -> str:
    """Create JWT token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")


# Content processing utilities
async def extract_content_from_messages(messages: List[SlackMessage]) -> ExtractedContent:
    """Extract structured content from Slack messages using AI with fallback"""
    start_time = time.time()
    
    try:
        # Record processing start
        monitoring_dashboard.record_slack_event()
        
        # Try to acquire OpenAI rate limit slot
        estimated_tokens = len(str(messages)) // 4  # Rough token estimate
        can_use_ai = await rate_limiter.acquire_request_slot(estimated_tokens, "gpt-4-turbo-preview")
        
        if can_use_ai:
            try:
                # Use AI extraction
                extracted = await slack_handler.extract_conversation_content(messages)
                
                # Record successful AI processing
                processing_time = time.time() - start_time
                await rate_limiter.release_request_slot(True, estimated_tokens, processing_time, "gpt-4-turbo-preview")
                monitoring_dashboard.record_openai_request(estimated_tokens, estimated_tokens * 0.00001)  # Rough cost
                
            except Exception as ai_error:
                logger.warning(f"AI extraction failed, using fallback: {str(ai_error)}")
                
                # Record AI failure and use fallback
                await rate_limiter.release_request_slot(False, 0, time.time() - start_time, "gpt-4-turbo-preview")
                
                # Use fallback handler
                message_dicts = [{
                    "user": msg.user_id,
                    "text": msg.text,
                    "ts": msg.timestamp.timestamp()
                } for msg in messages]
                
                fallback_result = await fallback_handler.extract_content_fallback(message_dicts)
                
                if fallback_result:
                    extracted = ExtractedContent(
                        decisions=fallback_result.get("decisions", []),
                        action_items=fallback_result.get("action_items", []),
                        key_insights=fallback_result.get("key_insights", []),
                        topic=fallback_result.get("title", "Conversation"),
                        summary=fallback_result.get("summary", "Summary not available"),
                        priority=5
                    )
                else:
                    raise Exception("Both AI and fallback extraction failed")
        else:
            logger.info("Rate limit reached, using fallback extraction")
            
            # Use fallback handler directly
            message_dicts = [{
                "user": msg.user_id,
                "text": msg.text,
                "ts": msg.timestamp.timestamp()
            } for msg in messages]
            
            fallback_result = await fallback_handler.extract_content_fallback(message_dicts)
            
            if fallback_result:
                extracted = ExtractedContent(
                    decisions=fallback_result.get("decisions", []),
                    action_items=fallback_result.get("action_items", []),
                    key_insights=fallback_result.get("key_insights", []),
                    topic=fallback_result.get("title", "Conversation"),
                    summary=fallback_result.get("summary", "Summary not available"),
                    priority=5
                )
            else:
                raise Exception("Fallback extraction failed")
        
        # Record successful processing
        processing_time = time.time() - start_time
        monitoring_dashboard.record_request(processing_time, True)
        
        logger.info(
            "Content extracted from messages",
            message_count=len(messages),
            decisions_count=len(extracted.decisions),
            action_items_count=len(extracted.action_items),
            processing_time=processing_time
        )
        
        return extracted
        
    except Exception as e:
        # Record failed processing
        processing_time = time.time() - start_time
        monitoring_dashboard.record_request(processing_time, False)
        
        # Send alert for extraction failures
        await alert_manager.trigger_alert(
            "Content Extraction Failed",
            f"Failed to extract content from {len(messages)} messages: {str(e)}",
            AlertSeverity.WARNING,
            "content_extraction"
        )
        
        logger.error("Failed to extract content from messages", error=str(e))
        raise HTTPException(status_code=500, detail="Content extraction failed")


async def create_notion_page(content: ExtractedContent, channel_name: str, thread_url: str) -> str:
    """Create a Notion page from extracted content with monitoring"""
    try:
        # Record Notion API call
        monitoring_dashboard.record_notion_api_call()
        
        # Use notion_handler for consistent document creation
        document_data = {
            "title": content.topic,
            "summary": content.summary,
            "decisions": content.decisions,
            "action_items": content.action_items,
            "key_insights": content.key_insights,
            "participants": content.participants,
            "priority": content.priority,
            "original_thread_url": thread_url
        }
        
        result = await notion_handler.create_document(document_data, channel_name)
        
        if result and "page_id" in result:
            # Record successful document creation
            monitoring_dashboard.record_document_creation()
            
            logger.info(
                "Notion page created successfully",
                page_id=result["page_id"],
                channel=channel_name,
                topic=content.topic
            )
            
            return result["page_id"]
        else:
            raise Exception("Document creation returned invalid result")
        
    except Exception as e:
        # Send alert for Notion failures
        await alert_manager.trigger_alert(
            "Notion Document Creation Failed",
            f"Failed to create Notion document for channel {channel_name}: {str(e)}",
            AlertSeverity.CRITICAL,
            "notion_creation"
        )
        
        logger.error("Failed to create Notion page", error=str(e))
        raise HTTPException(status_code=500, detail="Notion page creation failed")


# API Routes

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint with monitoring integration"""
    try:
        # Get health status from monitoring dashboard
        health_status = await monitoring_dashboard.perform_health_checks()
        
        # Get system metrics
        system_metrics = await monitoring_dashboard.collect_system_metrics()
        app_metrics = await monitoring_dashboard.collect_application_metrics()
        
        # Get alert status
        alert_status = alert_manager.get_alert_status()
        
        # Get rate limiter status
        rate_limiter_stats = rate_limiter.get_current_stats()
        
        # Determine overall health
        is_healthy = (
            health_status.status in ["healthy", "degraded"] and
            app_state.is_production_ready and
            alert_status["emergency_alerts"] == 0
        )
        
        response_data = {
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": datetime.utcnow(),
            "version": "2.0.0",
            "environment": ENVIRONMENT,
            "uptime_seconds": (datetime.utcnow() - app_state.startup_time).total_seconds(),
            "production_ready": app_state.is_production_ready,
            "health_checks": health_status.checks,
            "system_metrics": {
                "cpu_percent": system_metrics.cpu_percent,
                "memory_percent": system_metrics.memory_percent,
                "disk_percent": system_metrics.disk_percent
            },
            "application_metrics": {
                "documents_created": app_metrics.documents_created_total,
                "error_rate": app_metrics.error_rate,
                "response_time_avg": app_metrics.response_time_avg
            },
            "alerts": {
                "active_alerts": alert_status["active_alerts"],
                "critical_alerts": alert_status["critical_alerts"],
                "emergency_alerts": alert_status["emergency_alerts"]
            },
            "rate_limiter": {
                "requests_last_minute": rate_limiter_stats["requests_last_minute"],
                "requests_limit": rate_limiter_stats["requests_limit"],
                "circuit_breaker_open": rate_limiter_stats["circuit_breaker_open"]
            }
        }
        
        # Return appropriate HTTP status
        status_code = 200 if is_healthy else 503
        return JSONResponse(content=response_data, status_code=status_code)
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": "Health check system failure",
                "timestamp": datetime.utcnow()
            },
            status_code=503
        )


@app.post("/slack/events")
async def handle_slack_events(request: Request, background_tasks: BackgroundTasks):
    """Handle Slack webhook events"""
    try:
        # Verify Slack signature
        body = await request.body()
        signature_verifier = SignatureVerifier(SLACK_SIGNING_SECRET)
        
        if not signature_verifier.is_valid_request(body, request.headers):
            raise HTTPException(status_code=401, detail="Invalid Slack signature")
        
        # Parse event data
        event_data = await request.json()
        
        # Handle URL verification
        if event_data.get("type") == "url_verification":
            return {"challenge": event_data.get("challenge")}
        
        # Process event in background
        if event_data.get("type") == "event_callback":
            background_tasks.add_task(process_slack_event, event_data["event"])
        
        return {"status": "ok"}
        
    except Exception as e:
        logger.error("Failed to handle Slack event", error=str(e))
        raise HTTPException(status_code=500, detail="Event processing failed")


async def process_slack_event(event: Dict[str, Any]):
    """Process a Slack event in the background"""
    try:
        event_type = event.get("type")
        
        if event_type == "app_mention":
            await handle_app_mention(event)
        elif event_type == "message":
            await handle_message(event)
        
    except Exception as e:
        logger.error("Failed to process Slack event", event_type=event.get("type"), error=str(e))


async def handle_app_mention(event: Dict[str, Any]):
    """Handle @SlackToDoc mentions"""
    try:
        channel_id = event["channel"]
        thread_ts = event.get("thread_ts", event["ts"])
        
        # Get thread messages
        response = await app_state.slack_client.conversations_replies(
            channel=channel_id,
            ts=thread_ts
        )
        
        # Convert to SlackMessage objects
        messages = []
        for msg in response["messages"]:
            messages.append(SlackMessage(
                user_id=msg["user"],
                channel_id=channel_id,
                thread_ts=thread_ts,
                text=msg["text"],
                timestamp=datetime.fromtimestamp(float(msg["ts"]))
            ))
        
        # Extract content
        content = await extract_content_from_messages(messages)
        
        # Get channel info
        channel_info = await app_state.slack_client.conversations_info(channel=channel_id)
        channel_name = channel_info["channel"]["name"]
        
        # Create thread URL
        team_info = await app_state.slack_client.team_info()
        team_domain = team_info["team"]["domain"]
        thread_url = f"https://{team_domain}.slack.com/archives/{channel_id}/p{thread_ts.replace('.', '')}"
        
        # Create Notion page
        page_id = await create_notion_page(content, channel_name, thread_url)
        
        # Track user activity if they're a beta user
        # Note: In production, you'd have user mapping logic here
        
        # Send confirmation message
        await app_state.slack_client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=f"✅ Documentation created! View it here: https://notion.so/{page_id}\n\n📊 Performance: Processed {len(messages)} messages in {time.time() - time.time():.2f}s"
        )
        
        logger.info(
            "Successfully processed app mention",
            channel_id=channel_id,
            thread_ts=thread_ts,
            notion_page_id=page_id
        )
        
    except Exception as e:
        logger.error("Failed to handle app mention", error=str(e))


async def handle_message(event: Dict[str, Any]):
    """Handle regular messages for keyword detection"""
    # TODO: Implement keyword-based triggering
    pass


@app.get("/config")
async def get_configuration(user: Dict[str, Any] = Depends(verify_jwt_token)):
    """Get current configuration"""
    return app_state.config


@app.put("/config")
async def update_configuration(
    config_update: ConfigurationUpdate,
    user: Dict[str, Any] = Depends(verify_jwt_token)
):
    """Update configuration"""
    try:
        # Update configuration
        if config_update.monitored_channels is not None:
            app_state.config["slack"]["monitored_channels"] = config_update.monitored_channels
        
        if config_update.trigger_keywords is not None:
            app_state.config["slack"]["trigger_keywords"] = config_update.trigger_keywords
        
        if config_update.ai_model is not None:
            app_state.config["processing"]["ai_model"] = config_update.ai_model
        
        # Save to file (in production, use database)
        with open("config.yaml", "w") as f:
            yaml.dump(app_state.config, f)
        
        logger.info("Configuration updated", user_id=user.get("sub"))
        
        return {"status": "updated", "config": app_state.config}
        
    except Exception as e:
        logger.error("Failed to update configuration", error=str(e))
        raise HTTPException(status_code=500, detail="Configuration update failed")


@app.post("/notion/sync")
async def manual_notion_sync(
    channel_id: str,
    thread_ts: str,
    user: Dict[str, Any] = Depends(verify_jwt_token)
):
    """Manually trigger Notion synchronization"""
    try:
        # Get thread messages
        response = await app_state.slack_client.conversations_replies(
            channel=channel_id,
            ts=thread_ts
        )
        
        # Process messages and create documentation
        messages = []
        for msg in response["messages"]:
            messages.append(SlackMessage(
                user_id=msg["user"],
                channel_id=channel_id,
                thread_ts=thread_ts,
                text=msg["text"],
                timestamp=datetime.fromtimestamp(float(msg["ts"]))
            ))
        
        content = await extract_content_from_messages(messages)
        
        # Get channel info
        channel_info = await app_state.slack_client.conversations_info(channel=channel_id)
        channel_name = channel_info["channel"]["name"]
        
        # Create thread URL
        team_info = await app_state.slack_client.team_info()
        team_domain = team_info["team"]["domain"]
        thread_url = f"https://{team_domain}.slack.com/archives/{channel_id}/p{thread_ts.replace('.', '')}"
        
        # Create Notion page
        page_id = await create_notion_page(content, channel_name, thread_url)
        
        return {
            "status": "synced",
            "notion_page_id": page_id,
            "content": content.dict()
        }
        
    except Exception as e:
        logger.error("Manual sync failed", error=str(e))
        raise HTTPException(status_code=500, detail="Sync failed")


# Production monitoring and beta endpoints

@app.get("/metrics")
async def get_metrics():
    """Get comprehensive system metrics"""
    try:
        dashboard_data = await monitoring_dashboard.get_dashboard_data()
        return dashboard_data
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Metrics unavailable")


@app.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """Get metrics in Prometheus format"""
    try:
        prometheus_metrics = await monitoring_dashboard.export_metrics("prometheus")
        return Response(content=prometheus_metrics, media_type="text/plain")
    except Exception as e:
        logger.error("Failed to get Prometheus metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Prometheus metrics unavailable")


@app.get("/alerts")
async def get_alerts():
    """Get current alert status"""
    try:
        alert_status = alert_manager.get_alert_status()
        return alert_status
    except Exception as e:
        logger.error("Failed to get alerts", error=str(e))
        raise HTTPException(status_code=500, detail="Alert status unavailable")


@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, user: Dict[str, Any] = Depends(verify_jwt_token)):
    """Acknowledge an alert"""
    try:
        success = await alert_manager.acknowledge_alert(alert_id, user.get("sub", "unknown"))
        if success:
            return {"status": "acknowledged", "alert_id": alert_id}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    except Exception as e:
        logger.error("Failed to acknowledge alert", error=str(e))
        raise HTTPException(status_code=500, detail="Alert acknowledgment failed")


@app.post("/beta/signup")
async def beta_signup(email: str, name: str, company: str, team_size: int, use_case: str):
    """Beta user signup endpoint"""
    try:
        result = await beta_signup_handler.signup_beta_user(
            email=email,
            name=name,
            company=company,
            team_size=team_size,
            use_case=use_case
        )
        return result
    except Exception as e:
        logger.error("Beta signup failed", error=str(e))
        raise HTTPException(status_code=500, detail="Beta signup failed")


@app.get("/beta/stats")
async def get_beta_stats(user: Dict[str, Any] = Depends(verify_jwt_token)):
    """Get beta program statistics"""
    try:
        stats = beta_signup_handler.get_beta_stats()
        return stats
    except Exception as e:
        logger.error("Failed to get beta stats", error=str(e))
        raise HTTPException(status_code=500, detail="Beta stats unavailable")


@app.post("/beta/users/{user_id}/feedback")
async def submit_beta_feedback(user_id: str, feedback_type: str, rating: int, comments: str):
    """Submit beta user feedback"""
    try:
        success = await beta_signup_handler.submit_feedback(user_id, feedback_type, rating, comments)
        if success:
            return {"status": "submitted", "user_id": user_id}
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.error("Failed to submit feedback", error=str(e))
        raise HTTPException(status_code=500, detail="Feedback submission failed")


@app.get("/")
async def root():
    """Root endpoint with production system info"""
    uptime = (datetime.utcnow() - app_state.startup_time).total_seconds()
    return {
        "message": "SlackToDoc Production API",
        "version": "2.0.0",
        "environment": ENVIRONMENT,
        "uptime_seconds": uptime,
        "production_ready": app_state.is_production_ready,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
            "alerts": "/alerts",
            "beta_signup": "/beta/signup"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=ENVIRONMENT == "development",
        log_level=LOG_LEVEL.lower()
    )