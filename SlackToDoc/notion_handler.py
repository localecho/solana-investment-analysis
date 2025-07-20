"""
Notion Integration Handler for SlackToDoc Production
Handles Notion OAuth, workspace connections, and document creation with optimization
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import httpx
from notion_client import AsyncClient
from notion_client.errors import APIResponseError, RequestTimeoutError

# Configure logging
logger = logging.getLogger(__name__)

class NotionHandler:
    """Production-ready Notion integration with enterprise features"""
    
    def __init__(self):
        self.client_id = os.getenv("NOTION_CLIENT_ID")
        self.client_secret = os.getenv("NOTION_CLIENT_SECRET")
        self.redirect_uri = os.getenv("NOTION_REDIRECT_URI", "https://slacktodoc.com/notion/oauth/callback")
        
        # Default client for non-OAuth operations
        self.default_token = os.getenv("NOTION_TOKEN")
        self.default_client = AsyncClient(auth=self.default_token) if self.default_token else None
        
        # Rate limiting configuration
        self.rate_limit_requests = 3  # Notion allows 3 requests per second
        self.rate_limit_window = 1.0  # 1 second window
        self.last_request_times = []
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # Start with 1 second delay
        
        # Template configurations
        self.document_templates = {
            "meeting_notes": self.get_meeting_notes_template(),
            "decision_record": self.get_decision_record_template(),
            "action_items": self.get_action_items_template(),
            "knowledge_base": self.get_knowledge_base_template()
        }
    
    async def generate_oauth_url(self, state: str, team_id: str) -> str:
        """Generate Notion OAuth authorization URL"""
        try:
            if not self.client_id:
                raise ValueError("Notion OAuth not configured - missing client_id")
            
            base_url = "https://api.notion.com/v1/oauth/authorize"
            params = {
                "client_id": self.client_id,
                "response_type": "code",
                "owner": "user",
                "redirect_uri": self.redirect_uri,
                "state": state
            }
            
            # Build URL manually for better control
            param_string = "&".join([f"{k}={v}" for k, v in params.items()])
            oauth_url = f"{base_url}?{param_string}"
            
            logger.info(f"Generated Notion OAuth URL for team: {team_id}")
            return oauth_url
            
        except Exception as e:
            logger.error(f"Failed to generate Notion OAuth URL: {str(e)}")
            raise
    
    async def handle_oauth_callback(self, code: str, state: str) -> Dict[str, Any]:
        """Handle Notion OAuth callback and store workspace connection"""
        try:
            # Exchange code for access token
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.notion.com/v1/oauth/token",
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json"
                    },
                    json={
                        "grant_type": "authorization_code",
                        "code": code,
                        "redirect_uri": self.redirect_uri
                    },
                    auth=(self.client_id, self.client_secret),
                    timeout=30.0
                )
            
            if response.status_code != 200:
                logger.error(f"Notion OAuth token exchange failed: {response.status_code}")
                raise Exception(f"OAuth token exchange failed: {response.status_code}")
            
            oauth_response = response.json()
            
            # Extract workspace information
            workspace_data = {
                "access_token": oauth_response["access_token"],
                "token_type": oauth_response.get("token_type", "bearer"),
                "bot_id": oauth_response["bot_id"],
                "workspace_id": oauth_response["workspace_id"],
                "workspace_name": oauth_response["workspace_name"],
                "workspace_icon": oauth_response.get("workspace_icon"),
                "owner": oauth_response["owner"],
                "duplicated_template_id": oauth_response.get("duplicated_template_id"),
                "request_id": oauth_response.get("request_id"),
                "connected_at": datetime.utcnow().isoformat()
            }
            
            # Create Notion client for this workspace
            notion_client = AsyncClient(auth=workspace_data["access_token"])
            
            # Verify connection and get additional workspace info
            await self.verify_workspace_connection(notion_client, workspace_data)
            
            # Store workspace connection
            await self.store_workspace_connection(workspace_data, state)
            
            logger.info(f"Successfully connected Notion workspace: {workspace_data['workspace_name']}")
            
            return {
                "success": True,
                "workspace_id": workspace_data["workspace_id"],
                "workspace_name": workspace_data["workspace_name"],
                "bot_id": workspace_data["bot_id"]
            }
            
        except Exception as e:
            logger.error(f"Notion OAuth callback error: {str(e)}")
            raise
    
    async def verify_workspace_connection(self, client: AsyncClient, workspace_data: Dict[str, Any]) -> None:
        """Verify Notion workspace connection and permissions"""
        try:
            # Test basic API access
            users_response = await self.rate_limited_request(client, "users", "list")
            
            if not users_response.get("results"):
                logger.warning(f"No users found in workspace: {workspace_data['workspace_id']}")
            
            # Test database creation permissions
            try:
                # Try to search for existing databases
                search_response = await self.rate_limited_request(
                    client, "search", 
                    filter={"property": "object", "value": "database"},
                    page_size=1
                )
                
                workspace_data["can_create_databases"] = True
                workspace_data["existing_databases"] = len(search_response.get("results", []))
                
            except APIResponseError as e:
                if "insufficient_permissions" in str(e):
                    workspace_data["can_create_databases"] = False
                    logger.warning(f"Limited permissions in workspace: {workspace_data['workspace_id']}")
                else:
                    raise
            
            logger.info(f"Verified Notion workspace connection: {workspace_data['workspace_id']}")
            
        except Exception as e:
            logger.error(f"Workspace verification failed: {str(e)}")
            raise Exception(f"Failed to verify Notion workspace: {str(e)}")
    
    async def store_workspace_connection(self, workspace_data: Dict[str, Any], state: str) -> None:
        """Store Notion workspace connection in database"""
        try:
            from database import get_database
            
            db = await get_database()
            
            # Extract team_id from state (assuming format: team_id:timestamp)
            team_id = state.split(":")[0] if ":" in state else state
            
            # Encrypt access token
            encrypted_token = self.encrypt_token(workspace_data["access_token"])
            
            # Store workspace connection
            query = """
            UPDATE workspaces 
            SET 
                notion_workspace_id = $1,
                notion_workspace_name = $2,
                notion_access_token = $3,
                notion_bot_id = $4,
                notion_connected_at = $5,
                notion_owner_info = $6,
                updated_at = CURRENT_TIMESTAMP
            WHERE slack_team_id = $7
            """
            
            await db.execute(
                query,
                workspace_data["workspace_id"],
                workspace_data["workspace_name"],
                encrypted_token,
                workspace_data["bot_id"],
                workspace_data["connected_at"],
                json.dumps(workspace_data["owner"]),
                team_id
            )
            
            # Create default database for team documentation
            await self.create_default_database(workspace_data["access_token"], team_id)
            
            logger.info(f"Stored Notion workspace connection for team: {team_id}")
            
        except Exception as e:
            logger.error(f"Failed to store workspace connection: {str(e)}")
            raise
    
    async def create_default_database(self, access_token: str, team_id: str) -> str:
        """Create default database for team documentation"""
        try:
            client = AsyncClient(auth=access_token)
            
            # Database properties schema
            database_properties = {
                "Title": {
                    "title": {}
                },
                "Type": {
                    "select": {
                        "options": [
                            {"name": "Meeting Notes", "color": "blue"},
                            {"name": "Decision Record", "color": "green"},
                            {"name": "Action Items", "color": "yellow"},
                            {"name": "Knowledge Base", "color": "purple"}
                        ]
                    }
                },
                "Channel": {
                    "rich_text": {}
                },
                "Date Created": {
                    "date": {}
                },
                "Participants": {
                    "multi_select": {
                        "options": []
                    }
                },
                "Status": {
                    "select": {
                        "options": [
                            {"name": "Draft", "color": "gray"},
                            {"name": "Review", "color": "yellow"},
                            {"name": "Approved", "color": "green"},
                            {"name": "Archived", "color": "red"}
                        ]
                    }
                },
                "Priority": {
                    "select": {
                        "options": [
                            {"name": "Low", "color": "gray"},
                            {"name": "Medium", "color": "yellow"},
                            {"name": "High", "color": "orange"},
                            {"name": "Critical", "color": "red"}
                        ]
                    }
                },
                "Tags": {
                    "multi_select": {
                        "options": []
                    }
                },
                "Slack Thread": {
                    "url": {}
                }
            }
            
            # Create database
            database_response = await self.rate_limited_request(
                client, "databases",
                parent={"type": "page_id", "page_id": await self.get_or_create_parent_page(client)},
                title=[
                    {
                        "type": "text",
                        "text": {"content": "SlackToDoc Knowledge Base"}
                    }
                ],
                properties=database_properties,
                description=[
                    {
                        "type": "text",
                        "text": {"content": "Automatically generated documentation from Slack conversations"}
                    }
                ]
            )
            
            database_id = database_response["id"]
            
            # Store database ID in workspace settings
            await self.update_workspace_database_id(team_id, database_id)
            
            logger.info(f"Created default database for team: {team_id}")
            return database_id
            
        except Exception as e:
            logger.error(f"Failed to create default database: {str(e)}")
            raise
    
    async def get_or_create_parent_page(self, client: AsyncClient) -> str:
        """Get or create a parent page for the database"""
        try:
            # Search for existing SlackToDoc page
            search_response = await self.rate_limited_request(
                client, "search",
                query="SlackToDoc",
                filter={"property": "object", "value": "page"}
            )
            
            # Check if we found an existing page
            for result in search_response.get("results", []):
                if "SlackToDoc" in result.get("properties", {}).get("title", {}).get("title", [{}])[0].get("text", {}).get("content", ""):
                    return result["id"]
            
            # Create new parent page
            page_response = await self.rate_limited_request(
                client, "pages",
                parent={"type": "workspace", "workspace": True},
                properties={
                    "title": {
                        "title": [
                            {
                                "type": "text",
                                "text": {"content": "SlackToDoc Documentation"}
                            }
                        ]
                    }
                },
                children=[
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {"content": "This page contains automatically generated documentation from Slack conversations."}
                                }
                            ]
                        }
                    }
                ]
            )
            
            return page_response["id"]
            
        except Exception as e:
            logger.error(f"Failed to get/create parent page: {str(e)}")
            # Fallback to workspace root
            return None
    
    async def create_notion_document(self, extracted_content: Dict[str, Any], team_id: str) -> Optional[str]:
        """Create a Notion document from extracted content"""
        try:
            # Get workspace connection
            workspace = await self.get_workspace_connection(team_id)
            if not workspace:
                logger.error(f"No Notion workspace connected for team: {team_id}")
                return None
            
            client = AsyncClient(auth=workspace["notion_access_token"])
            
            # Determine document type
            doc_type = self.determine_document_type(extracted_content)
            
            # Get appropriate template
            template = self.document_templates.get(doc_type, self.document_templates["knowledge_base"])
            
            # Build document properties
            properties = {
                "Title": {
                    "title": [
                        {
                            "type": "text",
                            "text": {"content": extracted_content.get("title", "Slack Conversation Summary")}
                        }
                    ]
                },
                "Type": {
                    "select": {"name": self.get_type_display_name(doc_type)}
                },
                "Channel": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {"content": extracted_content.get("channel_name", "Unknown")}
                        }
                    ]
                },
                "Date Created": {
                    "date": {"start": datetime.utcnow().isoformat()}
                },
                "Status": {
                    "select": {"name": "Draft"}
                },
                "Priority": {
                    "select": {"name": extracted_content.get("priority", "Medium")}
                },
                "Slack Thread": {
                    "url": extracted_content.get("thread_url", "")
                }
            }
            
            # Add participants if available
            if extracted_content.get("participants"):
                properties["Participants"] = {
                    "multi_select": [
                        {"name": participant} for participant in extracted_content["participants"][:10]  # Limit to 10
                    ]
                }
            
            # Add tags if available
            if extracted_content.get("tags"):
                properties["Tags"] = {
                    "multi_select": [
                        {"name": tag} for tag in extracted_content["tags"][:10]  # Limit to 10
                    ]
                }
            
            # Build document content blocks
            content_blocks = await self.build_content_blocks(extracted_content, template)
            
            # Create the page
            database_id = workspace.get("notion_database_id")
            if not database_id:
                # Create default database if it doesn't exist
                database_id = await self.create_default_database(workspace["notion_access_token"], team_id)
            
            page_response = await self.rate_limited_request(
                client, "pages",
                parent={"database_id": database_id},
                properties=properties,
                children=content_blocks
            )
            
            page_url = page_response["url"]
            
            # Log successful creation
            await self.log_document_creation(team_id, page_response["id"], extracted_content)
            
            logger.info(f"Created Notion document: {page_url}")
            return page_url
            
        except Exception as e:
            logger.error(f"Failed to create Notion document: {str(e)}")
            return None
    
    async def build_content_blocks(self, extracted_content: Dict[str, Any], template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build Notion content blocks from extracted content"""
        blocks = []
        
        # Add summary section
        if extracted_content.get("summary"):
            blocks.extend([
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"type": "text", "text": {"content": "📋 Summary"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": extracted_content["summary"]}}]
                    }
                },
                {"object": "block", "type": "divider", "divider": {}}
            ])
        
        # Add decisions section
        if extracted_content.get("decisions"):
            blocks.append({
                "object": "block",
                "type": "heading_2", 
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "✅ Decisions Made"}}]
                }
            })
            
            for decision in extracted_content["decisions"]:
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": decision}}]
                    }
                })
            
            blocks.append({"object": "block", "type": "divider", "divider": {}})
        
        # Add action items section
        if extracted_content.get("action_items"):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "🎯 Action Items"}}]
                }
            })
            
            for action in extracted_content["action_items"]:
                blocks.append({
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"type": "text", "text": {"content": action}}],
                        "checked": False
                    }
                })
            
            blocks.append({"object": "block", "type": "divider", "divider": {}})
        
        # Add key insights section
        if extracted_content.get("key_insights"):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "💡 Key Insights"}}]
                }
            })
            
            for insight in extracted_content["key_insights"]:
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": insight}}]
                    }
                })
            
            blocks.append({"object": "block", "type": "divider", "divider": {}})
        
        # Add metadata section
        blocks.extend([
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": "📊 Metadata"}}]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {"type": "text", "text": {"content": f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"}},
                        {"type": "text", "text": {"content": f"**Participants:** {', '.join(extracted_content.get('participants', ['Unknown']))}\n"}},
                        {"type": "text", "text": {"content": f"**Channel:** {extracted_content.get('channel_name', 'Unknown')}\n"}},
                        {"type": "text", "text": {"content": f"**AI Confidence:** {extracted_content.get('confidence_score', 0.8):.1%}"}}
                    ]
                }
            }
        ])
        
        return blocks
    
    async def rate_limited_request(self, client: AsyncClient, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make rate-limited request to Notion API"""
        # Implement rate limiting
        now = datetime.utcnow().timestamp()
        
        # Remove old timestamps outside the window
        self.last_request_times = [
            timestamp for timestamp in self.last_request_times 
            if now - timestamp < self.rate_limit_window
        ]
        
        # Wait if we've hit the rate limit
        if len(self.last_request_times) >= self.rate_limit_requests:
            sleep_time = self.rate_limit_window - (now - self.last_request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        # Record this request
        self.last_request_times.append(now)
        
        # Make the request with retries
        for attempt in range(self.max_retries):
            try:
                if endpoint == "pages":
                    return await client.pages.create(**kwargs)
                elif endpoint == "databases":
                    return await client.databases.create(**kwargs)
                elif endpoint == "search":
                    return await client.search(**kwargs)
                elif endpoint == "users":
                    if kwargs.get("action") == "list":
                        return await client.users.list()
                else:
                    raise ValueError(f"Unknown endpoint: {endpoint}")
                    
            except (RequestTimeoutError, APIResponseError) as e:
                if attempt == self.max_retries - 1:
                    raise
                
                # Exponential backoff
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {delay}s: {str(e)}")
                await asyncio.sleep(delay)
    
    def determine_document_type(self, content: Dict[str, Any]) -> str:
        """Determine the type of document based on content"""
        if content.get("decisions") and len(content["decisions"]) > 0:
            return "decision_record"
        elif content.get("action_items") and len(content["action_items"]) > 2:
            return "action_items"
        elif "meeting" in content.get("title", "").lower():
            return "meeting_notes"
        else:
            return "knowledge_base"
    
    def get_type_display_name(self, doc_type: str) -> str:
        """Get display name for document type"""
        type_names = {
            "meeting_notes": "Meeting Notes",
            "decision_record": "Decision Record", 
            "action_items": "Action Items",
            "knowledge_base": "Knowledge Base"
        }
        return type_names.get(doc_type, "Knowledge Base")
    
    def get_meeting_notes_template(self) -> Dict[str, Any]:
        """Get template for meeting notes"""
        return {
            "sections": ["summary", "decisions", "action_items", "key_insights"],
            "priority": "medium"
        }
    
    def get_decision_record_template(self) -> Dict[str, Any]:
        """Get template for decision records"""
        return {
            "sections": ["summary", "decisions", "key_insights"],
            "priority": "high"
        }
    
    def get_action_items_template(self) -> Dict[str, Any]:
        """Get template for action items"""
        return {
            "sections": ["summary", "action_items"],
            "priority": "high"
        }
    
    def get_knowledge_base_template(self) -> Dict[str, Any]:
        """Get template for knowledge base entries"""
        return {
            "sections": ["summary", "key_insights"],
            "priority": "medium"
        }
    
    async def get_workspace_connection(self, team_id: str) -> Optional[Dict[str, Any]]:
        """Get Notion workspace connection for a team"""
        try:
            from database import get_database
            
            db = await get_database()
            
            query = """
            SELECT * FROM workspaces 
            WHERE slack_team_id = $1 AND notion_workspace_id IS NOT NULL
            """
            
            row = await db.fetchrow(query, team_id)
            
            if not row:
                return None
            
            workspace_data = dict(row)
            
            # Decrypt access token
            if workspace_data.get("notion_access_token"):
                workspace_data["notion_access_token"] = self.decrypt_token(workspace_data["notion_access_token"])
            
            return workspace_data
            
        except Exception as e:
            logger.error(f"Failed to get workspace connection: {str(e)}")
            return None
    
    async def update_workspace_database_id(self, team_id: str, database_id: str) -> None:
        """Update workspace with default database ID"""
        try:
            from database import get_database
            
            db = await get_database()
            
            query = """
            UPDATE workspaces 
            SET notion_database_id = $1, updated_at = CURRENT_TIMESTAMP
            WHERE slack_team_id = $2
            """
            
            await db.execute(query, database_id, team_id)
            
        except Exception as e:
            logger.error(f"Failed to update database ID: {str(e)}")
    
    async def log_document_creation(self, team_id: str, page_id: str, content: Dict[str, Any]) -> None:
        """Log document creation for analytics"""
        try:
            from database import get_database
            
            db = await get_database()
            
            query = """
            INSERT INTO documents (
                workspace_id, notion_page_id, title, document_type,
                content, is_published, published_at
            )
            SELECT 
                w.id, $2, $3, $4, $5, true, CURRENT_TIMESTAMP
            FROM workspaces w
            WHERE w.slack_team_id = $1
            """
            
            await db.execute(
                query,
                team_id,
                page_id,
                content.get("title", "Untitled"),
                self.determine_document_type(content),
                json.dumps(content)
            )
            
        except Exception as e:
            logger.error(f"Failed to log document creation: {str(e)}")
    
    def encrypt_token(self, token: str) -> str:
        """Encrypt token for secure storage"""
        try:
            import cryptography.fernet as fernet
            
            encryption_key = os.getenv("ENCRYPTION_KEY")
            if not encryption_key:
                logger.warning("No encryption key found, storing token in plain text")
                return token
            
            f = fernet.Fernet(encryption_key.encode())
            encrypted_token = f.encrypt(token.encode())
            return encrypted_token.decode()
            
        except Exception as e:
            logger.error(f"Token encryption failed: {str(e)}")
            return token
    
    def decrypt_token(self, encrypted_token: str) -> str:
        """Decrypt token from storage"""
        try:
            import cryptography.fernet as fernet
            
            encryption_key = os.getenv("ENCRYPTION_KEY")
            if not encryption_key:
                return encrypted_token
            
            f = fernet.Fernet(encryption_key.encode())
            decrypted_token = f.decrypt(encrypted_token.encode())
            return decrypted_token.decode()
            
        except Exception as e:
            logger.error(f"Token decryption failed: {str(e)}")
            return encrypted_token


# Notion handler instance
notion_handler = NotionHandler()