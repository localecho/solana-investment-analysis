"""
Slack OAuth Handler for SlackToDoc Production Deployment
Handles OAuth flow, token management, and workspace installation
"""

import os
import json
import asyncio
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import jwt
import httpx
from fastapi import HTTPException, Request
from slack_sdk.oauth import AuthorizeUrlGenerator, OAuthStateUtils
from slack_sdk.oauth.installation_store import Installation
from slack_sdk.oauth.state_store import OAuthStateStore
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError

# Configure logging
logger = logging.getLogger(__name__)

class SlackOAuthHandler:
    """Production-ready Slack OAuth handler with enterprise security"""
    
    def __init__(self):
        self.client_id = os.getenv("SLACK_CLIENT_ID")
        self.client_secret = os.getenv("SLACK_CLIENT_SECRET")
        self.signing_secret = os.getenv("SLACK_SIGNING_SECRET")
        self.redirect_uri = os.getenv("SLACK_REDIRECT_URI", "https://slacktodoc.com/slack/oauth/callback")
        
        # Validate required configuration
        if not all([self.client_id, self.client_secret, self.signing_secret]):
            raise ValueError("Missing required Slack OAuth configuration")
        
        # OAuth components
        self.authorize_url_generator = AuthorizeUrlGenerator(
            client_id=self.client_id,
            scopes=[
                "app_mentions:read",
                "channels:history",
                "channels:read", 
                "chat:write",
                "chat:write.public",
                "commands",
                "files:read",
                "groups:history",
                "groups:read",
                "im:history",
                "im:read",
                "im:write",
                "links:read",
                "links:write",
                "mpim:history",
                "mpim:read",
                "mpim:write",
                "reactions:read",
                "reactions:write",
                "team:read",
                "users:read",
                "users:read.email"
            ],
            user_scopes=[
                "channels:history",
                "groups:history",
                "im:history",
                "mpim:history"
            ]
        )
        
        # State management for security
        self.state_store = OAuthStateStore(expiration_seconds=600)  # 10 minutes
        self.state_utils = OAuthStateUtils()
        
    async def generate_install_url(self, team_id: Optional[str] = None) -> str:
        """Generate secure installation URL with state verification"""
        try:
            # Generate cryptographically secure state
            state = self.state_utils.build_authorize_url_state()
            
            # Store state for verification
            await self.state_store.async_issue(state)
            
            # Generate authorization URL
            url = self.authorize_url_generator.generate(
                state=state,
                redirect_uri=self.redirect_uri,
                team=team_id
            )
            
            logger.info(f"Generated install URL with state: {state[:10]}...")
            return url
            
        except Exception as e:
            logger.error(f"Failed to generate install URL: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate installation URL")
    
    async def handle_oauth_callback(self, code: str, state: str) -> Dict[str, Any]:
        """Handle OAuth callback with comprehensive security validation"""
        try:
            # Verify state to prevent CSRF attacks
            if not await self.state_store.async_consume(state):
                logger.warning(f"Invalid OAuth state received: {state[:10]}...")
                raise HTTPException(status_code=400, detail="Invalid OAuth state")
            
            # Exchange code for tokens
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://slack.com/api/oauth.v2.access",
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded"
                    },
                    data={
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "code": code,
                        "redirect_uri": self.redirect_uri
                    },
                    timeout=30.0
                )
            
            if response.status_code != 200:
                logger.error(f"OAuth token exchange failed: {response.status_code}")
                raise HTTPException(status_code=400, detail="OAuth token exchange failed")
            
            oauth_response = response.json()
            
            if not oauth_response.get("ok"):
                error = oauth_response.get("error", "unknown_error")
                logger.error(f"Slack OAuth error: {error}")
                raise HTTPException(status_code=400, detail=f"Slack OAuth error: {error}")
            
            # Extract installation data
            installation_data = {
                "app_id": oauth_response.get("app_id"),
                "enterprise_id": oauth_response.get("enterprise", {}).get("id"),
                "team_id": oauth_response["team"]["id"],
                "team_name": oauth_response["team"]["name"],
                "bot_token": oauth_response["access_token"],
                "bot_id": oauth_response["bot_user_id"],
                "bot_user_id": oauth_response["bot_user_id"],
                "is_enterprise_install": oauth_response.get("is_enterprise_install", False),
                "installed_at": datetime.utcnow().isoformat(),
                "installer_user_id": oauth_response.get("authed_user", {}).get("id"),
                "scope": oauth_response.get("scope", ""),
                "token_type": oauth_response.get("token_type", "bot")
            }
            
            # Add user token if present
            if "authed_user" in oauth_response and "access_token" in oauth_response["authed_user"]:
                installation_data["user_token"] = oauth_response["authed_user"]["access_token"]
                installation_data["user_id"] = oauth_response["authed_user"]["id"]
                installation_data["user_scope"] = oauth_response["authed_user"].get("scope", "")
            
            # Store installation in database
            await self.store_installation(installation_data)
            
            # Verify bot token by testing API call
            await self.verify_installation(installation_data["bot_token"], installation_data["team_id"])
            
            logger.info(f"Successfully installed in team: {installation_data['team_name']} ({installation_data['team_id']})")
            
            return {
                "success": True,
                "team_id": installation_data["team_id"],
                "team_name": installation_data["team_name"],
                "bot_user_id": installation_data["bot_user_id"]
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"OAuth callback error: {str(e)}")
            raise HTTPException(status_code=500, detail="OAuth callback processing failed")
    
    async def store_installation(self, installation_data: Dict[str, Any]) -> None:
        """Store installation data securely in database"""
        try:
            # Import here to avoid circular imports
            from database import get_database
            
            db = await get_database()
            
            # Encrypt sensitive tokens before storage
            encrypted_bot_token = self.encrypt_token(installation_data["bot_token"])
            encrypted_user_token = None
            
            if "user_token" in installation_data:
                encrypted_user_token = self.encrypt_token(installation_data["user_token"])
            
            # Upsert installation record
            query = """
            INSERT INTO workspaces (
                slack_team_id, slack_team_name, bot_token, bot_user_id,
                enterprise_id, is_enterprise_install, installer_user_id,
                user_token, user_id, scope, user_scope, installed_at,
                is_active, settings
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
            )
            ON CONFLICT (slack_team_id) 
            DO UPDATE SET
                slack_team_name = EXCLUDED.slack_team_name,
                bot_token = EXCLUDED.bot_token,
                bot_user_id = EXCLUDED.bot_user_id,
                user_token = EXCLUDED.user_token,
                user_id = EXCLUDED.user_id,
                scope = EXCLUDED.scope,
                user_scope = EXCLUDED.user_scope,
                updated_at = CURRENT_TIMESTAMP,
                is_active = true
            """
            
            await db.execute(
                query,
                installation_data["team_id"],
                installation_data["team_name"],
                encrypted_bot_token,
                installation_data["bot_user_id"],
                installation_data.get("enterprise_id"),
                installation_data.get("is_enterprise_install", False),
                installation_data.get("installer_user_id"),
                encrypted_user_token,
                installation_data.get("user_id"),
                installation_data.get("scope", ""),
                installation_data.get("user_scope", ""),
                installation_data["installed_at"],
                True,
                json.dumps({"auto_process": True, "ai_model": "gpt-4"})
            )
            
            logger.info(f"Stored installation for team: {installation_data['team_id']}")
            
        except Exception as e:
            logger.error(f"Failed to store installation: {str(e)}")
            raise
    
    async def verify_installation(self, bot_token: str, team_id: str) -> None:
        """Verify installation by testing bot token"""
        try:
            client = AsyncWebClient(token=bot_token)
            
            # Test auth and get bot info
            auth_response = await client.auth_test()
            
            if not auth_response["ok"]:
                raise Exception(f"Bot token verification failed: {auth_response.get('error')}")
            
            # Verify team ID matches
            if auth_response["team_id"] != team_id:
                raise Exception(f"Team ID mismatch: expected {team_id}, got {auth_response['team_id']}")
            
            logger.info(f"Installation verified for team: {team_id}")
            
        except SlackApiError as e:
            logger.error(f"Slack API error during verification: {e.response['error']}")
            raise Exception(f"Installation verification failed: {e.response['error']}")
        except Exception as e:
            logger.error(f"Installation verification error: {str(e)}")
            raise
    
    async def get_installation(self, team_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve installation data for a team"""
        try:
            from database import get_database
            
            db = await get_database()
            
            query = """
            SELECT * FROM workspaces 
            WHERE slack_team_id = $1 AND is_active = true
            """
            
            row = await db.fetchrow(query, team_id)
            
            if not row:
                return None
            
            # Decrypt tokens
            installation_data = dict(row)
            installation_data["bot_token"] = self.decrypt_token(installation_data["bot_token"])
            
            if installation_data.get("user_token"):
                installation_data["user_token"] = self.decrypt_token(installation_data["user_token"])
            
            return installation_data
            
        except Exception as e:
            logger.error(f"Failed to retrieve installation for team {team_id}: {str(e)}")
            return None
    
    async def uninstall(self, team_id: str) -> bool:
        """Handle app uninstallation"""
        try:
            from database import get_database
            
            db = await get_database()
            
            # Mark installation as inactive instead of deleting
            query = """
            UPDATE workspaces 
            SET is_active = false, updated_at = CURRENT_TIMESTAMP
            WHERE slack_team_id = $1
            """
            
            result = await db.execute(query, team_id)
            
            if result == "UPDATE 1":
                logger.info(f"Uninstalled from team: {team_id}")
                return True
            else:
                logger.warning(f"No installation found for team: {team_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to uninstall from team {team_id}: {str(e)}")
            return False
    
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
            return token  # Fallback to plain text
    
    def decrypt_token(self, encrypted_token: str) -> str:
        """Decrypt token from storage"""
        try:
            import cryptography.fernet as fernet
            
            encryption_key = os.getenv("ENCRYPTION_KEY")
            if not encryption_key:
                return encrypted_token  # Assume plain text
            
            f = fernet.Fernet(encryption_key.encode())
            decrypted_token = f.decrypt(encrypted_token.encode())
            return decrypted_token.decode()
            
        except Exception as e:
            logger.error(f"Token decryption failed: {str(e)}")
            return encrypted_token  # Return as-is if decryption fails
    
    async def refresh_token_if_needed(self, team_id: str) -> Optional[str]:
        """Refresh token if it's expired or close to expiry"""
        try:
            installation = await self.get_installation(team_id)
            if not installation:
                return None
            
            # Test current token
            client = AsyncWebClient(token=installation["bot_token"])
            try:
                await client.auth_test()
                return installation["bot_token"]  # Token is still valid
            except SlackApiError as e:
                if e.response.get("error") == "invalid_auth":
                    logger.warning(f"Token expired for team {team_id}, attempting refresh")
                    # Implement token refresh logic here if Slack supports it
                    # For now, mark installation as inactive
                    await self.uninstall(team_id)
                    return None
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"Token refresh check failed for team {team_id}: {str(e)}")
            return None
    
    async def validate_webhook_signature(self, request: Request) -> bool:
        """Validate Slack webhook signature for security"""
        try:
            import hmac
            import hashlib
            import time
            
            # Get headers
            timestamp = request.headers.get("X-Slack-Request-Timestamp")
            signature = request.headers.get("X-Slack-Signature")
            
            if not timestamp or not signature:
                logger.warning("Missing signature headers in Slack request")
                return False
            
            # Check timestamp to prevent replay attacks
            if abs(time.time() - int(timestamp)) > 60 * 5:  # 5 minutes
                logger.warning("Slack request timestamp too old")
                return False
            
            # Get request body
            body = await request.body()
            
            # Create signature
            sig_basestring = f"v0:{timestamp}:{body.decode()}"
            computed_signature = "v0=" + hmac.new(
                self.signing_secret.encode(),
                sig_basestring.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Secure comparison to prevent timing attacks
            if not hmac.compare_digest(computed_signature, signature):
                logger.warning("Invalid Slack webhook signature")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Webhook signature validation error: {str(e)}")
            return False


# OAuth handler instance
oauth_handler = SlackOAuthHandler()