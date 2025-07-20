"""
Slack Webhook Handler for SlackToDoc Production
Handles real-time Slack events, interactions, and slash commands
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import re
from fastapi import HTTPException, Request, BackgroundTasks
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError

from slack_oauth_handler import oauth_handler

# Configure logging
logger = logging.getLogger(__name__)

class SlackWebhookHandler:
    """Production-ready Slack webhook handler with enterprise features"""
    
    def __init__(self):
        self.signing_secret = os.getenv("SLACK_SIGNING_SECRET")
        self.bot_user_id = None  # Will be set per team
        
        # Event handlers mapping
        self.event_handlers = {
            "app_mention": self.handle_app_mention,
            "message": self.handle_message,
            "app_home_opened": self.handle_app_home_opened,
            "team_join": self.handle_team_join,
            "user_change": self.handle_user_change,
            "reaction_added": self.handle_reaction_added,
            "reaction_removed": self.handle_reaction_removed
        }
        
        # Interactive component handlers
        self.interactive_handlers = {
            "block_actions": self.handle_block_actions,
            "view_submission": self.handle_view_submission,
            "view_closed": self.handle_view_closed,
            "shortcut": self.handle_shortcut,
            "message_action": self.handle_message_action
        }
        
        # Slash command handlers
        self.command_handlers = {
            "/slacktodoc": self.handle_slacktodoc_command
        }
    
    async def handle_webhook(self, request: Request, background_tasks: BackgroundTasks) -> Dict[str, Any]:
        """Main webhook handler with security validation"""
        try:
            # Validate signature
            if not await oauth_handler.validate_webhook_signature(request):
                raise HTTPException(status_code=401, detail="Invalid signature")
            
            # Parse request body
            body = await request.body()
            payload = json.loads(body.decode())
            
            # Handle URL verification challenge
            if payload.get("type") == "url_verification":
                return {"challenge": payload["challenge"]}
            
            # Handle different webhook types
            webhook_type = payload.get("type")
            
            if webhook_type == "event_callback":
                # Add to background tasks for async processing
                background_tasks.add_task(self.process_event, payload)
                return {"status": "ok"}
            
            elif webhook_type == "interactive":
                # Handle interactive components
                background_tasks.add_task(self.process_interactive, payload)
                return {"status": "ok"}
            
            elif webhook_type == "slash_command":
                # Handle slash commands
                return await self.process_slash_command(payload)
            
            else:
                logger.warning(f"Unknown webhook type: {webhook_type}")
                return {"status": "unknown_type"}
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON in webhook payload")
            raise HTTPException(status_code=400, detail="Invalid JSON")
        except Exception as e:
            logger.error(f"Webhook processing error: {str(e)}")
            raise HTTPException(status_code=500, detail="Webhook processing failed")
    
    async def process_event(self, payload: Dict[str, Any]) -> None:
        """Process Slack events asynchronously"""
        try:
            event = payload.get("event", {})
            event_type = event.get("type")
            team_id = payload.get("team_id")
            
            if not team_id:
                logger.error("No team_id in event payload")
                return
            
            # Get installation data
            installation = await oauth_handler.get_installation(team_id)
            if not installation:
                logger.error(f"No installation found for team: {team_id}")
                return
            
            # Create Slack client
            client = AsyncWebClient(token=installation["bot_token"])
            
            # Handle the event
            handler = self.event_handlers.get(event_type)
            if handler:
                await handler(event, client, team_id)
            else:
                logger.info(f"No handler for event type: {event_type}")
                
        except Exception as e:
            logger.error(f"Event processing error: {str(e)}")
    
    async def handle_app_mention(self, event: Dict[str, Any], client: AsyncWebClient, team_id: str) -> None:
        """Handle @SlackToDoc mentions"""
        try:
            channel_id = event["channel"]
            thread_ts = event.get("thread_ts", event["ts"])
            user_id = event["user"]
            text = event.get("text", "")
            
            logger.info(f"App mention in {channel_id} by {user_id}")
            
            # Send immediate acknowledgment
            await client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text="👋 I'm on it! Analyzing this conversation and creating documentation..."
            )
            
            # Process the conversation
            await self.process_conversation_for_documentation(
                client, channel_id, thread_ts, team_id
            )
            
        except SlackApiError as e:
            logger.error(f"Slack API error in app mention: {e.response['error']}")
        except Exception as e:
            logger.error(f"App mention handling error: {str(e)}")
    
    async def handle_message(self, event: Dict[str, Any], client: AsyncWebClient, team_id: str) -> None:
        """Handle regular messages for passive monitoring"""
        try:
            # Skip bot messages
            if event.get("bot_id") or event.get("subtype") == "bot_message":
                return
            
            channel_id = event["channel"]
            text = event.get("text", "")
            
            # Check for trigger keywords that suggest important content
            trigger_keywords = [
                "decision", "decided", "agree", "consensus",
                "action item", "todo", "task", "assign",
                "important", "key point", "summary",
                "meeting", "conclusion", "next steps"
            ]
            
            if any(keyword in text.lower() for keyword in trigger_keywords):
                # This might be important - analyze the thread
                thread_ts = event.get("thread_ts", event["ts"])
                
                # Get conversation context
                messages = await self.get_thread_messages(client, channel_id, thread_ts)
                
                if len(messages) >= 3:  # Minimum threshold for documentation
                    await self.suggest_documentation(client, channel_id, thread_ts, team_id)
                    
        except Exception as e:
            logger.error(f"Message handling error: {str(e)}")
    
    async def handle_app_home_opened(self, event: Dict[str, Any], client: AsyncWebClient, team_id: str) -> None:
        """Handle App Home tab opening"""
        try:
            user_id = event["user"]
            
            # Create personalized home view
            home_view = {
                "type": "home",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*Welcome to SlackToDoc! 📚*\n\nI help transform your team conversations into organized Notion documentation automatically."
                        }
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*How to use SlackToDoc:*\n\n• Mention `@SlackToDoc` in any conversation\n• Use `/slacktodoc` command for quick actions\n• Set up automatic monitoring for important channels"
                        }
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "⚙️ Settings"
                                },
                                "action_id": "open_settings",
                                "style": "primary"
                            },
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "📊 View Stats"
                                },
                                "action_id": "view_stats"
                            },
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "❓ Help"
                                },
                                "action_id": "show_help"
                            }
                        ]
                    }
                ]
            }
            
            await client.views_publish(
                user_id=user_id,
                view=home_view
            )
            
        except SlackApiError as e:
            logger.error(f"App home error: {e.response['error']}")
        except Exception as e:
            logger.error(f"App home handling error: {str(e)}")
    
    async def handle_team_join(self, event: Dict[str, Any], client: AsyncWebClient, team_id: str) -> None:
        """Handle new team member joining"""
        try:
            user_id = event["user"]["id"]
            user_name = event["user"].get("real_name", event["user"].get("name", "New User"))
            
            # Send welcome DM
            welcome_message = f"""
Hello {user_name}! 👋

Welcome to the team! I'm SlackToDoc, and I help capture important conversations and turn them into organized documentation.

Here's how you can use me:
• Mention `@SlackToDoc` in any conversation to create documentation
• Use `/slacktodoc help` to see all available commands
• Check out the App Home tab for settings and stats

Feel free to ask if you have any questions!
            """
            
            await client.chat_postMessage(
                channel=user_id,
                text=welcome_message
            )
            
        except SlackApiError as e:
            logger.error(f"Team join handling error: {e.response['error']}")
        except Exception as e:
            logger.error(f"Team join error: {str(e)}")
    
    async def handle_user_change(self, event: Dict[str, Any], client: AsyncWebClient, team_id: str) -> None:
        """Handle user profile changes"""
        try:
            # Update user information in database
            user_data = event["user"]
            await self.update_user_info(team_id, user_data)
            
        except Exception as e:
            logger.error(f"User change handling error: {str(e)}")
    
    async def handle_reaction_added(self, event: Dict[str, Any], client: AsyncWebClient, team_id: str) -> None:
        """Handle reaction additions for engagement tracking"""
        try:
            reaction = event["reaction"]
            
            # Track reactions that might indicate important content
            important_reactions = ["important", "star", "bookmark", "pushpin", "memo", "document"]
            
            if reaction in important_reactions:
                channel_id = event["item"]["channel"]
                message_ts = event["item"]["ts"]
                
                # Suggest documentation for highly reacted messages
                await self.suggest_documentation_for_message(
                    client, channel_id, message_ts, team_id, f"Message with :{reaction}: reaction"
                )
                
        except Exception as e:
            logger.error(f"Reaction handling error: {str(e)}")
    
    async def handle_reaction_removed(self, event: Dict[str, Any], client: AsyncWebClient, team_id: str) -> None:
        """Handle reaction removal"""
        try:
            # Log reaction removal for analytics
            pass
        except Exception as e:
            logger.error(f"Reaction removal handling error: {str(e)}")
    
    async def process_interactive(self, payload: Dict[str, Any]) -> None:
        """Process interactive component callbacks"""
        try:
            interaction_type = payload.get("type")
            team_id = payload["team"]["id"]
            
            # Get installation
            installation = await oauth_handler.get_installation(team_id)
            if not installation:
                logger.error(f"No installation found for team: {team_id}")
                return
            
            client = AsyncWebClient(token=installation["bot_token"])
            
            # Handle the interaction
            handler = self.interactive_handlers.get(interaction_type)
            if handler:
                await handler(payload, client, team_id)
            else:
                logger.info(f"No handler for interaction type: {interaction_type}")
                
        except Exception as e:
            logger.error(f"Interactive processing error: {str(e)}")
    
    async def handle_block_actions(self, payload: Dict[str, Any], client: AsyncWebClient, team_id: str) -> None:
        """Handle block action interactions"""
        try:
            actions = payload.get("actions", [])
            user_id = payload["user"]["id"]
            
            for action in actions:
                action_id = action["action_id"]
                
                if action_id == "open_settings":
                    await self.show_settings_modal(client, payload["trigger_id"], user_id, team_id)
                elif action_id == "view_stats":
                    await self.show_stats_modal(client, payload["trigger_id"], user_id, team_id)
                elif action_id == "show_help":
                    await self.show_help_modal(client, payload["trigger_id"])
                    
        except Exception as e:
            logger.error(f"Block action handling error: {str(e)}")
    
    async def handle_view_submission(self, payload: Dict[str, Any], client: AsyncWebClient, team_id: str) -> None:
        """Handle modal view submissions"""
        try:
            view = payload["view"]
            callback_id = view["callback_id"]
            user_id = payload["user"]["id"]
            
            if callback_id == "settings_modal":
                await self.process_settings_submission(view, user_id, team_id)
            elif callback_id == "create_doc_modal":
                await self.process_doc_creation_submission(view, client, user_id, team_id)
                
        except Exception as e:
            logger.error(f"View submission handling error: {str(e)}")
    
    async def handle_view_closed(self, payload: Dict[str, Any], client: AsyncWebClient, team_id: str) -> None:
        """Handle modal view closures"""
        try:
            # Log modal closures for analytics
            pass
        except Exception as e:
            logger.error(f"View closure handling error: {str(e)}")
    
    async def handle_shortcut(self, payload: Dict[str, Any], client: AsyncWebClient, team_id: str) -> None:
        """Handle global and message shortcuts"""
        try:
            callback_id = payload["callback_id"]
            
            if callback_id == "create_doc_shortcut":
                await self.handle_create_doc_shortcut(payload, client, team_id)
            elif callback_id == "settings_shortcut":
                await self.handle_settings_shortcut(payload, client, team_id)
                
        except Exception as e:
            logger.error(f"Shortcut handling error: {str(e)}")
    
    async def handle_message_action(self, payload: Dict[str, Any], client: AsyncWebClient, team_id: str) -> None:
        """Handle message actions"""
        try:
            callback_id = payload["callback_id"]
            
            if callback_id == "create_doc_shortcut":
                # Create documentation from specific message
                message = payload["message"]
                channel_id = payload["channel"]["id"]
                
                await self.create_documentation_from_message(
                    client, channel_id, message, team_id
                )
                
        except Exception as e:
            logger.error(f"Message action handling error: {str(e)}")
    
    async def process_slash_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process slash commands"""
        try:
            command = payload["command"]
            team_id = payload["team_id"]
            user_id = payload["user_id"]
            text = payload.get("text", "").strip()
            
            # Get installation
            installation = await oauth_handler.get_installation(team_id)
            if not installation:
                return {
                    "response_type": "ephemeral",
                    "text": "❌ SlackToDoc is not properly installed. Please reinstall the app."
                }
            
            client = AsyncWebClient(token=installation["bot_token"])
            
            # Handle the command
            handler = self.command_handlers.get(command)
            if handler:
                return await handler(payload, client, team_id)
            else:
                return {
                    "response_type": "ephemeral",
                    "text": f"❌ Unknown command: {command}"
                }
                
        except Exception as e:
            logger.error(f"Slash command processing error: {str(e)}")
            return {
                "response_type": "ephemeral",
                "text": "❌ Command processing failed. Please try again."
            }
    
    async def handle_slacktodoc_command(self, payload: Dict[str, Any], client: AsyncWebClient, team_id: str) -> Dict[str, Any]:
        """Handle /slacktodoc slash command"""
        try:
            text = payload.get("text", "").strip().lower()
            user_id = payload["user_id"]
            channel_id = payload["channel_id"]
            
            if text == "help" or text == "":
                return {
                    "response_type": "ephemeral",
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "*SlackToDoc Commands:*\n\n• `/slacktodoc help` - Show this help\n• `/slacktodoc status` - Check service status\n• `/slacktodoc create` - Create documentation from current channel\n• `/slacktodoc settings` - Open settings\n• `/slacktodoc stats` - View usage statistics"
                            }
                        }
                    ]
                }
            
            elif text == "status":
                return {
                    "response_type": "ephemeral",
                    "text": "✅ SlackToDoc is running and ready to help!"
                }
            
            elif text == "create":
                # Trigger documentation creation for current channel
                await self.create_documentation_for_channel(client, channel_id, team_id)
                return {
                    "response_type": "ephemeral",
                    "text": "📝 Creating documentation for this channel's recent conversations..."
                }
            
            elif text == "settings":
                # Show settings modal
                return {
                    "response_type": "ephemeral",
                    "text": "⚙️ Opening settings... (Feature coming soon!)"
                }
            
            elif text == "stats":
                # Show usage statistics
                stats = await self.get_team_stats(team_id)
                return {
                    "response_type": "ephemeral",
                    "text": f"📊 *Usage Statistics:*\n• Documents created: {stats.get('documents_created', 0)}\n• Messages processed: {stats.get('messages_processed', 0)}\n• Active channels: {stats.get('active_channels', 0)}"
                }
            
            else:
                return {
                    "response_type": "ephemeral",
                    "text": f"❓ Unknown command: `{text}`. Use `/slacktodoc help` for available commands."
                }
                
        except Exception as e:
            logger.error(f"SlackToDoc command error: {str(e)}")
            return {
                "response_type": "ephemeral",
                "text": "❌ Command failed. Please try again."
            }
    
    async def process_conversation_for_documentation(self, client: AsyncWebClient, channel_id: str, thread_ts: str, team_id: str) -> None:
        """Process conversation and create Notion documentation"""
        try:
            # Get conversation messages
            messages = await self.get_thread_messages(client, channel_id, thread_ts)
            
            if len(messages) < 2:
                await client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text="ℹ️ Not enough conversation content to create meaningful documentation."
                )
                return
            
            # Extract content using AI
            from ai_processor import extract_content_from_messages
            
            extracted_content = await extract_content_from_messages(messages)
            
            if not extracted_content:
                await client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text="❌ Unable to extract meaningful content from this conversation."
                )
                return
            
            # Create Notion document
            from notion_handler import create_notion_document
            
            notion_url = await create_notion_document(extracted_content, team_id)
            
            if notion_url:
                await client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=f"✅ Documentation created! View it here: {notion_url}"
                )
            else:
                await client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text="❌ Failed to create Notion documentation. Please check your Notion integration."
                )
                
        except Exception as e:
            logger.error(f"Conversation processing error: {str(e)}")
            await client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text="❌ Error processing conversation. Please try again later."
            )
    
    async def get_thread_messages(self, client: AsyncWebClient, channel_id: str, thread_ts: str) -> List[Dict[str, Any]]:
        """Get all messages in a thread"""
        try:
            response = await client.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                limit=100
            )
            
            if response["ok"]:
                return response["messages"]
            else:
                logger.error(f"Failed to get thread messages: {response.get('error')}")
                return []
                
        except SlackApiError as e:
            logger.error(f"Error getting thread messages: {e.response['error']}")
            return []
    
    async def suggest_documentation(self, client: AsyncWebClient, channel_id: str, thread_ts: str, team_id: str) -> None:
        """Suggest documentation creation for important conversations"""
        try:
            # Send suggestion message with action button
            await client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "💡 This conversation seems important! Would you like me to create documentation?"
                        }
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "📝 Create Docs"
                                },
                                "action_id": "create_documentation",
                                "value": f"{channel_id}:{thread_ts}",
                                "style": "primary"
                            },
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "❌ No Thanks"
                                },
                                "action_id": "dismiss_suggestion"
                            }
                        ]
                    }
                ]
            )
            
        except Exception as e:
            logger.error(f"Documentation suggestion error: {str(e)}")
    
    async def update_user_info(self, team_id: str, user_data: Dict[str, Any]) -> None:
        """Update user information in database"""
        try:
            from database import get_database
            
            db = await get_database()
            
            query = """
            INSERT INTO users (workspace_id, slack_user_id, slack_username, email, last_active)
            SELECT id, $2, $3, $4, CURRENT_TIMESTAMP
            FROM workspaces WHERE slack_team_id = $1
            ON CONFLICT (workspace_id, slack_user_id)
            DO UPDATE SET
                slack_username = EXCLUDED.slack_username,
                email = EXCLUDED.email,
                last_active = CURRENT_TIMESTAMP
            """
            
            await db.execute(
                query,
                team_id,
                user_data["id"],
                user_data.get("name", ""),
                user_data.get("profile", {}).get("email", "")
            )
            
        except Exception as e:
            logger.error(f"User info update error: {str(e)}")
    
    async def get_team_stats(self, team_id: str) -> Dict[str, int]:
        """Get usage statistics for a team"""
        try:
            from database import get_database
            
            db = await get_database()
            
            # Get basic stats
            stats_query = """
            SELECT 
                COUNT(DISTINCT d.id) as documents_created,
                COUNT(DISTINCT m.id) as messages_processed,
                COUNT(DISTINCT c.id) as active_channels
            FROM workspaces w
            LEFT JOIN documents d ON w.id = d.workspace_id
            LEFT JOIN message_threads mt ON w.id = mt.workspace_id
            LEFT JOIN messages m ON mt.id = m.thread_id
            LEFT JOIN channels c ON w.id = c.workspace_id AND c.is_monitored = true
            WHERE w.slack_team_id = $1
            """
            
            row = await db.fetchrow(stats_query, team_id)
            
            return {
                "documents_created": row["documents_created"] or 0,
                "messages_processed": row["messages_processed"] or 0,
                "active_channels": row["active_channels"] or 0
            }
            
        except Exception as e:
            logger.error(f"Stats retrieval error: {str(e)}")
            return {"documents_created": 0, "messages_processed": 0, "active_channels": 0}


# Webhook handler instance
webhook_handler = SlackWebhookHandler()