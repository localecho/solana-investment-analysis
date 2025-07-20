"""
Beta User Signup and Onboarding Handler for SlackToDoc
Manages beta user registration, onboarding, and engagement tracking
"""

import os
import asyncio
import logging
import time
import secrets
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logger = logging.getLogger(__name__)

class UserStatus(Enum):
    """Beta user status"""
    PENDING = "pending"
    ACTIVE = "active"
    ONBOARDED = "onboarded"
    INACTIVE = "inactive"
    CHURNED = "churned"

@dataclass
class BetaUser:
    """Beta user data structure"""
    id: str
    email: str
    name: str
    company: str
    team_size: int
    use_case: str
    slack_team_id: Optional[str]
    notion_workspace_id: Optional[str]
    status: UserStatus
    signup_date: float
    onboarded_date: Optional[float]
    last_activity: Optional[float]
    documents_created: int
    engagement_score: float
    feedback: List[Dict[str, Any]]
    referral_code: str
    referred_by: Optional[str]

class BetaSignupHandler:
    """Production beta user management system"""
    
    def __init__(self):
        self.users: Dict[str, BetaUser] = {}
        self.waiting_list: List[str] = []
        self.referral_codes: Dict[str, str] = {}  # code -> user_id
        
        # Beta program configuration
        self.config = {
            "max_beta_users": int(os.getenv("MAX_BETA_USERS", "50")),
            "auto_approve": os.getenv("AUTO_APPROVE_BETA", "false").lower() == "true",
            "onboarding_timeout_days": 7,
            "inactive_threshold_days": 14,
            "churn_threshold_days": 30
        }
        
        # Email templates
        self.email_templates = {
            "signup_confirmation": self._get_signup_confirmation_template(),
            "beta_approval": self._get_beta_approval_template(),
            "onboarding_start": self._get_onboarding_start_template(),
            "onboarding_reminder": self._get_onboarding_reminder_template(),
            "engagement_followup": self._get_engagement_followup_template(),
            "feedback_request": self._get_feedback_request_template()
        }
        
        # Engagement tracking
        self.engagement_weights = {
            "documents_created": 10,
            "daily_active": 5,
            "slack_integration": 8,
            "notion_integration": 8,
            "support_interaction": 3,
            "feedback_provided": 15,
            "referral_made": 20
        }
    
    async def signup_beta_user(self, email: str, name: str, company: str, 
                              team_size: int, use_case: str, 
                              referred_by: Optional[str] = None) -> Dict[str, Any]:
        """Handle beta user signup"""
        try:
            # Validate input
            if not email or not name:
                return {"success": False, "error": "Email and name are required"}
            
            # Check if user already exists
            if email in [user.email for user in self.users.values()]:
                return {"success": False, "error": "Email already registered"}
            
            # Generate user ID and referral code
            user_id = f"beta_{int(time.time())}_{secrets.token_hex(4)}"
            referral_code = self._generate_referral_code()
            
            # Create beta user
            user = BetaUser(
                id=user_id,
                email=email,
                name=name,
                company=company,
                team_size=team_size,
                use_case=use_case,
                slack_team_id=None,
                notion_workspace_id=None,
                status=UserStatus.PENDING,
                signup_date=time.time(),
                onboarded_date=None,
                last_activity=time.time(),
                documents_created=0,
                engagement_score=0.0,
                feedback=[],
                referral_code=referral_code,
                referred_by=referred_by
            )
            
            # Store user
            self.users[user_id] = user
            self.referral_codes[referral_code] = user_id
            
            # Check if can be auto-approved
            can_approve = (
                len([u for u in self.users.values() if u.status == UserStatus.ACTIVE]) < self.config["max_beta_users"]
            )
            
            if can_approve and self.config["auto_approve"]:
                await self._approve_beta_user(user_id)
                status = "approved"
            else:
                self.waiting_list.append(user_id)
                status = "waiting_list"
            
            # Send confirmation email
            await self._send_signup_confirmation(user)
            
            # Track referral
            if referred_by:
                await self._track_referral(referred_by, user_id)
            
            logger.info(f"Beta user signed up: {email} ({status})")
            
            return {
                "success": True,
                "user_id": user_id,
                "status": status,
                "referral_code": referral_code,
                "position_in_queue": len(self.waiting_list) if status == "waiting_list" else 0
            }
            
        except Exception as e:
            logger.error(f"Beta signup failed: {str(e)}")
            return {"success": False, "error": "Signup failed"}
    
    async def _approve_beta_user(self, user_id: str) -> bool:
        """Approve a beta user"""
        try:
            if user_id not in self.users:
                return False
            
            user = self.users[user_id]
            user.status = UserStatus.ACTIVE
            
            # Remove from waiting list
            if user_id in self.waiting_list:
                self.waiting_list.remove(user_id)
            
            # Send approval email with onboarding instructions
            await self._send_beta_approval(user)
            
            # Schedule onboarding follow-up
            asyncio.create_task(self._schedule_onboarding_followup(user_id))
            
            logger.info(f"Beta user approved: {user.email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to approve beta user {user_id}: {str(e)}")
            return False
    
    async def link_slack_workspace(self, user_id: str, slack_team_id: str) -> bool:
        """Link user to Slack workspace"""
        try:
            if user_id not in self.users:
                return False
            
            user = self.users[user_id]
            user.slack_team_id = slack_team_id
            user.last_activity = time.time()
            
            # Update engagement score
            await self._update_engagement_score(user_id, "slack_integration")
            
            # Check if fully onboarded
            if user.notion_workspace_id:
                await self._complete_onboarding(user_id)
            
            logger.info(f"Slack workspace linked for user: {user.email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to link Slack workspace: {str(e)}")
            return False
    
    async def link_notion_workspace(self, user_id: str, notion_workspace_id: str) -> bool:
        """Link user to Notion workspace"""
        try:
            if user_id not in self.users:
                return False
            
            user = self.users[user_id]
            user.notion_workspace_id = notion_workspace_id
            user.last_activity = time.time()
            
            # Update engagement score
            await self._update_engagement_score(user_id, "notion_integration")
            
            # Check if fully onboarded
            if user.slack_team_id:
                await self._complete_onboarding(user_id)
            
            logger.info(f"Notion workspace linked for user: {user.email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to link Notion workspace: {str(e)}")
            return False
    
    async def _complete_onboarding(self, user_id: str):
        """Mark user as fully onboarded"""
        try:
            user = self.users[user_id]
            
            if user.status == UserStatus.ACTIVE:
                user.status = UserStatus.ONBOARDED
                user.onboarded_date = time.time()
                
                # Send onboarding completion email
                await self._send_onboarding_completion(user)
                
                logger.info(f"User onboarding completed: {user.email}")
                
        except Exception as e:
            logger.error(f"Failed to complete onboarding: {str(e)}")
    
    async def track_document_creation(self, user_id: str):
        """Track document creation for engagement"""
        try:
            if user_id not in self.users:
                return
            
            user = self.users[user_id]
            user.documents_created += 1
            user.last_activity = time.time()
            
            # Update engagement score
            await self._update_engagement_score(user_id, "documents_created")
            
        except Exception as e:
            logger.error(f"Failed to track document creation: {str(e)}")
    
    async def track_daily_activity(self, user_id: str):
        """Track daily activity for engagement"""
        try:
            if user_id not in self.users:
                return
            
            user = self.users[user_id]
            user.last_activity = time.time()
            
            # Update engagement score (but not too frequently)
            last_score_update = getattr(user, '_last_daily_score_update', 0)
            if time.time() - last_score_update > 86400:  # 24 hours
                await self._update_engagement_score(user_id, "daily_active")
                user._last_daily_score_update = time.time()
            
        except Exception as e:
            logger.error(f"Failed to track daily activity: {str(e)}")
    
    async def submit_feedback(self, user_id: str, feedback_type: str, 
                            rating: int, comments: str) -> bool:
        """Submit user feedback"""
        try:
            if user_id not in self.users:
                return False
            
            user = self.users[user_id]
            
            feedback_entry = {
                "type": feedback_type,
                "rating": rating,
                "comments": comments,
                "timestamp": time.time()
            }
            
            user.feedback.append(feedback_entry)
            user.last_activity = time.time()
            
            # Update engagement score
            await self._update_engagement_score(user_id, "feedback_provided")
            
            logger.info(f"Feedback submitted by user: {user.email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit feedback: {str(e)}")
            return False
    
    async def _update_engagement_score(self, user_id: str, action: str):
        """Update user engagement score"""
        try:
            user = self.users[user_id]
            weight = self.engagement_weights.get(action, 1)
            
            # Add points with decay for repeated actions
            if action == "documents_created":
                # Diminishing returns for document creation
                multiplier = max(0.1, 1.0 - (user.documents_created * 0.05))
                points = weight * multiplier
            else:
                points = weight
            
            user.engagement_score += points
            
            # Apply time decay (scores decay over time to keep them current)
            days_since_signup = (time.time() - user.signup_date) / 86400
            decay_factor = max(0.5, 1.0 - (days_since_signup * 0.01))
            user.engagement_score *= decay_factor
            
        except Exception as e:
            logger.error(f"Failed to update engagement score: {str(e)}")
    
    async def _schedule_onboarding_followup(self, user_id: str):
        """Schedule onboarding follow-up emails"""
        try:
            # Wait 24 hours, then send onboarding reminder if not completed
            await asyncio.sleep(86400)
            
            user = self.users.get(user_id)
            if user and user.status == UserStatus.ACTIVE:
                await self._send_onboarding_reminder(user)
            
            # Wait another 3 days for final reminder
            await asyncio.sleep(259200)
            
            user = self.users.get(user_id)
            if user and user.status == UserStatus.ACTIVE:
                await self._send_onboarding_final_reminder(user)
                
        except Exception as e:
            logger.error(f"Onboarding followup error: {str(e)}")
    
    async def _track_referral(self, referrer_code: str, new_user_id: str):
        """Track successful referral"""
        try:
            if referrer_code in self.referral_codes:
                referrer_id = self.referral_codes[referrer_code]
                if referrer_id in self.users:
                    # Update engagement score for referrer
                    await self._update_engagement_score(referrer_id, "referral_made")
                    logger.info(f"Referral tracked: {referrer_id} -> {new_user_id}")
                    
        except Exception as e:
            logger.error(f"Failed to track referral: {str(e)}")
    
    def _generate_referral_code(self) -> str:
        """Generate unique referral code"""
        while True:
            code = secrets.token_urlsafe(8)
            if code not in self.referral_codes:
                return code
    
    async def _send_signup_confirmation(self, user: BetaUser):
        """Send signup confirmation email"""
        template = self.email_templates["signup_confirmation"]
        subject = "Welcome to SlackToDoc Beta!"
        
        body = template.format(
            name=user.name,
            referral_code=user.referral_code
        )
        
        await self._send_email(user.email, subject, body)
    
    async def _send_beta_approval(self, user: BetaUser):
        """Send beta approval email"""
        template = self.email_templates["beta_approval"]
        subject = "You're in! SlackToDoc Beta Access Granted"
        
        body = template.format(
            name=user.name,
            setup_url="https://slacktodoc.com/setup"
        )
        
        await self._send_email(user.email, subject, body)
    
    async def _send_onboarding_reminder(self, user: BetaUser):
        """Send onboarding reminder email"""
        template = self.email_templates["onboarding_reminder"]
        subject = "Complete Your SlackToDoc Setup"
        
        body = template.format(
            name=user.name,
            setup_url="https://slacktodoc.com/setup"
        )
        
        await self._send_email(user.email, subject, body)
    
    async def _send_email(self, to_email: str, subject: str, body: str):
        """Send email using SMTP"""
        try:
            smtp_config = {
                "server": os.getenv("SMTP_SERVER"),
                "port": int(os.getenv("SMTP_PORT", "587")),
                "username": os.getenv("SMTP_USERNAME"),
                "password": os.getenv("SMTP_PASSWORD"),
                "from_email": os.getenv("BETA_FROM_EMAIL", "beta@slacktodoc.com")
            }
            
            if not all([smtp_config["server"], smtp_config["username"], smtp_config["password"]]):
                logger.warning("Email configuration incomplete, skipping email send")
                return
            
            msg = MIMEMultipart()
            msg["From"] = smtp_config["from_email"]
            msg["To"] = to_email
            msg["Subject"] = subject
            
            msg.attach(MIMEText(body, "html"))
            
            with smtplib.SMTP(smtp_config["server"], smtp_config["port"]) as server:
                server.starttls()
                server.login(smtp_config["username"], smtp_config["password"])
                server.send_message(msg)
            
            logger.info(f"Email sent to {to_email}: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
    
    def _get_signup_confirmation_template(self) -> str:
        """Get signup confirmation email template"""
        return """
        <html>
        <body>
            <h2>Welcome to SlackToDoc Beta!</h2>
            
            <p>Hi {name},</p>
            
            <p>Thanks for signing up for the SlackToDoc beta program! We're excited to have you on board.</p>
            
            <p>Your application is being reviewed, and we'll notify you as soon as you're approved for beta access.</p>
            
            <h3>What's Next?</h3>
            <ul>
                <li>We'll review your application within 24 hours</li>
                <li>You'll receive setup instructions once approved</li>
                <li>Join our beta community for updates and support</li>
            </ul>
            
            <h3>Refer Friends</h3>
            <p>Love what you see? Share your referral code with colleagues:</p>
            <p><strong>{referral_code}</strong></p>
            
            <p>Best regards,<br>The SlackToDoc Team</p>
        </body>
        </html>
        """
    
    def _get_beta_approval_template(self) -> str:
        """Get beta approval email template"""
        return """
        <html>
        <body>
            <h2>🎉 You're in! SlackToDoc Beta Access Granted</h2>
            
            <p>Hi {name},</p>
            
            <p>Congratulations! Your application for SlackToDoc beta has been approved.</p>
            
            <h3>Get Started Now</h3>
            <ol>
                <li><a href="{setup_url}">Set up your Slack integration</a></li>
                <li>Connect your Notion workspace</li>
                <li>Try creating your first document with @SlackToDoc</li>
            </ol>
            
            <h3>Support & Community</h3>
            <p>Need help? We're here for you:</p>
            <ul>
                <li>📧 Email: support@slacktodoc.com</li>
                <li>💬 Slack: #slacktodoc-beta</li>
                <li>📖 Documentation: docs.slacktodoc.com</li>
            </ul>
            
            <p>Ready to transform your team conversations into organized knowledge!<br>The SlackToDoc Team</p>
        </body>
        </html>
        """
    
    def _get_onboarding_start_template(self) -> str:
        """Get onboarding start email template"""
        return """
        <html>
        <body>
            <h2>Let's Get You Set Up with SlackToDoc</h2>
            
            <p>Hi {name},</p>
            
            <p>Ready to start capturing your team's knowledge automatically?</p>
            
            <h3>Quick Setup (5 minutes)</h3>
            <ol>
                <li><a href="{setup_url}">Install SlackToDoc in Slack</a></li>
                <li>Connect your Notion workspace</li>
                <li>Try it: mention @SlackToDoc in any conversation</li>
            </ol>
            
            <p>That's it! Your team conversations will start becoming searchable documentation.</p>
            
            <p>Questions? Reply to this email - we read every message.<br>The SlackToDoc Team</p>
        </body>
        </html>
        """
    
    def _get_onboarding_reminder_template(self) -> str:
        """Get onboarding reminder email template"""
        return """
        <html>
        <body>
            <h2>Don't Miss Out - Complete Your SlackToDoc Setup</h2>
            
            <p>Hi {name},</p>
            
            <p>We noticed you haven't completed your SlackToDoc setup yet. Don't worry - it only takes 5 minutes!</p>
            
            <h3>What You're Missing</h3>
            <ul>
                <li>⚡ Automatic documentation from Slack conversations</li>
                <li>🔍 Searchable knowledge base in Notion</li>
                <li>🤖 AI-powered content extraction</li>
                <li>📊 Team collaboration insights</li>
            </ul>
            
            <p><a href="{setup_url}" style="background-color: #007cba; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px;">Complete Setup Now</a></p>
            
            <p>Need help? Just reply to this email.<br>The SlackToDoc Team</p>
        </body>
        </html>
        """
    
    def _get_engagement_followup_template(self) -> str:
        """Get engagement follow-up email template"""
        return """
        <html>
        <body>
            <h2>How's SlackToDoc Working for You?</h2>
            
            <p>Hi {name},</p>
            
            <p>You've been using SlackToDoc for a while now, and we'd love to hear about your experience!</p>
            
            <h3>Your Usage Stats</h3>
            <ul>
                <li>📄 Documents created: {documents_created}</li>
                <li>⭐ Engagement score: {engagement_score}/100</li>
                <li>📅 Days active: {days_active}</li>
            </ul>
            
            <p>We're always improving based on user feedback. What would make SlackToDoc even better for your team?</p>
            
            <p><a href="{feedback_url}">Share Your Feedback</a></p>
            
            <p>Thanks for being part of our beta!<br>The SlackToDoc Team</p>
        </body>
        </html>
        """
    
    def _get_feedback_request_template(self) -> str:
        """Get feedback request email template"""
        return """
        <html>
        <body>
            <h2>Help Us Build the Perfect Documentation Tool</h2>
            
            <p>Hi {name},</p>
            
            <p>Your feedback as a beta user is incredibly valuable to us. Would you take 2 minutes to share your thoughts?</p>
            
            <h3>Quick Questions</h3>
            <ul>
                <li>How likely are you to recommend SlackToDoc? (0-10)</li>
                <li>What's your favorite feature?</li>
                <li>What would you change or improve?</li>
                <li>How has it impacted your team's workflow?</li>
            </ul>
            
            <p><a href="{feedback_url}" style="background-color: #28a745; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px;">Give Feedback</a></p>
            
            <p>As a thank you, beta users who provide feedback get priority access to new features!</p>
            
            <p>Appreciate your time,<br>The SlackToDoc Team</p>
        </body>
        </html>
        """
    
    def get_beta_stats(self) -> Dict[str, Any]:
        """Get beta program statistics"""
        users_by_status = {}
        for status in UserStatus:
            users_by_status[status.value] = len([u for u in self.users.values() if u.status == status])
        
        total_documents = sum(user.documents_created for user in self.users.values())
        avg_engagement = sum(user.engagement_score for user in self.users.values()) / len(self.users) if self.users else 0
        
        return {
            "total_signups": len(self.users),
            "users_by_status": users_by_status,
            "waiting_list_length": len(self.waiting_list),
            "total_documents_created": total_documents,
            "average_engagement_score": avg_engagement,
            "onboarding_completion_rate": len([u for u in self.users.values() if u.status == UserStatus.ONBOARDED]) / max(1, len([u for u in self.users.values() if u.status in [UserStatus.ACTIVE, UserStatus.ONBOARDED]])),
            "referral_rate": len([u for u in self.users.values() if u.referred_by]) / max(1, len(self.users))
        }


# Global beta signup handler instance
beta_signup_handler = BetaSignupHandler()