"""
AI Fallback Handler for SlackToDoc
Provides graceful degradation when AI services fail or are unavailable
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import re
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class FallbackConfig:
    """Configuration for fallback behavior"""
    enable_rule_based_extraction: bool = True
    enable_keyword_detection: bool = True
    enable_pattern_matching: bool = True
    enable_simple_summarization: bool = True
    max_fallback_attempts: int = 3
    fallback_confidence_threshold: float = 0.6

class AIFallbackHandler:
    """Handles graceful degradation when AI services are unavailable"""
    
    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        
        # Keyword patterns for different content types
        self.decision_keywords = [
            r'\b(?:decided?|decision|agree[ds]?|consensus|resolved?|concluded?)\b',
            r'\b(?:we\'ll|we will|going with|chosen|selected|approved?)\b',
            r'\b(?:final|official|confirmed?|settled?)\b'
        ]
        
        self.action_keywords = [
            r'\b(?:action item|todo|task|assign|responsible|owner|due)\b',
            r'\b(?:will do|going to|needs? to|should|must|have to)\b',
            r'\b(?:by \w+|deadline|complete|finish|deliver)\b'
        ]
        
        self.insight_keywords = [
            r'\b(?:learned?|insight|important|key|critical|note)\b',
            r'\b(?:remember|keep in mind|worth noting|good point)\b',
            r'\b(?:discovery|finding|realization|understanding)\b'
        ]
        
        # Meeting indicators
        self.meeting_indicators = [
            r'\b(?:meeting|standup|sync|review|retrospective|planning)\b',
            r'\b(?:agenda|minutes|notes|discussion|presentation)\b'
        ]
        
        # Priority indicators
        self.priority_indicators = {
            'high': [r'\b(?:urgent|critical|asap|priority|important|blocker)\b'],
            'medium': [r'\b(?:soon|next|upcoming|moderate)\b'],
            'low': [r'\b(?:later|future|nice to have|optional)\b']
        }
        
        # Common user name patterns
        self.user_patterns = [
            r'<@[A-Z0-9]+>',  # Slack user mentions
            r'\b[A-Z][a-z]+\b'  # Capitalized names
        ]
    
    async def extract_content_fallback(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract content using rule-based fallback methods"""
        try:
            if not messages:
                return None
            
            logger.info("Using fallback content extraction methods")
            
            # Combine all message text
            full_text = self._combine_message_text(messages)
            
            if not full_text.strip():
                return None
            
            # Extract different content types
            extracted = {
                "title": self._generate_title_fallback(full_text, messages),
                "summary": self._generate_summary_fallback(full_text),
                "decisions": self._extract_decisions_fallback(full_text),
                "action_items": self._extract_action_items_fallback(full_text),
                "key_insights": self._extract_insights_fallback(full_text),
                "participants": self._extract_participants_fallback(messages),
                "topics": self._extract_topics_fallback(full_text),
                "sentiment": self._analyze_sentiment_fallback(full_text),
                "priority": self._determine_priority_fallback(full_text),
                "confidence_score": self._calculate_fallback_confidence(full_text),
                "processing_method": "rule_based_fallback",
                "processed_at": datetime.utcnow().isoformat()
            }
            
            # Filter out empty results
            extracted = self._filter_empty_results(extracted)
            
            # Validate minimum content
            if not self._has_minimum_content(extracted):
                logger.warning("Fallback extraction produced insufficient content")
                return None
            
            logger.info(f"Fallback extraction completed with {extracted['confidence_score']:.2f} confidence")
            return extracted
            
        except Exception as e:
            logger.error(f"Fallback extraction failed: {str(e)}")
            return None
    
    def _combine_message_text(self, messages: List[Dict[str, Any]]) -> str:
        """Combine all message text into a single string"""
        texts = []
        
        for message in messages:
            text = message.get("text", "")
            if text and message.get("subtype") not in ["channel_join", "channel_leave", "bot_message"]:
                # Clean up text
                text = self._clean_text(text)
                if text.strip():
                    texts.append(text)
        
        return " ".join(texts)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove Slack formatting
        text = re.sub(r'<@[A-Z0-9]+>', '[User]', text)
        text = re.sub(r'<#[A-Z0-9]+\|([^>]+)>', r'#\1', text)
        text = re.sub(r'<(https?://[^|>]+)\|([^>]+)>', r'\2', text)
        text = re.sub(r'<(https?://[^>]+)>', r'\1', text)
        
        # Clean up code blocks and formatting
        text = re.sub(r'```[^`]*```', '[Code Block]', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _generate_title_fallback(self, text: str, messages: List[Dict[str, Any]]) -> str:
        """Generate a title using rule-based methods"""
        try:
            # Look for explicit meeting titles or topics
            meeting_match = re.search(r'(?:meeting|discussion|sync)(?:\s+(?:about|on|for|regarding))?\s+([^.!?]+)', text, re.IGNORECASE)
            if meeting_match:
                return f"Discussion: {meeting_match.group(1).strip().title()}"
            
            # Look for decision language
            decision_match = re.search(r'(?:decided?|decision)\s+(?:on|about|to)\s+([^.!?]+)', text, re.IGNORECASE)
            if decision_match:
                return f"Decision: {decision_match.group(1).strip().title()}"
            
            # Look for project or feature names
            project_match = re.search(r'(?:project|feature|issue)\s+([A-Z][a-zA-Z0-9\s-]+)', text)
            if project_match:
                return f"Project Discussion: {project_match.group(1).strip()}"
            
            # Fallback: Use first meaningful sentence
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                if len(sentence.strip()) > 10 and len(sentence.strip()) < 80:
                    return sentence.strip().title()
            
            # Final fallback
            return f"Conversation from {datetime.now().strftime('%Y-%m-%d')}"
            
        except Exception as e:
            logger.error(f"Title generation failed: {str(e)}")
            return "Slack Conversation"
    
    def _generate_summary_fallback(self, text: str) -> str:
        """Generate a summary using simple text analysis"""
        try:
            sentences = re.split(r'[.!?]+', text)
            
            # Find sentences with decision/conclusion language
            important_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                
                # Check for important keywords
                if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in 
                       self.decision_keywords + self.action_keywords[:2]):
                    important_sentences.append(sentence)
            
            if important_sentences:
                # Take first 2-3 important sentences
                summary_sentences = important_sentences[:3]
                summary = ". ".join(summary_sentences) + "."
                
                # Ensure reasonable length
                if len(summary) > 200:
                    summary = summary[:197] + "..."
                
                return summary
            
            # Fallback: Use first few sentences
            summary_sentences = [s.strip() for s in sentences[:3] if len(s.strip()) > 10]
            if summary_sentences:
                summary = ". ".join(summary_sentences) + "."
                if len(summary) > 200:
                    summary = summary[:197] + "..."
                return summary
            
            return "Team discussion with various topics covered."
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return "Conversation summary not available."
    
    def _extract_decisions_fallback(self, text: str) -> List[str]:
        """Extract decisions using pattern matching"""
        decisions = []
        
        try:
            sentences = re.split(r'[.!?]+', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                
                # Check for decision language
                for pattern in self.decision_keywords:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        # Clean up the sentence
                        decision = re.sub(r'^\w+\s*:', '', sentence).strip()
                        if len(decision) > 10:
                            decisions.append(decision)
                        break
            
            # Remove duplicates while preserving order
            seen = set()
            unique_decisions = []
            for decision in decisions:
                if decision.lower() not in seen:
                    seen.add(decision.lower())
                    unique_decisions.append(decision)
            
            return unique_decisions[:5]  # Limit to 5 decisions
            
        except Exception as e:
            logger.error(f"Decision extraction failed: {str(e)}")
            return []
    
    def _extract_action_items_fallback(self, text: str) -> List[str]:
        """Extract action items using pattern matching"""
        action_items = []
        
        try:
            sentences = re.split(r'[.!?]+', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                
                # Check for action language
                for pattern in self.action_keywords:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        # Try to identify assignee
                        assignee_match = re.search(r'\b([A-Z][a-z]+)\s+(?:will|should|needs? to|responsible)', sentence)
                        
                        action = sentence
                        if assignee_match:
                            assignee = assignee_match.group(1)
                            action = f"{assignee}: {sentence}"
                        
                        action_items.append(action)
                        break
            
            # Remove duplicates
            seen = set()
            unique_actions = []
            for action in action_items:
                if action.lower() not in seen:
                    seen.add(action.lower())
                    unique_actions.append(action)
            
            return unique_actions[:7]  # Limit to 7 action items
            
        except Exception as e:
            logger.error(f"Action item extraction failed: {str(e)}")
            return []
    
    def _extract_insights_fallback(self, text: str) -> List[str]:
        """Extract key insights using pattern matching"""
        insights = []
        
        try:
            sentences = re.split(r'[.!?]+', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 15:  # Insights should be more substantial
                    continue
                
                # Check for insight language
                for pattern in self.insight_keywords:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        insights.append(sentence)
                        break
            
            # Remove duplicates
            seen = set()
            unique_insights = []
            for insight in insights:
                if insight.lower() not in seen:
                    seen.add(insight.lower())
                    unique_insights.append(insight)
            
            return unique_insights[:5]  # Limit to 5 insights
            
        except Exception as e:
            logger.error(f"Insight extraction failed: {str(e)}")
            return []
    
    def _extract_participants_fallback(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract participant names from messages"""
        participants = set()
        
        try:
            for message in messages:
                user = message.get("user")
                if user and user != "USLACKBOT":
                    participants.add(user)
                
                # Also look for mentioned users in text
                text = message.get("text", "")
                mentions = re.findall(r'<@([A-Z0-9]+)>', text)
                participants.update(mentions)
            
            return list(participants)
            
        except Exception as e:
            logger.error(f"Participant extraction failed: {str(e)}")
            return []
    
    def _extract_topics_fallback(self, text: str) -> List[str]:
        """Extract topics using simple keyword analysis"""
        topics = []
        
        try:
            # Look for common topic indicators
            words = re.findall(r'\b[A-Z][a-z]+\b', text)
            
            # Count frequency of capitalized words (potential topics)
            word_freq = {}
            for word in words:
                if len(word) > 3 and word.lower() not in ['User', 'Code', 'Block']:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get most frequent words as topics
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            topics = [word for word, freq in sorted_words[:5] if freq > 1]
            
            # Add meeting type if detected
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.meeting_indicators):
                topics.append("Meeting")
            
            return topics
            
        except Exception as e:
            logger.error(f"Topic extraction failed: {str(e)}")
            return []
    
    def _analyze_sentiment_fallback(self, text: str) -> str:
        """Simple sentiment analysis using keyword matching"""
        try:
            positive_words = ['good', 'great', 'excellent', 'awesome', 'perfect', 'love', 'like', 'happy', 'excited']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'problem', 'issue', 'concern', 'worry']
            
            text_lower = text.lower()
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                return "positive"
            elif negative_count > positive_count:
                return "negative"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return "neutral"
    
    def _determine_priority_fallback(self, text: str) -> str:
        """Determine priority using keyword matching"""
        try:
            text_lower = text.lower()
            
            for priority, patterns in self.priority_indicators.items():
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        return priority
            
            # Default to medium if no indicators found
            return "medium"
            
        except Exception as e:
            logger.error(f"Priority determination failed: {str(e)}")
            return "medium"
    
    def _calculate_fallback_confidence(self, text: str) -> float:
        """Calculate confidence score for fallback extraction"""
        try:
            confidence = 0.4  # Base confidence for fallback
            
            # Increase confidence based on content indicators
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.decision_keywords):
                confidence += 0.15
            
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.action_keywords):
                confidence += 0.15
            
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.insight_keywords):
                confidence += 0.1
            
            # Length factor
            if len(text) > 500:
                confidence += 0.1
            elif len(text) < 100:
                confidence -= 0.1
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {str(e)}")
            return 0.4
    
    def _filter_empty_results(self, extracted: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out empty or meaningless results"""
        filtered = extracted.copy()
        
        # Remove empty lists
        for key in ["decisions", "action_items", "key_insights", "participants", "topics"]:
            if not filtered.get(key):
                filtered[key] = []
        
        # Ensure minimum string lengths
        if len(filtered.get("title", "")) < 5:
            filtered["title"] = "Slack Conversation"
        
        if len(filtered.get("summary", "")) < 10:
            filtered["summary"] = "Conversation summary not available."
        
        return filtered
    
    def _has_minimum_content(self, extracted: Dict[str, Any]) -> bool:
        """Check if extraction has minimum viable content"""
        content_score = 0
        
        if extracted.get("title") and len(extracted["title"]) > 5:
            content_score += 1
        
        if extracted.get("summary") and len(extracted["summary"]) > 20:
            content_score += 1
        
        if extracted.get("decisions"):
            content_score += 2
        
        if extracted.get("action_items"):
            content_score += 2
        
        if extracted.get("key_insights"):
            content_score += 1
        
        return content_score >= 2  # Minimum viable content
    
    async def create_simple_document_fallback(self, messages: List[Dict[str, Any]], 
                                            team_id: str) -> Optional[str]:
        """Create a simple document when AI extraction fails"""
        try:
            logger.info("Creating simple fallback document")
            
            # Basic document structure
            doc_content = {
                "title": f"Conversation from {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "summary": "Conversation captured from Slack (AI processing unavailable)",
                "raw_messages": [],
                "participants": set(),
                "message_count": len(messages),
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Process messages
            for message in messages:
                if message.get("subtype") not in ["channel_join", "channel_leave", "bot_message"]:
                    user = message.get("user", "Unknown")
                    text = self._clean_text(message.get("text", ""))
                    timestamp = message.get("ts", "")
                    
                    if text:
                        doc_content["raw_messages"].append({
                            "user": user,
                            "text": text,
                            "timestamp": timestamp
                        })
                        doc_content["participants"].add(user)
            
            doc_content["participants"] = list(doc_content["participants"])
            
            # This would integrate with Notion handler
            # For now, return a placeholder URL
            return f"https://notion.so/fallback-doc-{datetime.now().strftime('%Y%m%d%H%M')}"
            
        except Exception as e:
            logger.error(f"Fallback document creation failed: {str(e)}")
            return None


# Global fallback handler instance
fallback_handler = AIFallbackHandler()