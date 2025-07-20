"""
AI Content Processor for SlackToDoc
Handles OpenAI GPT-4 integration for content extraction with production optimizations
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import openai
from openai import AsyncOpenAI
import tiktoken

# Configure logging
logger = logging.getLogger(__name__)

class AIProcessor:
    """Production-ready AI processor with OpenAI GPT-4 integration"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Model configuration
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
        
        # Rate limiting and retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        self.request_timeout = 60.0
        
        # Token counting for cost optimization
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.max_input_tokens = 8000  # Leave room for response
        
        # Prompt templates
        self.prompts = {
            "extract_content": self.get_content_extraction_prompt(),
            "summarize": self.get_summarization_prompt(),
            "categorize": self.get_categorization_prompt(),
            "sentiment": self.get_sentiment_analysis_prompt()
        }
    
    async def extract_content_from_messages(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract structured content from Slack messages using GPT-4"""
        try:
            # Preprocess messages
            formatted_conversation = self.format_conversation_for_ai(messages)
            
            # Check token count and truncate if necessary
            formatted_conversation = self.optimize_token_usage(formatted_conversation)
            
            if not formatted_conversation.strip():
                logger.warning("Empty conversation after preprocessing")
                return None
            
            # Build the extraction prompt
            system_prompt = self.prompts["extract_content"]
            user_prompt = f"""
Please analyze this Slack conversation and extract structured information:

{formatted_conversation}

Please provide your response in the following JSON format:
{{
    "title": "Brief descriptive title for this conversation",
    "summary": "2-3 sentence summary of the main topic and outcome",
    "decisions": ["list of decisions made"],
    "action_items": ["list of action items with assignees if mentioned"],
    "key_insights": ["list of important insights or learnings"],
    "participants": ["list of participant names"],
    "topics": ["list of main topics discussed"],
    "sentiment": "positive/neutral/negative",
    "confidence_score": 0.85,
    "priority": "high/medium/low"
}}
            """
            
            # Make AI request with retries
            response = await self.make_ai_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.temperature
            )
            
            if not response:
                logger.error("No response from AI service")
                return None
            
            # Parse and validate response
            extracted_content = self.parse_ai_response(response)
            
            if not extracted_content:
                logger.error("Failed to parse AI response")
                return None
            
            # Enhance with metadata
            extracted_content.update({
                "processed_at": datetime.utcnow().isoformat(),
                "message_count": len(messages),
                "ai_model": self.model,
                "processing_time_ms": None  # Will be set by caller
            })
            
            logger.info(f"Successfully extracted content from {len(messages)} messages")
            return extracted_content
            
        except Exception as e:
            logger.error(f"Content extraction failed: {str(e)}")
            return None
    
    def format_conversation_for_ai(self, messages: List[Dict[str, Any]]) -> str:
        """Format Slack messages for AI processing"""
        try:
            formatted_lines = []
            
            for message in messages:
                # Skip system messages and bot messages
                if message.get("subtype") in ["channel_join", "channel_leave", "bot_message"]:
                    continue
                
                timestamp = message.get("ts", "")
                user = message.get("user", "Unknown")
                text = message.get("text", "")
                
                # Clean up text
                text = self.clean_message_text(text)
                
                if text.strip():
                    # Format: [timestamp] User: message
                    time_str = self.format_timestamp(timestamp)
                    formatted_lines.append(f"[{time_str}] {user}: {text}")
            
            return "\n".join(formatted_lines)
            
        except Exception as e:
            logger.error(f"Message formatting error: {str(e)}")
            return ""
    
    def clean_message_text(self, text: str) -> str:
        """Clean and normalize message text"""
        if not text:
            return ""
        
        # Remove Slack formatting
        import re
        
        # Remove user mentions like <@U1234567>
        text = re.sub(r'<@[A-Z0-9]+>', '[User]', text)
        
        # Remove channel mentions like <#C1234567|general>
        text = re.sub(r'<#[A-Z0-9]+\|([^>]+)>', r'#\1', text)
        
        # Remove links but keep text like <https://example.com|link text>
        text = re.sub(r'<(https?://[^|>]+)\|([^>]+)>', r'\2 (\1)', text)
        text = re.sub(r'<(https?://[^>]+)>', r'\1', text)
        
        # Remove special formatting
        text = re.sub(r'```([^`]+)```', r'[Code: \1]', text)
        text = re.sub(r'`([^`]+)`', r'[\1]', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def format_timestamp(self, timestamp: str) -> str:
        """Format Slack timestamp for display"""
        try:
            if not timestamp:
                return "Unknown"
            
            # Slack timestamps are in format "1234567890.123456"
            ts_float = float(timestamp)
            dt = datetime.fromtimestamp(ts_float)
            return dt.strftime("%H:%M")
            
        except (ValueError, TypeError):
            return "Unknown"
    
    def optimize_token_usage(self, text: str) -> str:
        """Optimize token usage to stay within limits"""
        try:
            # Count tokens
            tokens = self.encoding.encode(text)
            
            if len(tokens) <= self.max_input_tokens:
                return text
            
            logger.warning(f"Text too long ({len(tokens)} tokens), truncating to {self.max_input_tokens}")
            
            # Truncate to max tokens, trying to preserve complete messages
            truncated_tokens = tokens[:self.max_input_tokens]
            truncated_text = self.encoding.decode(truncated_tokens)
            
            # Try to end at a complete line
            lines = truncated_text.split('\n')
            if len(lines) > 1:
                truncated_text = '\n'.join(lines[:-1])
            
            return truncated_text + "\n\n[Note: Conversation truncated due to length]"
            
        except Exception as e:
            logger.error(f"Token optimization error: {str(e)}")
            return text[:10000]  # Fallback to character limit
    
    async def make_ai_request(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> Optional[str]:
        """Make AI request with retries and error handling"""
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    timeout=self.request_timeout
                )
                
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content.strip()
                else:
                    logger.error("Empty response from OpenAI")
                    return None
                    
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit hit (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff for rate limits
                    delay = self.retry_delay * (2 ** attempt) * 2
                    await asyncio.sleep(delay)
                    continue
                raise
                
            except openai.APITimeoutError as e:
                logger.warning(f"Request timeout (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                raise
                
            except openai.APIError as e:
                logger.error(f"OpenAI API error (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                raise
                
            except Exception as e:
                logger.error(f"Unexpected error in AI request (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay
                    await asyncio.sleep(delay)
                    continue
                raise
        
        logger.error(f"All {self.max_retries} attempts failed")
        return None
    
    def parse_ai_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse and validate AI response"""
        try:
            # Extract JSON from response (in case there's extra text)
            import re
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
            
            # Parse JSON
            parsed = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["title", "summary"]
            for field in required_fields:
                if field not in parsed or not parsed[field]:
                    logger.warning(f"Missing required field: {field}")
                    parsed[field] = "Not provided"
            
            # Ensure lists are actually lists
            list_fields = ["decisions", "action_items", "key_insights", "participants", "topics"]
            for field in list_fields:
                if field not in parsed:
                    parsed[field] = []
                elif not isinstance(parsed[field], list):
                    # Convert string to single-item list
                    parsed[field] = [str(parsed[field])]
            
            # Validate confidence score
            if "confidence_score" not in parsed:
                parsed["confidence_score"] = 0.8
            else:
                try:
                    parsed["confidence_score"] = float(parsed["confidence_score"])
                    if not 0 <= parsed["confidence_score"] <= 1:
                        parsed["confidence_score"] = 0.8
                except (ValueError, TypeError):
                    parsed["confidence_score"] = 0.8
            
            # Validate sentiment
            valid_sentiments = ["positive", "neutral", "negative"]
            if parsed.get("sentiment") not in valid_sentiments:
                parsed["sentiment"] = "neutral"
            
            # Validate priority
            valid_priorities = ["high", "medium", "low"]
            if parsed.get("priority") not in valid_priorities:
                parsed["priority"] = "medium"
            
            logger.info("Successfully parsed AI response")
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Response parsing error: {str(e)}")
            return None
    
    async def summarize_content(self, content: str, max_length: int = 200) -> Optional[str]:
        """Generate a concise summary of content"""
        try:
            system_prompt = self.prompts["summarize"]
            user_prompt = f"""
Please provide a concise summary (max {max_length} characters) of the following content:

{content}

Focus on the main points and key outcomes.
            """
            
            response = await self.make_ai_request(system_prompt, user_prompt, temperature=0.2)
            
            if response and len(response) <= max_length * 1.2:  # Allow some flexibility
                return response
            elif response:
                # Truncate if too long
                return response[:max_length-3] + "..."
            else:
                return None
                
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            return None
    
    async def categorize_content(self, content: str) -> Optional[str]:
        """Categorize content into predefined categories"""
        try:
            system_prompt = self.prompts["categorize"]
            user_prompt = f"""
Please categorize the following content into one of these categories:
- meeting_notes
- decision_record  
- action_items
- knowledge_base
- technical_discussion
- brainstorming
- announcement
- question_answer

Content:
{content}

Return only the category name.
            """
            
            response = await self.make_ai_request(system_prompt, user_prompt, temperature=0.1)
            
            valid_categories = [
                "meeting_notes", "decision_record", "action_items", "knowledge_base",
                "technical_discussion", "brainstorming", "announcement", "question_answer"
            ]
            
            if response and response.strip().lower() in valid_categories:
                return response.strip().lower()
            else:
                return "knowledge_base"  # Default category
                
        except Exception as e:
            logger.error(f"Categorization error: {str(e)}")
            return "knowledge_base"
    
    async def analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment of content"""
        try:
            system_prompt = self.prompts["sentiment"]
            user_prompt = f"""
Please analyze the sentiment of the following content and provide:
1. Overall sentiment (positive/neutral/negative)
2. Confidence score (0.0 to 1.0)
3. Key emotional indicators

Content:
{content}

Format as JSON:
{{"sentiment": "positive", "confidence": 0.85, "indicators": ["excited", "collaborative"]}}
            """
            
            response = await self.make_ai_request(system_prompt, user_prompt, temperature=0.2)
            
            if response:
                parsed = self.parse_ai_response(response)
                if parsed and "sentiment" in parsed:
                    return parsed
            
            # Default response
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "indicators": []
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return {"sentiment": "neutral", "confidence": 0.5, "indicators": []}
    
    def get_content_extraction_prompt(self) -> str:
        """Get system prompt for content extraction"""
        return """
You are an expert assistant that analyzes team conversations and extracts structured information. 
Your goal is to identify decisions made, action items assigned, key insights shared, and important discussion points.

Focus on:
1. Concrete decisions and agreements reached
2. Specific action items with clear ownership when mentioned
3. Important insights, learnings, or knowledge shared
4. Main topics and themes discussed
5. Overall sentiment and priority level

Be concise but comprehensive. Only include clear, actionable items in action_items. 
Provide confidence scores based on how clear and explicit the content is.
        """
    
    def get_summarization_prompt(self) -> str:
        """Get system prompt for summarization"""
        return """
You are an expert at creating concise, informative summaries of team conversations.
Focus on the main topic, key outcomes, and any important decisions or next steps.
Keep summaries clear and actionable.
        """
    
    def get_categorization_prompt(self) -> str:
        """Get system prompt for categorization"""
        return """
You are an expert at categorizing team conversations based on their content and purpose.
Consider the main intent, outcome, and structure of the conversation to determine the most appropriate category.
        """
    
    def get_sentiment_analysis_prompt(self) -> str:
        """Get system prompt for sentiment analysis"""
        return """
You are an expert at analyzing the emotional tone and sentiment of team conversations.
Consider word choice, context, and overall team dynamics to assess sentiment accurately.
        """
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get AI usage statistics"""
        try:
            # This would typically pull from a database or cache
            # For now, return basic stats
            return {
                "requests_today": 0,
                "tokens_used_today": 0,
                "average_response_time": 0.0,
                "success_rate": 1.0,
                "current_model": self.model
            }
        except Exception as e:
            logger.error(f"Failed to get usage stats: {str(e)}")
            return {}


# Global AI processor instance
ai_processor = AIProcessor()

# Convenience function for backward compatibility
async def extract_content_from_messages(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract content from messages using the global AI processor"""
    return await ai_processor.extract_content_from_messages(messages)