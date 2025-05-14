from crewai import Agent
from typing import Dict, List, Any, Optional
from crewai.memory import ShortTermMemory, LongTermMemory, EntityMemory
import json
from config.logging_config import get_logger
from config.settings import (
    get_agent_trigger_names,
    get_agent_default_name,
    is_supported_document_type,
    get_document_type_name,
    SUPPORTED_DOCUMENT_TYPES
)
from pydantic import Field

class Responder(Agent):
    short_term_memory: ShortTermMemory = Field(default_factory=ShortTermMemory)
    long_term_memory: LongTermMemory = Field(default_factory=LongTermMemory)
    entity_memory: EntityMemory = Field(default_factory=EntityMemory)
    logger: Any = Field(default_factory=lambda: get_logger("agents.responder", "agent_initialization"))
    default_name: str = Field(default_factory=get_agent_default_name)

    def __init__(self):
        self.logger.info("Initializing Responder agent", extra={
            "memory_systems": ["short_term", "long_term", "entity"]
        })
        
        super().__init__(
            role='Responder',
            goal='Provide accurate and helpful responses to user queries',
            backstory="""You are an expert responder with deep knowledge in various fields.
            Your role is to provide accurate, complete, and helpful responses to user queries.""",
            allow_delegation=False,
            memory={
                'short_term': self.short_term_memory,
                'long_term': self.long_term_memory,
                'entity': self.entity_memory
            }
        )

    def should_respond(self, message: Dict[str, Any]) -> bool:
        """
        Determine if the agent should respond to the message.
        
        Args:
            message: Dictionary containing message details including text and file info
            
        Returns:
            bool: True if should respond, False otherwise
        """
        try:
            # Always respond to document uploads
            if message.get("file"):
                file_type = message["file"].get("type", "").lower()
                return is_supported_document_type(file_type)
            
            # Check for trigger words in text messages
            if message_text := message.get("text", "").lower():
                return any(name in message_text for name in get_agent_trigger_names())
            
            return False
            
        except Exception as e:
            self.logger.error(
                f"Error checking response condition: {str(e)}",
                extra={"context": "should_respond"}
            )
            return False

    def prepare_document_response(self, document: Dict[str, Any], user_id: str = None) -> str:
        """
        Prepare a response for a document upload.
        
        Args:
            document: Dictionary containing document details and content
            user_id: Optional user identifier for personalization
            
        Returns:
            str: HTML-formatted response about the document
        """
        try:
            self.logger.info(
                "Starting document response preparation",
                extra={
                    "context": "document_response",
                    "document_type": document.get("type"),
                    "document_name": document.get("name"),
                    "user_id": user_id
                }
            )
            
            # Store document in memory
            self._store_document_memory(document, user_id)
            
            # Get document type and name
            doc_type = document.get("type", "").lower()
            doc_name = document.get("name", "document")
            
            # Create document summary
            self.logger.debug(
                "Creating document summary",
                extra={
                    "context": "document_summary",
                    "document_type": doc_type
                }
            )
            summary = self._create_document_summary(document)
            
            # Get historical context if available
            historical_context = {}
            if user_id:
                self.logger.debug(
                    "Retrieving historical context",
                    extra={
                        "context": "historical_context",
                        "user_id": user_id
                    }
                )
                historical_context = self._get_historical_context(user_id)
            
            # Build response components
            greeting = self._create_document_greeting(doc_type, doc_name)
            main_content = self._format_document_content(summary, historical_context)
            suggestions = self._generate_document_suggestions(document, historical_context)
            
            # Combine components
            response = f"{greeting}\n\n{main_content}"
            
            if suggestions:
                response += f"\n\n{suggestions}"
            
            response += f"\n\n{self._create_document_closing()}"
            
            self.logger.info(
                "Document response prepared successfully",
                extra={
                    "context": "document_response",
                    "response_length": len(response),
                    "has_suggestions": bool(suggestions)
                }
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                "Error preparing document response",
                extra={
                    "context": "document_response",
                    "error": str(e),
                    "document_type": document.get("type"),
                    "document_name": document.get("name")
                },
                exc_info=True
            )
            raise

    def _store_document_memory(self, document: Dict[str, Any], user_id: Optional[str]):
        """Store document information in memory systems."""
        try:
            # Store in short-term memory
            self.short_term_memory.add(
                "last_document",
                {
                    "type": document.get("type"),
                    "name": document.get("name"),
                    "timestamp": self._get_timestamp(),
                    "user_id": user_id
                }
            )
            
            # Update entity memory with document entities
            if content := document.get("content"):
                entities = self._identify_entities(content)
                for entity in entities:
                    self.entity_memory.update_entity(entity)
            
            # Store in long-term memory
            self.long_term_memory.add(
                f"document_{self._get_timestamp()}",
                {
                    "document": document,
                    "user_id": user_id,
                    "timestamp": self._get_timestamp()
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"Error storing document memory: {str(e)}",
                extra={"context": "store_document"}
            )

    def _create_document_greeting(self, doc_type: str, doc_name: str) -> str:
        """Create a greeting for document upload."""
        try:
            doc_type_name = get_document_type_name(doc_type)
            return f"📄 I've received your {doc_type_name} <b>'{doc_name}'</b>! Let me analyze it for you..."
            
        except Exception as e:
            self.logger.error(
                f"Error creating document greeting: {str(e)}",
                extra={"context": "document_greeting"}
            )
            return "📄 I've received your document! Let me analyze it for you..."

    def _create_document_summary(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the document content."""
        try:
            summary = {
                "main_topics": self._extract_main_topics(document),
                "key_points": self._extract_key_points(document),
                "entities": self._extract_entities(document),
                "metadata": self._extract_document_metadata(document)
            }
            return summary
            
        except Exception as e:
            self.logger.error(
                f"Error creating document summary: {str(e)}",
                extra={"context": "document_summary"}
            )
            return {}

    def _format_document_content(self, summary: Dict[str, Any],
                               historical_context: Dict[str, Any]) -> str:
        """Format document summary content."""
        try:
            content = ["<b>Document Summary:</b>"]
            
            # Add main topics
            if main_topics := summary.get("main_topics"):
                content.append("\n<b>Main Topics:</b>")
                content.append(self._format_list(main_topics))
            
            # Add key points
            if key_points := summary.get("key_points"):
                content.append("\n<b>Key Points:</b>")
                content.append(self._format_list(key_points))
            
            # Add relevant historical insights
            if historical_insights := self._get_document_historical_insights(
                summary, historical_context
            ):
                content.append("\n<b>Related Information:</b>")
                content.append(self._format_list(historical_insights))
            
            return "\n".join(content)
            
        except Exception as e:
            self.logger.error(
                f"Error formatting document content: {str(e)}",
                extra={"context": "format_document"}
            )
            return "<b>Error processing document content</b>"

    def _generate_document_suggestions(self, document: Dict[str, Any],
                                    historical_context: Dict[str, Any]) -> str:
        """Generate suggestions based on document content."""
        try:
            suggestions = []
            
            # Add content-based suggestions
            if content_suggestions := self._get_content_suggestions(document):
                suggestions.extend(content_suggestions)
            
            # Add historical-based suggestions
            if historical_suggestions := self._get_historical_suggestions(
                document, historical_context
            ):
                suggestions.extend(historical_suggestions)
            
            if not suggestions:
                return ""
            
            return "<b>Suggestions:</b>\n" + self._format_list(suggestions)
            
        except Exception as e:
            self.logger.error(
                f"Error generating document suggestions: {str(e)}",
                extra={"context": "document_suggestions"}
            )
            return ""

    def _create_document_closing(self) -> str:
        """Create a closing message for document response."""
        return "Would you like me to help you understand any specific part of the document? Just ask! 📚"

    def _extract_main_topics(self, document: Dict[str, Any]) -> List[str]:
        """Extract main topics from document content."""
        # TODO: Implement topic extraction logic
        return []

    def _extract_key_points(self, document: Dict[str, Any]) -> List[str]:
        """Extract key points from document content."""
        # TODO: Implement key points extraction logic
        return []

    def _extract_document_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from document."""
        # TODO: Implement metadata extraction logic
        return {}

    def _get_document_historical_insights(self, summary: Dict[str, Any],
                                        historical_context: Dict[str, Any]) -> List[str]:
        """Get historical insights relevant to the document."""
        # TODO: Implement historical insights logic
        return []

    def _get_content_suggestions(self, document: Dict[str, Any]) -> List[str]:
        """Generate suggestions based on document content."""
        # TODO: Implement content suggestions logic
        return []

    def _get_historical_suggestions(self, document: Dict[str, Any],
                                  historical_context: Dict[str, Any]) -> List[str]:
        """Generate suggestions based on historical context."""
        # TODO: Implement historical suggestions logic
        return []

    def prepare_response(self, plan: Dict, results: Dict,
                        review: Dict, user_id: str = None) -> str:
        """
        Prepare the final response message for Telegram with memory enhancement.
        
        Args:
            plan (Dict): Original plan
            results (Dict): Execution results
            review (Dict): Critic's review
            user_id (str): Optional user identifier for personalization
            
        Returns:
            str: HTML-formatted message for Telegram
        """
        try:
            self.logger.info(
                "Preparing memory-enhanced response",
                extra={"context": "prepare_response"}
            )
            
            # Store current interaction in short-term memory
            interaction_data = {
                "plan": plan,
                "results": results,
                "review": review,
                "user_id": user_id
            }
            self.short_term_memory.add("current_interaction", interaction_data)
            
            # Update entity memory with relevant entities
            entities = self._extract_entities(results)
            for entity in entities:
                self.entity_memory.update_entity(entity)
            
            # Get historical context
            historical_context = self._get_historical_context(user_id) if user_id else {}
            
            # Determine response characteristics
            response_type = plan["request_type"]
            tone = self._determine_tone(review["final_verdict"])
            
            # Build enhanced response components
            greeting = self._create_personalized_greeting(response_type, user_id)
            main_content = self._create_enhanced_content(response_type, results, historical_context)
            context = self._add_enhanced_context(results, historical_context)
            closing = self._create_adaptive_closing(tone, review, user_id)
            
            # Combine components with HTML formatting
            response = f"{greeting}\n\n{main_content}"
            
            if context:
                response += f"\n\n{context}"
                
            response += f"\n\n{closing}"
            
            # Store response in long-term memory
            self._store_response_memory(response, plan, results, review, user_id)
            
            self.logger.debug(
                "Memory-enhanced response prepared successfully",
                extra={"context": "prepare_response"}
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                f"Error preparing memory-enhanced response: {str(e)}",
                extra={"context": "prepare_response"}
            )
            raise

    def _extract_entities(self, results: Dict) -> List[Dict[str, Any]]:
        """Extract relevant entities from results for entity memory."""
        try:
            entities = []
            
            # Extract from research findings
            if findings := results.get("research_findings"):
                for finding in findings:
                    entities.extend(self._identify_entities(finding))
            
            # Extract from subgoal results
            for result in results.get("subgoal_results", []):
                if output := result.get("output"):
                    entities.extend(self._identify_entities(output))
                    
            return entities
            
        except Exception as e:
            self.logger.error(
                f"Error extracting entities: {str(e)}",
                extra={"context": "entity_extraction"}
            )
            return []

    def _identify_entities(self, text: str) -> List[Dict[str, Any]]:
        """Identify entities in text."""
        # TODO: Implement entity identification logic
        return []

    def _get_historical_context(self, user_id: str) -> Dict[str, Any]:
        """Retrieve relevant historical context from memory systems."""
        try:
            # Get recent interactions from short-term memory
            recent_interactions = self.short_term_memory.get_recent(user_id)
            
            # Get relevant long-term patterns
            long_term_patterns = self.long_term_memory.search_relevant(user_id)
            
            # Get user-specific entities
            user_entities = self.entity_memory.get_entities(user_id)
            
            return {
                "recent_interactions": recent_interactions,
                "long_term_patterns": long_term_patterns,
                "user_entities": user_entities
            }
            
        except Exception as e:
            self.logger.error(
                f"Error getting historical context: {str(e)}",
                extra={"context": "historical_context"}
            )
            return {}

    def _create_personalized_greeting(self, response_type: str, user_id: Optional[str]) -> str:
        """Create a personalized greeting using memory context."""
        try:
            base_greeting = self._create_greeting(response_type)
            
            if not user_id:
                return base_greeting
                
            # Get user-specific context
            user_context = self._get_user_context(user_id)
            user_entities = self.entity_memory.get_entities(user_id)
            
            # Personalize greeting based on context
            if user_context.get("preferred_tone"):
                return self._adapt_greeting(base_greeting, user_context["preferred_tone"])
                
            return base_greeting
            
        except Exception as e:
            self.logger.error(
                f"Error creating personalized greeting: {str(e)}",
                extra={"context": "personalized_greeting"}
            )
            return self._create_greeting(response_type)

    def _create_enhanced_content(self, response_type: str, results: Dict,
                               historical_context: Dict) -> str:
        """Create enhanced main content using memory context."""
        try:
            # Get base content
            content = []
            
            # Add research findings
            if results.get("research_findings"):
                findings = self._format_research_findings(results["research_findings"])
                content.append(findings)
            
            # Add historical insights if relevant
            if historical_insights := self._get_relevant_historical_insights(
                response_type, historical_context
            ):
                content.append(self._format_historical_insights(historical_insights))
            
            # Add subgoal results with context
            for result in results.get("subgoal_results", []):
                if result.get("output"):
                    content.append(result["output"])
                if result.get("research_data"):
                    content.extend(self._format_list(result["research_data"]))
            
            # Format based on response type
            return self._format_response_type(response_type, content)
            
        except Exception as e:
            self.logger.error(
                f"Error creating enhanced content: {str(e)}",
                extra={"context": "enhanced_content"}
            )
            return self._create_main_content(response_type, results)

    def _get_relevant_historical_insights(self, response_type: str,
                                        historical_context: Dict) -> List[str]:
        """Get relevant historical insights for current response."""
        try:
            insights = []
            
            # Check recent interactions
            if recent := historical_context.get("recent_interactions"):
                insights.extend(self._filter_relevant_insights(recent, response_type))
            
            # Check long-term patterns
            if patterns := historical_context.get("long_term_patterns"):
                insights.extend(self._extract_pattern_insights(patterns))
                
            return insights
            
        except Exception as e:
            self.logger.error(
                f"Error getting historical insights: {str(e)}",
                extra={"context": "historical_insights"}
            )
            return []

    def _filter_relevant_insights(self, interactions: List[Dict],
                                response_type: str) -> List[str]:
        """Filter relevant insights from interactions."""
        # TODO: Implement insight filtering logic
        return []

    def _extract_pattern_insights(self, patterns: List[Dict]) -> List[str]:
        """Extract insights from interaction patterns."""
        # TODO: Implement pattern insight extraction
        return []

    def _format_historical_insights(self, insights: List[str]) -> str:
        """Format historical insights for inclusion in response."""
        if not insights:
            return ""
            
        formatted = "<b>Related Insights:</b>\n"
        formatted += self._format_list(insights)
        return formatted

    def _format_response_type(self, response_type: str, content: List[str]) -> str:
        """Format response based on type with memory context."""
        formatters = {
            "question": self._format_answer,
            "task": self._format_task_result,
            "analysis": self._format_analysis,
            "advice": self._format_advice,
            "conversation": self._format_conversation
        }
        
        formatter = formatters.get(response_type, self._format_conversation)
        return formatter(content)

    def _store_response_memory(self, response: str, plan: Dict,
                             results: Dict, review: Dict, user_id: Optional[str]):
        """Store response data in long-term memory."""
        try:
            memory_data = {
                "response": response,
                "plan": plan,
                "results": results,
                "review": review,
                "user_id": user_id,
                "timestamp": self._get_timestamp()
            }
            
            self.long_term_memory.add(
                f"response_{self._get_timestamp()}",
                memory_data
            )
            
        except Exception as e:
            self.logger.error(
                f"Error storing response memory: {str(e)}",
                extra={"context": "store_memory"}
            )

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()

    def _adapt_greeting(self, base_greeting: str, preferred_tone: str) -> str:
        """Adapt greeting based on preferred tone."""
        # TODO: Implement greeting adaptation logic
        return base_greeting

    def _determine_tone(self, verdict: str) -> str:
        """Determine the response tone based on the review verdict."""
        if verdict == "excellent":
            return "enthusiastic"
        elif verdict == "good":
            return "positive"
        elif verdict == "fair":
            return "constructive"
        else:
            return "encouraging"

    def _create_greeting(self, response_type: str) -> str:
        """Create a contextual greeting."""
        greetings = {
            "question": "🤔 Let me help you with that!",
            "task": "💪 I'm on it!",
            "analysis": "🔍 I've looked into this carefully...",
            "advice": "💡 Here's what I think...",
            "conversation": "👋 Hey there!"
        }
        
        return greetings.get(response_type, "👋 Hi!")

    def _create_main_content(self, response_type: str, results: Dict) -> str:
        """Create the main response content."""
        content = []
        
        # Add research findings if available
        if results["research_findings"]:
            findings = self._format_research_findings(results["research_findings"])
            content.append(findings)
        
        # Add subgoal results
        for result in results["subgoal_results"]:
            if result["output"]:
                content.append(result["output"])
            if result["research_data"]:
                content.extend(self._format_list(result["research_data"]))
        
        # Format based on response type
        if response_type == "question":
            return self._format_answer(content)
        elif response_type == "task":
            return self._format_task_result(content)
        elif response_type == "analysis":
            return self._format_analysis(content)
        elif response_type == "advice":
            return self._format_advice(content)
        else:
            return self._format_conversation(content)

    def _add_relevant_context(self, results: Dict) -> str:
        """Add relevant context from additional findings."""
        if not results.get("additional_context"):
            return ""
            
        context = "<b>Additional Context:</b>\n"
        context += self._format_list(results["additional_context"])
        return context

    def _create_closing(self, tone: str, review: Dict) -> str:
        """Create a contextual closing message."""
        closings = {
            "enthusiastic": "🌟 Is there anything else you'd like to know? I'm here to help!",
            "positive": "😊 Let me know if you need any clarification!",
            "constructive": "💭 Feel free to ask follow-up questions!",
            "encouraging": "🤝 I'm here to help you succeed!"
        }
        
        # Add improvements if needed
        if review["improvements"]:
            improvements = self._format_list(review["improvements"][:2])
            return f"{improvements}\n\n{closings[tone]}"
            
        return closings[tone]

    def _format_research_findings(self, findings: List[str]) -> str:
        """Format research findings with HTML."""
        if not findings:
            return ""
            
        formatted = "<b>Key Findings:</b>\n"
        formatted += self._format_list(findings)
        return formatted

    def _format_answer(self, content: List[str]) -> str:
        """Format an answer response."""
        return "\n\n".join(content)

    def _format_task_result(self, content: List[str]) -> str:
        """Format a task result."""
        return "<b>Here's what I did:</b>\n" + self._format_list(content)

    def _format_analysis(self, content: List[str]) -> str:
        """Format an analysis response."""
        return "<b>Analysis Results:</b>\n" + self._format_list(content)

    def _format_advice(self, content: List[str]) -> str:
        """Format advice response."""
        return "<b>My Advice:</b>\n" + self._format_list(content)

    def _format_conversation(self, content: List[str]) -> str:
        """Format a conversational response."""
        return "\n\n".join(content)

    def _format_list(self, items: List[str]) -> str:
        """Format a list with bullet points."""
        return "\n".join(f"• {item}" for item in items if item) 