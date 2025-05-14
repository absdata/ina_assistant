from crewai import Agent, LLM
from typing import Dict, List, Any, Optional, ClassVar
from crewai.memory import ShortTermMemory, LongTermMemory, EntityMemory
import json
from config.logging_config import get_logger
from pydantic import Field, BaseModel
from utils.memory_config import create_memory_systems

class Critic(Agent):
    model_config: ClassVar[dict] = {"arbitrary_types_allowed": True}
    short_term_memory: ShortTermMemory = Field(default_factory=ShortTermMemory)
    long_term_memory: LongTermMemory = Field(default_factory=LongTermMemory)
    entity_memory: EntityMemory = Field(default_factory=EntityMemory)
    logger: Any = Field(default_factory=lambda: get_logger("agents.critic", "agent_initialization"))

    def __init__(self, llm: LLM = None, **data):
        # Initialize memory systems with Azure embeddings
        memory_systems = create_memory_systems()
        
        # Initialize logger
        logger = get_logger("agents.critic", "agent_initialization")
        
        # Initialize parent class
        super().__init__(
            role='Critic',
            goal='Evaluate and improve the quality of responses',
            backstory="""You are an expert critic with deep knowledge in various fields.
            Your role is to evaluate responses for accuracy, completeness, and helpfulness,
            while suggesting improvements when necessary.""",
            allow_delegation=False,
            llm=llm,
            memory=memory_systems,
            **data
        )
        
        # Store initialized instances
        self.short_term_memory = memory_systems['short_term']
        self.long_term_memory = memory_systems['long_term']
        self.entity_memory = memory_systems['entity']
        self.logger = logger
        
        self.logger.info("Initializing Critic agent", extra={
            "memory_systems": ["short_term", "long_term", "entity"]
        })

    def ina_evaluate_response(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a response and provide feedback using memory-enhanced analysis.
        
        Args:
            response: The response to evaluate
            context: Dictionary containing relevant context (user query, chat history, etc.)
            
        Returns:
            Dictionary containing evaluation results and suggestions
        """
        try:
            self.logger.info(
                "Evaluating response with memory context",
                extra={"context": "evaluate_response"}
            )

            # Store current interaction in short-term memory
            self.short_term_memory.add(
                "current_evaluation",
                {"response": response, "context": context}
            )

            # Update entity memory with any identified entities
            entities = self._extract_entities(response)
            for entity in entities:
                self.entity_memory.update_entity(entity)

            # Get historical context from long-term memory
            historical_context = self.long_term_memory.search_relevant(response)
            
            # Combine current context with historical insights
            enhanced_context = {
                **context,
                "historical_context": historical_context,
                "entities": entities
            }

            evaluation = {
                "quality_score": self._assess_quality(response),
                "accuracy_score": self._assess_accuracy(response, enhanced_context),
                "completeness_score": self._assess_completeness(response, enhanced_context),
                "suggestions": self._generate_suggestions(response, enhanced_context),
                "improvements": self._suggest_improvements(response),
                "historical_insights": self._get_historical_insights(response)
            }

            # Store evaluation in long-term memory
            self.long_term_memory.add(
                f"evaluation_{context.get('message_id', 'unknown')}",
                evaluation
            )

            self.logger.debug(
                f"Memory-enhanced evaluation completed: {evaluation}",
                extra={"context": "evaluate_response"}
            )

            return evaluation

        except Exception as e:
            self.logger.error(
                f"Error in memory-enhanced evaluation: {str(e)}",
                extra={"context": "evaluate_response"}
            )
            raise

    def _extract_entities(self, response: str) -> List[Dict[str, Any]]:
        """Extract entities from response for entity memory."""
        try:
            # Implement entity extraction logic here
            # Example structure:
            entities = []
            # TODO: Add entity extraction implementation
            return entities
        except Exception as e:
            self.logger.error(
                f"Error extracting entities: {str(e)}",
                extra={"context": "entity_extraction"}
            )
            return []

    def _get_historical_insights(self, response: str) -> List[Dict[str, Any]]:
        """Get relevant historical insights from long-term memory."""
        try:
            insights = self.long_term_memory.search_similar(response)
            return self._process_historical_insights(insights)
        except Exception as e:
            self.logger.error(
                f"Error getting historical insights: {str(e)}",
                extra={"context": "historical_insights"}
            )
            return []

    def _process_historical_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and filter historical insights for relevance."""
        try:
            processed_insights = []
            for insight in insights:
                if self._is_insight_relevant(insight):
                    processed_insights.append(self._format_insight(insight))
            return processed_insights
        except Exception as e:
            self.logger.error(
                f"Error processing historical insights: {str(e)}",
                extra={"context": "process_insights"}
            )
            return []

    def _is_insight_relevant(self, insight: Dict[str, Any]) -> bool:
        """Determine if a historical insight is relevant to current context."""
        # Implement relevance checking logic
        return True  # Placeholder

    def _format_insight(self, insight: Dict[str, Any]) -> Dict[str, Any]:
        """Format historical insight for current evaluation context."""
        # Implement insight formatting logic
        return insight  # Placeholder

    def _assess_quality(self, response: str) -> float:
        """Assess the overall quality of the response."""
        try:
            # Evaluate factors like clarity, coherence, and tone
            factors = {
                "clarity": self._evaluate_clarity(response),
                "coherence": self._evaluate_coherence(response),
                "tone": self._evaluate_tone(response)
            }
            
            # Calculate weighted average
            weights = {"clarity": 0.4, "coherence": 0.4, "tone": 0.2}
            score = sum(score * weights[factor] for factor, score in factors.items())
            
            return round(score, 2)
            
        except Exception as e:
            self.logger.error(
                f"Error assessing quality: {str(e)}",
                extra={"context": "assess_quality"}
            )
            raise

    def _assess_accuracy(self, response: str, context: Dict[str, Any]) -> float:
        """Assess the accuracy of the response against available context."""
        try:
            # Check factual consistency with context
            accuracy_score = self._check_factual_consistency(response, context)
            
            # Verify technical correctness if applicable
            if self._contains_technical_content(response):
                technical_score = self._verify_technical_accuracy(response)
                accuracy_score = (accuracy_score + technical_score) / 2
                
            return round(accuracy_score, 2)
            
        except Exception as e:
            self.logger.error(
                f"Error assessing accuracy: {str(e)}",
                extra={"context": "assess_accuracy"}
            )
            raise

    def _assess_completeness(self, response: str, context: Dict[str, Any]) -> float:
        """Assess how completely the response addresses the user's query."""
        try:
            # Check if all parts of the query are addressed
            query_coverage = self._check_query_coverage(response, context)
            
            # Evaluate depth of explanation
            depth_score = self._evaluate_depth(response)
            
            # Combined score
            completeness_score = (query_coverage + depth_score) / 2
            
            return round(completeness_score, 2)
            
        except Exception as e:
            self.logger.error(
                f"Error assessing completeness: {str(e)}",
                extra={"context": "assess_completeness"}
            )
            raise

    def _generate_suggestions(self, response: str, context: Dict[str, Any]) -> List[str]:
        """Generate specific suggestions for improving the response."""
        try:
            suggestions = []
            
            # Check for missing context
            if missing_context := self._identify_missing_context(response, context):
                suggestions.append(f"Add context about: {', '.join(missing_context)}")
            
            # Check for clarity improvements
            if clarity_issues := self._identify_clarity_issues(response):
                suggestions.extend(clarity_issues)
            
            # Check for technical improvements
            if technical_suggestions := self._get_technical_suggestions(response):
                suggestions.extend(technical_suggestions)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(
                f"Error generating suggestions: {str(e)}",
                extra={"context": "generate_suggestions"}
            )
            raise

    def _suggest_improvements(self, response: str) -> List[str]:
        """Suggest specific improvements for the response."""
        try:
            improvements = []
            
            # Style improvements
            if style_improvements := self._get_style_improvements(response):
                improvements.extend(style_improvements)
            
            # Structure improvements
            if structure_improvements := self._get_structure_improvements(response):
                improvements.extend(structure_improvements)
            
            # Content improvements
            if content_improvements := self._get_content_improvements(response):
                improvements.extend(content_improvements)
            
            return improvements
            
        except Exception as e:
            self.logger.error(
                f"Error suggesting improvements: {str(e)}",
                extra={"context": "suggest_improvements"}
            )
            raise

    # Helper methods for evaluation components
    def _evaluate_clarity(self, response: str) -> float:
        """Evaluate the clarity of the response."""
        # Implementation for clarity evaluation
        return 0.0  # Placeholder

    def _evaluate_coherence(self, response: str) -> float:
        """Evaluate the coherence of the response."""
        # Implementation for coherence evaluation
        return 0.0  # Placeholder

    def _evaluate_tone(self, response: str) -> float:
        """Evaluate the tone of the response."""
        # Implementation for tone evaluation
        return 0.0  # Placeholder

    def _check_factual_consistency(self, response: str, context: Dict[str, Any]) -> float:
        """Check factual consistency with context."""
        # Implementation for factual consistency check
        return 0.0  # Placeholder

    def _verify_technical_accuracy(self, response: str) -> float:
        """Verify technical accuracy of the response."""
        # Implementation for technical accuracy verification
        return 0.0  # Placeholder

    def _contains_technical_content(self, response: str) -> bool:
        """Check if response contains technical content."""
        # Implementation for technical content detection
        return False  # Placeholder

    def _check_query_coverage(self, response: str, context: Dict[str, Any]) -> float:
        """Check how well the response covers the query."""
        # Implementation for query coverage check
        return 0.0  # Placeholder

    def _evaluate_depth(self, response: str) -> float:
        """Evaluate the depth of the explanation."""
        # Implementation for depth evaluation
        return 0.0  # Placeholder

    def _identify_missing_context(self, response: str, context: Dict[str, Any]) -> List[str]:
        """Identify missing context in the response."""
        # Implementation for missing context identification
        return []  # Placeholder

    def _identify_clarity_issues(self, response: str) -> List[str]:
        """Identify clarity issues in the response."""
        # Implementation for clarity issues identification
        return []  # Placeholder

    def _get_technical_suggestions(self, response: str) -> List[str]:
        """Get technical improvement suggestions."""
        # Implementation for technical suggestions
        return []  # Placeholder

    def _get_style_improvements(self, response: str) -> List[str]:
        """Get style improvement suggestions."""
        # Implementation for style improvements
        return []  # Placeholder

    def _get_structure_improvements(self, response: str) -> List[str]:
        """Get structure improvement suggestions."""
        # Implementation for structure improvements
        return []  # Placeholder

    def _get_content_improvements(self, response: str) -> List[str]:
        """Get content improvement suggestions."""
        # Implementation for content improvements
        return []  # Placeholder 