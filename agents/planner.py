from crewai import Agent
from typing import List, Dict
from crewai import LLM
import json

class Planner(Agent):
    def __init__(self, llm: LLM = None):
        super().__init__(
            role="Strategic Planner",
            goal="Break down user requests into clear, actionable subgoals",
            backstory="""You are a strategic planner who excels at breaking down complex
            requests into smaller, manageable tasks. You analyze user intentions and
            create structured plans.""",
            allow_delegation=False,
            llm=llm
        )

    def analyze_request(self, message: str, chunks: List[str]) -> Dict:
        """
        Analyze the user's request and break it down into subgoals.
        
        Args:
            message (str): User's message
            chunks (List[str]): Relevant context chunks
            
        Returns:
            Dict: Plan with subgoals and context
        """
        # Clean the message (remove "Inna" or "Ina" prefix)
        clean_message = message.lower()
        if clean_message.startswith("inna"):
            clean_message = clean_message[4:].strip()
        elif clean_message.startswith("ina"):
            clean_message = clean_message[3:].strip()

        # Analyze the request type
        request_type = self._determine_request_type(clean_message)
        
        # Create subgoals based on request type
        subgoals = self._create_subgoals(request_type, clean_message)
        
        # Identify relevant context
        relevant_context = self._filter_relevant_chunks(chunks, clean_message)
        
        return {
            "original_message": message,
            "clean_message": clean_message,
            "request_type": request_type,
            "subgoals": subgoals,
            "context": relevant_context
        }

    def _determine_request_type(self, message: str) -> str:
        """Determine the type of request based on the message content."""
        message = message.lower()
        
        if any(q in message for q in ["what", "how", "why", "when", "where", "who"]):
            return "question"
        elif any(cmd in message for cmd in ["do", "create", "make", "build", "generate"]):
            return "task"
        elif any(word in message for word in ["analyze", "review", "evaluate", "assess"]):
            return "analysis"
        elif any(word in message for word in ["help", "suggest", "recommend", "advice"]):
            return "advice"
        else:
            return "conversation"

    def _create_subgoals(self, request_type: str, message: str) -> List[Dict]:
        """Create specific subgoals based on request type."""
        subgoals = []
        
        if request_type == "question":
            subgoals = [
                {
                    "type": "research",
                    "description": "Search for relevant information",
                    "priority": 1
                },
                {
                    "type": "analyze",
                    "description": "Analyze and synthesize findings",
                    "priority": 2
                },
                {
                    "type": "formulate",
                    "description": "Formulate clear and concise answer",
                    "priority": 3
                }
            ]
        elif request_type == "task":
            subgoals = [
                {
                    "type": "plan",
                    "description": "Create detailed task plan",
                    "priority": 1
                },
                {
                    "type": "execute",
                    "description": "Execute task steps",
                    "priority": 2
                },
                {
                    "type": "verify",
                    "description": "Verify task completion",
                    "priority": 3
                }
            ]
        elif request_type == "analysis":
            subgoals = [
                {
                    "type": "gather",
                    "description": "Gather relevant data",
                    "priority": 1
                },
                {
                    "type": "analyze",
                    "description": "Perform detailed analysis",
                    "priority": 2
                },
                {
                    "type": "summarize",
                    "description": "Summarize key findings",
                    "priority": 3
                }
            ]
        elif request_type == "advice":
            subgoals = [
                {
                    "type": "understand",
                    "description": "Understand the situation",
                    "priority": 1
                },
                {
                    "type": "research",
                    "description": "Research solutions",
                    "priority": 2
                },
                {
                    "type": "advise",
                    "description": "Provide actionable advice",
                    "priority": 3
                }
            ]
        else:  # conversation
            subgoals = [
                {
                    "type": "engage",
                    "description": "Engage in conversation",
                    "priority": 1
                },
                {
                    "type": "respond",
                    "description": "Generate appropriate response",
                    "priority": 2
                }
            ]
            
        return subgoals

    def _filter_relevant_chunks(self, chunks: List[str], message: str) -> List[str]:
        """Filter and prioritize relevant context chunks."""
        if not chunks:
            return []
            
        # Simple relevance scoring based on keyword matching
        scored_chunks = []
        keywords = set(message.lower().split())
        
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            relevance_score = len(keywords.intersection(chunk_words))
            scored_chunks.append((chunk, relevance_score))
        
        # Sort by relevance score and take top 3
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored_chunks[:3]] 