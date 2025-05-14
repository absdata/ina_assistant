from crewai import Agent
from typing import List, Dict
from services.file_handler import FileHandler
import json
from pydantic import Field

class Doer(Agent):
    file_handler: FileHandler = Field(default_factory=FileHandler)

    def __init__(self):
        super().__init__(
            role="Task Executor",
            goal="Execute tasks and perform research efficiently",
            backstory="""You are a highly efficient executor who gets things done.
            You excel at research, analysis, and completing tasks with attention to detail.""",
            allow_delegation=False
        )

    def execute_plan(self, plan: Dict) -> Dict:
        """
        Execute the plan created by the Planner.
        
        Args:
            plan (Dict): Plan with subgoals and context
            
        Returns:
            Dict: Results of execution
        """
        results = {
            "request_type": plan["request_type"],
            "subgoal_results": [],
            "research_findings": [],
            "additional_context": []
        }
        
        # Execute each subgoal
        for subgoal in plan["subgoals"]:
            result = self._execute_subgoal(
                subgoal,
                plan["clean_message"],
                plan["context"]
            )
            results["subgoal_results"].append(result)
            
            # Add any research findings
            if result.get("research_data"):
                results["research_findings"].extend(result["research_data"])
                
            # Add any additional context discovered
            if result.get("additional_context"):
                results["additional_context"].extend(result["additional_context"])
        
        return results

    def _execute_subgoal(self, subgoal: Dict, message: str,
                        context: List[str]) -> Dict:
        """Execute a single subgoal."""
        subgoal_type = subgoal["type"]
        result = {
            "type": subgoal_type,
            "description": subgoal["description"],
            "status": "completed",
            "output": None,
            "research_data": [],
            "additional_context": []
        }
        
        if subgoal_type == "research":
            result.update(self._do_research(message, context))
            
        elif subgoal_type == "analyze":
            result.update(self._analyze_data(message, context))
            
        elif subgoal_type == "execute":
            result.update(self._execute_task(message, context))
            
        elif subgoal_type == "gather":
            result.update(self._gather_data(message, context))
            
        elif subgoal_type in ["understand", "engage"]:
            result.update(self._process_context(message, context))
            
        return result

    def _do_research(self, message: str, context: List[str]) -> Dict:
        """Perform research using available context and search."""
        # Search for relevant information in context
        relevant_info = self.file_handler.search_context(message)
        
        return {
            "output": "Research completed",
            "research_data": relevant_info,
            "additional_context": []
        }

    def _analyze_data(self, message: str, context: List[str]) -> Dict:
        """Analyze data and extract insights."""
        insights = []
        
        # Combine message context and research context
        all_context = context + self.file_handler.search_context(message)
        
        # Extract key points and patterns
        for ctx in all_context:
            # Simple keyword-based analysis
            keywords = set(ctx.lower().split())
            if len(keywords.intersection(set(message.lower().split()))) > 0:
                insights.append(ctx)
        
        return {
            "output": "Analysis completed",
            "research_data": insights,
            "additional_context": []
        }

    def _execute_task(self, message: str, context: List[str]) -> Dict:
        """Execute a specific task."""
        # For now, just process the task description
        task_steps = []
        
        # Break down the task into steps
        task_words = message.split()
        current_step = []
        
        for word in task_words:
            current_step.append(word)
            if len(current_step) >= 5:  # Arbitrary step size
                task_steps.append(" ".join(current_step))
                current_step = []
        
        if current_step:
            task_steps.append(" ".join(current_step))
        
        return {
            "output": "Task execution completed",
            "research_data": task_steps,
            "additional_context": []
        }

    def _gather_data(self, message: str, context: List[str]) -> Dict:
        """Gather data from various sources."""
        gathered_data = []
        
        # Search in existing context
        relevant_context = self.file_handler.search_context(message)
        gathered_data.extend(relevant_context)
        
        return {
            "output": "Data gathering completed",
            "research_data": gathered_data,
            "additional_context": []
        }

    def _process_context(self, message: str, context: List[str]) -> Dict:
        """Process and understand context."""
        processed_context = []
        
        # Combine and process all available context
        all_context = context + self.file_handler.search_context(message)
        
        for ctx in all_context:
            # Simple relevance check
            if any(word in ctx.lower() for word in message.lower().split()):
                processed_context.append(ctx)
        
        return {
            "output": "Context processing completed",
            "research_data": processed_context,
            "additional_context": []
        } 