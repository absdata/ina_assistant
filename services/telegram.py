from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from services.file_handler import FileHandler
from agents.planner import Planner
from agents.doer import Doer
from agents.critic import Critic
from agents.responder import Responder
from crewai import Crew
from config.settings import TELEGRAM_BOT_TOKEN
from config.logging_config import get_logger
from utils.llm_config import create_llm_config
import asyncio
import json
import logging
from typing import Any, Dict
from datetime import datetime

class TelegramBot:
    def __init__(self):
        # Create LLM configuration
        llm_config = create_llm_config()
        
        # Initialize agents with LLM configuration
        self.file_handler = FileHandler()
        self.planner = Planner(llm=llm_config)
        self.doer = Doer(llm=llm_config)
        self.critic = Critic(llm=llm_config)
        self.responder = Responder(llm=llm_config)
        
        self.logger = get_logger("services.telegram", "bot_service")
        # Set logger to DEBUG level
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.info("Initializing Telegram bot service", extra={
            "agents": ["planner", "doer", "critic", "responder"],
            "services": ["file_handler"]
        })
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        user = update.message.from_user
        self.logger.info(
            "New user started the bot",
            extra={
                "context": "start_command",
                "user_id": user.id,
                "username": user.username,
                "chat_id": update.message.chat_id
            }
        )
        
        welcome_message = (
            "👋 Hi! I'm Inna, your startup co-founder and AI assistant. "
            "I'm here to help you with anything you need!\n\n"
            "Just start your message with 'Inna' or 'Ina' and I'll be happy to help."
        )
        await update.message.reply_text(welcome_message)

    def _log_agent_start(self, agent_name: str, task: Dict[str, Any]) -> None:
        """Log when an agent starts working on a task."""
        self.logger.debug(
            f"=== Agent {agent_name} Starting Work ===",
            extra={
                "agent_start": {
                    "agent": agent_name,
                    "task_description": task.get("description", ""),
                    "dependencies": task.get("dependencies", []),
                    "context": [
                        {
                            "role": ctx.get("role", "unknown"),
                            "content_preview": ctx.get("content", "")[:200]
                        } for ctx in task.get("context", [])
                    ]
                }
            }
        )

    def _log_agent_end(self, agent_name: str, output: str, task: Dict[str, Any]) -> None:
        """Log when an agent completes a task."""
        self.logger.debug(
            f"=== Agent {agent_name} Completed Task ===",
            extra={
                "agent_completion": {
                    "agent": agent_name,
                    "task_description": task.get("description", ""),
                    "output_preview": str(output)[:500] if output else "",
                    "dependencies": task.get("dependencies", []),
                    "output_length": len(str(output)) if output else 0
                }
            }
        )

    async def _process_task_output(self, task_outputs: list) -> None:
        """Process and log task outputs."""
        for idx, output in enumerate(task_outputs):
            self.logger.debug(
                f"=== Task {idx + 1} Output ===",
                extra={
                    "task_output": {
                        "task_index": idx,
                        "output_preview": str(output)[:500],
                        "output_length": len(str(output))
                    }
                }
            )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages with enhanced context awareness."""
        message = update.message
        text = message.text or ""
        user = message.from_user
        
        # Check if message starts with "Inna" or "Ina"
        if not text.lower().startswith(("inna", "ina")):
            return
            
        # Remove trigger words (Inna/Ina) from the beginning of the message
        cleaned_text = text.lower()
        original_text = text  # Store original text before cleaning
        for trigger in ["inna", "ina"]:
            if cleaned_text.startswith(trigger):
                text = text[len(trigger):].strip()
                break
        
        # Log the initial message with clear formatting
        self.logger.debug(
            "=== Starting Message Processing ===",
            extra={
                "message_details": {
                    "original_message": original_text,
                    "cleaned_message": text,
                    "user_info": {
                        "user_id": user.id,
                        "chat_id": message.chat_id,
                        "username": user.username
                    }
                }
            }
        )
        
        # Log the actual message content separately for visibility
        self.logger.info(
            f"Processing message: '{text}'",
        )
            
        try:
            # Process the message
            message_id, chunks = await self._process_message(message)
            
            # Log chunks with clear message reference
            self.logger.debug(
                "=== Retrieved File Chunks ===",
                extra={
                    "chunks_details": {
                        "message": text,
                        "total_chunks": len(chunks),
                        "chunks_preview": [
                            {
                                "index": idx,
                                "content": str(chunk)[:500],
                                "type": type(chunk).__name__
                            } for idx, chunk in enumerate(chunks[:2])
                        ] if chunks else []
                    }
                }
            )
            
            # Get comprehensive context
            conversation_context = await self._get_conversation_context(message)
            
            # Format conversation history for better readability
            formatted_history = []
            for msg in conversation_context["chat_context"]:
                created_at = msg.get("created_at", "")
                msg_text = msg.get("message_text", "")
                is_file = bool(msg.get("file_content"))
                entry = {
                    "timestamp": created_at,
                    "text": msg_text,
                    "has_file": is_file,
                    "type": "user_message"
                }
                formatted_history.append(entry)
            
            # Add file context
            for file in conversation_context["file_context"]:
                if file.get("file_content"):
                    formatted_history.append({
                        "timestamp": file.get("created_at", ""),
                        "text": f"[File: {file.get('file_name', 'unknown')}] {file.get('file_content', '')}",
                        "has_file": True,
                        "type": "file_content"
                    })

            # Sort by timestamp
            formatted_history.sort(key=lambda x: x["timestamp"])
            
            # Create a readable version of the history
            readable_history = "\n\n".join([
                f"[{entry['timestamp']}] {'📄 ' if entry['has_file'] else ''}{entry['text'][:500]}"
                for entry in formatted_history[-5:]  # Last 5 interactions
            ])

            # Prepare shared context with enhanced history
            shared_context = {
                "current_query": {
                    "text": text,
                    "timestamp": datetime.now().isoformat(),
                },
                "conversation_history": {
                    "messages": formatted_history,
                    "total_messages": len(conversation_context["chat_context"]),
                    "has_files": any(msg.get("file_content") for msg in conversation_context["chat_context"])
                },
                "available_context": {
                    "chunks": [str(chunk) for chunk in chunks],
                    "file_context": [
                        {
                            "name": file.get("file_name"),
                            "type": file.get("file_type"),
                            "content": file.get("file_content", "")[:1000]  # Preview of file content
                        }
                        for file in conversation_context["file_context"]
                        if file.get("file_content")
                    ]
                }
            }

            # Create base context with all required fields
            request_context = {
                "description": "Original user request and conversation history",
                "role": "system",
                "expected_output": "Understanding of the complete conversation context",
                "content": (
                    f"Original request: '{text}'\n\n"
                    f"Recent conversation history:\n{readable_history}"
                )
            }
            
            resources_context = {
                "description": "Available context and resources",
                "role": "system",
                "expected_output": "Access to all available contextual information",
                "content": json.dumps(shared_context)
            }

            # Define task-specific contexts
            planner_context = {
                "description": "Planning instructions",
                "role": "user",
                "expected_output": "A clear plan with specific steps for execution",
                "content": (
                    "Create a detailed plan that:\n"
                    "1. Addresses the user's request directly\n"
                    "2. Takes into account the conversation history\n"
                    "3. Utilizes available resources and context\n"
                    "4. Specifies clear steps for other agents to follow"
                )
            }

            doer_context = {
                "description": "Execution instructions",
                "role": "user",
                "expected_output": "Concrete results from following the plan",
                "content": (
                    "Execute the plan from the Strategic Planner while:\n"
                    "1. Following the specified steps\n"
                    "2. Maintaining conversation coherence\n"
                    "3. Using all available context and resources\n"
                    "4. Providing clear results for review"
                )
            }

            critic_context = {
                "description": "Review instructions",
                "role": "user",
                "expected_output": "Detailed analysis and improvement suggestions",
                "content": (
                    "Review the execution results while considering:\n"
                    "1. Accuracy and completeness of the response\n"
                    "2. Consistency with conversation history\n"
                    "3. Proper use of available context\n"
                    "4. Suggest specific improvements if needed"
                )
            }

            responder_context = {
                "description": "Response generation instructions",
                "role": "user",
                "expected_output": "A clear, coherent, and contextually appropriate response",
                "content": (
                    "Generate a final response that:\n"
                    "1. Directly addresses the user's request\n"
                    "2. Incorporates the execution results\n"
                    "3. Maintains conversation coherence\n"
                    "4. Uses appropriate tone and formatting"
                )
            }

            # Create tasks with properly structured contexts
            tasks = [
                {
                    "description": f"Analyze user request: '{text}' with conversation history",
                    "expected_output": "A structured plan for handling the user's request",
                    "agent": self.planner,
                    "context": [request_context, resources_context, planner_context]
                },
                {
                    "description": "Execute the plan with context awareness",
                    "expected_output": "Results from executing the plan",
                    "agent": self.doer,
                    "context": [request_context, resources_context, doer_context],
                    "dependencies": [0]
                },
                {
                    "description": "Review execution in conversation context",
                    "expected_output": "Analysis and improvements of the results",
                    "agent": self.critic,
                    "context": [request_context, resources_context, critic_context],
                    "dependencies": [0, 1]
                },
                {
                    "description": "Generate contextually appropriate response",
                    "expected_output": "A comprehensive and contextually appropriate response",
                    "agent": self.responder,
                    "context": [request_context, resources_context, responder_context],
                    "dependencies": [0, 1, 2]
                }
            ]

            # Log the setup for debugging
            self.logger.debug(
                "=== Conversation Context Setup ===",
                extra={
                    "setup_details": {
                        "current_query": text,
                        "history_sample": readable_history[:500],
                        "total_context": {
                            "messages": len(conversation_context["chat_context"]),
                            "files": len(conversation_context["file_context"]),
                            "chunks": len(chunks)
                        }
                    }
                }
            )

            crew = Crew(
                agents=[self.planner, self.doer, self.critic, self.responder],
                tasks=tasks,
                process_callbacks={
                    "on_task_start": lambda agent, task: self._log_agent_start(agent.name, task),
                    "on_task_end": lambda agent, output, task: self._log_agent_end(agent.name, output, task)
                },
                process=Process.sequential,
                verbose=True,
                memory=True
            )

            self.logger.debug(
                "=== Starting Crew Execution ===",
                extra={
                    "execution_details": {
                        "message": text,
                        "first_task": {
                            "description": tasks[0]["description"],
                            "agent": tasks[0]["agent"].__class__.__name__,
                            "context_preview": tasks[0]["context"][0]["content"]
                        },
                        "available_resources": {
                            "chunks": len(chunks),
                            "file_context": len(conversation_context["file_context"])
                        }
                    }
                }
            )
            
            # Get the response from the crew
            response = crew.kickoff()
            
            # Get all task outputs and process them
            task_outputs = response.tasks_output
            await self._process_task_output(task_outputs)
            
            final_response = str(task_outputs[-1])
            
            self.logger.debug(
                "=== Crew Execution Results ===",
                extra={
                    "execution_results": {
                        "original_query": text,
                        "total_tasks": len(task_outputs),
                        "final_response_preview": final_response[:500],
                        "final_response_length": len(final_response)
                    }
                }
            )
            
            # Send the response
            await message.reply_text(final_response, parse_mode='HTML')
            
        except Exception as e:
            self.logger.error(
                "=== Error Processing Message ===",
                extra={
                    "error_details": {
                        "error_message": str(e),
                        "error_type": type(e).__name__,
                        "original_message": text,
                        "user_id": user.id,
                        "chat_id": message.chat_id,
                        "stack_trace": True
                    }
                },
                exc_info=True
            )
            await message.reply_text(
                "I apologize, but I encountered an error while processing your request. "
                "Please try again later."
            )

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle document uploads with enhanced context awareness."""
        message = update.message
        document = message.document
        user = message.from_user
        
        self.logger.info(
            "Received document upload",
            extra={
                "context": "document_handling",
                "user_id": user.id,
                "chat_id": message.chat_id,
                "file_name": document.file_name,
                "file_size": document.file_size,
                "mime_type": document.mime_type
            }
        )
        
        # Check if file type is supported
        file_type = self._get_file_type(document.file_name)
        if not file_type:
            self.logger.warning(
                "Unsupported file type",
                extra={
                    "context": "document_handling",
                    "file_name": document.file_name,
                    "user_id": user.id
                }
            )
            await message.reply_text(
                "Sorry, I can only process PDF, DOCX, and TXT files at the moment."
            )
            return
            
        try:
            # Download the file
            self.logger.debug(
                "Downloading document",
                extra={
                    "context": "document_processing",
                    "file_type": file_type,
                    "file_id": document.file_id
                }
            )
            
            file = await context.bot.get_file(document.file_id)
            file_content = await file.download_as_bytearray()
            
            self.logger.debug(
                "Document downloaded successfully",
                extra={
                    "context": "document_processing",
                    "content_size": len(file_content)
                }
            )
            
            # Process the file with context
            message_id, chunks = await self._process_file(
                file_content=bytes(file_content),
                file_name=document.file_name,
                file_type=file_type,
                message=message
            )
            
            self.logger.info(
                "Document processed successfully",
                extra={
                    "context": "document_processing",
                    "message_id": message_id,
                    "chunks_count": len(chunks),
                    "file_type": file_type
                }
            )
            
            # Get file-specific context
            file_context = await self._get_file_context(message)
            
            # Enhanced response with file context
            response = (
                f"I've processed your {file_type.upper()} file and saved its contents. "
                f"I found {len(chunks)} relevant sections to learn from. "
                f"You have shared {len(file_context)} files with me so far. "
                "Feel free to ask me questions about any of them!"
            )
            
            await message.reply_text(response)
            
        except Exception as e:
            self.logger.error(
                "Error processing document",
                extra={
                    "context": "document_handling",
                    "error": str(e),
                    "file_name": document.file_name,
                    "file_type": file_type,
                    "user_id": user.id
                },
                exc_info=True
            )
            await message.reply_text(
                "I apologize, but I encountered an error while processing your file. "
                "Please try again later."
            )

    async def _process_message(self, message) -> tuple:
        """Process a text message with context."""
        return await asyncio.to_thread(
            self.file_handler.ina_process_message,
            message_text=message.text,
            user_id=message.from_user.id,
            chat_id=message.chat_id
        )

    async def _process_file(self, file_content, file_name, file_type, message) -> tuple:
        """Process an uploaded file with context."""
        return await asyncio.to_thread(
            self.file_handler.ina_process_file,
            file_content=file_content,
            file_name=file_name,
            file_type=file_type,
            user_id=message.from_user.id,
            chat_id=message.chat_id,
            message_text=message.caption or ""
        )

    async def _get_conversation_context(self, message) -> dict:
        """Get comprehensive conversation context."""
        return await asyncio.to_thread(
            self.file_handler.ina_get_conversation_context,
            user_id=message.from_user.id,
            chat_id=message.chat_id
        )

    async def _get_file_context(self, message) -> list:
        """Get file-specific context for the user."""
        return await asyncio.to_thread(
            self.file_handler.vector_store.ina_get_file_context,
            user_id=message.from_user.id
        )

    def _get_file_type(self, file_name: str) -> str:
        """Get the file type from the file name."""
        lower_name = file_name.lower()
        if lower_name.endswith('.pdf'):
            return 'pdf'
        elif lower_name.endswith('.docx'):
            return 'docx'
        elif lower_name.endswith('.txt'):
            return 'txt'
        return None

    def run(self):
        """Start the bot."""
        # Create application
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        application.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
        
        # Start the bot
        application.run_polling() 