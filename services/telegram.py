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
from typing import Any, Dict

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
        self.logger.info(
            f"Agent {agent_name} starting work",
            extra={
                "context": "agent_execution",
                "agent": agent_name,
                "task_description": task.get("description", ""),
                "expected_output": task.get("expected_output", ""),
                "stage": "start"
            }
        )

    def _log_agent_end(self, agent_name: str, output: str, task: Dict[str, Any]) -> None:
        """Log when an agent completes a task."""
        self.logger.info(
            f"Agent {agent_name} completed task",
            extra={
                "context": "agent_execution",
                "agent": agent_name,
                "output_preview": output[:200] if output else "",
                "task_description": task.get("description", ""),
                "stage": "complete"
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
        for trigger in ["inna", "ina"]:
            if cleaned_text.startswith(trigger):
                text = text[len(trigger):].strip()
                break
            
        try:
            self.logger.info(
                "Processing new message",
                extra={
                    "context": "message_handling",
                    "user_id": user.id,
                    "chat_id": message.chat_id,
                    "message_length": len(text),
                    "message_type": "text",
                    "original_message": text
                }
            )
            
            # Process the message
            message_id, chunks = await self._process_message(message)
            
            self.logger.debug(
                "Message processed",
                extra={
                    "context": "message_processing",
                    "message_id": message_id,
                    "chunks_count": len(chunks),
                    "chunks_content": chunks[:3]  # Log first 3 chunks for debugging
                }
            )
            
            # Get comprehensive context
            conversation_context = await self._get_conversation_context(message)
            
            self.logger.debug(
                "Retrieved conversation context",
                extra={
                    "context": "context_retrieval",
                    "user_context_size": len(conversation_context["user_context"]),
                    "chat_context_size": len(conversation_context["chat_context"]),
                    "file_context_size": len(conversation_context["file_context"]),
                    "user_context_sample": conversation_context["user_context"][:2],
                    "chat_context_sample": conversation_context["chat_context"][:2]
                }
            )
            
            # Create and run the crew with enhanced context and callbacks
            crew = Crew(
                agents=[self.planner, self.doer, self.critic, self.responder],
                tasks=[
                    {
                        "description": "Analyze user request and create a detailed plan",
                        "expected_output": "A structured plan for handling the user's request",
                        "agent": self.planner,
                        "context": [
                            {
                                "description": "User's request to analyze",
                                "expected_output": "Understanding of the request",
                                "role": "user",
                                "content": text
                            },
                            {
                                "description": "System context and available information",
                                "expected_output": "Context for planning",
                                "role": "system",
                                "content": json.dumps({
                                    "chunks": chunks,
                                    "user_context": conversation_context["user_context"],
                                    "chat_context": conversation_context["chat_context"],
                                    "file_context": conversation_context["file_context"]
                                })
                            }
                        ]
                    },
                    {
                        "description": "Execute the plan and process the request",
                        "expected_output": "Results from executing the plan",
                        "agent": self.doer,
                        "context": [
                            {
                                "description": "User's request to process",
                                "expected_output": "Understanding of the task",
                                "role": "user",
                                "content": text
                            },
                            {
                                "description": "System context and available information",
                                "expected_output": "Context for execution",
                                "role": "system",
                                "content": json.dumps({
                                    "chunks": chunks,
                                    "user_context": conversation_context["user_context"],
                                    "chat_context": conversation_context["chat_context"],
                                    "file_context": conversation_context["file_context"]
                                })
                            }
                        ]
                    },
                    {
                        "description": "Review and critique the execution results",
                        "expected_output": "Analysis and improvements of the results",
                        "agent": self.critic,
                        "context": [
                            {
                                "description": "User's original request",
                                "expected_output": "Understanding of requirements",
                                "role": "user",
                                "content": text
                            },
                            {
                                "description": "System context and execution results",
                                "expected_output": "Context for critique",
                                "role": "system",
                                "content": json.dumps({
                                    "chunks": chunks,
                                    "user_context": conversation_context["user_context"],
                                    "chat_context": conversation_context["chat_context"],
                                    "file_context": conversation_context["file_context"]
                                })
                            }
                        ]
                    },
                    {
                        "description": "Generate final response based on all previous work",
                        "expected_output": "A comprehensive and helpful response to the user's message",
                        "agent": self.responder,
                        "context": [
                            {
                                "description": "User's request to respond to",
                                "expected_output": "Understanding of user needs",
                                "role": "user",
                                "content": text
                            },
                            {
                                "description": "System context and previous agent outputs",
                                "expected_output": "Context for response generation",
                                "role": "system",
                                "content": json.dumps({
                                    "chunks": chunks,
                                    "user_context": conversation_context["user_context"],
                                    "chat_context": conversation_context["chat_context"],
                                    "file_context": conversation_context["file_context"]
                                })
                            }
                        ]
                    }
                ],
                process_callbacks={
                    "on_task_start": lambda agent, task: self._log_agent_start(agent.name, task),
                    "on_task_end": lambda agent, output, task: self._log_agent_end(agent.name, output, task)
                },
                verbose=True
            )
            
            self.logger.info(
                "Starting crew execution",
                extra={
                    "context": "crew_execution",
                    "agents": ["planner", "doer", "critic", "responder"],
                    "user_id": user.id,
                    "task_description": text
                }
            )
            
            # Get the response from the crew
            response = crew.kickoff()
            
            # Get all task outputs
            task_outputs = response.tasks_output
            final_response = task_outputs[-1]  # Last output is from the Responder
            
            # Log execution summary
            self.logger.info(
                "Crew execution completed",
                extra={
                    "context": "crew_execution",
                    "response_length": len(final_response),
                    "user_id": user.id,
                    "execution_summary": {
                        "total_tasks": len(task_outputs),
                        "task_outputs": [
                            {
                                "task_index": idx,
                                "output_length": len(str(output))
                            } for idx, output in enumerate(task_outputs)
                        ]
                    }
                }
            )
            
            # Send the response
            await message.reply_text(final_response, parse_mode='HTML')
            
        except Exception as e:
            self.logger.error(
                "Error processing message",
                extra={
                    "context": "message_handling",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "user_id": user.id,
                    "chat_id": message.chat_id
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