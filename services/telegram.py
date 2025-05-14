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
import asyncio
import logging

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self):
        self.file_handler = FileHandler()
        self.planner = Planner()
        self.doer = Doer()
        self.critic = Critic()
        self.responder = Responder()
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        welcome_message = (
            "👋 Hi! I'm Inna, your startup co-founder and AI assistant. "
            "I'm here to help you with anything you need!\n\n"
            "Just start your message with 'Inna' or 'Ina' and I'll be happy to help."
        )
        await update.message.reply_text(welcome_message)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages with enhanced context awareness."""
        message = update.message
        text = message.text or ""
        
        # Check if message starts with "Inna" or "Ina"
        if not text.lower().startswith(("inna", "ina")):
            return
            
        try:
            # Process the message
            message_id, chunks = await self._process_message(message)
            
            # Get comprehensive context
            conversation_context = await self._get_conversation_context(message)
            
            # Create and run the crew with enhanced context
            crew = Crew(
                agents=[self.planner, self.doer, self.critic, self.responder],
                tasks=[
                    {
                        "message": text,
                        "chunks": chunks,
                        "user_id": message.from_user.id,
                        "chat_id": message.chat_id,
                        "context": {
                            "current_chunks": chunks,
                            "user_context": conversation_context["user_context"],
                            "chat_context": conversation_context["chat_context"],
                            "file_context": conversation_context["file_context"]
                        }
                    }
                ]
            )
            
            # Get the response from the crew
            response = await crew.run()
            
            # Send the response
            await message.reply_text(response, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await message.reply_text(
                "I apologize, but I encountered an error while processing your request. "
                "Please try again later."
            )

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle document uploads with enhanced context awareness."""
        message = update.message
        document = message.document
        
        # Check if file type is supported
        file_type = self._get_file_type(document.file_name)
        if not file_type:
            await message.reply_text(
                "Sorry, I can only process PDF, DOCX, and TXT files at the moment."
            )
            return
            
        try:
            # Download the file
            file = await context.bot.get_file(document.file_id)
            file_content = await file.download_as_bytearray()
            
            # Process the file with context
            message_id, chunks = await self._process_file(
                file_content=bytes(file_content),
                file_name=document.file_name,
                file_type=file_type,
                message=message
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
            logger.error(f"Error processing document: {e}")
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