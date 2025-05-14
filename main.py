from services.telegram import TelegramBot
import logging

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    """Start the Inna AI assistant bot."""
    try:
        # Create and start the bot
        bot = TelegramBot()
        logger.info("Starting Inna AI assistant...")
        bot.run()
        
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise

if __name__ == "__main__":
    main() 