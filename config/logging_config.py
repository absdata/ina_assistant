import logging
import logging.handlers
import os
from datetime import datetime

def setup_logging():
    """Configure logging for both console and file output."""
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Create a formatter that includes timestamp, level, and context
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(context)s] - %(message)s'
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Create file handler with daily rotation
    log_file = os.path.join(logs_dir, f"ina_{datetime.now().strftime('%Y%m')}.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=30  # Keep logs for 30 days
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Create loggers for each component
    loggers = {
        "telegram": logging.getLogger("ina.telegram"),
        "vector_store": logging.getLogger("ina.vector_store"),
        "file_handler": logging.getLogger("ina.file_handler"),
        "embedding": logging.getLogger("ina.embedding"),
        "chunking": logging.getLogger("ina.chunking"),
        "planner": logging.getLogger("ina.agents.planner"),
        "doer": logging.getLogger("ina.agents.doer"),
        "critic": logging.getLogger("ina.agents.critic"),
        "responder": logging.getLogger("ina.agents.responder")
    }

    # Configure each logger
    for logger in loggers.values():
        logger.propagate = True  # Allow propagation to root logger
        logger.setLevel(logging.DEBUG)

    return loggers

class LoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter that adds context to log messages."""
    def process(self, msg, kwargs):
        # Add default context if none provided
        if 'context' not in self.extra:
            self.extra['context'] = 'general'
        return msg, kwargs

def get_logger(name: str, context: str = 'general') -> LoggerAdapter:
    """Get a logger with context."""
    logger = logging.getLogger(f"ina.{name}")
    return LoggerAdapter(logger, {'context': context}) 