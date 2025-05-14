import logging
import logging.handlers
import os
from datetime import datetime
import json

class CustomFormatter(logging.Formatter):
    """Custom formatter that properly handles structured data in extra fields."""
    def format(self, record):
        # Format the basic message
        if hasattr(record, 'message_details'):
            record.msg = f"\n{json.dumps(record.message_details, indent=2)}"
        elif hasattr(record, 'chunks_details'):
            record.msg = f"\n{json.dumps(record.chunks_details, indent=2)}"
        elif hasattr(record, 'context_details'):
            record.msg = f"\n{json.dumps(record.context_details, indent=2)}"
        elif hasattr(record, 'agent_context'):
            record.msg = f"\n{json.dumps(record.agent_context, indent=2)}"
        elif hasattr(record, 'execution_details'):
            record.msg = f"\n{json.dumps(record.execution_details, indent=2)}"
        elif hasattr(record, 'execution_results'):
            record.msg = f"\n{json.dumps(record.execution_results, indent=2)}"
        elif hasattr(record, 'error_details'):
            record.msg = f"\n{json.dumps(record.error_details, indent=2)}"
        
        # Add any extra fields that aren't already handled
        extra_fields = {
            key: value for key, value in record.__dict__.items()
            if key not in logging.LogRecord('').__dict__ 
            and key not in ['message_details', 'chunks_details', 'context_details', 
                          'agent_context', 'execution_details', 'execution_results', 
                          'error_details']
        }
        
        if extra_fields:
            record.msg = f"{record.msg}\nExtra: {json.dumps(extra_fields, indent=2)}"
        
        return super().format(record)

def setup_logging():
    """Configure logging for both console and file output."""
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Create a formatter that includes timestamp, level, and message
    formatter = CustomFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create console handler with colors for different levels
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)  # Changed to DEBUG to see all messages

    # Create file handler with daily rotation
    log_file = os.path.join(logs_dir, f"ina_{datetime.now().strftime('%Y%m')}.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=30
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    root_logger.handlers = []
    
    # Add our handlers
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
        logger.propagate = True
        logger.setLevel(logging.DEBUG)

    return loggers

def get_logger(name: str, context: str = 'general') -> logging.Logger:
    """Get a logger with context."""
    logger = logging.getLogger(f"ina.{name}")
    return logger  # Return the logger directly, no adapter needed anymore 