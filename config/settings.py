from dotenv import load_dotenv
import os
from typing import List

# Load environment variables
load_dotenv()

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
AZURE_EMBEDDING_DIMENSION = int(os.getenv("AZURE_EMBEDDING_DIMENSION", "2000"))

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_EMBEDDING_DIMENSION = int(os.getenv("SUPABASE_EMBEDDING_DIMENSION", "2000"))

# Agent Configuration
AGENT_TRIGGER_NAMES = os.getenv("AGENT_TRIGGER_NAMES", "ina,inna").lower().split(",")
AGENT_DEFAULT_NAME = os.getenv("AGENT_DEFAULT_NAME", "Ina")

# Document Types Configuration
SUPPORTED_DOCUMENT_TYPES = {
    "pdf": "PDF document",
    "docx": "Word document",
    "txt": "text file",
    "doc": "Word document"
}

# Validate required environment variables
required_vars = [
    "TELEGRAM_BOT_TOKEN",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
    "SUPABASE_URL",
    "SUPABASE_KEY"
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

def get_agent_trigger_names() -> List[str]:
    """Get the list of agent trigger names."""
    return AGENT_TRIGGER_NAMES

def get_agent_default_name() -> str:
    """Get the default agent name."""
    return AGENT_DEFAULT_NAME

def is_supported_document_type(file_type: str) -> bool:
    """Check if a file type is supported."""
    return file_type.lower() in SUPPORTED_DOCUMENT_TYPES

def get_document_type_name(file_type: str) -> str:
    """Get the friendly name for a document type."""
    return SUPPORTED_DOCUMENT_TYPES.get(file_type.lower(), "document") 