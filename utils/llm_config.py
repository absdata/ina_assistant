from crewai import LLM
import os
from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT_NAME
)

def create_llm_config() -> LLM:
    """Create LLM configuration for agents using Azure OpenAI."""
 #   return LLM(
 #       model=f"azure/{AZURE_OPENAI_DEPLOYMENT_NAME}",  # Prefix with 'azure/' to indicate Azure provider
 #       api_key=AZURE_OPENAI_API_KEY,
 #       base_url=AZURE_OPENAI_ENDPOINT,
 #       api_version=AZURE_OPENAI_API_VERSION,
 #   ) 

    return AzureChatOpenAI(
        openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY")
    )