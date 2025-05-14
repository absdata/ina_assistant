from crewai.memory import ShortTermMemory, LongTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from utils.azure_embeddings import AzureOpenAIEmbeddingFunction
import os
from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION
)

def create_memory_systems():
    """Create memory systems using Azure OpenAI embedding config."""

    embedder_config = {
        "provider": "azure",
        "config": {
            "model": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
            "api_key_env_var": os.getenv("AZURE_OPENAI_API_KEY"),
            "endpoint_env_var": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        }
    }

    memory_path = "./memory/"

    short_term_memory = ShortTermMemory(
        storage=RAGStorage(
            embedder_config=embedder_config,
            type="short_term",
            path=memory_path
        )
    )

    long_term_memory = LongTermMemory(
        storage=RAGStorage(
            embedder_config=embedder_config,
            type="long_term",  # отличие от short_term
            path=memory_path
        )
    )

    return {
        'short_term': short_term_memory,
        'long_term': long_term_memory
    }