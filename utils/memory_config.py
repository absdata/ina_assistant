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
        "provider": "openai",
        "config": {
            "model": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
        }
    }

    rag_storage = RAGStorage(
        embedder_config=embedder_config,
        type="short_term",
        path="/var/getinn/ina_assistant/memory/"
    )

    short_term = ShortTermMemory(storage=rag_storage)
    long_term = LongTermMemory(storage=rag_storage)
    entity = EntityMemory(storage=rag_storage)

    return {
        'short_term': short_term,
        'long_term': long_term,
        'entity': entity
    }