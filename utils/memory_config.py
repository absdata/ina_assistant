from crewai.memory import ShortTermMemory, LongTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from utils.azure_embeddings import AzureOpenAIEmbeddingFunction
from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION
)

def create_memory_systems():
    """Create memory systems with Azure OpenAI embeddings."""
    # Create Azure embedding function
    embedding_function = AzureOpenAIEmbeddingFunction(
        api_key=AZURE_OPENAI_API_KEY,
        endpoint=AZURE_OPENAI_ENDPOINT,
        deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION
    )
    
    # Create RAG storage with custom embedding function
    rag_storage = RAGStorage(embedding_function=embedding_function)
    
    # Create memory systems with RAG storage
    short_term = ShortTermMemory(storage=rag_storage)
    long_term = LongTermMemory(storage=rag_storage)
    entity = EntityMemory(storage=rag_storage)
    
    return {
        'short_term': short_term,
        'long_term': long_term,
        'entity': entity
    } 