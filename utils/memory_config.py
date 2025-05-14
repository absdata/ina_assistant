from crewai.memory import ShortTermMemory, LongTermMemory, EntityMemory
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
    
    # Create memory systems with custom embedding function
    short_term = ShortTermMemory(embedding_function=embedding_function)
    long_term = LongTermMemory(embedding_function=embedding_function)
    entity = EntityMemory(embedding_function=embedding_function)
    
    return {
        'short_term': short_term,
        'long_term': long_term,
        'entity': entity
    } 