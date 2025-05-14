from typing import List, Optional
import numpy as np
from openai import AzureOpenAI
from chromadb.api.types import Documents, EmbeddingFunction
from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION
)

class AzureOpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        batch_size: int = 128,
    ):
        self.api_key = api_key or AZURE_OPENAI_API_KEY
        self.endpoint = endpoint or AZURE_OPENAI_ENDPOINT
        self.deployment = deployment or AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        self.api_version = api_version or AZURE_OPENAI_API_VERSION
        self.batch_size = batch_size
        
        if not self.api_key:
            raise ValueError("Azure OpenAI API key is required")
        if not self.endpoint:
            raise ValueError("Azure OpenAI endpoint is required")
            
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )

    def __call__(self, texts: Documents) -> List[List[float]]:
        if not texts:
            return []
            
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.deployment,
                input=batch
            )
            embeddings.extend([data.embedding for data in response.data])
            
        return embeddings 