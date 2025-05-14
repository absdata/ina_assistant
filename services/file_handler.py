from typing import Tuple, List, Optional, Dict
from memory.chunking import TextChunker
from services.embedding import EmbeddingService
from memory.vector_store import VectorStore

class FileHandler:
    def __init__(self):
        self.chunker = TextChunker()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        
    def ina_process_file(self, file_content: bytes, file_name: str,
                    file_type: str, user_id: int, chat_id: int,
                    message_text: str) -> Tuple[str, List[str]]:
        """Process an uploaded file with context awareness."""
        # Parse and chunk the file
        full_text, chunks = self.chunker.parse_file(file_content, file_type)
        
        # Save the message and get message ID
        message_id = self.vector_store.ina_save_message(
            user_id=user_id,
            chat_id=chat_id,
            message_text=message_text,
            file_content=full_text,
            file_name=file_name,
            file_type=file_type
        )
        
        # Get embeddings for chunks
        embeddings = self.embedding_service.get_embeddings(chunks)
        
        # Save chunks and embeddings
        self.vector_store.ina_save_embeddings(message_id, chunks, embeddings)
        
        return message_id, chunks
        
    def ina_process_message(self, message_text: str, user_id: int,
                       chat_id: int) -> Tuple[str, List[str]]:
        """Process a text message with context awareness."""
        # Chunk the message text
        chunks = self.chunker.chunk_text(message_text)
        
        # Save the message and get message ID
        message_id = self.vector_store.ina_save_message(
            user_id=user_id,
            chat_id=chat_id,
            message_text=message_text
        )
        
        # Get embeddings for chunks
        embeddings = self.embedding_service.get_embeddings(chunks)
        
        # Save chunks and embeddings
        self.vector_store.ina_save_embeddings(message_id, chunks, embeddings)
        
        return message_id, chunks
        
    def ina_search_context(self, query: str, user_id: int = None,
                      chat_id: int = None, time_window: int = None,
                      limit: int = 5) -> List[str]:
        """
        Search for relevant context with user and time awareness.
        
        Args:
            query: Search query
            user_id: Filter by specific user
            chat_id: Filter by specific chat
            time_window: Time window in days
            limit: Maximum number of results
        """
        # Get query embedding
        query_embedding = self.embedding_service.get_embedding(query)
        
        # Search for similar chunks with context
        results = self.vector_store.ina_search_similar(
            query_embedding=query_embedding,
            limit=limit,
            user_id=user_id,
            time_window=time_window
        )
        
        # Extract and return the text chunks
        return [result["chunk_text"] for result in results]
        
    def ina_get_conversation_context(self, user_id: int, chat_id: int,
                               limit: int = 5) -> Dict[str, List[Dict]]:
        """
        Get comprehensive conversation context.
        
        Returns:
            Dict with user context, chat context, and file context
        """
        return {
            "user_context": self.vector_store.ina_get_user_context(user_id, limit),
            "chat_context": self.vector_store.ina_get_chat_context(chat_id, limit),
            "file_context": self.vector_store.ina_get_file_context(user_id)
        } 