from typing import List, Dict, Any
from supabase import create_client, Client
from config.settings import SUPABASE_URL, SUPABASE_KEY
from config.logging_config import get_logger
import uuid
from datetime import datetime, timedelta

class VectorStore:
    def __init__(self):
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.logger = get_logger("vector_store")

    def ina_save_message(self, user_id: int, chat_id: int, message_text: str,
                    file_content: str = None, file_name: str = None,
                    file_type: str = None) -> str:
        """Save a message to the database."""
        try:
            self.logger.info(
                f"Saving message for user {user_id} in chat {chat_id}",
                extra={"context": "save_message"}
            )
            
            data = {
                "user_id": user_id,
                "chat_id": chat_id,
                "message_text": message_text,
                "file_content": file_content,
                "file_name": file_name,
                "file_type": file_type
            }
            
            result = self.supabase.table("ina_messages").insert(data).execute()
            message_id = result.data[0]["id"]
            
            self.logger.debug(
                f"Message saved successfully with ID: {message_id}",
                extra={"context": "save_message"}
            )
            return message_id
            
        except Exception as e:
            self.logger.error(
                f"Error saving message: {str(e)}",
                extra={"context": "save_message"}
            )
            raise

    def ina_save_embeddings(self, message_id: str, chunks: List[str],
                       embeddings: List[List[float]]) -> None:
        """Save text chunks and their embeddings."""
        try:
            self.logger.info(
                f"Saving {len(chunks)} embeddings for message {message_id}",
                extra={"context": "save_embeddings"}
            )
            
            data = [
                {
                    "message_id": message_id,
                    "chunk_text": chunk,
                    "chunk_index": idx,
                    "embedding": embedding
                }
                for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ]
            
            self.supabase.table("ina_message_embeddings").insert(data).execute()
            
            self.logger.debug(
                "Embeddings saved successfully",
                extra={"context": "save_embeddings"}
            )
            
        except Exception as e:
            self.logger.error(
                f"Error saving embeddings: {str(e)}",
                extra={"context": "save_embeddings"}
            )
            raise

    def ina_search_similar(self, query_embedding: List[float], limit: int = 5,
                      threshold: float = 0.7, user_id: int = None,
                      time_window: int = None) -> List[Dict[str, Any]]:
        """
        Search for similar messages with user and time context.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            threshold: Similarity threshold (0-1)
            user_id: Filter by specific user
            time_window: Time window in days for context
        """
        try:
            self.logger.info(
                f"Searching similar messages (limit: {limit}, threshold: {threshold})",
                extra={"context": "search"}
            )
            
            # Base query
            query = self.supabase.table("ina_message_embeddings") \
                .select("""
                    id,
                    chunk_text,
                    chunk_index,
                    message_id,
                    ina_messages!inner(*)
                """)

            # Add user filter if specified
            if user_id:
                query = query.eq("ina_messages.user_id", user_id)
                self.logger.debug(
                    f"Filtering by user_id: {user_id}",
                    extra={"context": "search"}
                )

            # Add time window filter if specified
            if time_window:
                cutoff_date = (datetime.now() - timedelta(days=time_window)).isoformat()
                query = query.gte("ina_messages.created_at", cutoff_date)
                self.logger.debug(
                    f"Filtering by time window: {time_window} days",
                    extra={"context": "search"}
                )

            # Execute query
            result = query.execute()

            # Filter and sort results by similarity
            results = []
            for row in result.data:
                similarity = self._calculate_similarity(query_embedding, row["embedding"])
                if similarity >= threshold:
                    results.append({
                        "chunk_text": row["chunk_text"],
                        "similarity": similarity,
                        "message": row["ina_messages"],
                        "created_at": row["ina_messages"]["created_at"]
                    })
            
            results.sort(key=lambda x: x["similarity"], reverse=True)
            results = results[:limit]
            
            self.logger.info(
                f"Found {len(results)} similar messages",
                extra={"context": "search"}
            )
            
            return results
            
        except Exception as e:
            self.logger.error(
                f"Error searching similar messages: {str(e)}",
                extra={"context": "search"}
            )
            raise

    def ina_get_user_context(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent context for a specific user."""
        try:
            self.logger.info(
                f"Getting context for user {user_id} (limit: {limit})",
                extra={"context": "user_context"}
            )
            
            result = self.supabase.table("ina_messages") \
                .select("*") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
            
            self.logger.debug(
                f"Retrieved {len(result.data)} messages for user context",
                extra={"context": "user_context"}
            )
            
            return result.data
            
        except Exception as e:
            self.logger.error(
                f"Error getting user context: {str(e)}",
                extra={"context": "user_context"}
            )
            raise

    def ina_get_chat_context(self, chat_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent context for a specific chat."""
        try:
            self.logger.info(
                f"Getting context for chat {chat_id} (limit: {limit})",
                extra={"context": "chat_context"}
            )
            
            result = self.supabase.table("ina_messages") \
                .select("*") \
                .eq("chat_id", chat_id) \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
            
            self.logger.debug(
                f"Retrieved {len(result.data)} messages for chat context",
                extra={"context": "chat_context"}
            )
            
            return result.data
            
        except Exception as e:
            self.logger.error(
                f"Error getting chat context: {str(e)}",
                extra={"context": "chat_context"}
            )
            raise

    def ina_get_file_context(self, user_id: int, file_type: str = None) -> List[Dict[str, Any]]:
        """Get file-related context for a user."""
        try:
            self.logger.info(
                f"Getting file context for user {user_id}",
                extra={"context": "file_context"}
            )
            
            query = self.supabase.table("ina_messages") \
                .select("*") \
                .eq("user_id", user_id) \
                .not_.is_("file_content", "null")
                
            if file_type:
                query = query.eq("file_type", file_type)
                self.logger.debug(
                    f"Filtering by file type: {file_type}",
                    extra={"context": "file_context"}
                )
                
            result = query.execute()
            
            self.logger.debug(
                f"Retrieved {len(result.data)} files for context",
                extra={"context": "file_context"}
            )
            
            return result.data
            
        except Exception as e:
            self.logger.error(
                f"Error getting file context: {str(e)}",
                extra={"context": "file_context"}
            )
            raise

    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors using Supabase."""
        try:
            query = f"""
            SELECT 1 - (ARRAY{vec1}::vector <=> ARRAY{vec2}::vector) as similarity
            """
            result = self.supabase.rpc("calculate_similarity", {
                "vec1": vec1,
                "vec2": vec2
            }).execute()
            
            return result.data[0]["similarity"]
            
        except Exception as e:
            self.logger.error(
                f"Error calculating similarity: {str(e)}",
                extra={"context": "similarity"}
            )
            raise 