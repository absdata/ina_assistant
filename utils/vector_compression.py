import numpy as np
from sklearn.decomposition import PCA
from typing import List
from config.settings import SUPABASE_EMBEDDING_DIMENSION

class VectorCompressor:
    def __init__(self, target_dimensions: int = SUPABASE_EMBEDDING_DIMENSION):
        """Initialize the vector compressor with target dimensions."""
        self.target_dimensions = target_dimensions
        self.pca = None
        self.is_fitted = False

    def compress(self, vectors: List[List[float]]) -> List[List[float]]:
        """
        Compress vectors to target dimension using average pooling.
        This method works with any number of vectors.
        """
        if not vectors:
            return []
            
        vectors_array = np.array(vectors)
        original_dim = vectors_array.shape[1]
        
        if original_dim <= self.target_dimensions:
            # Pad with zeros if original dimension is smaller
            padding = np.zeros((vectors_array.shape[0], self.target_dimensions - original_dim))
            return np.hstack((vectors_array, padding)).tolist()
        
        # Calculate the size of each pooling window
        window_size = original_dim // self.target_dimensions
        remainder = original_dim % self.target_dimensions
        
        compressed_vectors = []
        for vector in vectors_array:
            # Reshape the vector to prepare for pooling
            reshaped = vector[:original_dim - remainder].reshape(-1, window_size)
            # Calculate mean for each window
            pooled = np.mean(reshaped, axis=1)
            
            # Handle the remainder if any
            if remainder:
                last_window_mean = np.mean(vector[original_dim - remainder:])
                pooled = np.append(pooled, last_window_mean)
            
            # Ensure we have exactly target_dimensions
            if len(pooled) < self.target_dimensions:
                padding = np.zeros(self.target_dimensions - len(pooled))
                pooled = np.append(pooled, padding)
            elif len(pooled) > self.target_dimensions:
                pooled = pooled[:self.target_dimensions]
            
            compressed_vectors.append(pooled.tolist())
        
        return compressed_vectors

    def compress_single(self, vector: List[float]) -> List[float]:
        """Compress a single vector."""
        if not vector:
            return []
            
        compressed = self.compress([vector])
        return compressed[0] if compressed else [] 