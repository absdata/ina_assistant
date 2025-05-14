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

    def fit(self, vectors: List[List[float]]):
        """Fit PCA on the input vectors."""
        if not vectors:
            return
            
        vectors_array = np.array(vectors)
        self.pca = PCA(n_components=self.target_dimensions)
        self.pca.fit(vectors_array)
        self.is_fitted = True

    def compress(self, vectors: List[List[float]]) -> List[List[float]]:
        """Compress vectors to target dimension using PCA."""
        if not vectors:
            return []
            
        vectors_array = np.array(vectors)
        
        # If not fitted or different input dimension, fit first
        if not self.is_fitted or self.pca.n_features_ != vectors_array.shape[1]:
            self.fit(vectors)
            
        # Transform vectors to lower dimension
        compressed = self.pca.transform(vectors_array)
        
        # Convert back to list format and ensure float values
        return compressed.tolist()

    def compress_single(self, vector: List[float]) -> List[float]:
        """Compress a single vector."""
        if not vector:
            return []
            
        compressed = self.compress([vector])
        return compressed[0] if compressed else [] 