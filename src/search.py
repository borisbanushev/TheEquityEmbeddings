import faiss
import numpy as np
import pandas as pd
from typing import List, Tuple

class EquitySearch:
    """
    Search engine for equity embeddings using FAISS.
    """
    def __init__(self, embeddings: np.ndarray, metadata: pd.DataFrame):
        self.embeddings = embeddings.astype('float32')
        self.metadata = metadata
        self.d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.d)
        self.index.add(self.embeddings)

    def query_by_index(self, idx: int, k: int = 5) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Find k nearest neighbors for a given index in the embedding space.
        """
        query_vector = self.embeddings[idx].reshape(1, -1)
        distances, indices = self.index.search(query_vector, k + 1)
        # Exclude the query itself (index 0 usually)
        return distances[0, 1:], self.metadata.iloc[indices[0, 1:]]

    def query_similar_tickers(self, ticker: str, k: int = 5) -> pd.DataFrame:
        """
        Search for companies similar to a specific ticker.
        """
        if 'ticker' not in self.metadata.columns:
            raise ValueError("Metadata must contain 'ticker' column.")
        
        idx = self.metadata[self.metadata['ticker'] == ticker].index
        if len(idx) == 0:
            raise ValueError(f"Ticker {ticker} not found in metadata.")
        
        _, results = self.query_by_index(idx[0], k)
        return results

    def save_index(self, path: str):
        faiss.write_index(self.index, path)

    def load_index(self, path: str):
        self.index = faiss.read_index(path)
