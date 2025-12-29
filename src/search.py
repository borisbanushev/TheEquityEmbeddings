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

    def query_dissimilar_tickers(self, ticker: str, k: int = 5) -> pd.DataFrame:
        """
        Search for companies LEAST similar (most dissimilar) to a specific ticker.
        These are the companies furthest away in the embedding space.
        """
        if 'ticker' not in self.metadata.columns:
            raise ValueError("Metadata must contain 'ticker' column.")

        idx = self.metadata[self.metadata['ticker'] == ticker].index
        if len(idx) == 0:
            raise ValueError(f"Ticker {ticker} not found in metadata.")

        # Calculate distances to all other companies
        query_vector = self.embeddings[idx[0]].reshape(1, -1)
        distances = np.linalg.norm(self.embeddings - query_vector, axis=1)

        # Get indices of k most dissimilar (largest distances), excluding self
        # Sort in descending order and skip the query itself
        sorted_indices = np.argsort(distances)[::-1]
        dissimilar_indices = [i for i in sorted_indices if i != idx[0]][:k]

        results = self.metadata.iloc[dissimilar_indices].copy()
        results['distance'] = distances[dissimilar_indices]

        return results

    def analyze_ticker(self, ticker: str, k: int = 5) -> dict:
        """
        Comprehensive analysis: find both most and least similar companies.

        Returns:
            dict with keys 'query_info', 'most_similar', 'least_similar'
        """
        if 'ticker' not in self.metadata.columns:
            raise ValueError("Metadata must contain 'ticker' column.")

        idx = self.metadata[self.metadata['ticker'] == ticker].index
        if len(idx) == 0:
            raise ValueError(f"Ticker {ticker} not found in metadata.")

        query_info = self.metadata.iloc[idx[0]].to_dict()

        # Get most similar
        distances_similar, most_similar = self.query_by_index(idx[0], k)
        most_similar = most_similar.copy()
        most_similar['distance'] = distances_similar

        # Get least similar
        least_similar = self.query_dissimilar_tickers(ticker, k)

        return {
            'query_info': query_info,
            'most_similar': most_similar,
            'least_similar': least_similar
        }

    def save_index(self, path: str):
        faiss.write_index(self.index, path)

    def load_index(self, path: str):
        self.index = faiss.read_index(path)
