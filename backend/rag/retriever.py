import os
import faiss
import numpy as np
from typing import List, Dict, Any

class MedicalRAGRetriever:
    """FAISS-backed vector search retriever for curated medical documents."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents: List[Dict[str, Any]] = []

    def add_documents(self, docs: List[Dict[str, Any]], embeddings: np.ndarray):
        if len(docs) != len(embeddings):
            raise ValueError("Document count and embeddings count must match.")
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.documents.extend(docs)

    def search(self, query_vector: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        distances, indices = self.index.search(np.array([query_vector], dtype=np.float32), k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents) and idx >= 0:
                doc = dict(self.documents[idx])
                doc["score"] = float(dist)
                results.append(doc)
        return results
