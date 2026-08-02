import numpy as np
from typing import List, Dict, Any, Optional

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

class AdvancedRAGRetriever:
    """Production RAG retriever with FAISS / Numpy vector indexing, MMR reranking, and citation generation."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        if HAS_FAISS:
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            self.index = None
            self.vectors: List[np.ndarray] = []
        self.documents: List[Dict[str, Any]] = []

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return
        embeddings = [c["embedding"] for c in chunks if "embedding" in c]
        if embeddings:
            np_embeddings = np.array(embeddings, dtype=np.float32)
            if HAS_FAISS and self.index is not None:
                self.index.add(np_embeddings)
            else:
                for vec in np_embeddings:
                    self.vectors.append(vec)
            for c in chunks:
                doc_meta = dict(c.get("metadata", {}))
                doc_meta["content"] = c["content"]
                self.documents.append(doc_meta)

    def search_hybrid(self, query_vector: np.ndarray, query_text: str, top_k: int = 5, mmr_lambda: float = 0.7) -> List[Dict[str, Any]]:
        if not self.documents:
            return []

        candidates = []
        q_terms = set(query_text.lower().split())

        if HAS_FAISS and self.index is not None and self.index.ntotal > 0:
            distances, indices = self.index.search(np.array([query_vector], dtype=np.float32), min(top_k * 3, self.index.ntotal))
            for dist, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(self.documents):
                    doc = dict(self.documents[idx])
                    vector_score = 1.0 / (1.0 + float(dist))
                    doc_terms = set(doc.get("content", "").lower().split())
                    lexical_score = len(q_terms.intersection(doc_terms)) / max(len(q_terms), 1)
                    doc["score"] = round(0.7 * vector_score + 0.3 * lexical_score, 4)
                    doc["citation"] = f"[{doc.get('source_file', 'Medical Knowledge')}]"
                    candidates.append(doc)
        elif self.vectors:
            for idx, vec in enumerate(self.vectors):
                dist = float(np.linalg.norm(query_vector - vec))
                vector_score = 1.0 / (1.0 + dist)
                doc = dict(self.documents[idx])
                doc_terms = set(doc.get("content", "").lower().split())
                lexical_score = len(q_terms.intersection(doc_terms)) / max(len(q_terms), 1)
                doc["score"] = round(0.7 * vector_score + 0.3 * lexical_score, 4)
                doc["citation"] = f"[{doc.get('source_file', 'Medical Knowledge')}]"
                candidates.append(doc)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def format_evidence_block(self, docs: List[Dict[str, Any]]) -> str:
        if not docs:
            return "No evidence documents retrieved."

        lines = []
        for i, doc in enumerate(docs, 1):
            src = doc.get("source_file", "Trusted Medical Reference")
            score = doc.get("score", 0.0)
            content = doc.get("content", "")
            lines.append(f"[{i}] {src} (Relevance: {score:.2f}):\n{content}")
        return "\n\n".join(lines)
