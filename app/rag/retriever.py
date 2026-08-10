"""
BAYMAX AI – Medical Retriever
==============================
Semantic retrieval across all medical knowledge collections.
Queries all collections in parallel, merges results, re-ranks by score.

Usage:
    from app.rag.retriever import MedicalRetriever
    retriever = MedicalRetriever(embedder, vector_store)
    results = retriever.retrieve("I have a fever and chest pain")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.rag.embedder import RAGEmbedder
from app.rag.vector_store import MedicalVectorStore, SearchResult
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class MedicalRetrievalResult:
    """
    Aggregated retrieval results from across all medical collections.

    Attributes:
        query:      Original query string.
        results:    All retrieved documents, sorted by score.
        context:    Pre-formatted context string for LLM injection.
    """
    query: str
    results: List[SearchResult] = field(default_factory=list)
    context: str = ""

    def to_context_string(self, max_chars: int = 3000) -> str:
        """
        Format retrieved results as a readable context block for the LLM.

        Args:
            max_chars: Maximum total characters in the output.

        Returns:
            Formatted context string.
        """
        if not self.results:
            return "No relevant medical information found in the knowledge base."

        lines = ["=== RETRIEVED MEDICAL KNOWLEDGE ===\n"]
        total = 0

        for i, r in enumerate(self.results, 1):
            source_label = r.collection.replace("_", " ").title()
            entry = (
                f"[{i}] Source: {source_label} (relevance: {r.score:.2f})\n"
                f"{r.text}\n"
            )
            if total + len(entry) > max_chars:
                lines.append("... [additional results truncated for length] ...")
                break
            lines.append(entry)
            total += len(entry)

        return "\n".join(lines)


class MedicalRetriever:
    """
    Semantic retrieval engine for the BAYMAX medical knowledge base.

    Queries all configured ChromaDB collections and merges + re-ranks results.

    Attributes:
        embedder:     RAGEmbedder instance for query encoding.
        vector_store: MedicalVectorStore instance.
        top_k:        Number of results to return per collection.
        min_score:    Minimum similarity score threshold (0–1).
        collections:  List of collection names to query.
    """

    def __init__(
        self,
        embedder: Optional[RAGEmbedder] = None,
        vector_store: Optional[MedicalVectorStore] = None,
        top_k: Optional[int] = None,
        min_score: float = 0.3,
    ) -> None:
        from config import settings

        self.embedder = embedder or RAGEmbedder()
        self.vector_store = vector_store or MedicalVectorStore()
        self.top_k = top_k or settings.RAG_TOP_K
        self.min_score = min_score
        self.collections = [
            settings.CHROMA_COLLECTION_DISEASE,
            settings.CHROMA_COLLECTION_SYMPTOM,
            settings.CHROMA_COLLECTION_MEDICINE,
            settings.CHROMA_COLLECTION_FIRSTAID,
            settings.CHROMA_COLLECTION_GENERAL,
        ]
        log.info(
            "MedicalRetriever initialized | collections={} | top_k={} | min_score={}",
            len(self.collections),
            self.top_k,
            self.min_score,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        collections: Optional[List[str]] = None,
    ) -> MedicalRetrievalResult:
        """
        Retrieve relevant medical documents for a query.

        Args:
            query:       Natural language query string.
            top_k:       Override the default top_k.
            collections: Override which collections to search.

        Returns:
            MedicalRetrievalResult with merged, ranked results.
        """
        if not query.strip():
            return MedicalRetrievalResult(query=query)

        k = top_k or self.top_k
        cols = collections or self.collections

        log.debug("Retrieving for query: '{}'", query[:100])

        # Clean the query to remove OCR/prompt noise for semantic search embedding
        search_query = query
        if "extracted text:" in query.lower() or len(query) > 250:
            import re
            lines = query.splitlines()
            medical_lines = []
            capture = False
            for line in lines:
                l_lower = line.lower()
                if any(h in l_lower for h in ["history", "medication", "symptom", "diagnosis", "report", "findings", "record"]):
                    capture = True
                if any(h in l_lower for h in ["please", "patient information", "contact number", "copyright"]):
                    capture = False
                if capture and line.strip():
                    clean_line = re.sub(r'[*#_`>|🚨🩹🫀🦟🩺🐍🌡️🫁📋💡🔥🐍😐🔲💡📋👤•¢«-]', '', line).strip()
                    if clean_line:
                        medical_lines.append(clean_line)
            if medical_lines:
                search_query = " ".join(medical_lines)
            else:
                words = [w for w in query.split() if w.lower() not in [
                    "i've", "uploaded", "medical", "document", "extracted", "text:", "text", "please", "analyse", "this", "and", "provide", "relevant", "information"
                ]]
                search_query = " ".join(words)

        log.debug("Extracted search query for RAG: '{}'", search_query[:150])

        # Embed the cleaned query once, reuse across all collections
        query_embedding = self.embedder.embed_single(search_query)

        # Search all collections
        all_results: List[SearchResult] = []
        for collection in cols:
            if not self.vector_store.collection_exists(collection):
                log.debug("Skipping empty collection: {}", collection)
                continue
            try:
                results = self.vector_store.query(
                    collection_name=collection,
                    query_embedding=query_embedding,
                    k=k,
                )
                all_results.extend(results)
            except Exception as exc:
                log.warning("Query failed for collection={}: {}", collection, exc)

        # Filter by minimum score
        filtered = [r for r in all_results if r.score >= self.min_score]

        # Global re-rank by score (descending)
        ranked = sorted(filtered, key=lambda r: r.score, reverse=True)

        # Deduplicate by text similarity (simple: drop near-identical chunks)
        deduped = self._deduplicate(ranked)

        result = MedicalRetrievalResult(
            query=query,
            results=deduped[:k * 2],  # Return up to 2x top_k across all sources
        )
        result.context = result.to_context_string()

        log.info(
            "Retrieval complete | query='{}' | found={} | after_filter={} | returned={}",
            query[:60],
            len(all_results),
            len(filtered),
            len(result.results),
        )
        return result

    def retrieve_by_collection(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Search a single specific collection.

        Args:
            query:      Query string.
            collection: Target collection name.
            top_k:      Number of results.

        Returns:
            List of SearchResult.
        """
        query_embedding = self.embedder.embed_single(query)
        return self.vector_store.query(
            collection_name=collection,
            query_embedding=query_embedding,
            k=top_k,
        )

    @staticmethod
    def _deduplicate(
        results: List[SearchResult],
        similarity_threshold: float = 0.95,
    ) -> List[SearchResult]:
        """
        Remove near-duplicate results based on text overlap.

        Args:
            results:              List of results to deduplicate.
            similarity_threshold: Jaccard similarity above which results are duplicates.

        Returns:
            Deduplicated list.
        """
        seen: List[set] = []
        deduped: List[SearchResult] = []

        for result in results:
            words = set(result.text.lower().split())
            is_duplicate = False
            for seen_words in seen:
                if len(words) == 0 or len(seen_words) == 0:
                    continue
                intersection = words & seen_words
                union = words | seen_words
                jaccard = len(intersection) / len(union)
                if jaccard >= similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduped.append(result)
                seen.append(words)

        return deduped
