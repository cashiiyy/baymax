"""
BAYMAX AI – Document Chunker
==============================
Splits long text documents into overlapping chunks suitable for embedding.
Uses LangChain's RecursiveCharacterTextSplitter under the hood.

Usage:
    from app.rag.chunker import DocumentChunker
    chunks = DocumentChunker().chunk_text("Long medical text...", metadata={...})
"""

from __future__ import annotations

from typing import List
from dataclasses import dataclass, field

from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class DocumentChunk:
    """
    A single chunk of a larger document, ready for embedding.

    Attributes:
        text:       The chunk text.
        metadata:   Associated metadata (source, type, record_id, etc.).
        chunk_id:   Auto-generated unique ID for this chunk.
    """
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            import hashlib
            self.chunk_id = hashlib.sha1(self.text.encode()).hexdigest()[:16]


class DocumentChunker:
    """
    Splits documents into overlapping text chunks for vector embedding.

    Uses RecursiveCharacterTextSplitter which respects sentence/paragraph
    boundaries before falling back to character splits.

    Attributes:
        chunk_size:    Target maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        from config import settings
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            length_function=len,
        )
        log.info(
            "DocumentChunker initialized | size={} overlap={}",
            self.chunk_size,
            self.chunk_overlap,
        )

    def chunk_text(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> List[DocumentChunk]:
        """
        Split a single text string into chunks.

        Args:
            text:     The document text to split.
            metadata: Metadata to attach to every resulting chunk.

        Returns:
            List of DocumentChunk objects.
        """
        if not text.strip():
            return []

        meta = metadata or {}
        raw_chunks = self._splitter.split_text(text)
        chunks = [
            DocumentChunk(
                text=chunk,
                metadata={**meta, "chunk_index": i, "total_chunks": len(raw_chunks)},
            )
            for i, chunk in enumerate(raw_chunks)
        ]
        log.debug(
            "Chunked document | source={} | {} → {} chunks",
            meta.get("source", "?"),
            len(text),
            len(chunks),
        )
        return chunks

    def chunk_records(
        self,
        records: List[dict],
        text_key: str = "text",
        metadata_keys: List[str] | None = None,
    ) -> List[DocumentChunk]:
        """
        Chunk a list of record dicts.

        Args:
            records:       List of dicts, each containing a text field.
            text_key:      Which field to use as the chunk text.
            metadata_keys: Which fields to keep as chunk metadata.

        Returns:
            Flattened list of all chunks from all records.
        """
        all_chunks: List[DocumentChunk] = []
        keys = metadata_keys or []

        for record in records:
            text = record.get(text_key, "")
            meta = {k: record.get(k) for k in keys if k in record}
            chunks = self.chunk_text(text, meta)
            all_chunks.extend(chunks)

        log.info(
            "Chunked {} records → {} total chunks",
            len(records),
            len(all_chunks),
        )
        return all_chunks
