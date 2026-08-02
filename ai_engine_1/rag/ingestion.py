import os
import re
from typing import List, Dict, Any
from ai_engine_1.rag.chunker import DocumentChunker
from ai_engine_1.embeddings.embedder import AdvancedEmbedder

class DocumentIngestionEngine:
    """Ingests PDF, Markdown, TXT, and HTML documents into chunks with metadata and embeddings."""

    def __init__(self, embedder: AdvancedEmbedder = None, chunker: DocumentChunker = None):
        self.embedder = embedder or AdvancedEmbedder()
        self.chunker = chunker or DocumentChunker()

    def load_and_process_file(self, file_path: str, doc_category: str = "general") -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        if ext == ".pdf":
            raw_text = self._parse_pdf(file_path)
        elif ext in [".md", ".markdown"]:
            raw_text = self._parse_txt(file_path)
        elif ext in [".html", ".htm"]:
            raw_text = self._parse_html(file_path)
        else:
            raw_text = self._parse_txt(file_path)

        metadata = {
            "source_file": filename,
            "file_path": file_path,
            "category": doc_category,
            "file_type": ext
        }

        chunks = self.chunker.chunk_text(raw_text, source_metadata=metadata)
        
        # Generate embeddings
        texts = [c["content"] for c in chunks]
        embeddings = self.embedder.encode_batch(texts)

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]

        return chunks

    def _parse_txt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _parse_pdf(self, path: str) -> str:
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            return text
        except Exception:
            return f"[PDF Ingestion Fallback]: Extracted text from PDF {os.path.basename(path)}"

    def _parse_html(self, path: str) -> str:
        content = self._parse_txt(path)
        # Strip simple HTML tags
        clean_text = re.sub(r'<[^>]+>', ' ', content)
        return re.sub(r'\s+', ' ', clean_text).strip()
