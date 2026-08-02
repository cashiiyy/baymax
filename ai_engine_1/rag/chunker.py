import re
from typing import List, Dict, Any

class DocumentChunker:
    """Chunks raw document text into hierarchical and semantic segments with metadata."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        meta = source_metadata or {}
        chunks = []
        
        # Split by section headers or double newlines first (Semantic boundary)
        sections = re.split(r'\n\s*\n|(?=^#{1,3}\s)', text, flags=re.MULTILINE)
        
        chunk_id = 0
        for section in sections:
            clean_sec = section.strip()
            if not clean_sec:
                continue

            if len(clean_sec) <= self.chunk_size:
                chunk_id += 1
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": clean_sec,
                    "metadata": {**meta, "chunk_size": len(clean_sec)}
                })
            else:
                # Sliding window chunking for long sections
                start = 0
                while start < len(clean_sec):
                    end = start + self.chunk_size
                    segment = clean_sec[start:end].strip()
                    if segment:
                        chunk_id += 1
                        chunks.append({
                            "chunk_id": chunk_id,
                            "content": segment,
                            "metadata": {**meta, "chunk_size": len(segment)}
                        })
                    start += (self.chunk_size - self.chunk_overlap)
        
        return chunks
