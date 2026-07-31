import numpy as np

class SentenceTransformerEmbedder:
    """Interface for generating text embeddings locally."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception:
                self._model = None

    def encode(self, text: str) -> np.ndarray:
        self._load_model()
        if self._model:
            return self._model.encode(text)
        # Fallback dummy embedding vector for testing/offline bootstrap
        np.random.seed(hash(text) % (2**32))
        return np.random.rand(384).astype(np.float32)
