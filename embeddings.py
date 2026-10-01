from functools import lru_cache

import numpy as np

from config import EMBED_MODEL


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer  # lazy import: slow to load
    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings, so dot product == cosine similarity."""
    return _get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)