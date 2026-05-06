import threading
import numpy as np

from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------ #
# Supported models                                                     #
# ------------------------------------------------------------------ #

SUPPORTED_MODELS = {
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "paraphrase-MiniLM-L6-v2"
}

# ------------------------------------------------------------------ #
# Thread-safe model loader                                             #
# ------------------------------------------------------------------ #

_model      = None
_model_lock = threading.Lock()


def get_model(model_name="all-MiniLM-L6-v2"):
    """
    Lazy-load and return the sentence transformer model.

    Thread-safe: uses a lock to prevent concurrent initialisation in
    multi-threaded test runners (e.g. pytest-xdist).

    Args:
        model_name (str): Name of the SentenceTransformer model.

    Returns:
        SentenceTransformer: Loaded model instance.

    Raises:
        ValueError: If model_name is not in SUPPORTED_MODELS.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Choose from: {sorted(SUPPORTED_MODELS)}"
        )

    global _model
    with _model_lock:
        if _model is None:
            _model = SentenceTransformer(model_name)

    return _model


# ------------------------------------------------------------------ #
# Cached text encoding                                                 #
# ------------------------------------------------------------------ #

@lru_cache(maxsize=10000)
def encode_text(text, model_name="all-MiniLM-L6-v2"):
    """
    Encode a single text into a normalised embedding with caching.

    The result is stored as a tuple (not a numpy array) because:
      - tuples are hashable, which lru_cache requires
      - tuples are immutable, preventing cache corruption from
        accidental in-place mutation by callers

    Args:
        text       (str): Input text to encode.
        model_name (str): SentenceTransformer model name.

    Returns:
        tuple: L2-normalised embedding as an immutable tuple of floats.
    """
    model     = get_model(model_name)
    embedding = model.encode(text, normalize_embeddings=True)
    return tuple(embedding)     # hashable + immutable


def get_cache_stats():
    """
    Return lru_cache statistics for encode_text.

    Useful for monitoring cache hit rate during large test runs
    to verify the cache is providing a meaningful speedup.

    Returns:
        dict: {
            "hits"    : int,
            "misses"  : int,
            "maxsize" : int,
            "currsize": int
        }
    """
    info = encode_text.cache_info()
    return {
        "hits":     info.hits,
        "misses":   info.misses,
        "maxsize":  info.maxsize,
        "currsize": info.currsize
    }


# ------------------------------------------------------------------ #
# Single-pair similarity                                               #
# ------------------------------------------------------------------ #

def similarity_score(expected, actual, model_name="all-MiniLM-L6-v2"):
    """
    Compute cosine similarity between two texts.

    Since embeddings are L2-normalised, cosine similarity reduces to a
    dot product — computed directly with np.dot for efficiency instead
    of using sklearn's batch-oriented cosine_similarity on a 1x1 matrix.

    Args:
        expected   (str): Reference text.
        actual     (str): Generated text.
        model_name (str): SentenceTransformer model name.

    Returns:
        float: Cosine similarity in [0.0, 1.0]. Returns 0.0 for empty inputs.
    """
    if not expected or not actual:
        return 0.0

    # Whitespace-only guard — encodes to a near-zero vector
    if not expected.strip() or not actual.strip():
        return 0.0

    e1 = np.array(encode_text(expected, model_name))
    e2 = np.array(encode_text(actual,   model_name))

    return float(np.dot(e1, e2))


# ------------------------------------------------------------------ #
# Batch similarity                                                     #
# ------------------------------------------------------------------ #

def batch_similarity(pairs, model_name="all-MiniLM-L6-v2"):
    """
    Compute cosine similarity scores for multiple (expected, actual) pairs.

    Encodes all texts in a single model.encode() call for efficiency.
    Empty or whitespace-only strings in a pair short-circuit to 0.0
    without being passed to the encoder.

    Args:
        pairs      (list[tuple]): [(expected, actual), ...]
        model_name (str):         SentenceTransformer model name.

    Returns:
        list[float]: Similarity scores in the same order as input pairs.
    """
    if not pairs:
        return []

    model = get_model(model_name)

    # Identify pairs that can be encoded (both sides non-empty)
    valid_indices = []
    texts_to_encode = []

    for idx, (expected, actual) in enumerate(pairs):
        exp_str = (expected or "").strip()
        act_str = (actual   or "").strip()

        if exp_str and act_str:
            valid_indices.append(idx)
            texts_to_encode.extend([exp_str, act_str])

    # Pre-fill all scores with 0.0 (covers empty-string pairs)
    scores = [0.0] * len(pairs)

    if not texts_to_encode:
        return scores

    # Single batched encode call for all valid texts
    embeddings = model.encode(texts_to_encode, normalize_embeddings=True)

    for batch_pos, pair_idx in enumerate(valid_indices):
        e1 = embeddings[batch_pos * 2]
        e2 = embeddings[batch_pos * 2 + 1]
        scores[pair_idx] = float(np.dot(e1, e2))

    return scores
