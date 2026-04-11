from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

# Global model cache
_model = None


def get_model(model_name="all-MiniLM-L6-v2"):
    """
    Lazy-load and return the sentence transformer model.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


@lru_cache(maxsize=10000)
def encode_text(text, model_name="all-MiniLM-L6-v2"):
    """
    Encode a single text into embedding with caching.
    """
    model = get_model(model_name)
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding


def similarity_score(expected, actual, model_name="all-MiniLM-L6-v2"):
    """
    Compute cosine similarity between two texts.

    Args:
        expected (str)
        actual (str)
        model_name (str)

    Returns:
        float
    """
    if not expected or not actual:
        return 0.0

    e1 = encode_text(expected, model_name)
    e2 = encode_text(actual, model_name)

    score = cosine_similarity([e1], [e2])[0][0]
    return float(score)


def batch_similarity(pairs, model_name="all-MiniLM-L6-v2"):
    """
    Compute similarity scores for multiple (expected, actual) pairs.

    Args:
        pairs (list of tuples): [(expected, actual), ...]

    Returns:
        list of float
    """
    if not pairs:
        return []

    model = get_model(model_name)

    # Flatten all texts
    texts = []
    for expected, actual in pairs:
        texts.extend([expected or "", actual or ""])

    embeddings = model.encode(texts, normalize_embeddings=True)

    scores = []
    for i in range(0, len(embeddings), 2):
        score = cosine_similarity(
            [embeddings[i]], [embeddings[i + 1]]
        )[0][0]
        scores.append(float(score))

    return scores