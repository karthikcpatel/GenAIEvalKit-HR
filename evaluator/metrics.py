import re
from collections import Counter


def tokenize(text, remove_stopwords=False):
    """
    Tokenize input text into lowercase word tokens.

    Args:
        text (str): Input string
        remove_stopwords (bool): Whether to remove common stopwords

    Returns:
        list: List of tokens
    """
    if not text:
        return []

    tokens = re.findall(r"\b\w+\b", text.lower())

    if remove_stopwords:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]

    return tokens


def compute_prf(expected_tokens, actual_tokens):
    """
    Compute precision, recall, and F1 score using token frequency (not just presence).

    Args:
        expected_tokens (list): Reference tokens
        actual_tokens (list): Generated tokens

    Returns:
        dict: {
            "precision": float,
            "recall": float,
            "f1": float
        }
    """

    # Handle edge cases
    if not expected_tokens and not actual_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not expected_tokens:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    if not actual_tokens:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

    expected_counts = Counter(expected_tokens)
    actual_counts = Counter(actual_tokens)

    # Intersection: minimum counts per token
    common = sum((expected_counts & actual_counts).values())

    total_actual = sum(actual_counts.values())
    total_expected = sum(expected_counts.values())

    precision = common / total_actual if total_actual > 0 else 0.0
    recall = common / total_expected if total_expected > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_prf_from_text(expected_text, actual_text, remove_stopwords=False):
    """
    Convenience wrapper: tokenize text and compute PRF.

    Args:
        expected_text (str)
        actual_text (str)
        remove_stopwords (bool)

    Returns:
        dict
    """
    expected_tokens = tokenize(expected_text, remove_stopwords=remove_stopwords)
    actual_tokens = tokenize(actual_text, remove_stopwords=remove_stopwords)

    return compute_prf(expected_tokens, actual_tokens)