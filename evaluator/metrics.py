import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ------------------------------------------------------------------ #
# Tokenization                                                         #
# ------------------------------------------------------------------ #

def tokenize(text, remove_stopwords=False):
    """
    Tokenize input text into lowercase word tokens.

    Args:
        text             (str):  Input string.
        remove_stopwords (bool): Whether to remove common English stopwords.
                                 Recommended for domain-specific evaluation
                                 to avoid common words inflating overlap scores.

    Returns:
        list: List of lowercase word tokens.
    """
    if not text:
        return []

    tokens = re.findall(r"\b\w+\b", text.lower())

    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]

    return tokens


def get_ngrams(tokens, n=2):
    """
    Generate n-grams from a token list.

    Useful for capturing phrase-level matches (e.g. "maternity leave",
    "notice period") that bag-of-words PRF misses.

    Args:
        tokens (list): List of string tokens.
        n      (int):  Ngram size (default: 2 for bigrams).

    Returns:
        list: List of ngram strings joined by spaces.
    """
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# ------------------------------------------------------------------ #
# Core PRF computation                                                 #
# ------------------------------------------------------------------ #

def compute_prf(expected_tokens, actual_tokens):
    """
    Compute precision, recall, and F1 using token frequency overlap.

    Token counts are used (not just presence), so repeated tokens
    are handled correctly.

    Edge case contract:
        - Both empty  → perfect score (1.0 / 1.0 / 1.0)
        - Either empty → all zeros (undefined, treated conservatively)

    Args:
        expected_tokens (list): Reference tokens.
        actual_tokens   (list): Generated tokens.

    Returns:
        dict: {
            "precision": float,
            "recall"   : float,
            "f1"       : float
        }
    """
    # Both empty — nothing to evaluate, treat as perfect match
    if not expected_tokens and not actual_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    # One side empty — no meaningful overlap possible
    if not expected_tokens or not actual_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    expected_counts = Counter(expected_tokens)
    actual_counts   = Counter(actual_tokens)

    # Intersection: minimum frequency per shared token
    common = sum((expected_counts & actual_counts).values())

    total_actual   = sum(actual_counts.values())
    total_expected = sum(expected_counts.values())

    precision = common / total_actual   if total_actual   > 0 else 0.0
    recall    = common / total_expected if total_expected > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4)
    }


def compute_prf_with_ngrams(expected_tokens, actual_tokens, ngram_n=2, ngram_weight=0.3):
    """
    Compute a weighted PRF that blends unigram and ngram F1 scores.

    Ngram matching captures phrase-level accuracy (e.g. "30 days",
    "annual leave", "PF contribution") that pure bag-of-words misses.

    Args:
        expected_tokens (list):  Reference tokens.
        actual_tokens   (list):  Generated tokens.
        ngram_n         (int):   Ngram size (default: 2).
        ngram_weight    (float): Weight for ngram F1 in the blended score
                                 (0.0 = unigrams only, 1.0 = ngrams only).

    Returns:
        dict: {
            "precision"     : float,  # unigram precision
            "recall"        : float,  # unigram recall
            "f1"            : float,  # blended f1
            "unigram_f1"    : float,
            "ngram_f1"      : float,
            "ngram_weight"  : float
        }
    """
    unigram_result = compute_prf(expected_tokens, actual_tokens)

    expected_ngrams = get_ngrams(expected_tokens, n=ngram_n)
    actual_ngrams   = get_ngrams(actual_tokens,   n=ngram_n)
    ngram_result    = compute_prf(expected_ngrams, actual_ngrams)

    blended_f1 = (
        (1 - ngram_weight) * unigram_result["f1"]
        + ngram_weight     * ngram_result["f1"]
    )

    return {
        "precision":    unigram_result["precision"],
        "recall":       unigram_result["recall"],
        "f1":           round(blended_f1, 4),
        "unigram_f1":   unigram_result["f1"],
        "ngram_f1":     ngram_result["f1"],
        "ngram_weight": ngram_weight
    }


# ------------------------------------------------------------------ #
# Convenience wrapper                                                  #
# ------------------------------------------------------------------ #

def compute_prf_from_text(expected_text, actual_text, remove_stopwords=True, use_ngrams=True):
    """
    Tokenize texts and compute PRF in one call.

    Args:
        expected_text    (str):  Ground truth answer.
        actual_text      (str):  Generated answer.
        remove_stopwords (bool): Strip common words before scoring
                                 (default: True for domain evaluation).
        use_ngrams       (bool): Blend in bigram F1 for phrase awareness
                                 (default: True).

    Returns:
        dict: PRF result from compute_prf or compute_prf_with_ngrams.
    """
    expected_tokens = tokenize(expected_text, remove_stopwords=remove_stopwords)
    actual_tokens   = tokenize(actual_text,   remove_stopwords=remove_stopwords)

    if use_ngrams:
        return compute_prf_with_ngrams(expected_tokens, actual_tokens)

    return compute_prf(expected_tokens, actual_tokens)


# ------------------------------------------------------------------ #
# Category-level aggregation                                           #
# ------------------------------------------------------------------ #

def aggregate_prf_by_category(results):
    """
    Compute average precision, recall, and F1 per HR category.

    Useful for identifying which knowledge domains the chatbot
    underperforms on across a full test run.

    Args:
        results (list[dict]): Each dict must have keys:
                              "category", "precision", "recall", "f1"

    Returns:
        dict[str, dict]: Category → averaged PRF scores.

    Example:
        {
            "leave":    {"precision": 0.72, "recall": 0.68, "f1": 0.70},
            "payroll":  {"precision": 0.61, "recall": 0.59, "f1": 0.60},
        }
    """
    from collections import defaultdict

    buckets = defaultdict(list)
    for r in results:
        buckets[r["category"]].append(r)

    aggregated = {}
    for category, rows in buckets.items():
        aggregated[category] = {
            "precision": round(sum(r["precision"] for r in rows) / len(rows), 4),
            "recall":    round(sum(r["recall"]    for r in rows) / len(rows), 4),
            "f1":        round(sum(r["f1"]        for r in rows) / len(rows), 4),
            "count":     len(rows)
        }

    return aggregated
