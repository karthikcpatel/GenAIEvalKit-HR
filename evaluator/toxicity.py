import re

# Weighted toxicity lexicon (can be expanded)
TOXIC_WORDS = {
    "hate": 0.6,
    "kill": 0.9,
    "stupid": 0.5,
}


def toxicity_score(text):
    """
    Compute a simple toxicity score based on keyword matches.

    Args:
        text (str)

    Returns:
        float: toxicity score between 0 and 1
    """
    if not text:
        return 0.0

    text = text.lower()
    score = 0.0

    for word, weight in TOXIC_WORDS.items():
        # Match whole words only
        if re.search(rf"\b{re.escape(word)}\b", text):
            score += weight

    return min(score, 1.0)


def is_toxic(text, threshold=0.5):
    """
    Determine whether text is toxic based on a threshold.

    Args:
        text (str)
        threshold (float)

    Returns:
        dict: {
            "toxic": bool,
            "score": float,
            "threshold": float,
            "matched_terms": list
        }
    """
    if not text:
        return {
            "toxic": False,
            "score": 0.0,
            "threshold": threshold,
            "matched_terms": []
        }

    text_lower = text.lower()
    score = 0.0
    matched_terms = []

    for word, weight in TOXIC_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text_lower):
            score += weight
            matched_terms.append(word)

    score = min(score, 1.0)
    is_flagged = score >= threshold

    return {
        "toxic": is_flagged,
        "score": score,
        "threshold": threshold,
        "matched_terms": matched_terms,
        "reason": (
            f"Detected toxic terms: {', '.join(matched_terms)}"
            if matched_terms else "No toxic terms detected"
        )
    }