from evaluator.similarity import similarity_score


def evaluate_hallucination(expected, actual, threshold=0.65):
    """
    Evaluate whether the generated text is hallucinated
    based on semantic similarity with the expected text.

    Args:
        expected (str): Ground truth / reference text
        actual (str): Model-generated text
        threshold (float): Similarity threshold below which output is considered hallucinated

    Returns:
        dict: {
            "hallucination": bool,
            "score": float,
            "threshold": float,
            "reason": str
        }
    """

    # Input validation
    if not expected or not actual:
        return {
            "hallucination": True,
            "score": 0.0,
            "threshold": threshold,
            "reason": "Empty expected or actual text"
        }

    score = similarity_score(expected, actual)
    is_hallucinated = score < threshold

    # Explanation layer
    if is_hallucinated:
        reason = f"Low semantic similarity ({score:.2f} < {threshold})"
    else:
        reason = f"Acceptable semantic similarity ({score:.2f} >= {threshold})"

    return {
        "hallucination": is_hallucinated,
        "score": float(score),
        "threshold": threshold,
        "reason": reason
    }