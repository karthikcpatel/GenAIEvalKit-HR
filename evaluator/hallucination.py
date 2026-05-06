from evaluator.similarity import similarity_score

# Category-aware thresholds
CATEGORY_THRESHOLDS = {
    "leave": 0.70,
    "payroll": 0.72,
    "grievance": 0.65,
    "benefits": 0.68,
    "onboarding": 0.65,
    "default": 0.65
}

# Minimum length ratio to flag overly short responses
MIN_LENGTH_RATIO = 0.2


def evaluate_hallucination(expected, actual, category="default", threshold=None):
    """
    Evaluate whether the generated text is hallucinated based on
    semantic similarity with the expected text.

    NOTE: This function uses semantic similarity as a proxy for hallucination
    detection. It is best used alongside context grounding checks
    (compare answer vs retrieved context) for more complete coverage.
    For stricter fact-level verification, consider integrating RAGAS
    faithfulness metric.

    Args:
        expected  (str): Ground truth / reference text.
        actual    (str): Model-generated text.
        category  (str): HR domain category for threshold selection.
        threshold (float | None): Override threshold. If None, resolved
                                  from CATEGORY_THRESHOLDS by category.

    Returns:
        dict: {
            "hallucination"    : bool,
            "score"            : float,
            "threshold"        : float,
            "reason"           : str,
            "expected_snippet" : str,   # first 100 chars, for logging
            "actual_snippet"   : str    # first 100 chars, for logging
        }
    """
    # ------------------------------------------------------------------ #
    # 1. Input validation                                                  #
    # ------------------------------------------------------------------ #
    if not expected or not actual:
        return {
            "hallucination": True,
            "score": 0.0,
            "threshold": threshold or CATEGORY_THRESHOLDS.get(category, CATEGORY_THRESHOLDS["default"]),
            "reason": "Empty expected or actual text",
            "expected_snippet": (expected or "")[:100],
            "actual_snippet": (actual or "")[:100]
        }

    # ------------------------------------------------------------------ #
    # 2. Resolve threshold                                                 #
    # ------------------------------------------------------------------ #
    if threshold is None:
        threshold = CATEGORY_THRESHOLDS.get(
            category.lower() if category else "default",
            CATEGORY_THRESHOLDS["default"]
        )

    # ------------------------------------------------------------------ #
    # 3. Length / coverage guard                                           #
    # Short responses (e.g. "Please check the HR portal.") can score      #
    # deceptively high due to neutral phrasing.                           #
    # ------------------------------------------------------------------ #
    expected_word_count = len(expected.split())
    actual_word_count = len(actual.split())
    length_ratio = actual_word_count / max(expected_word_count, 1)

    if length_ratio < MIN_LENGTH_RATIO:
        return {
            "hallucination": True,
            "score": 0.0,
            "threshold": threshold,
            "reason": (
                f"Response too short to be meaningful "
                f"(length ratio: {length_ratio:.2f}, minimum: {MIN_LENGTH_RATIO})"
            ),
            "expected_snippet": expected[:100],
            "actual_snippet": actual[:100]
        }

    # ------------------------------------------------------------------ #
    # 4. Semantic similarity check                                         #
    # ------------------------------------------------------------------ #
    score = similarity_score(expected, actual)
    is_hallucinated = score < threshold

    if is_hallucinated:
        reason = (
            f"Low semantic similarity ({score:.2f} < {threshold}) "
            f"[category={category}]"
        )
    else:
        reason = (
            f"Acceptable semantic similarity ({score:.2f} >= {threshold}) "
            f"[category={category}]"
        )

    return {
        "hallucination": is_hallucinated,
        "score": float(score),
        "threshold": threshold,
        "reason": reason,
        "expected_snippet": expected[:100],
        "actual_snippet": actual[:100]
    }
