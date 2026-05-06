import pytest
import allure

from evaluator.similarity import (
    similarity_score,
    batch_similarity,
    encode_text,
    get_cache_stats
)

# ------------------------------------------------------------------ #
# Named threshold constants                                            #
# Defined here so the rationale is clear and adjustments are          #
# centralised if the underlying model is ever swapped.                #
# ------------------------------------------------------------------ #

IDENTICAL_THRESHOLD  = 0.95   # exact same text → near-perfect similarity
PARAPHRASE_THRESHOLD = 0.50   # rephrased but same meaning → still similar
UNRELATED_CEILING    = 0.40   # unrelated topics → should score below this
EMPTY_SCORE          = 0.0    # any empty / whitespace input → always 0.0


# ================================================================== #
# similarity_score — unit tests                                        #
# ================================================================== #

@allure.feature("HR Chatbot Evaluation")
@allure.story("Semantic Similarity")
class TestSimilarityScore:

    def test_identical_texts(self):
        """
        Identical strings should produce near-perfect similarity.
        """
        s = similarity_score("Leave policy", "Leave policy")
        assert s >= IDENTICAL_THRESHOLD, (
            f"Expected >= {IDENTICAL_THRESHOLD} for identical texts, got {s:.4f}"
        )

    def test_paraphrase(self):
        """
        Semantically equivalent but differently worded texts should
        still score above the paraphrase threshold.
        """
        s = similarity_score(
            "Apply leave through HR portal",
            "You can request leave via HR system"
        )
        assert s >= PARAPHRASE_THRESHOLD, (
            f"Expected >= {PARAPHRASE_THRESHOLD} for paraphrases, got {s:.4f}"
        )

    def test_unrelated_texts(self):
        """
        Unrelated topics should score well below the unrelated ceiling.
        """
        s = similarity_score("Leave policy", "Stock market investment")
        assert s < UNRELATED_CEILING, (
            f"Expected < {UNRELATED_CEILING} for unrelated texts, got {s:.4f}"
        )

    def test_empty_expected(self):
        """
        Empty expected string should return 0.0 immediately.
        """
        s = similarity_score("", "some response")
        assert s == EMPTY_SCORE

    def test_empty_actual(self):
        """
        Empty actual string should return 0.0 immediately.
        """
        s = similarity_score("Leave policy", "")
        assert s == EMPTY_SCORE

    def test_both_empty(self):
        """
        Both strings empty should return 0.0.
        """
        s = similarity_score("", "")
        assert s == EMPTY_SCORE

    def test_whitespace_only_expected(self):
        """
        Whitespace-only expected string should return 0.0.
        Encoding whitespace produces a near-zero vector with undefined
        cosine similarity — guard returns 0.0 before encoding.
        """
        s = similarity_score("   ", "Leave policy")
        assert s == EMPTY_SCORE

    def test_whitespace_only_actual(self):
        """
        Whitespace-only actual string should return 0.0.
        """
        s = similarity_score("Leave policy", "   ")
        assert s == EMPTY_SCORE

    def test_numeric_identical(self):
        """
        Numeric strings that are identical should score high.
        Relevant for HR contexts like policy numbers or durations.
        """
        s = similarity_score("30", "30")
        assert s >= IDENTICAL_THRESHOLD, (
            f"Expected >= {IDENTICAL_THRESHOLD} for identical numeric text, got {s:.4f}"
        )

    def test_numeric_different(self):
        """
        Different numeric strings should score low.
        '30 days' and '90 days' mean very different things in HR policy.
        """
        s = similarity_score("30", "90")
        assert s < PARAPHRASE_THRESHOLD, (
            f"Expected < {PARAPHRASE_THRESHOLD} for different numbers, got {s:.4f}"
        )

    def test_score_is_float(self):
        """
        Return value must always be a Python float, not a numpy scalar.
        """
        s = similarity_score("Leave policy", "HR leave request")
        assert isinstance(s, float)

    def test_score_in_valid_range(self):
        """
        Cosine similarity of normalised embeddings must be in [0.0, 1.0].
        """
        s = similarity_score("Annual leave entitlement", "Paid time off")
        assert 0.0 <= s <= 1.0


# ================================================================== #
# batch_similarity — unit tests                                        #
# ================================================================== #

@allure.feature("HR Chatbot Evaluation")
@allure.story("Semantic Similarity")
class TestBatchSimilarity:

    def test_basic_batch(self):
        """
        A mixed batch: one identical pair and one unrelated pair.
        """
        pairs  = [
            ("Leave policy", "Leave policy"),
            ("Leave policy", "Stock market investment")
        ]
        scores = batch_similarity(pairs)

        assert len(scores) == 2
        assert scores[0] >= IDENTICAL_THRESHOLD, (
            f"Identical pair: expected >= {IDENTICAL_THRESHOLD}, got {scores[0]:.4f}"
        )
        assert scores[1] < UNRELATED_CEILING, (
            f"Unrelated pair: expected < {UNRELATED_CEILING}, got {scores[1]:.4f}"
        )

    def test_empty_pairs_list(self):
        """
        An empty input list should return an empty list immediately.
        """
        assert batch_similarity([]) == []

    def test_empty_string_in_pair(self):
        """
        A pair with an empty expected string should return 0.0
        without raising an error.
        """
        scores = batch_similarity([("", "some response")])
        assert scores == [0.0]

    def test_whitespace_only_in_pair(self):
        """
        A pair where one side is whitespace-only should return 0.0.
        """
        scores = batch_similarity([("   ", "Leave policy")])
        assert scores == [0.0]

    def test_mixed_valid_and_empty_pairs(self):
        """
        A batch mixing valid pairs and empty-string pairs should handle
        each correctly and preserve order.
        """
        pairs = [
            ("Leave policy", "Leave policy"),   # valid → high score
            ("", "Leave policy"),               # empty → 0.0
            ("Notice period", "Notice period")  # valid → high score
        ]
        scores = batch_similarity(pairs)

        assert len(scores) == 3
        assert scores[0] >= IDENTICAL_THRESHOLD
        assert scores[1] == 0.0
        assert scores[2] >= IDENTICAL_THRESHOLD

    def test_batch_returns_floats(self):
        """
        All returned values must be Python floats.
        """
        pairs  = [("HR policy", "HR policy")]
        scores = batch_similarity(pairs)
        assert all(isinstance(s, float) for s in scores)

    def test_batch_scores_in_valid_range(self):
        """
        All scores must be in [0.0, 1.0].
        """
        pairs = [
            ("Leave policy", "Annual leave entitlement"),
            ("Payroll processing", "Salary disbursement")
        ]
        scores = batch_similarity(pairs)
        assert all(0.0 <= s <= 1.0 for s in scores)


# ================================================================== #
# Cache behaviour tests                                                #
# ================================================================== #

@allure.feature("HR Chatbot Evaluation")
@allure.story("Semantic Similarity")
class TestCacheBehaviour:

    def test_cache_is_populated_after_call(self):
        """
        After encoding a text, the lru_cache should contain at least
        one entry.
        """
        encode_text.cache_clear()
        similarity_score("Leave policy", "Leave policy")
        stats = get_cache_stats()
        assert stats["currsize"] > 0

    def test_cache_hit_on_repeated_call(self):
        """
        Encoding the same text twice should produce a cache hit on the
        second call.
        """
        encode_text.cache_clear()

        similarity_score("Leave policy", "Leave policy")
        stats_after_first = get_cache_stats()

        similarity_score("Leave policy", "Leave policy")
        stats_after_second = get_cache_stats()

        assert stats_after_second["hits"] > stats_after_first["hits"], (
            "Expected at least one cache hit on repeated call"
        )

    def test_cache_stats_keys(self):
        """
        get_cache_stats should return a dict with the expected keys.
        """
        stats = get_cache_stats()
        assert set(stats.keys()) == {"hits", "misses", "maxsize", "currsize"}
