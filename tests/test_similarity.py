import pytest
from evaluator.similarity import similarity_score


def test_similarity_identical():
    s = similarity_score("Leave policy", "Leave policy")
    assert s > 0.9


def test_similarity_paraphrase():
    s = similarity_score(
        "Apply leave through HR portal",
        "You can request leave via HR system"
    )
    assert s > 0.5


def test_similarity_different():
    s = similarity_score(
        "Leave policy",
        "Stock market investment"
    )
    assert s < 0.4


def test_similarity_empty():
    s = similarity_score("", "text")
    assert s == 0.0