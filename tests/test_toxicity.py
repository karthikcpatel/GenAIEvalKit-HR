import pytest
import pandas as pd
from evaluator.toxicity import is_toxic, toxicity_score


# -----------------------------
# Unit Tests (Core Functionality)
# -----------------------------

def test_toxic_basic():
    result = is_toxic("I hate you")

    assert result["toxic"] is True
    assert result["score"] > 0


def test_non_toxic_text():
    result = is_toxic("Have a great day!")

    assert result["toxic"] is False
    assert result["score"] == 0


def test_word_boundary():
    # "kill" should NOT match "skill"
    result = is_toxic("I have great skills")

    assert result["toxic"] is False


def test_multiple_toxic_words():
    result = is_toxic("I hate you, you are stupid")

    assert result["toxic"] is True
    assert result["score"] > 0.5


def test_threshold_behavior():
    # Lower threshold should flag toxicity
    result = is_toxic("stupid", threshold=0.3)
    assert result["toxic"] is True

    # Higher threshold should NOT flag
    result = is_toxic("stupid", threshold=0.9)
    assert result["toxic"] is False


def test_empty_input():
    result = is_toxic("")

    assert result["toxic"] is False
    assert result["score"] == 0.0


def test_toxicity_score_function():
    score = toxicity_score("I hate this")

    assert score > 0
    assert score <= 1.0


# -----------------------------
# Dataset-Based Test (HR Suite)
# -----------------------------

@pytest.mark.integration
def test_hr_prompts_are_safe():
    """
    Ensure HR prompts themselves are non-toxic.
    """

    df = pd.read_csv("prompts/hr_prompt_suite.csv")

    toxic_prompts = []

    for row in df.itertuples():
        prompt = row.prompt

        result = is_toxic(prompt)

        if result["toxic"]:
            toxic_prompts.append({
                "prompt": prompt,
                "score": result["score"]
            })

    assert not toxic_prompts, f"Toxic prompts found:\n{toxic_prompts}"


@pytest.mark.integration
def test_expected_answers_are_safe():
    """
    Ensure expected HR answers are safe.
    """

    df = pd.read_csv("prompts/hr_prompt_suite.csv")

    toxic_answers = []

    for row in df.itertuples():
        expected = row.expected_answer

        result = is_toxic(expected)

        if result["toxic"]:
            toxic_answers.append({
                "answer": expected,
                "score": result["score"]
            })

    assert not toxic_answers, f"Toxic expected answers found:\n{toxic_answers}"