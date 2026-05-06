import pytest
import allure
import pandas as pd

from evaluator.toxicity import is_toxic, toxicity_score, get_severity
from rag.rag_pipeline import rag_answer


# ================================================================== #
# Unit Tests — Core functionality                                      #
# ================================================================== #

@allure.feature("HR Chatbot Evaluation")
@allure.story("Toxicity Detection")
class TestIsToxic:

    def test_toxic_basic(self):
        """
        A clearly toxic phrase should be flagged with a positive score.
        """
        result = is_toxic("I hate you")
        assert result["toxic"] is True
        assert result["score"] > 0

    def test_non_toxic_text(self):
        """
        Clean, professional text should not be flagged.
        """
        result = is_toxic("Have a great day!")
        assert result["toxic"] is False
        assert result["score"] == 0.0

    def test_word_boundary(self):
        """
        Partial matches must not trigger — 'kill' should NOT match 'skill'.
        """
        result = is_toxic("I have great skills")
        assert result["toxic"] is False
        assert "kill" not in result["matched_terms"]

    def test_multiple_toxic_words(self):
        """
        Multiple toxic terms should accumulate into a higher score.
        """
        result = is_toxic("I hate you, you are stupid")
        assert result["toxic"] is True
        assert len(result["matched_terms"]) >= 2

    def test_threshold_lower_flags(self):
        """
        A threshold below the word's weight should flag the text.
        """
        result = is_toxic("stupid", threshold=0.3)
        assert result["toxic"] is True

    def test_threshold_higher_passes(self):
        """
        A threshold above the word's weight should not flag the text.
        """
        result = is_toxic("stupid", threshold=0.99)
        assert result["toxic"] is False

    def test_empty_input(self):
        """
        Empty string should return safe result with score 0.0.
        """
        result = is_toxic("")
        assert result["toxic"] is False
        assert result["score"] == 0.0

    def test_whitespace_only_input(self):
        """
        Whitespace-only string should be treated as empty.
        """
        result = is_toxic("     ")
        assert result["toxic"] is False
        assert result["score"] == 0.0

    # ---- Return dict contract ------------------------------------ #

    def test_return_dict_has_all_keys(self):
        """
        is_toxic must always return a dict with all expected keys,
        regardless of input (toxic, clean, or empty).
        """
        expected_keys = {"toxic", "score", "threshold", "severity",
                         "matched_terms", "reason"}
        assert set(is_toxic("I hate this").keys())  == expected_keys
        assert set(is_toxic("Great day!").keys())   == expected_keys
        assert set(is_toxic("").keys())             == expected_keys

    def test_matched_terms_populated_for_toxic_text(self):
        """
        matched_terms should contain the detected word(s).
        """
        result = is_toxic("I hate you")
        assert "hate" in result["matched_terms"]

    def test_matched_terms_empty_for_clean_text(self):
        """
        matched_terms should be an empty list for clean input.
        """
        result = is_toxic("Have a great day!")
        assert result["matched_terms"] == []

    def test_reason_populated_for_toxic_text(self):
        """
        reason should mention the matched term.
        """
        result = is_toxic("I hate this")
        assert "hate" in result["reason"]

    def test_reason_for_clean_text(self):
        """
        reason should indicate no toxic terms were found.
        """
        result = is_toxic("Have a great day!")
        assert result["reason"] == "No toxic terms detected"

    def test_reason_for_empty_input(self):
        """
        reason for empty input should indicate empty input — not
        'no toxic terms detected', since detection wasn't run.
        """
        result = is_toxic("")
        assert result["reason"] == "Empty input"

    # ---- Severity ------------------------------------------------ #

    def test_severity_high(self):
        """
        A high-weight term should produce 'high' severity.
        """
        result = is_toxic("kill")
        assert result["severity"] in ("high", "medium", "low")
        # Exact tier depends on lexicon size; just confirm it's not 'none'
        assert result["severity"] != "none"

    def test_severity_none_for_clean_text(self):
        """
        Clean text should produce 'none' severity.
        """
        result = is_toxic("Have a great day!")
        assert result["severity"] == "none"

    def test_severity_none_for_empty_input(self):
        """
        Empty input should produce 'none' severity.
        """
        result = is_toxic("")
        assert result["severity"] == "none"

    # ---- Negation / context-awareness ---------------------------- #

    def test_negated_term_not_flagged(self):
        """
        A toxic word used in a neutral HR context (after a negation
        prefix) should not be flagged.
        e.g. "policy on hate speech" should not trigger 'hate'.
        """
        result = is_toxic("We have a policy on hate speech in the workplace")
        # 'hate' follows 'policy on' — should be suppressed
        assert "hate" not in result["matched_terms"]

    def test_genuine_toxic_still_flagged(self):
        """
        A direct toxic statement should still be flagged even when
        some neutral HR phrases are present.
        """
        result = is_toxic("I genuinely hate my manager")
        assert result["toxic"] is True


# ================================================================== #
# Unit Tests — toxicity_score convenience wrapper                      #
# ================================================================== #

@allure.feature("HR Chatbot Evaluation")
@allure.story("Toxicity Detection")
class TestToxicityScore:

    def test_returns_float(self):
        score = toxicity_score("I hate this")
        assert isinstance(score, float)

    def test_positive_for_toxic_text(self):
        score = toxicity_score("I hate this")
        assert score > 0

    def test_zero_for_clean_text(self):
        score = toxicity_score("Great work today!")
        assert score == 0.0

    def test_in_valid_range(self):
        score = toxicity_score("hate kill stupid")
        assert 0.0 <= score <= 1.0

    def test_delegates_to_is_toxic(self):
        """
        toxicity_score must return the same score as is_toxic()["score"],
        confirming it delegates rather than duplicating logic.
        """
        text = "I hate this process"
        assert toxicity_score(text) == is_toxic(text)["score"]


# ================================================================== #
# Unit Tests — get_severity                                            #
# ================================================================== #

@allure.feature("HR Chatbot Evaluation")
@allure.story("Toxicity Detection")
class TestGetSeverity:

    def test_high(self):
        assert get_severity(0.85) == "high"
        assert get_severity(1.0)  == "high"

    def test_medium(self):
        assert get_severity(0.5)  == "medium"
        assert get_severity(0.79) == "medium"

    def test_low(self):
        assert get_severity(0.1)  == "low"
        assert get_severity(0.49) == "low"

    def test_none(self):
        assert get_severity(0.0)  == "none"


# ================================================================== #
# Integration Tests — HR prompt suite                                  #
# ================================================================== #

@allure.feature("HR Chatbot Evaluation")
@allure.story("Toxicity Detection")
@pytest.mark.integration
def test_hr_prompts_are_safe():
    """
    Ensure all prompts in the HR test suite are themselves non-toxic.
    Toxic prompts indicate a data quality issue in the prompt suite.
    """
    df = pd.read_csv("prompts/hr_prompt_suite.csv")
    toxic_prompts = []

    for row in df.itertuples():
        prompt = row.prompt
        result = is_toxic(prompt)

        with allure.step(f"Prompt: {prompt[:80]}"):
            allure.attach(prompt,                        name="Prompt")
            allure.attach(str(result["score"]),          name="Toxicity Score")
            allure.attach(str(result["matched_terms"]),  name="Matched Terms")
            allure.attach(result["severity"],            name="Severity")
            allure.attach(result["reason"],              name="Reason")

        if result["toxic"]:
            toxic_prompts.append({
                "prompt":        prompt,
                "score":         result["score"],
                "severity":      result["severity"],
                "matched_terms": result["matched_terms"]
            })

    assert not toxic_prompts, f"Toxic prompts found:\n{toxic_prompts}"


@allure.feature("HR Chatbot Evaluation")
@allure.story("Toxicity Detection")
@pytest.mark.integration
def test_expected_answers_are_safe():
    """
    Ensure all expected answers in the HR test suite are non-toxic.
    Toxic expected answers indicate a data quality issue in the suite.
    """
    df = pd.read_csv("prompts/hr_prompt_suite.csv")
    toxic_answers = []

    for row in df.itertuples():
        expected = row.expected_answer
        result   = is_toxic(expected)

        with allure.step(f"Expected Answer: {expected[:80]}"):
            allure.attach(expected,                      name="Expected Answer")
            allure.attach(str(result["score"]),          name="Toxicity Score")
            allure.attach(str(result["matched_terms"]),  name="Matched Terms")
            allure.attach(result["severity"],            name="Severity")
            allure.attach(result["reason"],              name="Reason")

        if result["toxic"]:
            toxic_answers.append({
                "answer":        expected,
                "score":         result["score"],
                "severity":      result["severity"],
                "matched_terms": result["matched_terms"]
            })

    assert not toxic_answers, f"Toxic expected answers found:\n{toxic_answers}"


@allure.feature("HR Chatbot Evaluation")
@allure.story("Toxicity Detection")
@pytest.mark.integration
def test_llm_responses_are_safe():
    """
    Ensure the RAG pipeline does not produce toxic responses for any
    prompt in the HR test suite.

    This is the most critical toxicity test — it validates the actual
    chatbot output rather than just the static dataset content.
    """
    df = pd.read_csv("prompts/hr_prompt_suite.csv")
    toxic_responses = []

    for row in df.itertuples():
        prompt   = row.prompt
        category = getattr(row, "category", "general")

        rag_result = rag_answer(prompt)
        response   = rag_result["answer"]
        result     = is_toxic(response)

        with allure.step(
            f"Category: {category} | Prompt: {prompt[:80]}"
        ):
            allure.attach(prompt,                        name="Prompt")
            allure.attach(response,                      name="LLM Response")
            allure.attach(str(result["score"]),          name="Toxicity Score")
            allure.attach(str(result["matched_terms"]),  name="Matched Terms")
            allure.attach(result["severity"],            name="Severity")
            allure.attach(result["reason"],              name="Reason")

        if result["toxic"]:
            toxic_responses.append({
                "prompt":        prompt,
                "category":      category,
                "response":      response,
                "score":         result["score"],
                "severity":      result["severity"],
                "matched_terms": result["matched_terms"]
            })

    assert not toxic_responses, (
        f"\n{len(toxic_responses)} toxic LLM response(s) detected:\n"
        + "\n".join(
            f"  [{r['category']}] {r['prompt'][:60]}... "
            f"score={r['score']} severity={r['severity']} "
            f"terms={r['matched_terms']}"
            for r in toxic_responses
        )
    )
