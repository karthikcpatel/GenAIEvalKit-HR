import pandas as pd
import pytest
import allure

from llm.llm_client import get_llm_response
from evaluator.metrics import compute_prf, tokenize


@allure.feature("HR Chatbot Evaluation")
@allure.story("Token-level PRF Evaluation")
@pytest.mark.integration
def test_prompt_prf():
    """
    Integration test: evaluates token-level precision, recall, and F1
    for LLM responses against expected answers from dataset.
    """

    df = pd.read_csv("prompts/hr_prompt_suite.csv")

    failures = []

    for row in df.itertuples():

        prompt = row.prompt
        expected = row.expected_answer
        category = row.category
        difficulty = getattr(row, "difficulty", "medium")

        # Generate LLM response
        response = get_llm_response(prompt)

        # Tokenize
        expected_tokens = tokenize(expected)
        actual_tokens = tokenize(response)

        # Compute PRF
        result = compute_prf(expected_tokens, actual_tokens)

        precision = result["precision"]
        recall = result["recall"]
        f1 = result["f1"]

        # Dynamic threshold based on difficulty
        if difficulty == "easy":
            threshold = 0.6
        elif difficulty == "hard":
            threshold = 0.4
        else:
            threshold = 0.2

        # Allure reporting
        with allure.step(f"Category: {category} | Difficulty: {difficulty}"):
            allure.attach(prompt, name="Prompt")
            allure.attach(expected, name="Expected Answer")
            allure.attach(response, name="LLM Response")
            allure.attach(str(result), name="PRF Metrics")

        # Collect failures instead of failing early
        if f1 < threshold:
            failures.append({
                "prompt": prompt,
                "category": category,
                "f1": round(f1, 3),
                "threshold": threshold,
                "precision": round(precision, 3),
                "recall": round(recall, 3)
            })

    # Final assertion
    assert not failures, f"\nLow PRF scores detected:\n{failures}"