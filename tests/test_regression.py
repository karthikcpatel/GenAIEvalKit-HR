import pandas as pd
import pytest
import allure

from llm.llm_client import get_llm_response
from evaluator.similarity import similarity_score


# Baseline threshold (can be tuned or stored externally)
BASELINE_SIMILARITY = 0.7


@allure.feature("HR Chatbot Evaluation")
@allure.story("Regression Testing")
@pytest.mark.integration
def test_regression_hr_dataset():
    """
    Ensure model responses remain semantically consistent
    with expected answers from HR dataset.
    """

    df = pd.read_csv("prompts/hr_prompt_suite.csv")

    failures = []

    for row in df.itertuples():

        prompt = row.prompt
        expected = row.expected_answer
        category = row.category
        difficulty = getattr(row, "difficulty", "medium")

        response = get_llm_response(prompt)

        score = similarity_score(expected, response)

        # Dynamic threshold based on difficulty
        if difficulty == "easy":
            threshold = 0.65
        elif difficulty == "hard":
            threshold = 0.6
        else:
            threshold = BASELINE_SIMILARITY

        with allure.step(f"Category: {category} | Difficulty: {difficulty}"):
            allure.attach(prompt, name="Prompt")
            allure.attach(expected, name="Expected Answer")
            allure.attach(response, name="LLM Response")
            allure.attach(str(score), name="Similarity Score")

        if score < threshold:
            failures.append({
                "prompt": prompt,
                "category": category,
                "score": round(score, 3),
                "threshold": threshold
            })

    assert not failures, f"\nRegression failures detected:\n{failures}"