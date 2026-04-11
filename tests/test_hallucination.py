import pandas as pd
import allure
import pytest

from llm.llm_client import get_llm_response
from evaluator.hallucination import evaluate_hallucination


@allure.feature("HR Chatbot Evaluation")
@allure.story("Hallucination Detection")
@pytest.mark.integration
def test_no_hallucination():

    df = pd.read_csv("prompts/hr_prompt_suite.csv")

    failures = []

    for row in df.itertuples():

        prompt = row.prompt
        expected = row.expected_answer
        category = row.category
        difficulty = getattr(row, "difficulty", "medium")

        # Dynamic threshold based on difficulty
        if difficulty == "easy":
            threshold = 0.6
        elif difficulty == "hard":
            threshold = 0.4
        else:
            threshold = 0.55

        with allure.step(f"Category: {category} | Difficulty: {difficulty}"):

            response = get_llm_response(prompt)

            result = evaluate_hallucination(
                expected,
                response,
                threshold=threshold
            )

            allure.attach(prompt, name="Prompt")
            allure.attach(expected, name="Expected Answer")
            allure.attach(response, name="LLM Response")
            allure.attach(str(result), name="Evaluation Result")

            if result["hallucination"]:
                failures.append({
                    "prompt": prompt,
                    "score": result["score"],
                    "reason": result["reason"]
                })

    # Soft assertion (aggregate failures)
    assert not failures, f"Hallucinations detected:\n{failures}"