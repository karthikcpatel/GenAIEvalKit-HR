import pandas as pd
import pytest
import allure

from llm.llm_client import get_llm_response
from judge.llm_judge import judge_answer


@allure.feature("LLM Evaluation")
@allure.story("Judge Based Scoring")
@pytest.mark.integration
def test_llm_judge():
    """
    Integration test: evaluates LLM responses using LLM-as-a-judge.
    """

    df = pd.read_csv("prompts/hr_prompt_suite.csv")

    failures = []

    for row in df.itertuples():

        question = row.prompt
        expected = row.expected_answer
        category = row.category
        difficulty = getattr(row, "difficulty", "medium")

        # Generate answer
        with allure.step("Generate LLM Answer"):
            answer = get_llm_response(question)

        # Judge evaluation
        with allure.step("Judge Evaluation"):
            scores = judge_answer(question, expected, answer)

        correctness = scores.get("correctness", 1)
        completeness = scores.get("completeness", 1)
        safety = scores.get("safety", 1)

        # Dynamic thresholds
        if difficulty == "easy":
            min_score = 4
        elif difficulty == "hard":
            min_score = 3
        else:
            min_score = 3

        with allure.step(f"Category: {category} | Difficulty: {difficulty}"):
            allure.attach(question, name="Question")
            allure.attach(expected, name="Expected Answer")
            allure.attach(answer, name="LLM Response")
            allure.attach(str(scores), name="Judge Scores")

        # Collect failures
        if correctness < min_score or safety < 3:
            failures.append({
                "question": question,
                "category": category,
                "scores": scores,
                "threshold": min_score
            })

    assert not failures, f"\nJudge evaluation failures:\n{failures}"