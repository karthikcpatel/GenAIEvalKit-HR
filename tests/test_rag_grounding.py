import pandas as pd
import pytest
import allure

from rag.rag_pipeline import rag_answer
from evaluator.similarity import similarity_score


@allure.feature("RAG Evaluation")
@allure.story("Grounded Response Validation")
@pytest.mark.integration
def test_rag_grounded():
    """
    Ensure RAG responses are grounded in retrieved context.
    """

    df = pd.read_csv("prompts/hr_prompt_suite.csv")

    failures = []

    for row in df.itertuples():

        query = row.prompt
        expected = row.expected_answer
        difficulty = getattr(row, "difficulty", "medium")

        result = rag_answer(query)

        answer = result["answer"]
        contexts = result["contexts"]

        # Combine all retrieved context
        context_text = " ".join(contexts)

        # Measure grounding (answer vs context)
        grounding_score = similarity_score(context_text, answer)

        # Dynamic threshold
        if difficulty == "easy":
            threshold = 0.5
        elif difficulty == "hard":
            threshold = 0.4
        else:
            threshold = 0.3

        with allure.step(f"Query: {query}"):
            allure.attach(query, name="Query")
            allure.attach(expected, name="Expected Answer")
            allure.attach(answer, name="Generated Answer")
            allure.attach(context_text, name="Retrieved Context")
            allure.attach(str(grounding_score), name="Grounding Score")

        if grounding_score < threshold:
            failures.append({
                "query": query,
                "score": round(grounding_score, 3),
                "threshold": threshold
            })

    assert not failures, f"\nGrounding failures detected:\n{failures}"