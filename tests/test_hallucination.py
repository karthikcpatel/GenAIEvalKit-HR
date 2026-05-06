import pandas as pd
import allure
import pytest

from rag.rag_pipeline import rag_answer
from evaluator.hallucination import evaluate_hallucination

# ------------------------------------------------------------------ #
# Difficulty-based thresholds                                          #
# Higher difficulty = slightly more lenient, but never trivially low. #
# ------------------------------------------------------------------ #
DIFFICULTY_THRESHOLDS = {
    "easy": 0.70,
    "medium": 0.65,
    "hard": 0.58
}

# Minimum grounding threshold (answer vs retrieved context)
GROUNDING_THRESHOLD = 0.40


@allure.feature("HR Chatbot Evaluation")
@allure.story("Hallucination Detection")
@pytest.mark.integration
def test_no_hallucination():
    """
    Validate that LLM responses do not hallucinate relative to:
      1. The ground-truth expected answer  (faithfulness check)
      2. The retrieved RAG context         (grounding check)

    Both checks are run per prompt. Failures from either are aggregated
    and reported together at the end of the test run.
    """
    df = pd.read_csv("prompts/hr_prompt_suite.csv")

    failures = []

    for row in df.itertuples():
        prompt = row.prompt
        expected = row.expected_answer
        category = getattr(row, "category", "default")

        # ---------------------------------------------------------- #
        # Normalise difficulty — guard against NaN / blank cells      #
        # ---------------------------------------------------------- #
        difficulty = str(
            getattr(row, "difficulty", "medium") or "medium"
        ).strip().lower()

        threshold = DIFFICULTY_THRESHOLDS.get(difficulty, DIFFICULTY_THRESHOLDS["medium"])

        # ---------------------------------------------------------- #
        # Run RAG pipeline — mirrors production conditions exactly    #
        # ---------------------------------------------------------- #
        rag_result = rag_answer(prompt)
        response = rag_result["answer"]
        contexts = rag_result["contexts"]
        context_text = " ".join(contexts)

        # ---------------------------------------------------------- #
        # Check 1: Answer vs expected (faithfulness)                  #
        # ---------------------------------------------------------- #
        faithfulness_eval = evaluate_hallucination(
            expected=expected,
            actual=response,
            category=category,
            threshold=threshold
        )

        # ---------------------------------------------------------- #
        # Check 2: Answer vs retrieved context (grounding)            #
        # Ensures the response is rooted in what was actually         #
        # retrieved, not generated from model parametric memory.      #
        # ---------------------------------------------------------- #
        grounding_eval = evaluate_hallucination(
            expected=context_text,
            actual=response,
            category=category,
            threshold=GROUNDING_THRESHOLD
        )

        # ---------------------------------------------------------- #
        # Allure reporting — attach all scores for every case         #
        # (not only failures) so near-misses are visible in reports   #
        # ---------------------------------------------------------- #
        with allure.step(f"Category: {category} | Difficulty: {difficulty} | Prompt: {prompt[:80]}"):
            allure.attach(prompt,    name="Prompt")
            allure.attach(expected,  name="Expected Answer")
            allure.attach(response,  name="LLM Response")
            allure.attach(
                context_text or "(no context retrieved)",
                name="Retrieved Context"
            )
            allure.attach(
                (
                    f"Faithfulness  — score: {faithfulness_eval['score']:.3f} "
                    f"| threshold: {faithfulness_eval['threshold']} "
                    f"| hallucinated: {faithfulness_eval['hallucination']}\n"
                    f"Reason: {faithfulness_eval['reason']}\n\n"
                    f"Grounding     — score: {grounding_eval['score']:.3f} "
                    f"| threshold: {grounding_eval['threshold']} "
                    f"| hallucinated: {grounding_eval['hallucination']}\n"
                    f"Reason: {grounding_eval['reason']}"
                ),
                name="Evaluation Summary"
            )

        # ---------------------------------------------------------- #
        # Collect failures from both checks                           #
        # ---------------------------------------------------------- #
        if faithfulness_eval["hallucination"]:
            failures.append({
                "check": "faithfulness",
                "prompt": prompt,
                "category": category,
                "difficulty": difficulty,
                "score": faithfulness_eval["score"],
                "threshold": faithfulness_eval["threshold"],
                "reason": faithfulness_eval["reason"]
            })

        if grounding_eval["hallucination"]:
            failures.append({
                "check": "grounding",
                "prompt": prompt,
                "category": category,
                "difficulty": difficulty,
                "score": grounding_eval["score"],
                "threshold": grounding_eval["threshold"],
                "reason": grounding_eval["reason"]
            })

    # Soft assertion — aggregate all failures before raising
    assert not failures, (
        f"\n{len(failures)} hallucination/grounding failure(s) detected:\n"
        + "\n".join(
            f"  [{f['check'].upper()}] {f['prompt'][:60]}... "
            f"score={f['score']:.3f} threshold={f['threshold']} — {f['reason']}"
            for f in failures
        )
    )
