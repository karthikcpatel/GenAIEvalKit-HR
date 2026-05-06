import pandas as pd
import pytest
import allure

from rag.rag_pipeline import rag_answer, FALLBACK_MESSAGE
from evaluator.similarity import similarity_score

# ------------------------------------------------------------------ #
# Thresholds                                                           #
# ------------------------------------------------------------------ #

# Grounding: similarity between the generated answer and the
# retrieved context. Measures whether the answer is rooted in
# what was actually retrieved rather than model parametric memory.
GROUNDING_THRESHOLDS = {
    "easy":   0.65,
    "medium": 0.60,
    "hard":   0.55
}

# Faithfulness: similarity between the generated answer and the
# ground-truth expected answer. Measures whether the answer says
# the right thing regardless of what was retrieved.
FAITHFULNESS_THRESHOLDS = {
    "easy":   0.65,
    "medium": 0.60,
    "hard":   0.55
}

# Phrases that indicate a fallback / deferral response.
# Grounding and faithfulness checks are skipped for these — a
# generic deferral has no meaningful signal to measure.
FALLBACK_INDICATORS = [
    "please check the hr portal",
    "i don't have enough information"
]


def _is_fallback(answer):
    """Return True if the answer is a generic fallback deferral."""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in FALLBACK_INDICATORS)


@allure.feature("RAG Evaluation")
@allure.story("Grounded Response Validation")
@pytest.mark.integration
def test_rag_grounded():
    """
    Validate that RAG pipeline responses are:
      1. Grounded in the retrieved context
         (answer similarity vs context chunks)
      2. Faithful to the ground-truth expected answer
         (answer similarity vs expected answer)

    Fallback responses ("Please check the HR portal...") are detected
    and skipped — they carry no meaningful grounding signal.

    All failures across both checks are collected before asserting.
    """
    df = pd.read_csv("prompts/hr_prompt_suite.csv")

    failures = []

    for row in df.itertuples():

        query    = row.prompt
        expected = row.expected_answer
        category = getattr(row, "category", "general")

        # ---------------------------------------------------------- #
        # Normalise difficulty — guard against NaN / blank cells      #
        # ---------------------------------------------------------- #
        difficulty = str(
            getattr(row, "difficulty", "medium") or "medium"
        ).strip().lower()

        grounding_threshold   = GROUNDING_THRESHOLDS.get(
            difficulty, GROUNDING_THRESHOLDS["medium"]
        )
        faithfulness_threshold = FAITHFULNESS_THRESHOLDS.get(
            difficulty, FAITHFULNESS_THRESHOLDS["medium"]
        )

        # ---------------------------------------------------------- #
        # RAG pipeline call                                           #
        # ---------------------------------------------------------- #
        result       = rag_answer(query)
        answer       = result["answer"]
        contexts     = result["contexts"]
        ret_scores   = result["retrieval_scores"]
        ret_cats     = result["categories"]
        context_text = " ".join(contexts)

        # ---------------------------------------------------------- #
        # Fallback detection — skip scoring for deferral responses   #
        # ---------------------------------------------------------- #
        if _is_fallback(answer):
            with allure.step(
                f"[FALLBACK] Category: {category} | Query: {query[:60]}"
            ):
                allure.attach(query,   name="Query")
                allure.attach(answer,  name="Generated Answer (Fallback)")
                allure.attach(
                    "Skipped — fallback response detected",
                    name="Grounding Status"
                )
            continue

        # ---------------------------------------------------------- #
        # Score computation                                           #
        # ---------------------------------------------------------- #
        grounding_score    = similarity_score(context_text, answer)
        faithfulness_score = similarity_score(expected,     answer)

        # ---------------------------------------------------------- #
        # Allure reporting — all fields attached for every row       #
        # ---------------------------------------------------------- #
        with allure.step(
            f"Category: {category} | Difficulty: {difficulty} | "
            f"Query: {query[:60]}"
        ):
            allure.attach(query,          name="Query")
            allure.attach(expected,       name="Expected Answer")
            allure.attach(answer,         name="Generated Answer")
            allure.attach(
                context_text or "(no context)",
                name="Retrieved Context"
            )
            allure.attach(
                str(ret_scores),
                name="Retrieval Scores"
            )
            allure.attach(
                ret_cats[0] if ret_cats else "unknown",
                name="Top Retrieved Category"
            )
            allure.attach(
                (
                    f"Grounding    : {grounding_score:.3f}  "
                    f"(threshold: {grounding_threshold})\n"
                    f"Faithfulness : {faithfulness_score:.3f}  "
                    f"(threshold: {faithfulness_threshold})\n"
                    f"Difficulty   : {difficulty}"
                ),
                name="Score Summary"
            )

        # ---------------------------------------------------------- #
        # Failure collection                                          #
        # ---------------------------------------------------------- #
        fail_reason = None

        if grounding_score < grounding_threshold:
            fail_reason = (
                f"grounding={grounding_score:.3f} "
                f"< threshold={grounding_threshold}"
            )
        elif faithfulness_score < faithfulness_threshold:
            fail_reason = (
                f"faithfulness={faithfulness_score:.3f} "
                f"< threshold={faithfulness_threshold}"
            )

        if fail_reason:
            failures.append({
                "query":               query,
                "category":            ret_cats[0] if ret_cats else "unknown",
                "difficulty":          difficulty,
                "grounding_score":     round(grounding_score,    3),
                "faithfulness_score":  round(faithfulness_score, 3),
                "grounding_threshold": grounding_threshold,
                "faith_threshold":     faithfulness_threshold,
                "reason":              fail_reason
            })

    # ---------------------------------------------------------------- #
    # Soft assertion — all failures surfaced in one run                 #
    # ---------------------------------------------------------------- #
    assert not failures, (
        f"\n{len(failures)} RAG grounding/faithfulness failure(s) detected:\n"
        + "\n".join(
            f"  [{f['category']} / {f['difficulty']}] "
            f"{f['query'][:60]}... "
            f"G={f['grounding_score']} F={f['faithfulness_score']} "
            f"— {f['reason']}"
            for f in failures
        )
    )
