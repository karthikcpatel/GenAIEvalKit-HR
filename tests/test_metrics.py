import pandas as pd
import pytest
import allure

from rag.rag_pipeline import rag_answer
from evaluator.metrics import compute_prf_from_text, aggregate_prf_by_category

# ------------------------------------------------------------------ #
# Thresholds                                                           #
# ------------------------------------------------------------------ #

# F1 thresholds by difficulty — easy questions have unambiguous answers
# so a higher bar is appropriate; hard questions allow slightly more
# variance but still require meaningful overlap.
DIFFICULTY_F1_THRESHOLDS = {
    "easy":   0.65,
    "medium": 0.55,
    "hard":   0.50
}

# Independent floors for precision and recall.
# Prevents a high-precision / low-recall response (or vice versa)
# from masking a poor result behind an acceptable F1.
MIN_PRECISION = 0.40
MIN_RECALL    = 0.45


@allure.feature("HR Chatbot Evaluation")
@allure.story("Token-level PRF Evaluation")
@pytest.mark.integration
def test_prompt_prf():
    """
    Integration test: evaluates token-level precision, recall, and F1
    for RAG responses against expected answers from the prompt suite.

    Evaluation uses:
      - Stopword removal to avoid common words inflating scores
      - Bigram blending to capture phrase-level accuracy
      - Independent precision and recall floors alongside F1
      - Category-level aggregation attached to the Allure report
    """

    df = pd.read_csv("prompts/hr_prompt_suite.csv")

    failures     = []
    all_results  = []   # for category-level aggregation

    for row in df.itertuples():

        prompt   = row.prompt
        expected = row.expected_answer
        category = getattr(row, "category", "general")

        # ---------------------------------------------------------- #
        # Normalise difficulty — guard against NaN / blank cells      #
        # ---------------------------------------------------------- #
        difficulty = str(
            getattr(row, "difficulty", "medium") or "medium"
        ).strip().lower()

        f1_threshold = DIFFICULTY_F1_THRESHOLDS.get(
            difficulty, DIFFICULTY_F1_THRESHOLDS["medium"]
        )

        # ---------------------------------------------------------- #
        # Run full RAG pipeline — mirrors production conditions       #
        # ---------------------------------------------------------- #
        rag_result = rag_answer(prompt)
        response   = rag_result["answer"]

        # ---------------------------------------------------------- #
        # Compute PRF                                                  #
        # Stopword removal + bigram blending enabled by default        #
        # ---------------------------------------------------------- #
        result = compute_prf_from_text(
            expected_text=expected,
            actual_text=response,
            remove_stopwords=True,
            use_ngrams=True
        )

        precision    = result["precision"]
        recall       = result["recall"]
        f1           = result["f1"]
        unigram_f1   = result.get("unigram_f1", f1)
        ngram_f1     = result.get("ngram_f1",   f1)

        # ---------------------------------------------------------- #
        # Allure reporting — attach scores for every row              #
        # ---------------------------------------------------------- #
        with allure.step(
            f"Category: {category} | Difficulty: {difficulty} | Prompt: {prompt[:80]}"
        ):
            allure.attach(prompt,   name="Prompt")
            allure.attach(expected, name="Expected Answer")
            allure.attach(response, name="LLM Response")
            allure.attach(
                (
                    f"Precision   : {precision:.3f}  (floor: {MIN_PRECISION})\n"
                    f"Recall      : {recall:.3f}  (floor: {MIN_RECALL})\n"
                    f"F1 (blended): {f1:.3f}  (threshold: {f1_threshold})\n"
                    f"Unigram F1  : {unigram_f1:.3f}\n"
                    f"Ngram F1    : {ngram_f1:.3f}\n"
                    f"Difficulty  : {difficulty}"
                ),
                name="PRF Metrics"
            )

        # ---------------------------------------------------------- #
        # Accumulate for category aggregation                         #
        # ---------------------------------------------------------- #
        all_results.append({
            "category":  category,
            "precision": precision,
            "recall":    recall,
            "f1":        f1
        })

        # ---------------------------------------------------------- #
        # Failure detection — F1 gate + independent P/R floors        #
        # ---------------------------------------------------------- #
        fail_reason = None

        if f1 < f1_threshold:
            fail_reason = f"F1={f1:.3f} < threshold={f1_threshold}"
        elif precision < MIN_PRECISION:
            fail_reason = f"Precision={precision:.3f} < floor={MIN_PRECISION}"
        elif recall < MIN_RECALL:
            fail_reason = f"Recall={recall:.3f} < floor={MIN_RECALL}"

        if fail_reason:
            failures.append({
                "prompt":     prompt,
                "category":   category,
                "difficulty": difficulty,
                "f1":         round(f1,        3),
                "precision":  round(precision, 3),
                "recall":     round(recall,    3),
                "threshold":  f1_threshold,
                "reason":     fail_reason
            })

    # ---------------------------------------------------------------- #
    # Category-level aggregation — attached to Allure for trend review  #
    # ---------------------------------------------------------------- #
    category_summary = aggregate_prf_by_category(all_results)

    summary_lines = [
        f"{cat:20s} | precision={v['precision']:.3f} | "
        f"recall={v['recall']:.3f} | f1={v['f1']:.3f} | n={v['count']}"
        for cat, v in sorted(category_summary.items())
    ]

    with allure.step("Category-level PRF Summary"):
        allure.attach(
            "\n".join(summary_lines),
            name="Category Aggregation",
            attachment_type=allure.attachment_type.TEXT
        )

    # ---------------------------------------------------------------- #
    # Final soft assertion — all failures surfaced in one run           #
    # ---------------------------------------------------------------- #
    assert not failures, (
        f"\n{len(failures)} PRF failure(s) detected:\n"
        + "\n".join(
            f"  [{f['category']} / {f['difficulty']}] {f['prompt'][:60]}... "
            f"P={f['precision']} R={f['recall']} F1={f['f1']} — {f['reason']}"
            for f in failures
        )
    )
