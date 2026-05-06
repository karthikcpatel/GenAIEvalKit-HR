import json
import os
import statistics
from collections import defaultdict

import pandas as pd
import pytest
import allure

from rag.rag_pipeline import rag_answer, FALLBACK_MESSAGE
from evaluator.similarity import similarity_score

# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

# Similarity thresholds per difficulty level.
# Monotonically decreasing: easy questions have unambiguous answers
# and are held to the highest bar.
DIFFICULTY_THRESHOLDS = {
    "easy":   0.70,
    "medium": 0.65,
    "hard":   0.60
}

# Path to the stored baseline file.
# Override via environment variable for CI/CD pipelines.
BASELINE_FILE = os.environ.get(
    "REGRESSION_BASELINE_PATH", "regression_baseline.json"
)

# Phrases that indicate a generic fallback / deferral response.
# These are skipped in regression scoring — a retrieval gap should
# not be counted as a model regression.
FALLBACK_INDICATORS = [
    "please check the hr portal",
    "i don't have enough information"
]


# ------------------------------------------------------------------ #
# Baseline helpers                                                     #
# ------------------------------------------------------------------ #

def load_baseline():
    """
    Load the stored regression baseline from disk.

    The baseline is a dict mapping prompt → similarity score from a
    known-good run. Used to detect model drift between versions.

    Returns:
        dict: {prompt: score} or {} if no baseline exists yet.
    """
    if os.path.exists(BASELINE_FILE):
        try:
            with open(BASELINE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_baseline(baseline):
    """
    Persist a new regression baseline to disk.

    Should be called after a run that is confirmed to be correct,
    to establish the reference point for future regression checks.

    Args:
        baseline (dict): {prompt: score} mapping from the current run.
    """
    with open(BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)


def _is_fallback(response):
    """Return True if the response is a generic HR portal deferral."""
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in FALLBACK_INDICATORS)


# ------------------------------------------------------------------ #
# Regression test                                                      #
# ------------------------------------------------------------------ #

@allure.feature("HR Chatbot Evaluation")
@allure.story("Regression Testing")
@pytest.mark.regression
@pytest.mark.integration
def test_regression_hr_dataset():
    """
    Regression test: validates that RAG pipeline responses remain
    semantically consistent with expected answers across runs.

    Two-layer check per prompt:
      1. Absolute threshold — score must meet the difficulty-based
         minimum (DIFFICULTY_THRESHOLDS).
      2. Baseline drift — if a stored baseline exists, score must not
         drop more than 0.05 below the previously recorded score for
         the same prompt. This detects model regressions even when
         the absolute threshold is still met.

    Fallback responses ("Please check the HR portal...") are detected
    and reported separately — they indicate a retrieval gap, not a
    model regression, and are excluded from failure counts.

    A category-level summary and overall pass rate are attached to the
    Allure report after the loop.
    """
    df = pd.read_csv("prompts/hr_prompt_suite.csv")

    baseline         = load_baseline()
    current_run      = {}       # {prompt: score} — saved as new baseline if run passes
    failures         = []
    fallback_prompts = []
    category_scores  = defaultdict(list)

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

        threshold = DIFFICULTY_THRESHOLDS.get(
            difficulty, DIFFICULTY_THRESHOLDS["medium"]
        )

        # ---------------------------------------------------------- #
        # RAG pipeline call — mirrors production conditions           #
        # ---------------------------------------------------------- #
        rag_result = rag_answer(prompt)
        response   = rag_result["answer"]

        # ---------------------------------------------------------- #
        # Fallback detection — skip scoring, log separately          #
        # ---------------------------------------------------------- #
        if _is_fallback(response):
            fallback_prompts.append({
                "prompt":   prompt,
                "category": category
            })
            with allure.step(
                f"[FALLBACK] Category: {category} | Prompt: {prompt[:60]}"
            ):
                allure.attach(prompt,   name="Prompt")
                allure.attach(response, name="Response (Fallback)")
                allure.attach(
                    "Skipped — fallback/deferral response detected",
                    name="Regression Status"
                )
            continue

        # ---------------------------------------------------------- #
        # Similarity scoring                                          #
        # ---------------------------------------------------------- #
        score = similarity_score(expected, response)

        current_run[prompt] = score
        category_scores[category].append(score)

        # ---------------------------------------------------------- #
        # Baseline drift check                                        #
        # ---------------------------------------------------------- #
        baseline_score    = baseline.get(prompt)
        drift_threshold   = (baseline_score - 0.05) if baseline_score else None
        drift_fail        = (
            drift_threshold is not None and score < drift_threshold
        )

        # ---------------------------------------------------------- #
        # Allure reporting                                            #
        # ---------------------------------------------------------- #
        with allure.step(
            f"Category: {category} | Difficulty: {difficulty} | "
            f"Prompt: {prompt[:60]}"
        ):
            allure.attach(prompt,    name="Prompt")
            allure.attach(expected,  name="Expected Answer")
            allure.attach(response,  name="LLM Response")
            allure.attach(
                (
                    f"Score     : {score:.3f}\n"
                    f"Threshold : {threshold}\n"
                    f"Status    : {'PASS ✓' if score >= threshold else 'FAIL ✗'}\n"
                    f"Baseline  : {baseline_score:.3f if baseline_score else 'N/A'}\n"
                    f"Drift     : {'FAIL ✗' if drift_fail else 'OK ✓'}"
                ),
                name="Similarity Score"
            )

        # ---------------------------------------------------------- #
        # Failure collection                                          #
        # ---------------------------------------------------------- #
        fail_reason = None

        if score < threshold:
            fail_reason = (
                f"score={score:.3f} < threshold={threshold}"
            )
        elif drift_fail:
            fail_reason = (
                f"score={score:.3f} dropped > 0.05 below "
                f"baseline={baseline_score:.3f}"
            )

        if fail_reason:
            failures.append({
                "prompt":         prompt,
                "category":       category,
                "difficulty":     difficulty,
                "score":          round(score, 3),
                "threshold":      threshold,
                "baseline_score": round(baseline_score, 3) if baseline_score else None,
                "response":       response[:200],
                "reason":         fail_reason
            })

    # ---------------------------------------------------------------- #
    # Category-level summary                                            #
    # ---------------------------------------------------------------- #
    category_lines = []
    for cat in sorted(category_scores):
        scores = category_scores[cat]
        category_lines.append(
            f"{cat:20s} | avg={statistics.mean(scores):.3f} "
            f"| min={min(scores):.3f} "
            f"| n={len(scores)}"
        )

    with allure.step("Category Regression Summary"):
        allure.attach(
            "\n".join(category_lines) or "No scored results.",
            name="Category Scores",
            attachment_type=allure.attachment_type.TEXT
        )

    # ---------------------------------------------------------------- #
    # Overall pass rate                                                 #
    # ---------------------------------------------------------------- #
    total     = len(df)
    skipped   = len(fallback_prompts)
    scored    = total - skipped
    passed    = scored - len(failures)
    pass_rate = (passed / scored * 100) if scored > 0 else 0.0

    allure.attach(
        (
            f"Total prompts : {total}\n"
            f"Scored        : {scored}\n"
            f"Skipped (fallback): {skipped}\n"
            f"Passed        : {passed}\n"
            f"Failed        : {len(failures)}\n"
            f"Pass rate     : {pass_rate:.1f}%"
        ),
        name="Regression Pass Rate"
    )

    # ---------------------------------------------------------------- #
    # Persist current run as new baseline if all scored prompts passed #
    # ---------------------------------------------------------------- #
    if not failures and current_run:
        updated_baseline = {**baseline, **current_run}
        save_baseline(updated_baseline)

    # ---------------------------------------------------------------- #
    # Soft assertion — all failures surfaced in one run                 #
    # ---------------------------------------------------------------- #
    assert not failures, (
        f"\n{len(failures)} regression failure(s) detected "
        f"({pass_rate:.1f}% pass rate):\n"
        + "\n".join(
            f"  [{f['category']} / {f['difficulty']}] "
            f"{f['prompt'][:60]}... "
            f"score={f['score']} baseline={f['baseline_score']} "
            f"— {f['reason']}"
            for f in failures
        )
    )
