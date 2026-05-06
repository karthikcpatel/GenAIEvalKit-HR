import time
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
import allure

from rag.rag_pipeline import rag_answer

# ------------------------------------------------------------------ #
# SLA Constants                                                        #
# Defined at module level for visibility and easy configuration.      #
# ------------------------------------------------------------------ #

SLA_PER_REQUEST_S = 10.0   # max acceptable latency per individual request
SLA_AVERAGE_S     = 6.0    # max acceptable mean latency across the full run
SLA_P95_S         = 8.0    # max acceptable 95th percentile latency


@allure.feature("HR Chatbot Performance")
@allure.story("Latency SLA Validation")
@pytest.mark.performance
def test_latency():
    """
    Performance test: measures end-to-end RAG pipeline latency across
    the full prompt suite and validates SLA thresholds.

    Metrics validated:
      - Per-request latency  < SLA_PER_REQUEST_S
      - Mean latency         < SLA_AVERAGE_S
      - P95 latency          < SLA_P95_S

    Additional outputs attached to Allure:
      - Per-row latency, difficulty, response length
      - Per-category latency breakdown (avg, max, count)
      - Full aggregate summary (mean, max, p95)
    """
    df = pd.read_csv("prompts/hr_prompt_suite.csv")

    if df.empty:
        pytest.skip("Prompt suite is empty — no latency data to collect")

    # ---------------------------------------------------------------- #
    # Warmup — excluded from measurement                                #
    # The first call in a cold process is slower due to connection      #
    # setup and SDK initialisation. One warmup request stabilises       #
    # subsequent timings.                                               #
    # ---------------------------------------------------------------- #
    try:
        rag_answer(df.iloc[0]["prompt"])
    except Exception:
        pass   # warmup failure should not block the test

    # ---------------------------------------------------------------- #
    # Measurement loop                                                  #
    # ---------------------------------------------------------------- #
    latencies          = []
    slow_requests      = []
    category_latencies = defaultdict(list)

    for row in df.itertuples():

        prompt   = row.prompt
        category = getattr(row, "category", "general")

        # Normalise difficulty — guard against NaN / blank cells
        difficulty = str(
            getattr(row, "difficulty", "medium") or "medium"
        ).strip().lower()

        # ---- Timed RAG pipeline call ----------------------------- #
        start    = time.perf_counter()           # monotonic high-res timer
        result   = rag_answer(prompt)
        latency  = time.perf_counter() - start
        response = result["answer"]

        latencies.append(latency)
        category_latencies[category].append(latency)

        # ---- Allure step ----------------------------------------- #
        with allure.step(
            f"Category: {category} | Difficulty: {difficulty} | "
            f"Prompt: {prompt[:60]}"
        ):
            allure.attach(prompt,                         name="Prompt")
            allure.attach(response,                       name="LLM Response")
            allure.attach(f"{latency:.3f}s",              name="Latency")
            allure.attach(difficulty,                     name="Difficulty")
            allure.attach(str(len(response)),             name="Response Length (chars)")
            allure.attach(
                "SLOW ⚠️" if latency > SLA_PER_REQUEST_S else "OK ✓",
                name="SLA Status"
            )

        # ---- Track slow requests --------------------------------- #
        if latency > SLA_PER_REQUEST_S:
            slow_requests.append({
                "prompt":          prompt,
                "category":        category,
                "difficulty":      difficulty,
                "latency":         round(latency, 3),
                "response_length": len(response)
            })

    # ---------------------------------------------------------------- #
    # Guard against empty latency list                                  #
    # ---------------------------------------------------------------- #
    if not latencies:
        pytest.skip("No latency data collected — all rows may have been skipped")

    # ---------------------------------------------------------------- #
    # Aggregate metrics                                                 #
    # ---------------------------------------------------------------- #
    avg_latency = statistics.mean(latencies)
    max_latency = max(latencies)
    p95_latency = float(np.percentile(latencies, 95))

    # ---- Aggregate summary attachment ---------------------------- #
    allure.attach(
        (
            f"Mean : {avg_latency:.3f}s  (SLA: < {SLA_AVERAGE_S}s)\n"
            f"P95  : {p95_latency:.3f}s  (SLA: < {SLA_P95_S}s)\n"
            f"Max  : {max_latency:.3f}s\n"
            f"Count: {len(latencies)}\n"
            f"Slow : {len(slow_requests)} request(s) > {SLA_PER_REQUEST_S}s"
        ),
        name="Latency Summary"
    )

    # ---- Per-category breakdown ---------------------------------- #
    category_lines = []
    for cat in sorted(category_latencies):
        lats = category_latencies[cat]
        category_lines.append(
            f"{cat:20s} | avg={statistics.mean(lats):.3f}s "
            f"| max={max(lats):.3f}s "
            f"| p95={float(np.percentile(lats, 95)):.3f}s "
            f"| n={len(lats)}"
        )

    with allure.step("Per-Category Latency Breakdown"):
        allure.attach(
            "\n".join(category_lines),
            name="Category Latency Breakdown",
            attachment_type=allure.attachment_type.TEXT
        )

    # ---------------------------------------------------------------- #
    # SLA assertions — all failures collected before asserting          #
    # so a single run surfaces all violations at once                  #
    # ---------------------------------------------------------------- #
    sla_failures = []

    if avg_latency >= SLA_AVERAGE_S:
        sla_failures.append(
            f"Mean latency {avg_latency:.3f}s exceeds SLA of {SLA_AVERAGE_S}s"
        )

    if p95_latency >= SLA_P95_S:
        sla_failures.append(
            f"P95 latency {p95_latency:.3f}s exceeds SLA of {SLA_P95_S}s"
        )

    if slow_requests:
        slow_lines = "\n".join(
            f"  [{r['category']} / {r['difficulty']}] "
            f"{r['prompt'][:60]}... "
            f"latency={r['latency']}s "
            f"response_len={r['response_length']}"
            for r in slow_requests
        )
        sla_failures.append(
            f"{len(slow_requests)} request(s) exceeded "
            f"per-request SLA of {SLA_PER_REQUEST_S}s:\n{slow_lines}"
        )

    assert not sla_failures, (
        f"\n{len(sla_failures)} SLA violation(s) detected:\n"
        + "\n\n".join(sla_failures)
    )
