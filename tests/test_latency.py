import time
import pandas as pd
import pytest
import statistics
import allure

from llm.llm_client import get_llm_response


@allure.feature("HR Chatbot Performance")
@allure.story("Latency SLA Validation")
@pytest.mark.performance
def test_latency():
    """
    Performance test: measures latency across dataset
    and validates SLA using aggregated metrics.
    """

    df = pd.read_csv("prompts/hr_prompt_suite.csv")

    SLA_THRESHOLD = 10  # seconds (per request upper bound)
    AVG_THRESHOLD = 6   # average latency SLA

    latencies = []
    slow_requests = []

    for row in df.itertuples():

        prompt = row.prompt
        category = row.category

        with allure.step(f"Category: {category}"):

            start = time.time()
            response = get_llm_response(prompt)
            latency = time.time() - start

            latencies.append(latency)

            allure.attach(prompt, name="Prompt")
            allure.attach(response, name="LLM Response")
            allure.attach(f"{latency:.2f}", name="Latency Seconds")

            # Track slow requests (don't fail immediately)
            if latency > SLA_THRESHOLD:
                slow_requests.append({
                    "prompt": prompt,
                    "latency": round(latency, 2)
                })

    # Aggregate metrics
    avg_latency = statistics.mean(latencies)
    max_latency = max(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies)) - 1]

    # Attach summary
    allure.attach(
        f"Avg: {avg_latency:.2f}, Max: {max_latency:.2f}, P95: {p95_latency:.2f}",
        name="Latency Summary"
    )

    # Assertions
    assert avg_latency < AVG_THRESHOLD, f"Average latency too high: {avg_latency:.2f}s"
    assert not slow_requests, f"Slow requests detected:\n{slow_requests}"