import json
import re
from llm.llm_client import get_llm_response
from judge.judge_prompts import JUDGE_PROMPT


def extract_json(text):
    """
    Extract JSON object from LLM response.
    """
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None


def validate_scores(result):
    """
    Ensure scores are within 1–5 range.
    """
    def clamp(x):
        try:
            return max(1, min(5, int(x)))
        except:
            return 1

    return {
        "correctness": clamp(result.get("correctness")),
        "completeness": clamp(result.get("completeness")),
        "safety": clamp(result.get("safety")),
        "reasoning": result.get("reasoning", {})
    }


def judge_answer(question, expected, answer, retries=2):
    """
    Evaluate answer using LLM judge.

    Returns structured evaluation with fallback safety.
    """

    prompt = JUDGE_PROMPT.format(
        question=question,
        expected=expected,
        answer=answer
    )

    last_response = None

    for _ in range(retries + 1):
        response = get_llm_response(prompt)
        last_response = response

        parsed = extract_json(response)

        if parsed:
            return validate_scores(parsed)

    # fallback with debug info
    return {
        "correctness": 1,
        "completeness": 1,
        "safety": 1,
        "reasoning": {
            "error": "Failed to parse LLM response",
            "raw_response": last_response
        }
    }