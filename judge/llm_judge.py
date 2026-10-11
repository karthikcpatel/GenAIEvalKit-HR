import json
import re
import time

from llm.llm_client import get_llm_response
from judge.judge_prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT

# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

SCORE_FIELDS = ("correctness", "completeness", "safety")
SCORE_MIN    = 1
SCORE_MAX    = 5


# ------------------------------------------------------------------ #
# JSON extraction                                                      #
# ------------------------------------------------------------------ #

def extract_json(text):
    """
    Extract a JSON object from an LLM response string.

    Attempts strict parsing first, then falls back to extracting
    the first {...} block from the response.

    Args:
        text (str): Raw LLM response.

    Returns:
        dict | None: Parsed dict if successful, None otherwise.
    """
    if not text:
        return None

    # Attempt 1 — strict parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 2 — extract first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ------------------------------------------------------------------ #
# Score validation                                                     #
# ------------------------------------------------------------------ #

def _clamp(value):
    """
    Clamp a value to the valid score range [SCORE_MIN, SCORE_MAX].

    Returns SCORE_MIN if value cannot be converted to int.

    Args:
        value: Raw score value from parsed JSON.

    Returns:
        int: Clamped score.
    """
    try:
        return max(SCORE_MIN, min(SCORE_MAX, int(value)))
    except (TypeError, ValueError):
        return SCORE_MIN


def validate_scores(result):
    """
    Validate and normalise a parsed judge result dict.

    - Clamps each score field to [SCORE_MIN, SCORE_MAX]
    - Records any missing score fields explicitly rather than
      silently defaulting, so callers can distinguish a genuine
      low score from a missing field
    - Normalises the reasoning field to a dict regardless of what
      the LLM returned

    Args:
        result (dict): Parsed JSON from the LLM judge.

    Returns:
        dict: {
            "correctness"  : int,
            "completeness" : int,
            "safety"       : int,
            "reasoning"    : dict,
            "missing_fields": list   # only present if fields were absent
        }
    """
    missing = [f for f in SCORE_FIELDS if f not in result]

    # Normalise reasoning to dict
    reasoning = result.get("reasoning", {})
    if not isinstance(reasoning, dict):
        reasoning = {"raw": str(reasoning)}

    validated = {
        "correctness":  _clamp(result.get("correctness",  SCORE_MIN)),
        "completeness": _clamp(result.get("completeness", SCORE_MIN)),
        "safety":       _clamp(result.get("safety",       SCORE_MIN)),
        "reasoning":    reasoning
    }

    if missing:
        validated["missing_fields"] = missing

    return validated


# ------------------------------------------------------------------ #
# Judge entry point                                                    #
# ------------------------------------------------------------------ #

def judge_answer(question, expected, answer, retries=2):
    """
    Evaluate an LLM answer using an LLM-as-judge approach.

    Sends a structured prompt to the judge LLM and parses the
    returned JSON scores. Retries on parse failure with exponential
    backoff. Returns a safe fallback result if all attempts fail.

    Args:
        question (str): The original user question.
        expected (str): Ground truth / reference answer.
        answer   (str): The LLM-generated answer to evaluate.
        retries  (int): Number of additional attempts after the first
                        (default: 2, so up to 3 total attempts).

    Returns:
        dict: {
            "correctness"  : int,   # 1–5
            "completeness" : int,   # 1–5
            "safety"       : int,   # 1–5
            "reasoning"    : dict,
            "attempts"     : int,   # how many tries were needed
            "missing_fields": list  # only if fields were absent in response
        }
    """
    # ---- Input validation ---------------------------------------- #
    if not question or not expected or not answer:
        return {
            "correctness":  SCORE_MIN,
            "completeness": SCORE_MIN,
            "safety":       SCORE_MIN,
            "reasoning":    {"error": "One or more inputs are empty or None"},
            "attempts":     0
        }

    prompt        = JUDGE_USER_PROMPT.format(
        question=question,
        expected=expected,
        answer=answer
    )
    last_response = None

    for attempt in range(retries + 1):
        # The rubric and output contract live in the system prompt —
        # without it the judge scores with no criteria.
        response      = get_llm_response(
            prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=0
        )
        last_response = response
        parsed        = extract_json(response)

        if parsed:
            result           = validate_scores(parsed)
            result["attempts"] = attempt + 1
            return result

        # Exponential backoff before retrying (skip after last attempt)
        if attempt < retries:
            time.sleep(2 ** attempt)   # 1s → 2s

    # ---- All attempts exhausted ---------------------------------- #
    return {
        "correctness":  SCORE_MIN,
        "completeness": SCORE_MIN,
        "safety":       SCORE_MIN,
        "reasoning": {
            "error":        "Failed to parse LLM judge response after all retries",
            "raw_response": last_response
        },
        "attempts": retries + 1
    }
