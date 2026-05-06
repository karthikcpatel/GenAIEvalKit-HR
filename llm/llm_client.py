import logging
import os
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------ #
# Logging                                                              #
# ------------------------------------------------------------------ #

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

# Supported Claude models — validated before each call to prevent
# silent 400 errors from misspelled or outdated model names.
SUPPORTED_MODELS = {
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-haiku-4-5-20251001",
}

# HTTP status codes that are transient and worth retrying.
# 400 / 401 / 422 are client errors that will never succeed on retry.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Base URL override — useful for routing through a proxy or
# an internal gateway without modifying source code.
BASE_URL = os.environ.get("LLM_BASE_URL", None)   # None = Anthropic default


# ------------------------------------------------------------------ #
# Client factory                                                       #
# ------------------------------------------------------------------ #

def _make_client():
    """
    Instantiate an Anthropic client using the API key from the
    environment. Key is read at call time (not import time) so that
    test fixtures which set env vars dynamically are respected.

    Returns:
        anthropic.Anthropic: Configured client instance.

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file or set it as an environment variable."
        )

    kwargs = {"api_key": api_key}
    if BASE_URL:
        kwargs["base_url"] = BASE_URL

    return anthropic.Anthropic(**kwargs)


# ------------------------------------------------------------------ #
# LLM call                                                             #
# ------------------------------------------------------------------ #

def get_llm_response(
    prompt,
    system_prompt=None,
    model="claude-sonnet-4-6",
    temperature=0,
    max_tokens=1024,
    retries=2,
):
    """
    Send a prompt to Claude and return the text response.

    Uses the official Anthropic Python SDK (anthropic package).
    Retries only on transient errors (429, 5xx) with exponential
    backoff. Raises on non-retryable errors so callers receive a
    clear exception rather than silently evaluating an error string.

    Args:
        prompt        (str):   User message.
        system_prompt (str):   Optional system instruction.
        model         (str):   Claude model name (default: claude-sonnet-4-6).
        temperature   (float): Sampling temperature (0 = deterministic).
        max_tokens    (int):   Maximum tokens in the response (default: 1024).
        retries       (int):   Additional attempts after the first
                               (default: 2, so up to 3 total attempts).

    Returns:
        str: The model's text response, stripped of leading/trailing whitespace.

    Raises:
        ValueError:          If model name is unsupported or API key is missing.
        anthropic.APIError:  For non-retryable API errors (4xx except 429).
        RuntimeError:        If all retry attempts are exhausted.
    """
    # ---- Model validation ---------------------------------------- #
    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model}'. "
            f"Choose from: {sorted(SUPPORTED_MODELS)}"
        )

    # ---- Input validation ---------------------------------------- #
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")

    # ---- Build message list -------------------------------------- #
    messages = [{"role": "user", "content": prompt}]

    # ---- Retry loop ---------------------------------------------- #
    client     = _make_client()
    last_error = None

    for attempt in range(retries + 1):
        try:
            kwargs = {
                "model":       model,
                "max_tokens":  max_tokens,
                "messages":    messages,
                "temperature": temperature,
            }

            # system is a top-level param in the Anthropic SDK,
            # not a message role
            if system_prompt:
                kwargs["system"] = system_prompt

            response = client.messages.create(**kwargs)

            # ---- Response validation ----------------------------- #
            if not response.content:
                last_error = "API returned empty content list"
                logger.warning(
                    "[LLM Client] Attempt %d/%d — empty content: %s",
                    attempt + 1, retries + 1, last_error
                )
                if attempt < retries:
                    time.sleep(2 ** attempt)
                continue

            text = response.content[0].text
            if not text or not text.strip():
                last_error = "API returned blank text in content[0]"
                logger.warning(
                    "[LLM Client] Attempt %d/%d — blank text: %s",
                    attempt + 1, retries + 1, last_error
                )
                if attempt < retries:
                    time.sleep(2 ** attempt)
                continue

            return text.strip()

        except anthropic.RateLimitError as e:
            # 429 — transient, always retry with backoff
            last_error = f"Rate limit exceeded: {e}"
            logger.warning(
                "[LLM Client] Attempt %d/%d — rate limited: %s",
                attempt + 1, retries + 1, last_error
            )

        except anthropic.InternalServerError as e:
            # 5xx — transient server error, retry
            last_error = f"Server error: {e}"
            logger.warning(
                "[LLM Client] Attempt %d/%d — server error: %s",
                attempt + 1, retries + 1, last_error
            )

        except anthropic.AuthenticationError as e:
            # 401 — invalid API key, no point retrying
            raise anthropic.AuthenticationError(
                f"[LLM Client] Authentication failed — check ANTHROPIC_API_KEY: {e}",
                response=e.response,
                body=e.body
            ) from e

        except anthropic.BadRequestError as e:
            # 400 — malformed request, no point retrying
            raise anthropic.BadRequestError(
                f"[LLM Client] Bad request — check prompt and parameters: {e}",
                response=e.response,
                body=e.body
            ) from e

        except anthropic.APIConnectionError as e:
            # Network-level failure — retry
            last_error = f"Connection error: {e}"
            logger.warning(
                "[LLM Client] Attempt %d/%d — connection error: %s",
                attempt + 1, retries + 1, last_error
            )

        except anthropic.APIError as e:
            # Catch-all for any other Anthropic API error
            last_error = f"API error: {e}"
            logger.warning(
                "[LLM Client] Attempt %d/%d — API error: %s",
                attempt + 1, retries + 1, last_error
            )

        # ---- Exponential backoff before next attempt ------------- #
        if attempt < retries:
            sleep_time = 2 ** attempt   # 1s → 2s
            logger.info(
                "[LLM Client] Retrying in %ds (attempt %d/%d)...",
                sleep_time, attempt + 2, retries + 1
            )
            time.sleep(sleep_time)

    # ---- All attempts exhausted ---------------------------------- #
    raise RuntimeError(
        f"[LLM Client] All {retries + 1} attempt(s) failed. "
        f"Last error: {last_error}"
    )
