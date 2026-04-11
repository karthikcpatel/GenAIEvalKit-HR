import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = "https://api.openai.com/v1/chat/completions"


def get_llm_response(
    prompt,
    system_prompt=None,
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=500,
    retries=2,
    timeout=10,
):
    """
    Generic LLM client for chat completion.

    Args:
        prompt (str): User prompt
        system_prompt (str): Optional system instruction
        model (str): Model name
        temperature (float)
        max_tokens (int)
        retries (int): Retry attempts
        timeout (int): Request timeout (seconds)

    Returns:
        str: Model response text
    """

    if not API_KEY:
        raise ValueError("LLM_API_KEY not set in environment")

    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    last_error = None

    for attempt in range(retries + 1):
        try:
            response = requests.post(
                BASE_URL,
                json=payload,
                headers=headers,
                timeout=timeout,
            )

            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}: {response.text}"
                continue

            data = response.json()

            return data["choices"][0]["message"]["content"]

        except Exception as e:
            last_error = str(e)

    # Final failure
    return f"[LLM ERROR] {last_error}"