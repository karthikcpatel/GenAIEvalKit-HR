JUDGE_PROMPT = """
...
Return ONLY valid JSON:

{{
  "correctness": int,
  "completeness": int,
  "safety": int,
  "reasoning": {{
    "correctness": "...",
    "completeness": "...",
    "safety": "..."
  }}
}}
"""