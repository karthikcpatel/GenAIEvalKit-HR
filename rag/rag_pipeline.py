from rag.retriever import retrieve
from llm.llm_client import get_llm_response

# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

# Minimum retrieval similarity score for a context chunk to be
# included in the prompt. Chunks below this threshold are likely
# irrelevant and increase hallucination risk.
MIN_CONTEXT_SCORE = 0.35

# Maximum length (characters) of the raw query passed into the prompt.
# Prevents excessively long or adversarial inputs from distorting
# prompt structure.
MAX_QUERY_LENGTH = 500

# Canonical fallback message used in both the early-return path and
# the in-prompt instruction, ensuring downstream fallback detection
# can rely on a single consistent string.
FALLBACK_MESSAGE = (
    "I don't have enough information. "
    "Please check the HR portal for accurate information."
)


# ------------------------------------------------------------------ #
# RAG pipeline                                                         #
# ------------------------------------------------------------------ #

def rag_answer(query):
    """
    Generate an HR assistant response using retrieved knowledge.

    Pipeline:
      1. Retrieve top-k context chunks from the HR knowledge base
      2. Filter out chunks below MIN_CONTEXT_SCORE
      3. Build a numbered, bounded prompt from the filtered chunks
      4. Call the LLM with temperature=0 for deterministic output
      5. Return the answer alongside retrieval metadata

    Args:
        query (str): Employee question.

    Returns:
        dict: {
            "question"        : str,
            "answer"          : str,
            "contexts"        : list[str],
            "retrieval_scores": list[float],
            "categories"      : list[str]
        }
    """
    # ---- Input sanitisation -------------------------------------- #
    query_safe = (query or "").strip()[:MAX_QUERY_LENGTH]

    if not query_safe:
        return {
            "question":         query,
            "answer":           FALLBACK_MESSAGE,
            "contexts":         [],
            "retrieval_scores": [],
            "categories":       []
        }

    # ---- Retrieval ----------------------------------------------- #
    retrieval = retrieve(query_safe)

    raw_contexts = retrieval["contexts"]
    raw_scores   = retrieval["scores"]
    raw_cats     = retrieval["categories"]
    raw_kp       = retrieval.get("key_points", [])

    # ---- Score-based filtering ----------------------------------- #
    # Zip contexts with their scores and keep only high-quality chunks.
    # This prevents low-relevance context from misleading the LLM.
    filtered = [
        (ctx, score, cat, kp)
        for ctx, score, cat, kp in zip(raw_contexts, raw_scores, raw_cats, raw_kp or [""] * len(raw_contexts))
        if score >= MIN_CONTEXT_SCORE
    ]

    if not filtered:
        return {
            "question":         query,
            "answer":           FALLBACK_MESSAGE,
            "contexts":         [],
            "retrieval_scores": [],
            "categories":       []
        }

    contexts   = [f[0] for f in filtered]
    scores     = [f[1] for f in filtered]
    categories = [f[2] for f in filtered]
    key_points = [f[3] for f in filtered]

    # ---- Prompt construction ------------------------------------- #
    # Numbered context blocks make chunk boundaries explicit so the
    # LLM can distinguish between potentially conflicting sources.
    context_blocks = "\n\n".join(
        f"[Context {i + 1}]\n{ctx}"
        for i, ctx in enumerate(contexts)
    )

    # Append key points as supplementary structured hints if available
    key_points_section = ""
    non_empty_kp = [kp for kp in key_points if kp]
    if non_empty_kp:
        key_points_section = (
            "\n\nKey policy points for reference:\n"
            + "\n".join(f"- {kp}" for kp in non_empty_kp)
        )

    prompt = f"""
You are an HR assistant. You have been provided with {len(contexts)}
context passage(s) retrieved in order of relevance.

Instructions:
- Answer the question strictly using the provided context.
- Prioritise information from earlier context passages (higher relevance).
- Do not combine conflicting information from different passages —
  use the most relevant one.
- If the answer is not clearly available in the context, respond with:
  "{FALLBACK_MESSAGE}"
- Do not invent facts, policies, or figures not present in the context.

Context:
{context_blocks}{key_points_section}

Question:
{query_safe}
"""

    # ---- LLM call ------------------------------------------------ #
    response = get_llm_response(
        prompt,
        system_prompt="You are a helpful and safe HR assistant.",
        temperature=0
    )

    return {
        "question":         query,
        "answer":           response,
        "contexts":         contexts,
        "retrieval_scores": scores,
        "categories":       categories
    }
