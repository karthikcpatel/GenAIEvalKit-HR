import csv
import os
import threading

import numpy as np

from evaluator.similarity import encode_text, similarity_score

# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

# Path to the HR knowledge base CSV.
# Override via environment variable for CI/CD or alternative fixtures.
DEFAULT_KB_PATH = os.environ.get(
    "HR_KB_PATH", "prompts/hr_prompt_suite.csv"
)

# Required columns in the KB CSV — validated at load time.
REQUIRED_COLUMNS = {"prompt", "expected_answer", "category"}

# Minimum similarity score for a retrieved chunk to be included in
# results. Entries below this threshold are discarded so the RAG
# pipeline never receives irrelevant context.
MIN_SCORE = 0.35

# ------------------------------------------------------------------ #
# Global KB state                                                      #
# ------------------------------------------------------------------ #

KB             = []          # list of dicts, one per KB entry
_kb_embeddings = None        # pre-computed question embeddings (np.ndarray)
_kb_lock       = threading.Lock()


# ------------------------------------------------------------------ #
# KB loading                                                           #
# ------------------------------------------------------------------ #

def load_kb(csv_path=DEFAULT_KB_PATH):
    """
    Load the HR knowledge base from a CSV file into the global KB list
    and pre-compute question embeddings for fast retrieval.

    Thread-safe: uses a lock to prevent concurrent initialisation in
    multi-threaded test runners (e.g. pytest-xdist).

    The KB is loaded only once per process. Call reload_kb() to force
    a fresh load (e.g. to swap in a test fixture CSV).

    Args:
        csv_path (str): Path to the HR prompt suite CSV.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError:        If required columns are missing.
        RuntimeError:      For any other load failure.
    """
    global KB, _kb_embeddings

    with _kb_lock:
        if KB:
            return  # already loaded — skip

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"[Retriever] KB file not found at '{csv_path}'. "
                f"Check the path or set the HR_KB_PATH environment variable."
            )

        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                # ---- Schema validation --------------------------- #
                missing_cols = REQUIRED_COLUMNS - set(reader.fieldnames or [])
                if missing_cols:
                    raise ValueError(
                        f"[Retriever] KB CSV is missing required columns: "
                        f"{missing_cols}. Found: {reader.fieldnames}"
                    )

                for row in reader:
                    KB.append({
                        "question":   row["prompt"],
                        # "answer" values serve as LLM context passages —
                        # named "answer" internally but exposed as "contexts"
                        # in the retrieve() return dict for clarity downstream.
                        "answer":     row["expected_answer"],
                        "category":   row["category"],
                        "key_points": row.get("key_points", "")
                    })

        except (FileNotFoundError, ValueError):
            raise   # re-raise typed errors as-is

        except Exception as e:
            raise RuntimeError(
                f"[Retriever] Failed to load KB from '{csv_path}': {e}"
            ) from e

        # ---- Pre-compute question embeddings --------------------- #
        # Embeddings are computed once at load time so retrieve() can
        # use a batched dot-product instead of O(n) sequential calls.
        _kb_embeddings = np.array([
            encode_text(item["question"]) for item in KB
        ])


def reload_kb(csv_path=DEFAULT_KB_PATH):
    """
    Force a fresh KB load, discarding any cached entries and embeddings.

    Useful in tests that need to swap in a smaller fixture CSV without
    restarting the process.

    Args:
        csv_path (str): Path to the replacement KB CSV.
    """
    global KB, _kb_embeddings
    with _kb_lock:
        KB.clear()
        _kb_embeddings = None
    load_kb(csv_path)


# ------------------------------------------------------------------ #
# Retrieval                                                            #
# ------------------------------------------------------------------ #

def retrieve(query, top_k=3):
    """
    Retrieve the top-k most relevant KB entries for a given query.

    Scoring strategy:
      - Computes similarity between the query and each KB entry's
        question AND answer, taking the maximum of the two scores.
        This improves recall when the user's phrasing matches the
        answer text more closely than the stored question.
      - Uses pre-computed question embeddings and batched dot-product
        for efficiency (O(1) matrix multiply vs O(n) sequential calls).
      - Filters out entries below MIN_SCORE to prevent low-quality
        context from reaching the LLM.

    Args:
        query  (str): Employee question.
        top_k  (int): Maximum number of results to return.

    Returns:
        dict: {
            "contexts"  : list[str],    # answer texts (LLM context passages)
            "questions" : list[str],    # original KB questions
            "categories": list[str],
            "scores"    : list[float],
            "key_points": list[str]
        }
        All lists are empty if no entries meet the MIN_SCORE threshold.
    """
    load_kb()

    if not KB or _kb_embeddings is None:
        return {
            "contexts":   [],
            "questions":  [],
            "categories": [],
            "scores":     [],
            "key_points": []
        }

    # ---- Query embedding ----------------------------------------- #
    query_vec = np.array(encode_text(query))   # shape: (embedding_dim,)

    # ---- Batched question similarity ----------------------------- #
    # _kb_embeddings shape: (n_entries, embedding_dim)
    # Embeddings are L2-normalised → dot product = cosine similarity
    question_scores = (_kb_embeddings @ query_vec).tolist()

    # ---- Answer similarity (sequential — not pre-indexed) -------- #
    # Answer texts are longer and more varied than questions, so
    # pre-indexing them has diminishing returns for typical KB sizes.
    answer_scores = [
        similarity_score(query, item["answer"]) for item in KB
    ]

    # ---- Combined score: max of question and answer similarity --- #
    combined_scores = [
        max(q, a) for q, a in zip(question_scores, answer_scores)
    ]

    # ---- Sort and filter ----------------------------------------- #
    scored = sorted(
        zip(KB, combined_scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_results = [
        (item, score) for item, score in scored[:top_k]
        if score >= MIN_SCORE
    ]

    if not top_results:
        return {
            "contexts":   [],
            "questions":  [],
            "categories": [],
            "scores":     [],
            "key_points": []
        }

    return {
        # "contexts" contains the expected answers from the KB —
        # these are the reference texts passed to the LLM as context,
        # not the original questions.
        "contexts":   [r[0]["answer"]     for r in top_results],
        "questions":  [r[0]["question"]   for r in top_results],
        "categories": [r[0]["category"]   for r in top_results],
        "scores":     [r[1]               for r in top_results],
        "key_points": [r[0]["key_points"] for r in top_results]
    }
