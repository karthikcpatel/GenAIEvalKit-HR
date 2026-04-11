from rag.retriever import retrieve
from llm.llm_client import get_llm_response


def rag_answer(query):
    """
    Generate HR assistant response using retrieved knowledge.
    """

    retrieval = retrieve(query)
    contexts = retrieval["contexts"]

    # Handle no context case
    if not contexts:
        return {
            "answer": "I don't have enough information. Please check the HR portal.",
            "contexts": [],
            "retrieval_scores": [],
            "categories": []
        }

    context_text = "\n".join(contexts)

    prompt = f"""
You are an HR assistant.

Answer the question strictly using the provided context.
If the answer is not clearly available, say:
"Please check the HR portal for accurate information."

Context:
{context_text}

Question:
{query}
"""

    response = get_llm_response(
        prompt,
        system_prompt="You are a helpful and safe HR assistant.",
        temperature=0
    )

    return {
        "answer": response,
        "contexts": contexts,
        "retrieval_scores": retrieval["scores"],
        "categories": retrieval["categories"]
    }