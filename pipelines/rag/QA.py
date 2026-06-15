from typing import List, Dict


# -----------------------------
# Build context from chunks
# -----------------------------
def build_context(chunks: List[Dict]) -> str:
    return "\n\n---\n\n".join([c["text"] for c in chunks])


# -----------------------------
# Core QA function (RAG)
# -----------------------------
def answer_question(query: str, index, chunks, top_k=3):

    # import here to avoid circular imports
    from pipelines.rag.embeddings import search

    # 1. retrieve relevant chunks
    retrieved = search(query, index, chunks, top_k=top_k)

    context = build_context(retrieved)

    # 2. structured clinical prompt
    prompt = f"""
You are a clinical research assistant.

Use ONLY the provided context to answer the question.
Do NOT use external knowledge.

If the answer is a list, return bullet points.
If the information is not present, say: "Not found in protocol".

Be precise and clinical.

---------------- CONTEXT ----------------
{context}
----------------------------------------

QUESTION:
{query}

ANSWER:
"""

    return prompt