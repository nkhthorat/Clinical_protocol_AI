from pipelines.rag.embeddings import search


class ClinicalProtocolQA:
    def __init__(self, index, chunks, llm=None):
        self.index = index
        self.chunks = chunks
        self.llm = llm  # your Qwen wrapper

    # -----------------------------
    # STEP 1: Retrieve context
    # -----------------------------
    def retrieve(self, question, top_k=5):
        results = search(
            question,
            self.index,
            self.chunks,
            top_k=top_k
        )
        return results

    # -----------------------------
    # STEP 2: Build prompt safely
    # -----------------------------
    def build_prompt(self, question, retrieved_chunks):

        # SAFE: dict → text extraction
        context = "\n\n---\n\n".join(
            chunk["text"] if isinstance(chunk, dict) else str(chunk)
            for chunk in retrieved_chunks
        )

        prompt = f"""
You are a clinical research assistant.

Use ONLY the provided protocol context to answer.

RULES:
- Do NOT use external knowledge
- If information is missing, say "Not found in protocol"
- Be precise and clinical
- Prefer bullet points for lists
- Do not hallucinate

---------------- CONTEXT ----------------

{context}

----------------------------------------

QUESTION:
{question}

ANSWER:
""".strip()

        return prompt

    # -----------------------------
    # STEP 3: Call LLM
    # -----------------------------
    def generate(self, prompt):
        if self.llm is None:
            raise ValueError("LLM not initialized in ClinicalProtocolQA")

        return self.llm.generate(prompt)

    # -----------------------------
    # MAIN API
    # -----------------------------
    def ask(self, question, top_k=5):

        # 1. Retrieve
        retrieved_chunks = self.retrieve(question, top_k=top_k)

        # 2. Build prompt
        prompt = self.build_prompt(question, retrieved_chunks)

        # 3. Generate answer
        return self.generate(prompt)