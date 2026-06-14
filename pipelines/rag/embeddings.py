import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict


# -----------------------------
# Load embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Create embeddings
# -----------------------------
def embed_chunks(chunks: List[Dict]) -> np.ndarray:
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return np.array(embeddings).astype("float32")


# -----------------------------
# Build FAISS index
# -----------------------------
def build_faiss_index(embeddings: np.ndarray):

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index


# -----------------------------
# Save RAG system (index + chunks)
# -----------------------------
def save_rag(index, chunks, folder="rag_store"):

    os.makedirs(folder, exist_ok=True)

    # save FAISS index
    faiss.write_index(index, f"{folder}/index.faiss")

    # save metadata (chunks)
    with open(f"{folder}/chunks.json", "w") as f:
        json.dump(chunks, f)


# -----------------------------
# Load RAG system
# -----------------------------
def load_rag(folder="rag_store"):

    index = faiss.read_index(f"{folder}/index.faiss")

    with open(f"{folder}/chunks.json", "r") as f:
        chunks = json.load(f)

    return index, chunks


# -----------------------------
# Search function
# -----------------------------
def search(query: str, index, chunks, top_k=5):

    query_vec = model.encode([query]).astype("float32")

    distances, indices = index.search(query_vec, top_k)

    results = []

    for i in indices[0]:
        results.append(chunks[i])

    return results