import re
from typing import List, Dict


def clean_text(text: str) -> str:
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# -----------------------------
# Simple semantic chunking
# -----------------------------
def chunk_text(text: str, max_words: int = 250) -> List[Dict]:

    text = clean_text(text)

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        words = sentence.split()
        length = len(words)

        if current_length + length > max_words:
            chunks.append({
                "text": " ".join(current_chunk)
            })
            current_chunk = [sentence]
            current_length = length
        else:
            current_chunk.append(sentence)
            current_length += length

    if current_chunk:
        chunks.append({
            "text": " ".join(current_chunk)
        })

    return chunks