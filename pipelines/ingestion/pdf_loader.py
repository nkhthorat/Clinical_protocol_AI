# pipelines/ingestion/pdf_loader.py

from pipelines.utils import check_pymupdf

check_pymupdf()

import fitz


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)

    text = []

    for page in doc:
        text.append(page.get_text())

    return "\n".join(text)