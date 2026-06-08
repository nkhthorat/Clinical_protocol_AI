def check_pymupdf():
    try:
        import fitz
        return True
    except ImportError:
        raise ImportError(
            "PyMuPDF is not installed. Run: pip install pymupdf"
        )