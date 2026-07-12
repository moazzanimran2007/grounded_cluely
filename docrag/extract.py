"""
Step 1: ingest one file and pull out its raw text.

Text and markdown files are read straight off disk. PDFs go through pypdf, a
pure Python reader with no system dependencies, which keeps setup simple and
free. We do not try to be clever about layout here. We just want the words.
"""

import os


def extract_text(path):
    """
    Return the full text of a .txt, .md, or .pdf file as one string.
    Unsupported types stop with a clear message rather than a confusing error.
    """
    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}")

    lower = path.lower()
    if lower.endswith(".txt") or lower.endswith(".md"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    if lower.endswith(".pdf"):
        return _extract_pdf(path)

    raise SystemExit(
        f"Unsupported file type: {path}. Use a .txt, .md, or .pdf file."
    )


def _extract_pdf(path):
    """Read every page of a PDF and join the page texts with newlines."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        # extract_text can return None on an image only page, so guard it.
        pages.append(page.extract_text() or "")
    return "\n".join(pages)
