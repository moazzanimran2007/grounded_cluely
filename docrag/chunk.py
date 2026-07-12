"""
Step 2: split the text into overlapping chunks.

Why chunk at all: embedding models and the answer model both work better on
focused passages than on a whole document. Why overlap: if we cut on a hard
boundary, a sentence that explains an idea could be split across two chunks and
neither would hold the full thought. A small overlap means any given idea lands
whole inside at least one chunk.

We measure size in words. It is not as precise as counting tokens, but it is
simple, predictable, and good enough for retrieval.
"""

import re

from .config import CHUNK_OVERLAP_WORDS, CHUNK_WORDS


def normalize_whitespace(text):
    """Collapse runs of spaces, tabs, and newlines into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap_words=CHUNK_OVERLAP_WORDS):
    """
    Slice text into windows of chunk_words words. Each window starts
    (chunk_words - overlap_words) words after the previous one, so neighbours
    share overlap_words words. Returns a list of strings.
    """
    words = normalize_whitespace(text).split(" ")
    if words == [""]:
        return []

    # The step is how far we slide the window forward each time. It must be at
    # least 1, otherwise we would loop forever.
    step = max(1, chunk_words - overlap_words)

    chunks = []
    start = 0
    while start < len(words):
        window = words[start:start + chunk_words]
        chunks.append(" ".join(window))
        # Stop once this window reached the end of the text.
        if start + chunk_words >= len(words):
            break
        start += step
    return chunks
