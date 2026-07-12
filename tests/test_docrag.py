"""
No-key tests for the docrag pieces that are pure logic: chunking and the cosine
ranking. These need no API and no network, so they run in the same pytest pass
as the Task 1 tests. The embed and answer steps are covered by the live demo,
since those genuinely need the model.
"""

import numpy as np

from docrag.chunk import chunk_text
from docrag.retrieve import cosine_top_k


def test_chunking_count_and_overlap():
    # 20 words, windows of 10, overlap of 3, so the window slides 7 each time:
    # 0..10, 7..17, 14..20. That is three chunks.
    words = [f"w{i}" for i in range(20)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_words=10, overlap_words=3)
    assert len(chunks) == 3

    # The overlap is real: the last 3 words of one chunk are the first 3 words
    # of the next.
    first = chunks[0].split()
    second = chunks[1].split()
    assert first[-3:] == second[:3]


def test_chunking_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("    \n  ") == []


def test_cosine_top_k_ranks_by_similarity():
    # Row 0 points the same way as the query, so it must rank first. Row 2 is a
    # 45 degree match, row 1 is orthogonal and least similar.
    matrix = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    query = [1.0, 0.05]

    ranked = cosine_top_k(query, matrix, k=3)
    order = [row for row, _ in ranked]
    assert order[0] == 0
    assert order[-1] == 1

    # Scores come back sorted high to low.
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)
