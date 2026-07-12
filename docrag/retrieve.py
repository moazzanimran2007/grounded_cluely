"""
Step 4: given a question, find the chunks most like it.

We embed the question, then compare its vector against every stored chunk vector
using cosine similarity. Cosine similarity measures the angle between two
vectors, so it scores by direction (meaning) and ignores length. A score near
1.0 means very similar, near 0.0 means unrelated.

For a few hundred chunks this brute force comparison is instant, so there is no
need for an index structure yet.
"""

import numpy as np

from .config import TOP_K
from .store import embed_texts, load_index


def cosine_top_k(query_vec, matrix, k):
    """
    Return the k most similar rows of matrix to query_vec, as a list of
    (row_index, score) pairs sorted from best to worst.
    """
    q = np.ascontiguousarray(query_vec, dtype=np.float64)
    m = np.ascontiguousarray(matrix, dtype=np.float64)

    # Normalize to unit length so the dot product equals the cosine similarity.
    # Guard against a zero length vector (norm 0) so we never divide by zero.
    q_norm = np.linalg.norm(q) or 1.0
    q_unit = q / q_norm
    m_norms = np.linalg.norm(m, axis=1)
    m_norms[m_norms == 0] = 1.0
    m_unit = m / m_norms[:, None]

    # One similarity score per row. We multiply and sum per row instead of using
    # the matmul operator, which keeps the result identical but avoids a spurious
    # NumPy warning that some versions emit from matmul on these shapes.
    scores = (m_unit * q_unit).sum(axis=1)
    best = np.argsort(-scores)[:k]  # indices of the highest scores
    return [(int(i), float(scores[i])) for i in best]


def retrieve(source_path, question, k=TOP_K):
    """
    Return the top k chunks for a question as a list of dicts with id, score,
    and text. Highest score first.
    """
    index = load_index(source_path)
    if index is None:
        raise SystemExit(
            "No index found for that file. Build it first with the index step."
        )

    query_vec = embed_texts([question], task_type="RETRIEVAL_QUERY")[0]
    matrix = [c["embedding"] for c in index["chunks"]]

    results = []
    for row, score in cosine_top_k(query_vec, matrix, k):
        chunk = index["chunks"][row]
        results.append({"id": chunk["id"], "score": score, "text": chunk["text"]})
    return results
