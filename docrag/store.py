"""
Step 3: embed the chunks and store them in a local JSON index.

The store is the simplest thing that works: one JSON file per document, holding
the chunk text and its embedding side by side. No hosted vector database yet.
For a few hundred chunks, loading this file and comparing vectors in memory is
instant, and you can open the file and read exactly what is stored.

Idempotency: we record a hash of the source file. Re-running on an unchanged
file sees the same hash and skips re-embedding, so nothing is duplicated and no
API calls are wasted. Change the file and it re-embeds.
"""

import hashlib
import json
import os

from .config import EMBED_MODEL, INDEX_DIR, get_client

# Gemini accepts up to 100 pieces of text per embedding request. We batch so a
# long document still goes up in a handful of calls.
EMBED_BATCH = 100


def file_hash(path):
    """A SHA-256 fingerprint of the file contents, read in small blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def index_path_for(source_path):
    """The JSON index file for a given source document."""
    stem = os.path.splitext(os.path.basename(source_path))[0]
    return os.path.join(INDEX_DIR, stem + ".json")


def embed_texts(texts, task_type):
    """
    Turn a list of strings into a list of embedding vectors.

    task_type tells Gemini how the text will be used. RETRIEVAL_DOCUMENT for the
    stored chunks and RETRIEVAL_QUERY for the question. Using the matching types
    measurably improves which chunks come back as relevant.
    """
    from google.genai import types

    client = get_client()
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vectors.extend(e.values for e in resp.embeddings)
    return vectors


def build_index(source_path, chunks, force=False):
    """
    Embed the chunks and write the index file. Returns (record, did_embed).

    If an index already exists for this exact file (same hash) and force is
    False, we leave it alone and return did_embed=False. That is the idempotent
    path: re-running is a no-op.
    """
    os.makedirs(INDEX_DIR, exist_ok=True)
    idx_path = index_path_for(source_path)
    src_hash = file_hash(source_path)

    if os.path.exists(idx_path) and not force:
        with open(idx_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("source_hash") == src_hash:
            return existing, False

    embeddings = embed_texts(chunks, task_type="RETRIEVAL_DOCUMENT")
    record = {
        "source_path": os.path.abspath(source_path),
        "source_hash": src_hash,
        "embed_model": EMBED_MODEL,
        "chunks": [
            {"id": i, "text": text, "embedding": vector}
            for i, (text, vector) in enumerate(zip(chunks, embeddings))
        ],
    }
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(record, f)
    return record, True


def load_index(source_path):
    """Return the saved index for a document, or None if it has not been built."""
    idx_path = index_path_for(source_path)
    if not os.path.exists(idx_path):
        return None
    with open(idx_path, "r", encoding="utf-8") as f:
        return json.load(f)
