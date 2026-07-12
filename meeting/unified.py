"""
Unified retrieval: answer a question from BOTH the uploaded documents and the
live meeting transcript, and make the answer aware of where each fact came from.

How it works:
  - Embed the question once.
  - Compare it against the document chunks (already embedded and stored) and
    against the transcript chunks (embedded fresh each query, since the
    transcript keeps growing).
  - Merge the two sets of hits and sort by similarity, so the strongest matches
    win regardless of which source they came from.
  - Hand the top hits to the model, tagged by source, with instructions to cite
    the source, to surface conflicts, and to say plainly when neither source has
    the answer.
"""

from docrag.answer import NOT_FOUND
from docrag.config import CHAT_MODEL, get_client
from docrag.retrieve import cosine_top_k
from docrag.store import embed_texts, load_index

GROUNDING_INSTRUCTION = (
    "You are an assistant in a live meeting. You are given two kinds of context: "
    "passages from uploaded DOCUMENTS, and lines from the live MEETING TRANSCRIPT, "
    "where [ME] is the user and [THEM] is other people. "
    "Answer using only the context provided, never outside knowledge. "
    "When a fact comes from a document, say so, for example 'based on the document'. "
    "When it comes from the meeting, say so, for example 'earlier in the meeting it was said'. "
    "If the documents and the transcript disagree, present both versions and say "
    "which came from where, do not silently pick one. "
    f"If neither source contains the answer, reply exactly: {NOT_FOUND}"
)


def unified_retrieve(question, doc_path, memory, k_each=4):
    """
    Return a single ranked list of hits drawn from both sources. Each hit is a
    dict with source ('document' or 'transcript'), id, score, and text.
    """
    question_vec = embed_texts([question], task_type="RETRIEVAL_QUERY")[0]
    hits = []

    # Documents: compare against the stored embeddings, no re-embedding needed.
    doc_index = load_index(doc_path) if doc_path else None
    if doc_index:
        matrix = [c["embedding"] for c in doc_index["chunks"]]
        for row, score in cosine_top_k(question_vec, matrix, k_each):
            chunk = doc_index["chunks"][row]
            hits.append(
                {"source": "document", "id": chunk["id"], "score": score, "text": chunk["text"]}
            )

    # Transcript: embed the current chunks fresh, since it grows as people talk.
    transcript_chunks = memory.chunks()
    if transcript_chunks:
        vectors = embed_texts(
            [c["text"] for c in transcript_chunks], task_type="RETRIEVAL_DOCUMENT"
        )
        for row, score in cosine_top_k(question_vec, vectors, k_each):
            chunk = transcript_chunks[row]
            hits.append(
                {"source": "transcript", "id": chunk["id"], "score": score, "text": chunk["text"]}
            )

    # Strongest matches first, whichever source they came from.
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits


def answer_from_sources(question, hits):
    """
    Ask the model for an answer grounded in the given hits. Returns the answer
    text. If there are no hits at all, returns the not found sentence without
    calling the model.
    """
    if not hits:
        return NOT_FOUND

    blocks = []
    for h in hits:
        header = "DOCUMENT" if h["source"] == "document" else "MEETING TRANSCRIPT"
        blocks.append(f"[{header} chunk {h['id']}]\n{h['text']}")
    context = "\n\n".join(blocks)

    prompt = (
        f"{GROUNDING_INSTRUCTION}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    client = get_client()
    resp = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return (resp.text or "").strip()


def is_not_found(answer):
    """True if the model abstained, so the caller can trigger the web hook."""
    return "could not find that in the document" in answer.lower()


def web_search_hook(question):
    """
    Web search fallback, reached only when the documents and the transcript both
    came up empty (the "neither" branch).

    It uses Gemini's built in Google Search grounding, so it needs no second API
    key: the same GEMINI_API_KEY does the search, reads the pages, and writes a
    grounded answer in one call, and hands back the source URLs to cite.

    Returns a dict {answer, sources} on success, or None when the web has nothing
    useful, so the caller can abstain instead of inventing.
    """
    from google.genai import types

    client = get_client()
    prompt = (
        "Use web search to answer the question factually and concisely. "
        "If you cannot find a reliable answer, reply exactly: "
        f"{NOT_FOUND}\n\nQuestion: {question}"
    )
    try:
        resp = client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
    except Exception:  # noqa: BLE001  a search or quota failure means abstain
        return None

    text = (resp.text or "").strip()
    if not text or is_not_found(text):
        return None
    return {"answer": text, "sources": _web_sources(resp)}


def _web_sources(resp):
    """Pull the cited web pages (title and url) out of the grounding metadata."""
    sources = []
    try:
        metadata = resp.candidates[0].grounding_metadata
        for chunk in metadata.grounding_chunks or []:
            web = getattr(chunk, "web", None)
            if web and web.uri:
                sources.append({"title": web.title or web.uri, "uri": web.uri})
    except (AttributeError, IndexError, TypeError):
        pass

    # De duplicate by url while keeping order.
    seen = set()
    unique = []
    for s in sources:
        if s["uri"] not in seen:
            seen.add(s["uri"])
            unique.append(s)
    return unique
