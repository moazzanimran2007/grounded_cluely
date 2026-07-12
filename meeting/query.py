"""
Shared question orchestration for every front end (terminal, overlay, web).

It reuses the existing unified retrieval and answer functions unchanged. It only
decides the branch (documents and transcript, else web, else nothing) and returns
the answer text plus a list of source lines for display. Keeping this in one
place means all front ends ask questions identically.
"""

from .unified import answer_from_sources, is_not_found, unified_retrieve, web_search_hook

TOP_FOR_PROMPT = 6


def answer_for_ui(question, doc_path, memory):
    """Return (answer_text, source_lines) for a typed question."""
    hits = unified_retrieve(question, doc_path, memory)
    top = hits[:TOP_FOR_PROMPT]
    answer = answer_from_sources(question, top)

    if not top or is_not_found(answer):
        web = web_search_hook(question)
        if web is None:
            return ("I could not find that in your documents, the meeting, or on the web.", [])
        sources = [f"web: {s['title']} - {s['uri']}" for s in web["sources"]]
        return ("From the web: " + web["answer"], sources)

    sources = []
    for h in top:
        tag = "document" if h["source"] == "document" else "meeting"
        snippet = h["text"].replace("\n", " / ")
        if len(snippet) > 110:
            snippet = snippet[:110] + "..."
        sources.append(f"{tag} chunk {h['id']} (score {h['score']:.2f}): {snippet}")
    return (answer, sources)
