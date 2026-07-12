"""
The meeting assistant: capture the meeting live while you type questions, and
answer them grounded in both your documents and what has been said so far.

Run from the project root:
    python -m meeting.assistant --doc path/to/your/document.pdf
    python -m meeting.assistant                 (transcript only, no document)

While the meeting captures in the background, type a question and press Enter.
Commands:
    :transcript   show the meeting transcript so far
    :quit         stop and exit
"""

import argparse

from docrag.chunk import chunk_text
from docrag.extract import extract_text
from docrag.store import build_index, load_index

from .capture import LiveCapture
from .summary import save_meeting
from .transcript_memory import TranscriptMemory
from .unified import answer_from_sources, is_not_found, unified_retrieve, web_search_hook

TOP_FOR_PROMPT = 6


def ensure_doc_indexed(path):
    if load_index(path) is None:
        chunks = chunk_text(extract_text(path))
        build_index(path, chunks)


def _print_sources(hits):
    print("Grounded on:")
    for h in hits:
        tag = "document" if h["source"] == "document" else "meeting"
        snippet = h["text"].replace("\n", " / ")
        if len(snippet) > 130:
            snippet = snippet[:130] + "..."
        print(f"  [{tag} chunk {h['id']}  score {h['score']:.3f}]  {snippet}")


def answer_one(question, doc_path, memory):
    """Retrieve from both sources, answer, and print the answer with its sources."""
    hits = unified_retrieve(question, doc_path, memory)
    top = hits[:TOP_FOR_PROMPT]
    answer = answer_from_sources(question, top)

    if not top or is_not_found(answer):
        # The "neither" branch: docs and transcript had nothing. Fall through to
        # web search. This is the only place web search is ever consulted, so a
        # web answer can never blend into a document or meeting answer.
        web = web_search_hook(question)
        if web is None:
            print("Answer: I could not find that in your documents, the meeting, or on the web.")
            return
        print("Answer (from the web): " + web["answer"])
        if web["sources"]:
            print("Web sources:")
            for s in web["sources"]:
                print(f"  {s['title']} - {s['uri']}")
        else:
            print("Web sources: (the model did not return citations)")
        return

    print("Answer: " + answer)
    _print_sources(top)


def run_query_loop(memory, doc_path):
    """
    The main thread loop. Blocks on input while capture keeps running on the
    background thread. This is the part that must not, and does not, pause audio.
    """
    print("\nType a question and press Enter. Commands: :transcript, :quit\n")
    while True:
        try:
            line = input("question> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line == ":quit":
            break
        if line == ":transcript":
            print(memory.render() or "(transcript is empty so far)")
            continue
        try:
            answer_one(line, doc_path, memory)
        except Exception as e:  # noqa: BLE001  surface API errors without crashing the loop
            print(f"(could not answer: {e})")


def run_session(memory, doc_path):
    """
    Run one meeting: capture in the background, take questions on the main thread,
    and on ANY exit path (clean :quit or Ctrl+C) stop capture and save a summary.

    The try/finally is the shutdown handler. Whether the loop returns normally or
    a KeyboardInterrupt unwinds through it, the finally block always runs, so an
    interrupted meeting still gets summarized from whatever transcript exists.
    """
    capture = LiveCapture(on_line=lambda label, text: memory.add(label, text))
    status = capture.start()
    print(
        f"Capturing. Mic alive: {status['mic_alive']}, "
        f"system alive: {status['system_alive']} at {status['system_rate']} Hz."
    )
    if not status["mic_alive"] or not status["system_alive"]:
        print("Warning: a capture stream may be down. Answers from the meeting may be empty.")

    try:
        run_query_loop(memory, doc_path)
    except KeyboardInterrupt:
        print("\nInterrupted. Wrapping up the meeting.")
    finally:
        print("\nStopping capture.")
        capture.stop()
        print("Writing the meeting summary...")
        path = save_meeting(memory)
        print(f"Meeting summary saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Live meeting assistant")
    parser.add_argument("--doc", help="path to a document to ground answers in")
    args = parser.parse_args()

    memory = TranscriptMemory()

    doc_path = None
    if args.doc:
        doc_path = args.doc
        print(f"Indexing document: {doc_path}")
        ensure_doc_indexed(doc_path)

    run_session(memory, doc_path)


if __name__ == "__main__":
    main()
