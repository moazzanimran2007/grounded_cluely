"""
A small command line so you can run each step on its own and watch it work.

  python -m docrag extract  FILE
  python -m docrag chunk    FILE
  python -m docrag index    FILE [--force]
  python -m docrag retrieve FILE "your question"
  python -m docrag ask      FILE "your question"
  python -m docrag demo     FILE QUESTIONS_FILE

extract and chunk need no API key. The rest need GEMINI_API_KEY in .env.
"""

import argparse

from .answer import answer
from .chunk import chunk_text
from .extract import extract_text
from .retrieve import retrieve
from .store import build_index, load_index

PREVIEW_CHARS = 400


def ensure_index(path):
    """Build the index for a file if it does not exist yet, then return it."""
    index = load_index(path)
    if index is not None:
        return index
    chunks = chunk_text(extract_text(path))
    record, _ = build_index(path, chunks)
    return record


def cmd_extract(args):
    text = extract_text(args.file)
    print(f"Extracted {len(text)} characters from {args.file}.")
    print("First few hundred characters:\n")
    print(text[:PREVIEW_CHARS])


def cmd_chunk(args):
    chunks = chunk_text(extract_text(args.file))
    print(f"Split into {len(chunks)} chunks.\n")
    # Show the first chunk and the last one so you can see the shape and the
    # overlap without scrolling through everything.
    if chunks:
        print("----- first chunk -----")
        print(chunks[0])
    if len(chunks) > 1:
        print("\n----- last chunk -----")
        print(chunks[-1])


def cmd_index(args):
    chunks = chunk_text(extract_text(args.file))
    record, did_embed = build_index(args.file, chunks, force=args.force)
    if did_embed:
        print(f"Embedded and stored {len(record['chunks'])} chunks.")
    else:
        print(
            f"Already indexed ({len(record['chunks'])} chunks), source unchanged. "
            "Nothing re-embedded. Use --force to rebuild."
        )


def cmd_retrieve(args):
    ensure_index(args.file)
    results = retrieve(args.file, args.question)
    print(f"Top {len(results)} chunks for: {args.question}\n")
    for r in results:
        print(f"[chunk {r['id']}  score {r['score']:.3f}]")
        print(r["text"][:300])
        print()


def cmd_ask(args):
    ensure_index(args.file)
    results = retrieve(args.file, args.question)
    reply = answer(args.question, results)
    print(f"Q: {args.question}")
    print(f"A: {reply}")
    print(f"\n(grounded on chunks: {', '.join(str(r['id']) for r in results)})")


def cmd_demo(args):
    ensure_index(args.file)
    with open(args.questions, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    print(f"Running {len(questions)} questions against {args.file}\n")
    for q in questions:
        results = retrieve(args.file, q)
        reply = answer(q, results)
        chunk_ids = ", ".join(str(r["id"]) for r in results)
        print(f"Q: {q}")
        print(f"A: {reply}")
        print(f"   (chunks {chunk_ids})\n")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="docrag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="step 1: show extracted text")
    p.add_argument("file")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("chunk", help="step 2: show the chunks")
    p.add_argument("file")
    p.set_defaults(func=cmd_chunk)

    p = sub.add_parser("index", help="step 3: embed and store")
    p.add_argument("file")
    p.add_argument("--force", action="store_true", help="rebuild even if unchanged")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("retrieve", help="step 4: show retrieved chunks")
    p.add_argument("file")
    p.add_argument("question")
    p.set_defaults(func=cmd_retrieve)

    p = sub.add_parser("ask", help="step 5: grounded answer")
    p.add_argument("file")
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("demo", help="step 6: run a list of questions")
    p.add_argument("file")
    p.add_argument("questions")
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
