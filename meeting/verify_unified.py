"""
Guided verification for the unified assistant.

It loads a known document, captures a short fake meeting where you say a few
known facts out loud, then has you ask questions that can only be answered from
the transcript, only from the document, or from both. You confirm each answer
pulls from the right source.

Run from the project root, headphones recommended:
    python -m meeting.verify_unified

The document facts and the facts you will speak are deliberately different, so
the source of every answer is unambiguous.
"""

import os
import time

from .assistant import ensure_doc_indexed, run_query_loop
from .capture import LiveCapture
from .transcript_memory import TranscriptMemory

SAMPLE_DOC = os.path.join(os.path.dirname(__file__), "sample", "northwind_brief.txt")

# Facts that live ONLY in the document.
DOC_FACTS = [
    "launch date is March 14, 2026",
    "subscription price is 49 dollars per month",
    "vendor contact is Dana Reyes",
]

# Facts you will SAY OUT LOUD, which live only in the meeting transcript.
SPOKEN_FACTS = [
    "The client for this project is Acme Corporation.",
    "Our total budget is fifty thousand dollars.",
    "We agreed to lower the price to 39 dollars per month.",
]

# Questions to try, with where the answer should come from.
TEST_QUESTIONS = [
    ("Who is the client?", "transcript only, expect Acme Corporation"),
    ("When is the launch date?", "document only, expect March 14"),
    ("What is the product name and who is the client?", "both, Northwind from doc and Acme from meeting"),
    ("What is the monthly price?", "conflict, expect both 49 from the doc and 39 from the meeting"),
    ("What is the office address?", "neither, expect a plain could not find"),
]


def main():
    print("Unified assistant verification")
    print("This loads a document, captures a few spoken facts, then lets you ask.")
    print("Headphones recommended so the mic does not echo system audio.\n")

    print(f"Indexing the sample document: {os.path.basename(SAMPLE_DOC)}")
    ensure_doc_indexed(SAMPLE_DOC)

    memory = TranscriptMemory()
    capture = LiveCapture(on_line=lambda label, text: memory.add(label, text))
    status = capture.start()
    print(
        f"Capturing. Mic alive: {status['mic_alive']}, "
        f"system alive: {status['system_alive']} at {status['system_rate']} Hz.\n"
    )
    if not status["mic_alive"]:
        print("Mic does not seem alive. Fix mic permission and rerun.")
        capture.stop()
        return

    try:
        # Phase 1: speak the known facts into the mic.
        print("=" * 66)
        print("PHASE 1: say these lines out loud, clearly, one at a time:")
        print("=" * 66)
        for fact in SPOKEN_FACTS:
            print(f"   {fact}")
        input("\nPress Enter AFTER you have said all three... ")

        print("Waiting a moment for the last lines to transcribe...")
        time.sleep(3)

        print("\nMeeting transcript captured so far:")
        print(memory.render() or "   (nothing captured, check your mic)")

        # Phase 2: ask questions and check the source of each answer.
        print()
        print("=" * 66)
        print("PHASE 2: ask these questions and check the source of each answer:")
        print("=" * 66)
        for question, expectation in TEST_QUESTIONS:
            print(f"   {question:<45} -> {expectation}")
        print()
        print("The document holds these facts (so you can check doc only answers):")
        for fact in DOC_FACTS:
            print(f"   {fact}")

        run_query_loop(memory, SAMPLE_DOC)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping capture.")
        capture.stop()


if __name__ == "__main__":
    main()
