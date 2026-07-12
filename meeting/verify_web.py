"""
Guided verification for the web search fallback.

It confirms three things:
  1. A question that is NOT in your document or your spoken transcript falls
     through to web search, returns a correct answer, and is tagged as coming
     from the web with a source.
  2. A document question still answers from the document, web does not hijack it.
  3. A transcript question still answers from the meeting, web does not hijack it.

While all of this runs, audio capture keeps going on the background thread, so
the web call (which happens on the main thread, like every answer) does not pause
the meeting.

Run from the project root, headphones recommended:
    python -m meeting.verify_web
"""

import os
import time

from .assistant import ensure_doc_indexed, run_query_loop
from .capture import LiveCapture
from .transcript_memory import TranscriptMemory

SAMPLE_DOC = os.path.join(os.path.dirname(__file__), "sample", "northwind_brief.txt")

SPOKEN_FACT = "The client for this project is Acme Corporation."

TEST_QUESTIONS = [
    ("Who is the current president of the United States?",
     "WEB only, expect an Answer (from the web) with a source"),
    ("When is the launch date?",
     "DOCUMENT only, expect based on the document, NOT the web"),
    ("Who is the client?",
     "TRANSCRIPT only, expect earlier in the meeting, NOT the web"),
]


def main():
    print("Web search fallback verification")
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
        print("=" * 66)
        print("PHASE 1: say this line out loud, clearly:")
        print("=" * 66)
        print(f"   {SPOKEN_FACT}")
        input("\nPress Enter AFTER you have said it... ")
        print("Waiting a moment for it to transcribe...")
        time.sleep(3)
        print("\nMeeting transcript so far:")
        print(memory.render() or "   (nothing captured, check your mic)")

        print()
        print("=" * 66)
        print("PHASE 2: ask these and check the source tag on each answer:")
        print("=" * 66)
        for question, expectation in TEST_QUESTIONS:
            print(f"   {question:<48} -> {expectation}")
        print()
        print("A PASS is: the first answer is tagged (from the web) with a source,")
        print("and the other two answer from the document and the meeting with no web.")
        print()

        run_query_loop(memory, SAMPLE_DOC)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping capture.")
        capture.stop()


if __name__ == "__main__":
    main()
