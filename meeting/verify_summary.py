"""
Guided verification for end of meeting summarization.

Run a short session, say a few decisions and action items out loud, then end the
meeting. A timestamped summary file should appear in the meetings folder, with
the decisions and action items you actually said and nothing invented.

Test both exit paths, one per run:
  - Run once and end by typing :quit. Confirm a summary file was saved.
  - Run again and end by pressing Ctrl+C. Confirm a summary is STILL saved from
    the partial transcript.

Run from the project root, headphones not required (this uses your mic only):
    python -m meeting.verify_summary
"""

from .assistant import run_session
from .transcript_memory import TranscriptMemory

SUGGESTED_LINES = [
    "We decided to launch the product on March 14.",
    "We agreed to use PostgreSQL for the database.",
    "I will own the backend integration and finish it by Friday.",
    "Dana will write the launch announcement.",
]


def main():
    print("Summary verification: run a short meeting, then end it two ways.")
    print()
    print("After it says Capturing, say these lines out loud, clearly:")
    for line in SUGGESTED_LINES:
        print(f"   {line}")
    print()
    print("Then END THE MEETING one of two ways:")
    print("   type :quit and press Enter   (clean quit), OR")
    print("   press Ctrl+C                 (interrupted)")
    print()
    print("Either way, a timestamped summary should be saved and its path printed.")
    print("Check that the decisions and action items match what you said, that the")
    print("backend item is owned by ME and the announcement by Dana, and that")
    print("nothing was invented.")
    print()
    print("Run this once ending with :quit, then run it again ending with Ctrl+C.")
    print()

    memory = TranscriptMemory()
    run_session(memory, None)


if __name__ == "__main__":
    main()
