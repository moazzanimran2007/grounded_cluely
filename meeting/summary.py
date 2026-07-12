"""
End of meeting summarization.

When a session ends, this reads the full labeled transcript out of memory and
writes a timestamped file holding a structured summary plus the raw transcript.
It reuses the same Gemini chat model the rest of the app uses, so there is no new
dependency and no new key.

It does not touch capture, retrieval, or the query loop. It only reads the
transcript memory at shutdown.
"""

import os
import time

from docrag.config import CHAT_MODEL, get_client

# meetings/ sits at the project root, next to the meeting package.
MEETINGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "meetings"
)

# Below this many real spoken words, treat the meeting as too short to summarize
# rather than asking the model to invent something from almost nothing.
MIN_WORDS = 12

SUMMARY_INSTRUCTION = (
    "You are summarizing a meeting from its transcript. Each line is tagged [ME] "
    "for the user and [THEM] for other participants, with a timestamp. "
    "Write the summary in exactly these three sections, with these headings:\n\n"
    "Recap:\n"
    "A short prose paragraph on what the meeting was about.\n\n"
    "Key decisions:\n"
    "A bullet list of decisions that were actually made. If none were made, write None.\n\n"
    "Action items:\n"
    "A bullet list of tasks people committed to. After each item put the owner in "
    "parentheses. Use the speaker when it is clear who committed, for example (ME) "
    "or (THEM), or a name if the transcript names one. If the owner is not clear, "
    "write (owner unspecified). If there are no action items, write None.\n\n"
    "Ground everything strictly in the transcript. Do not invent decisions or "
    "action items that were not said. Do not use outside knowledge. "
    "Do not use em dashes anywhere, use commas or periods instead."
)


# Built from code points (0x2014 em dash, 0x2013 en dash) so this source file
# itself contains no dash characters.
_EM_DASH = chr(0x2014)
_EN_DASH = chr(0x2013)


def _strip_dashes(text):
    """Replace em and en dashes so the saved file never contains them."""
    return text.replace(_EM_DASH, " - ").replace(_EN_DASH, "-")


def generate_summary(transcript_text):
    """Ask the model for the three part summary, grounded in the transcript."""
    prompt = (
        f"{SUMMARY_INSTRUCTION}\n\n"
        f"Transcript:\n{transcript_text}\n\n"
        "Summary:"
    )
    client = get_client()
    resp = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return (resp.text or "").strip() or "The model returned an empty summary."


def _file_content(readable_time, summary, transcript):
    return (
        "# Meeting summary\n\n"
        f"Saved: {readable_time}\n\n"
        f"{summary}\n\n"
        "## Full transcript\n\n"
        f"{transcript or '(no transcript was captured)'}\n"
    )


def save_meeting(memory, meetings_dir=MEETINGS_DIR):
    """
    Summarize the transcript in memory and save it. Returns the file path.

    Always writes a file, even on an empty or failed run, so a meeting is never
    lost. The raw transcript is saved alongside the summary in the same file.
    """
    os.makedirs(meetings_dir, exist_ok=True)

    lines = memory.lines()
    transcript = memory.render()
    real_words = sum(len(row["text"].split()) for row in lines)

    if real_words < MIN_WORDS:
        summary = (
            "Meeting too short or empty to summarize. No decisions or action "
            "items were captured."
        )
    else:
        try:
            summary = generate_summary(transcript)
        except Exception as e:  # noqa: BLE001  never lose the transcript over an API error
            summary = (
                "Summary could not be generated (" + str(e)[:120] + "). "
                "The full transcript is preserved below."
            )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    readable = time.strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(meetings_dir, f"meeting_{stamp}.md")

    content = _strip_dashes(_file_content(readable, summary, transcript))
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
