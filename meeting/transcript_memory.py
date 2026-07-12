"""
The rolling meeting transcript, held in memory as it streams.

This is the second knowledge source alongside your documents. Every final line
from either stream ([ME] or [THEM]) is appended with a rough timestamp. For
retrieval, the lines are grouped into small chunks the same way a document is,
so the same cosine similarity search works on both.

It is shared between two threads: the capture thread writes, the query thread
reads. A lock makes that safe, so a half written transcript is never read.
"""

import threading
import time


class TranscriptMemory:
    def __init__(self, lines_per_chunk=4):
        self._lines = []
        self._lock = threading.Lock()
        self.lines_per_chunk = lines_per_chunk

    def add(self, label, text):
        """Append one final line. Called from the capture thread."""
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._lines.append(
                {
                    "t": time.time(),
                    "clock": time.strftime("%H:%M:%S"),
                    "label": label,
                    "text": text,
                }
            )

    def lines(self):
        with self._lock:
            return list(self._lines)

    def is_empty(self):
        with self._lock:
            return not self._lines

    def render(self):
        """The whole transcript as readable text."""
        return "\n".join(
            f"[{r['clock']}] [{r['label']}] {r['text']}" for r in self.lines()
        )

    def chunks(self):
        """
        Group consecutive lines into retrieval chunks, keeping the speaker labels
        and times inside each chunk so the model can see who said what and roughly
        when. Returns a list of {id, text}.
        """
        rows = self.lines()
        chunks = []
        for start in range(0, len(rows), self.lines_per_chunk):
            group = rows[start:start + self.lines_per_chunk]
            text = "\n".join(
                f"[{r['clock']}] [{r['label']}] {r['text']}" for r in group
            )
            chunks.append({"id": start // self.lines_per_chunk, "text": text})
        return chunks
