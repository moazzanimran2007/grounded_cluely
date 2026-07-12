"""
Launch the Swift system audio helper and read its frames into a queue.

How the pieces fit, since the subprocess part is new:

  - subprocess.Popen starts the compiled Swift binary as a separate program.
    We connect to its two output streams: stdout carries the raw binary audio,
    stderr carries human and machine readable status lines.
  - The helper prints a "META sample_rate=NNNNN" line on stderr. We read that
    first so we know the real sample rate before connecting to Deepgram. The
    rate is not hardcoded anywhere: it comes from the helper every run.
  - One background thread keeps reading stdout and dropping audio chunks into a
    thread safe queue, the same kind of queue the mic callback fills. From the
    transcription code's point of view, this queue is just another audio source.
"""

import os
import queue
import re
import signal
import subprocess
import sys
import threading

# native/systemaudiotap sits next to this package, one level up.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HELPER = os.path.join(PROJECT_ROOT, "native", "systemaudiotap")

# Matches both the machine line "sample_rate=44100" and the human line
# "sample rate 44100 Hz", so it works whether or not the helper was rebuilt.
_RATE_RE = re.compile(r"sample[_ ]rate[=\s]+(\d+)")


class SystemAudioSource:
    """
    Owns the helper subprocess and exposes its audio as a queue of byte chunks.
    Call start() to begin (it returns the sample rate), then read audio_q. Call
    stop() when finished.
    """

    def __init__(self, helper_path=DEFAULT_HELPER, block_bytes=4096):
        self.helper_path = helper_path
        self.block_bytes = block_bytes
        self.audio_q = queue.Queue()
        self.sample_rate = None
        self.proc = None

    def start(self):
        if not os.path.exists(self.helper_path):
            raise SystemExit(
                f"System audio helper not found at {self.helper_path}. "
                "Build it with: sh native/build.sh"
            )

        self.proc = subprocess.Popen(
            [self.helper_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Learn the sample rate from stderr before anything else reads stderr.
        self.sample_rate = self._read_sample_rate()

        # Now hand stdout and the rest of stderr to background threads.
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        return self.sample_rate

    def _read_sample_rate(self, max_lines=50):
        for _ in range(max_lines):
            line = self.proc.stderr.readline()
            if not line:
                break
            text = line.decode(errors="replace")
            match = _RATE_RE.search(text)
            if match:
                return int(match.group(1))
        raise SystemExit(
            "The helper did not report a sample rate. It may have failed to "
            "start or lacks the system audio permission."
        )

    def _pump_stdout(self):
        # Read fixed size blocks of raw PCM and queue them. A None marks the end
        # so the sender knows to stop, the same sentinel the mic path uses.
        while True:
            block = self.proc.stdout.read(self.block_bytes)
            if not block:
                self.audio_q.put(None)
                break
            self.audio_q.put(block)

    def _drain_stderr(self):
        # Keep stderr flowing so the helper never blocks on a full pipe. We stay
        # quiet here, but surface a line if it looks like an error.
        for line in self.proc.stderr:
            text = line.decode(errors="replace").strip()
            if "ERROR" in text:
                print(f"[system helper] {text}", file=sys.stderr)

    def stop(self):
        self.audio_q.put(None)
        if self.proc and self.proc.poll() is None:
            # SIGINT is what the helper installs a clean shutdown handler for.
            self.proc.send_signal(signal.SIGINT)
