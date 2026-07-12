"""
LiveCapture: run both audio streams in the background and feed every final line
to a callback, so the rest of the program can do other work (like answering
questions) without ever pausing the audio.

The concurrency, plainly:
  - The mic and the system helper each fill their own thread safe queue.
  - One BACKGROUND thread runs an asyncio loop with the two Deepgram streams.
    That thread does nothing but capture and transcribe, forever, until stopped.
  - Your MAIN thread is free. It can block on input() or spend two seconds
    calling the answer model, and the background thread keeps capturing the
    whole time. The only shared state is whatever your on_line callback touches,
    which is the lock protected transcript memory.

This is why the query path cannot starve the audio path: they are different
threads, and the audio one never waits on the query one.
"""

import asyncio
import queue
import threading
import time

from listen import SAMPLE_RATE, make_mic_stream

from .session import transcribe_stream
from .system_source import SystemAudioSource


def _drain(q):
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


class LiveCapture:
    def __init__(self, on_line):
        # on_line(label, text) is called for each final transcript line.
        self.on_line = on_line
        self.mic_q = queue.Queue()
        self.system = SystemAudioSource()
        self.mic_stream = None
        self.system_rate = None
        self.error = None
        self._thread = None

    def _record(self, label, transcript):
        if transcript.is_final and transcript.text.strip():
            self.on_line(label, transcript.text)

    def start(self):
        """
        Start both audio sources, do a quick liveness sample, then start the
        background transcription thread. Returns a small status dict.
        """
        def on_mic_frames(indata, frames, time_info, status):
            self.mic_q.put(bytes(indata))

        self.mic_stream = make_mic_stream(on_mic_frames)
        # Start the microphone BEFORE the system audio helper. The helper creates
        # a CoreAudio aggregate device, and creating that while the default mic is
        # opening can make the mic stream start hang on macOS. Opening the mic
        # first, while CoreAudio is in a clean state, avoids that collision.
        self.mic_stream.start()
        self.system_rate = self.system.start()

        # Sample for a moment to confirm frames are flowing before we drain.
        time.sleep(1.5)
        mic_alive = not self.mic_q.empty()
        system_alive = self.system.proc.poll() is None

        # Clear the warm up backlog so the transcript starts clean.
        _drain(self.mic_q)
        _drain(self.system.audio_q)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        return {
            "system_rate": self.system_rate,
            "mic_alive": mic_alive,
            "system_alive": system_alive,
        }

    def _run(self):
        async def both():
            await asyncio.gather(
                transcribe_stream(self.mic_q, SAMPLE_RATE, lambda t: self._record("ME", t)),
                transcribe_stream(
                    self.system.audio_q, self.system_rate, lambda t: self._record("THEM", t)
                ),
            )

        try:
            asyncio.run(both())
        except BaseException as error:  # noqa: BLE001  capture so caller can report
            self.error = error

    def stop(self):
        self.mic_q.put(None)
        self.system.audio_q.put(None)
        self.system.stop()
        if self.mic_stream is not None:
            self.mic_stream.stop()
            self.mic_stream.close()
        if self._thread is not None:
            self._thread.join(timeout=5)
