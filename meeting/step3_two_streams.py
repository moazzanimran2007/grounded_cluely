"""
Step 3: transcribe mic and system audio at once, labeled.

Two separate streams run at the same time:
  [ME]   your microphone, captured by sounddevice at 16000 Hz
  [THEM] system audio, captured by the Swift helper at its own rate

Each stream has its own Deepgram connection, so the two are never mixed into one
blob. The labeling is the whole point: the transcript records who said what.

We print only final lines, each stamped with the time it arrived, so the result
reads as a clean meeting transcript in time order. Interim guesses are skipped
here because two streams overwriting one live line would be unreadable.

Run from the project root:
    python -m meeting.step3_two_streams
"""

import asyncio
import queue
import time

from listen import SAMPLE_RATE, make_mic_stream

from .session import transcribe_stream
from .system_source import SystemAudioSource


def make_labeled_printer():
    """Return a function that prints final transcripts tagged with a label."""
    def handle(label, transcript):
        # Skip interim guesses and empty results; we want clean final lines.
        if not transcript.is_final or not transcript.text.strip():
            return
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] [{label}] {transcript.text}")
    return handle


async def run():
    printer = make_labeled_printer()

    # Mic source: feed the sounddevice callback into its own queue.
    mic_q = queue.Queue()

    def on_mic_frames(indata, frames, time_info, status):
        mic_q.put(bytes(indata))

    mic_stream = make_mic_stream(on_mic_frames)

    # System source: the Swift helper, at whatever rate it reports.
    system = SystemAudioSource()
    system_rate = system.start()
    print(f"Listening. [ME] mic at {SAMPLE_RATE} Hz, [THEM] system at {system_rate} Hz.")
    print("Talk, and play audio, and watch labeled lines appear. Press Ctrl+C to stop.\n")

    try:
        with mic_stream:
            # Both streams run concurrently. Each result is printed the moment it
            # arrives, so the two interleave naturally in time order.
            await asyncio.gather(
                transcribe_stream(mic_q, SAMPLE_RATE, lambda t: printer("ME", t)),
                transcribe_stream(system.audio_q, system_rate, lambda t: printer("THEM", t)),
            )
    finally:
        mic_q.put(None)
        system.stop()


def main():
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nStopping.")


if __name__ == "__main__":
    main()
