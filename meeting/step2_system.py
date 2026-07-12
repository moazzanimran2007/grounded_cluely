"""
Step 2: transcribe system audio on its own.

Launches the Swift helper, reads its frames, and sends them through the same
Deepgram path the mic uses. Play any audio on your Mac (a video, a call) and
watch it get transcribed live in the terminal.

Run from the project root:
    python -m meeting.step2_system

Interim guesses refine in place, final lines commit, exactly like the mic
script. Press Ctrl+C to stop.
"""

import asyncio

from listen import render_transcript

from .session import transcribe_stream
from .system_source import SystemAudioSource


def main():
    source = SystemAudioSource()
    rate = source.start()
    print(f"System audio capture started at {rate} Hz.")
    print("Play some audio on your Mac to see it transcribed. Press Ctrl+C to stop.\n")

    try:
        asyncio.run(
            transcribe_stream(source.audio_q, rate, render_transcript)
        )
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        source.stop()


if __name__ == "__main__":
    main()
