"""
Run one audio source through the shared Deepgram path.

This is the seam that lets the mic and the system audio reuse the exact same
transcription code. It takes any queue of PCM byte chunks plus the sample rate
that audio is at, opens a Deepgram connection tuned to that rate, and pumps the
audio through listen.py's sender and receiver. Each parsed transcript is handed
to on_transcript, which the caller uses to label or print it.
"""

import asyncio

from websockets.asyncio.client import connect

from listen import build_deepgram_url, receiver, require_api_key, sender


async def transcribe_stream(audio_q, sample_rate, on_transcript, channels=1):
    """
    Transcribe everything that arrives on audio_q until a None sentinel, calling
    on_transcript(Transcript) for each result. sample_rate comes from the source
    so Deepgram decodes the bytes correctly.
    """
    api_key = require_api_key()
    headers = {"Authorization": f"Token {api_key}"}
    url = build_deepgram_url(sample_rate=sample_rate, channels=channels)

    async with connect(url, additional_headers=headers) as ws:
        # sender drains the queue into Deepgram, receiver reads results back.
        # Exactly the same two coroutines the mic uses.
        await asyncio.gather(
            sender(ws, audio_q),
            receiver(ws, on_transcript),
        )
