"""
Task 1: capture mic audio, stream it to Deepgram, print the live transcript.

The whole thing is three moving parts:

  1. sounddevice grabs raw audio from your mic in small chunks.
  2. A background queue hands those chunks to the network code without
     blocking the microphone.
  3. A websocket sends the chunks to Deepgram and reads transcripts back.

Run it, talk, and watch your words appear. Press Ctrl+C to stop.

The functions below are written as small, separate pieces on purpose. That
keeps the live loop readable, and it also lets the test suite exercise each
piece (key loading, message parsing, audio chunking, the send/receive glue)
without a real mic or a real network connection.
"""

import asyncio
import json
import os
import queue
import sys
import wave
from collections import namedtuple

import sounddevice as sd
# Use the modern asyncio client explicitly. The bare "websockets.connect" still
# points at the older legacy client, which does not accept additional_headers.
from websockets.asyncio.client import connect
from dotenv import load_dotenv

# Pull DEEPGRAM_API_KEY out of the .env file into the environment.
load_dotenv()

# Audio settings. These three values MUST match what we tell Deepgram in the
# URL below, otherwise it will try to decode our bytes with the wrong shape.
SAMPLE_RATE = 16000   # 16 kHz: plenty for speech, half the data of 32 kHz.
CHANNELS = 1          # Mono. One voice source, no need for stereo.
BLOCK_SIZE = 4000     # Frames per chunk. 4000 / 16000 = 0.25s of audio per send.
SAMPLE_WIDTH = 2      # Bytes per sample. 2 = 16-bit (int16), what we ask for.

# Deepgram's live endpoint. Everything after the "?" is configuration:
#   model=nova-2        their current general-purpose model
#   encoding=linear16   raw 16-bit PCM, which is exactly what sounddevice gives us
#   sample_rate         MUST match the rate of the audio we actually send
#   channels            MUST match the channel count we actually send
#   interim_results     send partial guesses as you speak, not just final lines
#   punctuate           add commas and periods for readability
#
# We build the URL from a function because different audio sources run at
# different sample rates. The mic is 16000, but the system audio tap reports its
# own rate (44100 or 48000 depending on the output device), so the system stream
# passes that rate in here instead of assuming one.
def build_deepgram_url(sample_rate=SAMPLE_RATE, channels=CHANNELS):
    return (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-2"
        "&encoding=linear16"
        f"&sample_rate={sample_rate}"
        f"&channels={channels}"
        "&interim_results=true"
        "&punctuate=true"
    )


# The mic path keeps using this constant, unchanged.
DEEPGRAM_URL = build_deepgram_url()

# A small, named container for one transcript result. "text" is the words,
# "is_final" tells us whether Deepgram considers this line settled or still a
# live guess that may change.
Transcript = namedtuple("Transcript", ["text", "is_final"])

# A thread-safe mailbox. The microphone callback (which runs on its own thread)
# drops audio bytes in here, and our network code picks them up. queue.Queue is
# the standard tool for passing data between threads safely.
audio_queue: "queue.Queue[bytes]" = queue.Queue()


def require_api_key(env=None):
    """
    Return the Deepgram API key, or exit with a clear message if it is missing.

    We read from a dict (os.environ by default) so a test can pass its own env
    and check both the success and the failure path. Raising SystemExit with a
    message is Python's clean way to stop a script: it prints the message to
    stderr and exits non-zero, instead of crashing with a confusing traceback.
    """
    if env is None:
        env = os.environ
    key = env.get("DEEPGRAM_API_KEY")
    if not key:
        raise SystemExit(
            "No DEEPGRAM_API_KEY found. Copy .env.example to .env and paste your key."
        )
    return key


def parse_transcript(message):
    """
    Turn one raw Deepgram websocket message into a Transcript, or None.

    Returns None (rather than raising) for anything that is not usable text:
    non-Results messages, empty transcripts, error payloads, and malformed
    JSON. The live loop and the tests both rely on this never blowing up on a
    surprising message.
    """
    try:
        data = json.loads(message)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    # Deepgram tags transcription payloads with type "Results". Anything else
    # (Metadata, errors, keepalives) carries no words for us.
    if data.get("type") != "Results":
        return None
    try:
        text = data["channel"]["alternatives"][0]["transcript"]
    except (KeyError, IndexError, TypeError):
        return None
    if not text:
        return None
    return Transcript(text=text, is_final=bool(data.get("is_final")))


def render_transcript(t, out=None):
    """
    Print one Transcript the way the live terminal should show it.

    Interim guesses overwrite themselves in place (carriage return, no newline)
    so the line refines as you speak. A final result commits to its own line.
    out defaults to the current stdout, resolved at call time so a test can
    capture it.
    """
    if out is None:
        out = sys.stdout
    if t.is_final:
        out.write("\r" + t.text + "\n")
    else:
        out.write("\r" + t.text)
    out.flush()


def read_wav_chunks(path, block_size=BLOCK_SIZE):
    """
    Read a WAV file and hand it back as the same kind of fixed-size int16 byte
    chunks the microphone produces. This is what lets the audio tests run off a
    real file instead of live hardware.

    Returns (sample_rate, channels, sample_width, chunks) so a test can assert
    the audio shape. Every chunk except possibly the last holds exactly
    block_size frames.
    """
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()  # bytes per sample, 2 means int16
        raw = wav.readframes(wav.getnframes())

    bytes_per_frame = channels * sample_width
    chunk_bytes = block_size * bytes_per_frame
    chunks = [raw[i:i + chunk_bytes] for i in range(0, len(raw), chunk_bytes)]
    return sample_rate, channels, sample_width, chunks


def make_mic_stream(on_frames):
    """
    Build (but do not start) a sounddevice input stream that calls on_frames
    with each chunk of raw int16 mic bytes.

    Pulled into its own function so the live code and the tests share one
    definition of "the audio shape we ask the device for". A test mocks
    sd.RawInputStream and checks these exact arguments.
    """
    return sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        callback=on_frames,
    )


def on_audio(indata, frames, time_info, status):
    """
    sounddevice calls this automatically every time it has a fresh chunk of
    mic audio. It runs on a separate audio thread, so the only safe thing to do
    here is hand the bytes off quickly and return. No printing, no networking.
    """
    if status:
        # Buffer overflows etc. Worth seeing, but not fatal.
        print(f"[audio status] {status}", file=sys.stderr)
    # indata is a raw bytes buffer because we use RawInputStream.
    audio_queue.put(bytes(indata))


async def sender(ws, audio_q=None):
    """
    Pull audio chunks off the queue and send them to Deepgram as binary frames.

    queue.get() blocks until a chunk is available, but blocking is forbidden in
    async code because it would freeze the whole event loop. So we run the
    blocking get() in a thread pool with run_in_executor and await the result.
    That keeps the receiver free to print transcripts at the same time.

    A None on the queue is the shutdown signal: we tell Deepgram we are done so
    it flushes the last transcript, then stop.
    """
    if audio_q is None:
        audio_q = audio_queue
    loop = asyncio.get_event_loop()
    while True:
        chunk = await loop.run_in_executor(None, audio_q.get)
        if chunk is None:
            await ws.send(json.dumps({"type": "CloseStream"}))
            return
        await ws.send(chunk)


async def receiver(ws, on_transcript=None):
    """
    Read messages from Deepgram, parse each one, and hand any real transcript
    to on_transcript. Defaults to printing live in the terminal.

    The on_transcript hook is what lets other code (the acceptance script, the
    pipeline test) tap the same stream of words without changing this loop.
    """
    if on_transcript is None:
        on_transcript = render_transcript
    async for message in ws:
        t = parse_transcript(message)
        if t is not None:
            on_transcript(t)


async def main():
    api_key = require_api_key()

    # Open the mic. RawInputStream gives us plain bytes (int16 PCM) instead of a
    # numpy array, which is exactly the format Deepgram wants, so no conversion.
    stream = make_mic_stream(on_audio)

    # Authenticate with a header. Deepgram expects "Token <your key>".
    headers = {"Authorization": f"Token {api_key}"}

    print("Connecting to Deepgram...")
    async with connect(DEEPGRAM_URL, additional_headers=headers) as ws:
        print("Connected. Start talking. Press Ctrl+C to stop.\n")
        with stream:
            # Run sending and receiving at the same time. If either one finishes
            # or errors, we stop both.
            await asyncio.gather(sender(ws), receiver(ws))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Ctrl+C: drop the sentinel so sender() can close the stream cleanly.
        audio_queue.put(None)
        print("\nStopped.")
