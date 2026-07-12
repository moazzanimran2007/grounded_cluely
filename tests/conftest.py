"""
Shared test helpers and fixtures.

conftest.py is a special pytest file: anything defined here is available to
every test in this folder without importing it.
"""

import json
import os

import pytest


@pytest.fixture
def wav_path():
    """Absolute path to the real WAV fixture used by the audio tests."""
    return os.path.join(os.path.dirname(__file__), "fixtures", "sample_speech.wav")


def results_message(text, is_final):
    """
    Build a JSON string shaped like a real Deepgram "Results" message. Used by
    the transcription and pipeline tests so we never need a live connection.
    """
    return json.dumps(
        {
            "type": "Results",
            "is_final": is_final,
            "channel": {"alternatives": [{"transcript": text}]},
        }
    )


class FakeWebSocket:
    """
    A stand-in for the Deepgram websocket.

    It records everything our code sends (so a test can confirm audio frames
    went out) and, when iterated with "async for", yields a fixed list of
    canned messages (so our receiver has something to parse). This is the whole
    trick that lets the pipeline run with no network.
    """

    def __init__(self, incoming_messages=None):
        self._incoming = list(incoming_messages or [])
        self.sent = []

    async def send(self, data):
        self.sent.append(data)

    def __aiter__(self):
        self._iter = iter(self._incoming)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration
