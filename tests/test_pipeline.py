"""
Automated test 3: pipeline glue.

With the audio source and the API both faked, prove that audio frames flowing
IN come out as bytes sent to the websocket, and that transcript messages coming
back come OUT as printed text. This is the end to end loop minus the hardware
and the network.
"""

import asyncio
import json
import queue

import listen
from conftest import FakeWebSocket, results_message


def test_audio_frames_flow_in_and_transcript_text_flows_out(capsys):
    # Messages the fake Deepgram will "send" back to us, in order:
    incoming = [
        json.dumps({"type": "Metadata"}),          # ignored, not a transcript
        results_message("hello", is_final=False),    # interim guess
        results_message("hello world", is_final=True),  # final line
    ]
    ws = FakeWebSocket(incoming)

    # Two real audio chunks go in, then None to tell the sender to stop.
    audio_q = queue.Queue()
    audio_q.put(b"\x01\x02" * 200)
    audio_q.put(b"\x03\x04" * 200)
    audio_q.put(None)

    async def run():
        # sender drains the audio queue into the fake ws; receiver reads the
        # canned messages back out and prints them. Run them together.
        await asyncio.gather(
            listen.sender(ws, audio_q),
            listen.receiver(ws),
        )

    asyncio.run(run())

    # OUT side: the final transcript text reached stdout.
    printed = capsys.readouterr().out
    assert "hello world" in printed

    # IN side: both audio chunks were forwarded to the websocket as binary,
    # and the sender signed off with a CloseStream message.
    binary_frames = [s for s in ws.sent if isinstance(s, (bytes, bytearray))]
    assert len(binary_frames) == 2
    assert any("CloseStream" in s for s in ws.sent if isinstance(s, str))


def test_empty_and_error_messages_do_not_break_the_pipeline(capsys):
    # A run that is mostly junk: empty transcripts, an error, metadata. The one
    # real line should still print, and nothing should raise.
    incoming = [
        results_message("", is_final=False),
        json.dumps({"type": "Error", "description": "bad"}),
        results_message("", is_final=True),
        results_message("the only real words", is_final=True),
    ]
    ws = FakeWebSocket(incoming)

    audio_q = queue.Queue()
    audio_q.put(None)  # no audio, just close immediately

    async def run():
        await asyncio.gather(listen.sender(ws, audio_q), listen.receiver(ws))

    asyncio.run(run())

    printed = capsys.readouterr().out
    assert "the only real words" in printed
