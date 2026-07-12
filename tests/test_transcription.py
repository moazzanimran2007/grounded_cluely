"""
Automated test 2: transcription client parsing.

We feed parse_transcript the kinds of messages Deepgram actually sends and
confirm it pulls out the right text, tells partial from final, and never
crashes on empty, malformed, or error messages.
"""

import json

import listen
from conftest import results_message


def test_parses_final_result():
    t = listen.parse_transcript(results_message("hello world", is_final=True))
    assert t is not None
    assert t.text == "hello world"
    assert t.is_final is True


def test_parses_interim_result_as_not_final():
    t = listen.parse_transcript(results_message("hello", is_final=False))
    assert t is not None
    assert t.text == "hello"
    assert t.is_final is False


def test_empty_transcript_returns_none():
    # Deepgram emits empty Results during silence. Those carry no words, so we
    # drop them rather than printing blank lines.
    assert listen.parse_transcript(results_message("", is_final=True)) is None


def test_non_results_message_returns_none():
    # Metadata and keepalive messages are not transcripts.
    metadata = json.dumps({"type": "Metadata", "request_id": "abc"})
    assert listen.parse_transcript(metadata) is None


def test_error_payload_returns_none_without_crashing():
    error = json.dumps({"type": "Error", "description": "something went wrong"})
    assert listen.parse_transcript(error) is None


def test_malformed_json_returns_none_without_crashing():
    assert listen.parse_transcript("this is not json {{{") is None


def test_missing_fields_return_none_without_crashing():
    # A Results message with no alternatives should not raise a KeyError.
    broken = json.dumps({"type": "Results", "channel": {"alternatives": []}})
    assert listen.parse_transcript(broken) is None
