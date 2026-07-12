"""
Automated test 4: config and keys.

A missing API key must fail loudly with a clear message, not crash somewhere
deep in the network code with a confusing traceback.
"""

import pytest

import listen


def test_missing_key_exits_with_a_clear_message():
    # An empty environment means no key. require_api_key should stop the program
    # with SystemExit and a message that names the thing to fix.
    with pytest.raises(SystemExit) as exit_info:
        listen.require_api_key(env={})

    message = str(exit_info.value)
    assert "DEEPGRAM_API_KEY" in message
    assert ".env" in message


def test_blank_key_is_treated_as_missing():
    # A key set to an empty string is just as broken as no key at all.
    with pytest.raises(SystemExit):
        listen.require_api_key(env={"DEEPGRAM_API_KEY": ""})


def test_present_key_is_returned():
    key = listen.require_api_key(env={"DEEPGRAM_API_KEY": "test-key-123"})
    assert key == "test-key-123"
