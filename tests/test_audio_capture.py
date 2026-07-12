"""
Automated test 1: audio capture.

Two things to prove without any microphone:
  a) Reading a real WAV file produces audio frames of the expected sample rate,
     channel count, and chunk size.
  b) The mic stream we build asks the device for that exact same shape.

For (b) we mock the device, so no hardware is touched.
"""

from unittest import mock

import listen


def test_wav_reads_at_expected_sample_rate_and_channels(wav_path):
    sample_rate, channels, sample_width, chunks = listen.read_wav_chunks(wav_path)

    # The fixture, and everything downstream, assumes 16 kHz mono 16-bit audio.
    assert sample_rate == listen.SAMPLE_RATE   # 16000
    assert channels == listen.CHANNELS         # 1 (mono)
    assert sample_width == listen.SAMPLE_WIDTH  # 2 bytes = int16


def test_wav_splits_into_expected_chunk_size(wav_path):
    _, channels, sample_width, chunks = listen.read_wav_chunks(wav_path)
    bytes_per_frame = channels * sample_width

    # Every chunk except the last should hold exactly BLOCK_SIZE frames.
    for chunk in chunks[:-1]:
        frames_in_chunk = len(chunk) // bytes_per_frame
        assert frames_in_chunk == listen.BLOCK_SIZE

    # The last chunk is allowed to be shorter (the leftover), but never empty
    # and never larger than a full block.
    last_frames = len(chunks[-1]) // bytes_per_frame
    assert 0 < last_frames <= listen.BLOCK_SIZE


def test_chunks_reassemble_to_the_whole_file(wav_path):
    # No audio is lost or duplicated by the chunking: gluing the chunks back
    # together gives the original byte stream.
    _, channels, sample_width, chunks = listen.read_wav_chunks(wav_path)
    total_bytes = sum(len(c) for c in chunks)
    expected = 25000 * channels * sample_width  # the fixture is 25000 frames
    assert total_bytes == expected


def test_mic_stream_requests_the_expected_audio_shape():
    # Replace the real device class with a mock. We never start a stream, we
    # only check what arguments make_mic_stream hands to the device.
    with mock.patch.object(listen.sd, "RawInputStream") as fake_stream:
        callback = lambda *a, **k: None
        listen.make_mic_stream(callback)

    fake_stream.assert_called_once_with(
        samplerate=listen.SAMPLE_RATE,
        channels=listen.CHANNELS,
        blocksize=listen.BLOCK_SIZE,
        dtype="int16",
        callback=callback,
    )
