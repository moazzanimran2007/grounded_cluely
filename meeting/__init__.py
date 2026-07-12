"""
meeting: capture and transcribe more than one audio source at once.

This package adds system audio (what other people say in a call) on top of the
mic transcription that listen.py already does. It does not reimplement the
Deepgram path. It reuses listen.py's sender, receiver, and connection, and only
adds two things:

  system_source.py  launches the Swift helper and reads its audio frames
  session.py        runs one audio source through the shared Deepgram path
  step2_system.py   step 2: transcribe system audio on its own
  step3_two_streams transcribe mic and system at once, labeled [ME] and [THEM]
"""
