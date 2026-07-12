"""
Guided verification for step 3: the two labeled streams.

This runs the full two stream pipeline live (your mic as [ME], system audio as
[THEM]) and walks you through four manual stages, pausing between each, with
clear PASS or FAIL guidance. Before any manual stage it runs an automated sanity
layer that confirms both capture streams are actually alive and emitting frames,
so that a blank transcript later means nobody talked, not that a stream died.

Run from the project root:
    python -m meeting.verify_step3

Important for clean results: if your microphone can hear your speakers, it will
pick up the video audio and tag it [ME], which looks like a mislabel but is
really an echo. Use headphones for the cleanest separation.

Overall PASS means all of these:
  - Stage 1: your speech is tagged [ME], nothing is tagged [THEM].
  - Stage 2: the video is tagged [THEM], nothing is tagged [ME].
  - Stage 3: both labels are correct and the lines are in roughly correct time
    order, not bunched or reordered.
  - Stage 4: silence produces no garbage lines.
  - Sanity: both streams were confirmed alive and emitting frames.
"""

import queue
import threading
import time

import asyncio

from listen import SAMPLE_RATE, make_mic_stream, require_api_key

from .session import transcribe_stream
from .system_source import SystemAudioSource

# Shared, time stamped final transcript lines, filled by the live pipeline thread
# and read by the stage flow on the main thread.
events = []
events_lock = threading.Lock()

# Set if the background pipeline dies (for example a bad key or a dropped
# Deepgram connection), so we can report it instead of showing a silent blank.
pipeline_error = None


def record(label, transcript):
    """Append one final transcript line, tagged with its stream label."""
    if not transcript.is_final or not transcript.text.strip():
        return
    with events_lock:
        events.append(
            {"t": time.monotonic(), "label": label, "text": transcript.text}
        )


def events_since(start_time):
    with events_lock:
        return [e for e in events if e["t"] >= start_time]


def run_pipeline(mic_q, system_q, system_rate):
    """
    Run both transcription streams at once until each queue receives a None.
    Lives on a background thread so the main thread can prompt you between
    stages while transcription keeps flowing.
    """
    global pipeline_error

    async def both():
        await asyncio.gather(
            transcribe_stream(mic_q, SAMPLE_RATE, lambda t: record("ME", t)),
            transcribe_stream(system_q, system_rate, lambda t: record("THEM", t)),
        )

    try:
        asyncio.run(both())
    except BaseException as error:  # noqa: BLE001  report any failure plainly
        pipeline_error = error


# ----- small console helpers ---------------------------------------------------

def banner(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def ask_yes_no(question):
    while True:
        answer = input(f"{question} [y/n] ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please type y or n.")


def show(captured):
    if not captured:
        print("   (no transcript lines captured)")
        return
    base = captured[0]["t"]
    for e in captured:
        offset = e["t"] - base
        print(f"   +{offset:5.1f}s  [{e['label']}]  {e['text']}")


def countdown(seconds, message):
    for remaining in range(seconds, 0, -1):
        print(f"\r{message} {remaining:2d}s ", end="", flush=True)
        time.sleep(1)
    print("\r" + " " * 60 + "\r", end="")


def drain(q):
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


# ----- automated sanity layer --------------------------------------------------

def sanity_check(mic_q, system):
    """
    Confirm both streams are alive and delivering frames. Returns True to
    continue, False to stop. Fails fast with a clear message if either is dead.
    """
    print("Sanity check: confirming both capture streams are alive...")
    mic_before = mic_q.qsize()
    system_before = system.audio_q.qsize()
    time.sleep(3)
    mic_frames = mic_q.qsize() - mic_before
    system_frames = system.audio_q.qsize() - system_before

    ok = True

    if mic_frames > 0:
        print(f"  mic stream: ALIVE, {mic_frames} audio chunks in 3 seconds.")
    else:
        print("  mic stream: DEAD, no audio chunks arrived.")
        print("  Check microphone permission for your terminal and that a mic exists.")
        ok = False

    if system.proc.poll() is not None:
        print("  system helper: DEAD, the helper process has exited.")
        print("  Rebuild it with sh native/build.sh and confirm the audio permission.")
        ok = False
    elif system_frames > 0:
        print(f"  system helper: ALIVE, {system_frames} audio chunks in 3 seconds.")
    else:
        print("  system helper: process is alive but emitted no frames yet.")
        print("  That is normal only if nothing has played audio. It will deliver")
        print("  frames as soon as audio plays, so the stages below will exercise it.")

    return ok


# ----- the four manual stages --------------------------------------------------

# How long to keep collecting after a stage's action ends, so Deepgram's final
# lines (which arrive about two seconds late) land in the stage that produced
# them instead of spilling into the next one.
SETTLE_SECONDS = 3


def capture_window(seconds, action_message):
    """
    Capture for a fixed number of seconds, then wait SETTLE_SECONDS more for late
    final lines. Returns the lines that arrived across the whole window. This is
    what keeps each stage's words inside that stage.
    """
    start = time.monotonic()
    countdown(seconds, action_message)
    countdown(SETTLE_SECONDS, "Waiting for the last lines to arrive...")
    return events_since(start)


def stage1_mic_only():
    banner("STAGE 1 of 4: mic only, expect [ME]")
    print("PASS looks like: the sentence you speak appears tagged [ME], and")
    print("nothing is tagged [THEM].")
    print()
    print("Keep everything else silent: no video, no music.")
    input("Press Enter, then say: the quick brown fox jumps over the lazy dog ")
    captured = capture_window(8, "Speak now...")

    print("\nLines captured this stage:")
    show(captured)
    me_lines = [e for e in captured if e["label"] == "ME"]
    them_lines = [e for e in captured if e["label"] == "THEM"]
    if not me_lines:
        print("\nNo [ME] lines appeared, so the mic was not transcribed. That is a FAIL.")
        return False
    if them_lines:
        print("\nNote: some lines were tagged [THEM] while only your mic was active.")
        print("On speakers that is your mic hearing an echo, not a label bug.")
    return ask_yes_no("Did your sentence appear tagged [ME] with nothing wrongly [THEM]?")


def stage2_system_only():
    banner("STAGE 2 of 4: system only, expect [THEM]")
    print("PASS looks like: the video's speech appears tagged [THEM], and")
    print("nothing is tagged [ME].")
    print()
    print("Start a YouTube video with clear speech, and do not talk yourself.")
    input("Press Enter once the video is playing... ")
    captured = capture_window(15, "Capturing the video...")

    print("\nLines captured this stage:")
    show(captured)
    me_lines = [e for e in captured if e["label"] == "ME"]
    them_lines = [e for e in captured if e["label"] == "THEM"]
    if not them_lines:
        print("\nNo [THEM] lines appeared, so system audio was not transcribed. FAIL.")
        return False
    if me_lines:
        print("\nNote: some lines were tagged [ME] while only the video played.")
        print("On speakers that is your mic hearing the video. Headphones fix it.")
    return ask_yes_no("Did the video appear tagged [THEM] with nothing wrongly [ME]?")


def stage3_both():
    banner("STAGE 3 of 4: both at once, expect correct labels and time order")
    print("PASS looks like: your words are [ME], the video is [THEM], and the")
    print("lines are in roughly the order things were said, not bunched or jumbled.")
    print()
    print("Play the video AND talk over it for about 20 seconds.")
    input("Press Enter to start, then talk over the video... ")
    captured = capture_window(20, "Capturing both. Talk over the video...")

    print("\nLines captured this stage (offset shows arrival time):")
    show(captured)
    me_lines = [e for e in captured if e["label"] == "ME"]
    them_lines = [e for e in captured if e["label"] == "THEM"]
    if not me_lines or not them_lines:
        print("\nExpected both [ME] and [THEM] lines. Missing one means a stream")
        print("did not transcribe during the overlap. That is a FAIL.")
        return False
    return ask_yes_no(
        "Are your words [ME], the video [THEM], and the order roughly right?"
    )


def stage4_silence():
    banner("STAGE 4 of 4: silence, expect no garbage")
    print("PASS looks like: during silence, no made up lines and no random labels.")
    print()
    print("Stop the video and stay completely silent.")
    input("Press Enter, then be silent... ")
    captured = capture_window(10, "Listening to silence...")

    if not captured:
        print("\nNo lines were produced during silence. Clean PASS.")
        return True
    print("\nSome lines appeared during silence:")
    show(captured)
    return ask_yes_no("Was that little enough to count as no garbage?")


# ----- runner ------------------------------------------------------------------

def main():
    print("Step 3 verification: two labeled streams, [ME] and [THEM].")
    print("You will be asked to speak and to play a video. Headphones recommended.\n")

    # Fail fast and clearly if the Deepgram key is missing.
    require_api_key()

    # Start the mic stream and the system helper. Both fill their own queue.
    mic_q = queue.Queue()

    def on_mic_frames(indata, frames, time_info, status):
        mic_q.put(bytes(indata))

    mic_stream = make_mic_stream(on_mic_frames)
    system = SystemAudioSource()
    system_rate = system.start()
    mic_stream.start()

    # Automated sanity layer before any manual stage.
    if not sanity_check(mic_q, system):
        print("\nFAIL: a capture stream is not running. Fix the above and rerun.")
        mic_stream.stop()
        mic_stream.close()
        system.stop()
        return

    print(f"\nBoth streams alive. Mic at {SAMPLE_RATE} Hz, system at {system_rate} Hz.")

    # Clear the few seconds of audio gathered during the sanity wait so each
    # stage starts fresh, then start transcribing in the background.
    drain(mic_q)
    drain(system.audio_q)
    pipeline = threading.Thread(
        target=run_pipeline,
        args=(mic_q, system.audio_q, system_rate),
        daemon=True,
    )
    pipeline.start()
    time.sleep(1.5)  # let both Deepgram connections open

    if pipeline_error is not None:
        print(f"\nFAIL: the transcription pipeline could not start: {pipeline_error}")
        mic_stream.stop()
        mic_stream.close()
        system.stop()
        return

    # Run the four manual stages.
    results = [
        ("Stage 1, mic tagged [ME]", stage1_mic_only()),
        ("Stage 2, system tagged [THEM]", stage2_system_only()),
        ("Stage 3, both correct and in order", stage3_both()),
        ("Stage 4, silence stays clean", stage4_silence()),
    ]

    # Tear down.
    mic_q.put(None)
    system.audio_q.put(None)
    system.stop()
    mic_stream.stop()
    mic_stream.close()
    pipeline.join(timeout=5)

    # Summary.
    banner("Results")
    all_passed = True
    for name, passed in results:
        mark = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{mark}]  {name}")

    print()
    if all_passed:
        print("Overall PASS. Labels are correct, time order holds, silence is clean,")
        print("and both streams were confirmed alive. Step 3 is verified.")
    else:
        print("Overall FAIL. See the stage notes above for what to fix.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nVerification stopped.")
