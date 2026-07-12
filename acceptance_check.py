"""
Manual acceptance checklist for Task 1.

The pytest suite proves the parts work in isolation. This script proves the
real thing works with your actual voice, mic, and network: the checks a
machine cannot judge for you. It runs the live pipeline for short bursts and
walks you through four checks, combining an automatic measurement with a
simple yes or no from you.

Run it with:  python acceptance_check.py
Talk when it tells you to. Press Ctrl+C any time to bail out.

What a PASS means is printed before each check, so you always know what you are
looking for.
"""

import asyncio
import queue
import time

from websockets.asyncio.client import connect

from listen import (
    DEEPGRAM_URL,
    make_mic_stream,
    receiver,
    require_api_key,
    sender,
)


# ----- the live session runner -------------------------------------------------

async def run_session(seconds, on_event, on_ready=None):
    """
    Run the real capture-and-transcribe loop for a fixed number of seconds.

    on_event is called with each Transcript as it arrives. on_ready, if given,
    is called the instant the mic is open and the connection is live, which is
    the right moment to tell you to start speaking.
    """
    api_key = require_api_key()
    headers = {"Authorization": f"Token {api_key}"}
    audio_q = queue.Queue()

    def on_frames(indata, frames, time_info, status):
        audio_q.put(bytes(indata))

    stream = make_mic_stream(on_frames)

    async with connect(DEEPGRAM_URL, additional_headers=headers) as ws:
        with stream:
            if on_ready:
                on_ready()
            send_task = asyncio.create_task(sender(ws, audio_q))
            recv_task = asyncio.create_task(receiver(ws, on_event))

            await asyncio.sleep(seconds)

            # Ask the sender to flush and stop, then give Deepgram a moment to
            # return any last final lines before we tear everything down.
            audio_q.put(None)
            await asyncio.sleep(0.8)

            for task in (recv_task, send_task):
                task.cancel()
            for task in (recv_task, send_task):
                try:
                    await task
                except asyncio.CancelledError:
                    pass


def record_session(seconds, on_ready=None, echo=True):
    """
    Run one session and return the list of (timestamp, Transcript) events it
    produced. timestamp is a monotonic clock reading in seconds.

    Returns (events, error). error is None on success, or a short string if the
    session could not run (bad key, no network, mic blocked).
    """
    events = []

    def on_event(t):
        events.append((time.monotonic(), t))
        if echo and t.is_final:
            print("   heard:", t.text)

    try:
        asyncio.run(run_session(seconds, on_event, on_ready=on_ready))
        return events, None
    except SystemExit as e:
        return events, str(e)
    except Exception as e:  # noqa: BLE001  (we want to report any failure plainly)
        return events, f"{type(e).__name__}: {e}"


# ----- small console helpers ---------------------------------------------------

def ask_yes_no(question):
    while True:
        answer = input(f"{question} [y/n] ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please type y or n.")


def wait_for_enter(message):
    input(message)


def banner(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ----- the four checks ---------------------------------------------------------

def check_known_sentence():
    banner("Check 1 of 4: a known sentence appears")
    print("PASS looks like: the sentence you speak shows up as text, roughly")
    print("word for word, in the lines below.")
    print()
    print("When you press Enter, wait a second for it to connect, then say")
    print("clearly:  the quick brown fox jumps over the lazy dog")
    wait_for_enter("Press Enter to start...")

    events, error = record_session(12)
    if error:
        print(f"\nCould not run the session: {error}")
        return False

    if not any(t.is_final for _, t in events):
        print("\nNo final transcript came back. That is a fail.")
        return False

    return ask_yes_no("\nDid your sentence appear correctly above?")


def check_latency():
    banner("Check 2 of 4: near real time latency")
    print("PASS looks like: text starts appearing within about a second or two")
    print("of you speaking. We flag anything over 3 seconds, since the")
    print("requirement is near real time.")
    print()
    print("When you press Enter it will connect, then print 'Speak now'. The")
    print("instant you see that, say the word: testing")
    wait_for_enter("Press Enter to start...")

    start = {}

    def on_ready():
        start["t0"] = time.monotonic()
        print("\n>>> Speak now: say 'testing' <<<\n")

    events, error = record_session(8, on_ready=on_ready, echo=False)
    if error:
        print(f"\nCould not run the session: {error}")
        return False

    if not events or "t0" not in start:
        print("\nNo transcript came back, so latency could not be measured. Fail.")
        return False

    first_event_time = events[0][0]
    latency = first_event_time - start["t0"]
    print(f"First text arrived about {latency:.1f} seconds after 'Speak now'.")
    print("(This includes your reaction time, so the real latency is a bit lower.)")

    within_target = latency <= 3.0
    if within_target:
        print("That is within the near real time target.")
    else:
        print("That is slower than the near real time target of a few seconds.")
    return within_target


def check_sustained():
    banner("Check 3 of 4: keeps going for 30 seconds")
    print("PASS looks like: it transcribes the whole time without freezing or")
    print("stopping. We confirm text was still arriving in the final stretch.")
    print()
    print("When you press Enter, talk continuously for 30 seconds. Read")
    print("anything nearby out loud if you run out of things to say.")
    wait_for_enter("Press Enter to start...")

    start = {}

    def on_ready():
        start["t0"] = time.monotonic()

    events, error = record_session(30, on_ready=on_ready)
    if error:
        print(f"\nCould not run the session: {error}")
        return False

    if not events or "t0" not in start:
        print("\nNo transcripts arrived at all. Fail.")
        return False

    t0 = start["t0"]
    times = [ts - t0 for ts, _ in events]
    total = len(events)
    last = max(times)
    print(f"\nGot {total} transcript updates. Last one arrived at about")
    print(f"{last:.0f} seconds into the 30 second run.")

    # Did not freeze early if we got a healthy number of updates and at least
    # one of them landed in the final 10 seconds.
    kept_going = total >= 5 and last >= 20.0
    if kept_going:
        print("Text kept arriving through to the end, so it did not freeze.")
    else:
        print("Updates stopped early, which suggests a freeze or a drop.")

    confirmed = ask_yes_no("Did the text keep up the whole time without freezing?")
    return kept_going and confirmed


def check_silence():
    banner("Check 4 of 4: silence stays quiet")
    print("PASS looks like: when you say nothing, no made up words appear. A")
    print("stray word or two from background noise is tolerable, a stream of")
    print("text is not.")
    print()
    print("When you press Enter, stay silent for about 8 seconds.")
    wait_for_enter("Press Enter to start...")

    events, error = record_session(8)
    if error:
        print(f"\nCould not run the session: {error}")
        return False

    noise = [t.text for _, t in events if t.is_final and t.text.strip()]
    if not noise:
        print("\nNo text was produced during silence. Clean pass.")
        return True

    print("\nSome text appeared during silence:")
    for line in noise:
        print(f"   {line}")
    return ask_yes_no("Was that little enough to call it a pass?")


# ----- runner ------------------------------------------------------------------

def main():
    print("Task 1 acceptance checklist")
    print("This needs your Deepgram key in .env, a working mic, and a network")
    print("connection. You will be asked to speak during each check.")

    checks = [
        ("Known sentence appears", check_known_sentence),
        ("Near real time latency", check_latency),
        ("Sustained for 30 seconds", check_sustained),
        ("Silence stays quiet", check_silence),
    ]

    results = []
    for name, fn in checks:
        results.append((name, fn()))

    banner("Results")
    all_passed = True
    for name, passed in results:
        mark = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{mark}]  {name}")

    print()
    if all_passed:
        print("All manual checks passed. Task 1 is verified end to end.")
    else:
        print("One or more checks failed. See the notes above each result.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nChecklist stopped.")
