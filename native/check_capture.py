"""
Step 1 verifier for the Swift system audio helper.

It runs ./systemaudiotap for a few seconds, reads the raw PCM it emits, and
measures the signal level. Frames flowing is not enough: macOS hands back silent
frames until you grant the system audio recording permission, so this checks the
actual loudness, not just that bytes arrived.

Run it, and while it counts down, play any audio on your Mac (a video, music):

    python native/check_capture.py
"""

import os
import signal
import subprocess
import sys
import threading
import time

import numpy as np

HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "systemaudiotap")
SECONDS = 6


def main():
    if not os.path.exists(HELPER):
        print("Helper not built. Compile it first (see native/build.sh).")
        sys.exit(1)

    proc = subprocess.Popen([HELPER], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Collect the binary audio on one thread and echo the helper's diagnostics on
    # another, so neither blocks the timed capture.
    audio = bytearray()

    def read_audio():
        while True:
            block = proc.stdout.read(4096)
            if not block:
                break
            audio.extend(block)

    def read_errors():
        for line in proc.stderr:
            sys.stderr.write("[helper] " + line.decode(errors="replace"))

    threading.Thread(target=read_audio, daemon=True).start()
    threading.Thread(target=read_errors, daemon=True).start()

    print(f"Capturing for {SECONDS} seconds. Play some audio on your Mac NOW.")
    time.sleep(SECONDS)

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()

    samples = np.frombuffer(bytes(audio), dtype=np.int16)
    if len(samples) == 0:
        print("\nFAIL: no audio frames arrived at all.")
        print("The device never started. Make sure some audio was actually playing.")
        return

    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    print(f"\nCaptured {len(samples)} samples, signal level (rms): {rms:.1f}")

    if rms > 30:
        print("PASS: real system audio was captured. Step 1 works.")
    else:
        print("SILENT: frames flowed but every sample is silence.")
        print("This almost always means the system audio recording permission is")
        print("not granted to the app you ran this from (usually Terminal).")
        print("Fix it: System Settings, Privacy and Security. Enable your terminal")
        print("under Microphone, and under Screen and System Audio Recording if shown.")
        print("Then fully quit and reopen the terminal and run this again.")


if __name__ == "__main__":
    main()
