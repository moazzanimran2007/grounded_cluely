# Grounded Cluely

A local meeting assistant for macOS. It listens to your live calls, both your voice
and the other people's, keeps a labeled running transcript, and answers questions
you type, grounded in your uploaded documents, the meeting so far, and the web.
Everything runs on your own machine.

## What it does

- **Listens to both sides.** Your microphone is captured as `[ME]`, the other
  participants' audio is captured straight off your Mac's system output as
  `[THEM]`, using a native Core Audio tap with no virtual audio driver to install.
- **Grounded answers.** Ask a question and it retrieves from your documents and the
  live transcript, tells you which source each answer came from, and falls back to
  a web search only when neither has the answer.
- **Meeting summary.** When you end a meeting it saves a timestamped summary
  (recap, decisions, action items) plus the full transcript to the `meetings/`
  folder.
- **Local browser UI.** The whole thing runs as a page at `127.0.0.1:8000`, served
  by a small local server. Nothing is hosted online.

## Requirements

- **macOS 14.4 or newer** (the Core Audio tap needs it). Developed on macOS 15.
- **Python 3.11+** (developed on 3.13).
- **Xcode or the Command Line Tools**, to compile the small Swift capture helper.
- A free **Deepgram** key (live transcription) and a free **Google Gemini** key
  (embeddings, answers, summaries, web search).

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/grounded-cluely.git
cd grounded-cluely

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-web.txt -r docrag/requirements.txt

sh native/build.sh          # compiles native/systemaudiotap (the system audio helper)

cp .env.example .env         # then open .env and paste your two API keys
```

(`requirements-dev.txt` for the tests and `requirements-overlay.txt` for the
optional PyQt overlay are not needed to run the app.)

## Run

```bash
source .venv/bin/activate
python server.py
```

Open **http://127.0.0.1:8000** in Safari or Chrome. Then:

1. Optionally load or upload a document to ground answers in.
2. Click **Start meeting**. The first run, macOS asks for Microphone permission,
   and the system audio helper needs the system audio recording permission granted
   to your terminal (System Settings, Privacy and Security).
3. Talk, and let the call play. **Use headphones** so your mic does not pick up the
   other side, which keeps the `[ME]` and `[THEM]` labels clean.
4. Ask questions in the panel. Each answer shows its source.
5. Click **End meeting** to stop and save the summary. Its file path is shown.

## Honest limitations

- **Screen capture hiding is best effort.** On macOS 15+ ScreenCaptureKit ignores
  the exclusion flag, so the window may still appear (or as a black box) under
  modern screen recording and apps like Zoom.
- **Free tier hiccups.** Gemini's free tier occasionally returns a transient
  "high demand" error. Just ask again.
- **Clean labels need headphones.** On speakers, the mic hears the call and
  mislabels it as your voice.

## How it was built (reference)

The sections below document each capability in the order it was built, with its own
verification steps. They are reference, not required reading to run the app.

## Task 1: listen and transcribe

A single script, `listen.py`, captures your microphone and streams it to Deepgram,
printing the transcript live in the terminal. No UI, no documents, no web search yet.
This proves the core listen-and-transcribe loop that everything else sits on.

### Setup (about 10 minutes)

1. Create a virtual environment and install the dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Get a Deepgram API key at https://console.deepgram.com. Signup gives you free
   credit and does not need a card.

3. Copy the example env file and paste your key into it:

   ```bash
   cp .env.example .env
   ```

   Then open `.env` and replace `your_key_here` with your real key.

4. Run it:

   ```bash
   python listen.py
   ```

   Talk into your mic. Your words appear as you speak, refining as Deepgram hears
   more, then locking in as final lines. Press Ctrl+C to stop.

### How it works

Three parts, kept deliberately small so you can read all of it:

- **`sounddevice`** grabs raw audio from the mic in quarter-second chunks.
- **A thread-safe queue** passes those chunks from the audio thread to the network
  code, so capturing audio never waits on the network.
- **A websocket** streams the chunks to Deepgram and reads JSON transcripts back.

Interim results print on a line that keeps overwriting itself; final results commit
to their own line.

## Verifying Task 1

There are two layers of tests, kept separate on purpose. One runs with no hardware
and proves the code is correct. The other needs you and your mic, and proves the
real thing works.

### Layer 1: automated tests (no mic, no network)

These run in well under a second against a real WAV fixture and a fake Deepgram,
so they never touch your microphone or the internet.

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt   # one time, installs pytest
python -m pytest
```

A passing result is the line `16 passed` at the bottom, all green. The four groups:

- **Audio capture** (`tests/test_audio_capture.py`): reading a real WAV yields
  frames at 16 kHz, mono, 16-bit, split into the expected chunk size, and the
  mic stream asks the device for that same shape. PASS: these assertions hold.
- **Transcription parsing** (`tests/test_transcription.py`): the code pulls the
  right text out of Deepgram messages, tells partial from final, and returns
  nothing (instead of crashing) on empty, malformed, or error messages.
- **Pipeline glue** (`tests/test_pipeline.py`): audio frames pushed in come out
  as bytes sent to the websocket, and transcript messages coming back come out as
  printed text. PASS: the expected words appear in captured stdout.
- **Config and keys** (`tests/test_config.py`): a missing or blank key stops the
  program with a clear message naming `DEEPGRAM_API_KEY` and `.env`, not a silent
  crash.

### Layer 2: manual acceptance checklist (you and your mic)

This runs the live pipeline in short bursts and walks you through the checks a
machine cannot judge. It needs your key in `.env`, a working mic, and a network
connection.

```bash
source .venv/bin/activate
python acceptance_check.py
```

It speaks plainly before each check about what a PASS looks like, then prints a
PASS or FAIL summary at the end. The four checks:

1. **Known sentence**: you say a fixed sentence, you confirm it appeared. PASS:
   your words show up as text.
2. **Latency**: it times the gap from "Speak now" to the first text and flags
   anything over 3 seconds, since the requirement is near real time. PASS: text
   starts within a second or two.
3. **Sustained 30 seconds**: you talk continuously and it confirms text was still
   arriving at the end. PASS: no freeze, transcription keeps up the whole time.
4. **Silence**: you stay quiet and it checks that no made up text appears. PASS:
   silence produces little or no text.

## Task 2: document context (RAG)

A separate module, `docrag/`, that answers questions grounded in a single
uploaded document. It is fully independent of the audio code. Later, the live
transcript can feed the same retrieve-and-answer step as just another source.

It uses Google Gemini on its free tier: `gemini-embedding-001` for embeddings and
`gemini-2.5-flash` for the answer. Anthropic has no first-party embeddings API
and OpenAI has no free tier, so Gemini is the cheapest single key that covers
both halves of a RAG loop. The vectors live in a small local JSON file per
document, not a hosted vector database, to keep it cheap and easy to inspect.

### Setup

```bash
source .venv/bin/activate
pip install -r docrag/requirements.txt   # pypdf, google-genai, numpy
```

Get a free key at https://aistudio.google.com/apikey and add it to `.env`:

```
GEMINI_API_KEY=your_key_here
```

### The six steps, each runnable on its own

The first two need no key, so you can see extraction and chunking before
spending anything.

```bash
# Step 1: extract text and preview it
python -m docrag extract docrag/sample/grounded_cluely_overview.txt

# Step 2: chunk it and inspect the chunks
python -m docrag chunk docrag/sample/grounded_cluely_overview.txt

# Step 3: embed and store (idempotent: re-running an unchanged file is a no-op)
python -m docrag index docrag/sample/grounded_cluely_overview.txt

# Step 4: see which chunks a question retrieves
python -m docrag retrieve docrag/sample/grounded_cluely_overview.txt "what sample rate is used?"

# Step 5: get a grounded answer
python -m docrag ask docrag/sample/grounded_cluely_overview.txt "what sample rate is used?"

# Step 6: run a list of known-answer questions and judge accuracy
python -m docrag demo docrag/sample/grounded_cluely_overview.txt docrag/sample/sample_questions.txt
```

### Judging accuracy (step 6)

`docrag/sample/sample_questions.txt` holds seven questions whose answers you can
check against the sample document. The last one, about a "payment network", is
not in the document on purpose. A correct grounded system answers the first six
from the text and replies "I could not find that in the document." to the last
one rather than inventing an answer.

To try it on your own file, point any command at a `.txt`, `.md`, or `.pdf` path
instead of the sample.

The pure logic (chunk overlap, cosine ranking) is also covered by no-key unit
tests in `tests/test_docrag.py`, which run in the same `python -m pytest` pass.

## Checking both tasks: the web interface

`server.py` is a small local web page that lets you exercise Task 1 and Task 2
by hand, side by side. It is a thin layer over the same `listen.py` and `docrag`
code, so it is a quick way to confirm both work.

```bash
source .venv/bin/activate
pip install -r requirements-web.txt   # fastapi, uvicorn, python-multipart
python server.py
```

Then open http://127.0.0.1:8000 in your browser. Two dots at the top right show
whether each API key is detected.

- **Task 1 panel**: click Start, allow microphone access the first time, and talk.
  Interim words appear in amber and refine, final lines lock in below. Click Stop
  when done. PASS: your speech shows up as text within a second or two.
- **Task 2 panel**: click Load sample document (or upload your own `.txt`, `.md`,
  or `.pdf`), then type a question. PASS: on-topic questions get a grounded
  answer with the source chunks shown beneath, and off-topic questions come back
  as "I could not find that in the document."

## What is next

- **System audio**, not just your mic. To transcribe what other people say on a
  call, macOS needs a virtual audio device like BlackHole to route call audio into
  `sounddevice`. That is the next step for the meeting listener.
- **Merge the two**: feed the live transcript into the docrag retrieve-and-answer
  step so questions can be grounded in the meeting plus your documents at once.
- **Web search** (Task 3) as the fallback source when neither the transcript nor
  the documents can answer.
