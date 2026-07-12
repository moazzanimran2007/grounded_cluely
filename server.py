"""
A small local web interface to check that Task 1 and Task 2 work.

Run it with:  python server.py
Then open http://127.0.0.1:8000 in your browser.

Task 1 panel: click Start, talk, and watch the transcript appear. The server
captures this machine's mic, streams it to Deepgram, and pushes the text back to
the browser over a websocket.

Task 2 panel: load the sample document (or upload your own), then ask questions
and see the grounded answer plus the chunks it was built from.

This file is a thin layer. All the real work lives in listen.py (Task 1) and the
docrag package (Task 2). The interface just wires them to a web page.
"""

import asyncio
import os
import queue
import time

import uvicorn
from fastapi import Body, FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from websockets.asyncio.client import connect

# Task 1 pieces.
from listen import (
    DEEPGRAM_URL,
    make_mic_stream,
    parse_transcript,
    require_api_key,
    sender,
)

# Task 2 pieces.
from docrag.answer import answer
from docrag.chunk import chunk_text
from docrag.extract import extract_text
from docrag.retrieve import retrieve
from docrag.store import build_index, load_index

# Meeting assistant pieces (the main local web app). None of these are modified
# here, the server only drives them.
from meeting.assistant import ensure_doc_indexed
from meeting.capture import LiveCapture
from meeting.query import answer_for_ui
from meeting.summary import save_meeting
from meeting.transcript_memory import TranscriptMemory

HERE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(HERE, "uploads")
SAMPLE_DOC = os.path.join(HERE, "docrag", "sample", "grounded_cluely_overview.txt")

app = FastAPI()


# ----- the page ----------------------------------------------------------------

def _serve(page):
    with open(os.path.join(HERE, "web", page), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/")
async def home():
    # The meeting assistant is the main app.
    return _serve("meeting.html")


@app.get("/demo")
async def demo():
    # The original Task 1 and Task 2 check page, kept for reference.
    return _serve("index.html")


@app.get("/api/status")
async def status():
    """Tell the page which keys are present so it can show readiness."""
    return {
        "deepgram": bool(os.getenv("DEEPGRAM_API_KEY")),
        "gemini": bool(os.getenv("GEMINI_API_KEY")),
    }


# ----- Task 2: document question answering -------------------------------------

def _index_file(path):
    """Extract, chunk, and embed a file. Returns (record, did_embed)."""
    chunks = chunk_text(extract_text(path))
    return build_index(path, chunks)


@app.post("/api/use-sample")
async def use_sample():
    # Indexing does blocking work (file read plus an embedding call), so run it
    # off the event loop with to_thread to keep the server responsive.
    record, did_embed = await asyncio.to_thread(_index_file, SAMPLE_DOC)
    return {
        "file": SAMPLE_DOC,
        "name": os.path.basename(SAMPLE_DOC),
        "chunks": len(record["chunks"]),
        "reused": not did_embed,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        f.write(await file.read())
    try:
        record, did_embed = await asyncio.to_thread(_index_file, dest)
    except SystemExit as e:
        # Unsupported file type and similar are raised as SystemExit with a
        # clear message. Surface it instead of crashing the request.
        return JSONResponse({"error": str(e)}, status_code=400)
    return {
        "file": dest,
        "name": file.filename,
        "chunks": len(record["chunks"]),
        "reused": not did_embed,
    }


@app.post("/api/ask")
async def ask(payload: dict = Body(...)):
    file = payload.get("file")
    question = (payload.get("question") or "").strip()
    if not file or not question:
        return JSONResponse(
            {"error": "Load a document and type a question first."}, status_code=400
        )

    try:
        if load_index(file) is None:
            await asyncio.to_thread(_index_file, file)
        results = await asyncio.to_thread(retrieve, file, question)
        reply = await asyncio.to_thread(answer, question, results)
    except Exception as e:  # noqa: BLE001  surface API errors (like rate limits)
        return JSONResponse({"error": str(e)}, status_code=500)

    return {
        "answer": reply,
        "chunks": [
            {"id": r["id"], "score": round(r["score"], 3), "text": r["text"]}
            for r in results
        ],
    }


# ----- Task 1: live transcription over a websocket -----------------------------

async def _forward_transcripts(deepgram_ws, browser_ws):
    """Read Deepgram messages, parse them, and push transcripts to the browser."""
    async for message in deepgram_ws:
        t = parse_transcript(message)
        if t:
            await browser_ws.send_json(
                {"type": "transcript", "text": t.text, "is_final": t.is_final}
            )


async def _wait_for_stop(browser_ws):
    """Return when the browser asks to stop or disconnects."""
    try:
        while True:
            msg = await browser_ws.receive_text()
            if msg == "stop":
                return
    except WebSocketDisconnect:
        return


@app.websocket("/ws/transcribe")
async def transcribe(browser_ws: WebSocket):
    await browser_ws.accept()

    try:
        api_key = require_api_key()
    except SystemExit as e:
        await browser_ws.send_json({"type": "error", "message": str(e)})
        await browser_ws.close()
        return

    # The mic callback drops raw audio bytes into this queue; the sender drains
    # it. Same pattern as the standalone listener.
    audio_q = queue.Queue()

    def on_frames(indata, frames, time_info, status_flags):
        audio_q.put(bytes(indata))

    stream = make_mic_stream(on_frames)
    headers = {"Authorization": f"Token {api_key}"}

    try:
        async with connect(DEEPGRAM_URL, additional_headers=headers) as deepgram_ws:
            await browser_ws.send_json({"type": "status", "message": "listening"})
            with stream:
                send_task = asyncio.create_task(sender(deepgram_ws, audio_q))
                forward_task = asyncio.create_task(
                    _forward_transcripts(deepgram_ws, browser_ws)
                )
                control_task = asyncio.create_task(_wait_for_stop(browser_ws))

                # Run until the browser stops us or Deepgram closes the stream.
                await asyncio.wait(
                    {forward_task, control_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                audio_q.put(None)  # let the sender flush and close cleanly
                for task in (send_task, forward_task, control_task):
                    task.cancel()
    except Exception as e:  # noqa: BLE001  report connection or mic failures
        try:
            await browser_ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await browser_ws.close()
        except Exception:
            pass


# ----- the meeting assistant (the main local web app) --------------------------

class MeetingSession:
    """
    Holds the one active meeting. A local app has a single user, so one session
    is enough. It owns the transcript memory and the live capture, plus a queue
    of new lines for the websocket to push to the page as they are spoken.

    Concurrency note: LiveCapture runs audio on its own background thread. The
    on_line callback fires there, drops the line into a thread safe queue, and the
    websocket coroutine drains that queue. So the audio thread and the web event
    loop never block each other, the same pattern used everywhere else.
    """

    def __init__(self):
        self.memory = None
        self.capture = None
        self.doc_path = None
        self.line_queue = None
        self.active = False

    def start(self, doc_path):
        if self.active:
            return {"already_running": True}
        self.memory = TranscriptMemory()
        self.line_queue = queue.Queue()
        self.doc_path = doc_path or None

        def on_line(label, text):
            self.memory.add(label, text)
            self.line_queue.put(
                {"label": label, "text": text, "clock": time.strftime("%H:%M:%S")}
            )

        self.capture = LiveCapture(on_line=on_line)
        status = self.capture.start()
        self.active = True
        status["doc"] = os.path.basename(self.doc_path) if self.doc_path else None
        return status

    def stop(self):
        """Stop capture. Touches sounddevice, so must run on the main thread."""
        if not self.active:
            return False
        self.active = False
        self.capture.stop()
        if self.line_queue is not None:
            self.line_queue.put(None)  # unblock the websocket loop
        return True

    def summarize(self):
        """Write and read back the summary. Network only, safe off the loop."""
        path = save_meeting(self.memory)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            content = "(summary file could not be read back)"
        return {"path": path, "content": content}


SESSION = MeetingSession()


@app.post("/api/meeting/start")
async def meeting_start(payload: dict = Body(default={})):
    doc = (payload or {}).get("doc")
    if doc and load_index(doc) is None:
        # Indexing is network only (embeddings), so it is safe off the loop.
        await asyncio.to_thread(ensure_doc_indexed, doc)
    # IMPORTANT: capture uses sounddevice, whose start() must run on the main
    # thread on macOS or it hangs. uvicorn runs this handler on the main thread,
    # so we call start() directly here rather than via a worker thread. It blocks
    # the loop for a second or two while capture warms up, which is fine at the
    # start of a meeting.
    status = SESSION.start(doc)
    return status


@app.post("/api/meeting/ask")
async def meeting_ask(payload: dict = Body(...)):
    question = (payload.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "Type a question first."}, status_code=400)
    if SESSION.memory is None:
        return JSONResponse({"error": "Start a meeting first."}, status_code=400)
    try:
        answer_text, sources = await asyncio.to_thread(
            answer_for_ui, question, SESSION.doc_path, SESSION.memory
        )
    except Exception as e:  # noqa: BLE001  surface API errors like rate limits
        return JSONResponse({"error": str(e)}, status_code=500)
    return {"answer": answer_text, "sources": sources}


@app.post("/api/meeting/end")
async def meeting_end():
    # Stop capture on the main thread (sounddevice again), then run the summary
    # network call off the loop so it does not freeze the page.
    if not SESSION.stop():
        return JSONResponse({"error": "No active meeting."}, status_code=400)
    result = await asyncio.to_thread(SESSION.summarize)
    return result


@app.websocket("/ws/meeting")
async def meeting_ws(browser_ws: WebSocket):
    await browser_ws.accept()
    loop = asyncio.get_event_loop()

    # Send any transcript that already exists, so a page reload shows history.
    if SESSION.memory is not None:
        for row in SESSION.memory.lines():
            await browser_ws.send_json(
                {"type": "line", "label": row["label"], "text": row["text"], "clock": row["clock"]}
            )

    try:
        while True:
            q = SESSION.line_queue
            if q is None:
                await asyncio.sleep(0.2)
                continue
            line = await loop.run_in_executor(None, q.get)
            if line is None:  # the meeting ended
                await browser_ws.send_json({"type": "ended"})
                break
            await browser_ws.send_json({"type": "line", **line})
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        try:
            await browser_ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    print("Open http://127.0.0.1:8000 in your browser.")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
