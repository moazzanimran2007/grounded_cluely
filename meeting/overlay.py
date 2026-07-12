"""
A floating, always on top overlay front end for the meeting assistant.

This is a NEW front end. It does not change capture, retrieval, transcript
memory, summarization, or the query logic. It starts the same LiveCapture, reads
the same TranscriptMemory, asks questions through the same unified retrieval, and
saves a summary on exit through the same save_meeting. The overlay is only a
window around all of that.

Three threads cooperate here, and they only ever talk through Qt signals or the
lock protected transcript memory:
  - the Qt UI thread runs the window and the event loop,
  - LiveCapture's background thread keeps capturing and transcribing audio,
  - a short lived worker thread runs each question's network calls so the UI
    never freezes while an answer is being fetched.
Because the answer work runs off the UI thread, and the audio runs on its own
thread, typing a question never blocks audio capture.

Run from the project root:
    python -m meeting.overlay --doc path/to/document.pdf
    python -m meeting.overlay

Hotkeys (global, work even when the overlay is not focused):
    Cmd+Shift+J   show or hide the overlay
    Cmd+Shift+K   expand or collapse the panel
"""

import argparse
import html
import sys
import threading

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from pynput import keyboard

from .assistant import ensure_doc_indexed
from .capture import LiveCapture
from .summary import save_meeting
from .transcript_memory import TranscriptMemory
from .query import answer_for_ui

# Honest one line note shown in the UI about screen capture hiding.
CAPTURE_NOTE = (
    "Capture hiding is best effort. On macOS 15 and newer, screen recording and "
    "apps like Zoom can still see this panel, sometimes as a black box."
)


class Overlay(QWidget):
    # Signals emitted from worker or hotkey threads, delivered to the UI thread.
    answer_ready = pyqtSignal(str, list)
    answer_failed = pyqtSignal(str)
    hotkey_toggle_visibility = pyqtSignal()
    hotkey_toggle_expand = pyqtSignal()

    def __init__(self, memory, doc_path):
        super().__init__()
        self.memory = memory
        self.doc_path = doc_path
        self.capture = None       # set by main after start
        self._hotkeys = None
        self._cleaned = False
        self._drag_offset = None
        self.expanded = True

        # Frameless tool window that floats above other apps. Tool keeps it out
        # of the dock and the app switcher.
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._build_ui()

        self.answer_ready.connect(self._show_answer)
        self.answer_failed.connect(self._show_error)
        self.hotkey_toggle_visibility.connect(self.toggle_visibility)
        self.hotkey_toggle_expand.connect(self.toggle_expand)

        self._set_expanded(True)

    # ----- building the window -------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # A styled container gives us rounded corners over the transparent window.
        self.panel = QFrame()
        self.panel.setObjectName("panel")
        outer.addWidget(self.panel)

        root = QVBoxLayout(self.panel)
        root.setContentsMargins(12, 10, 12, 12)
        root.setSpacing(8)

        header = QHBoxLayout()
        self.title = QLabel("Grounded Cluely")
        self.title.setObjectName("title")
        self.collapse_btn = QPushButton("collapse")
        self.collapse_btn.setObjectName("smallbtn")
        self.collapse_btn.clicked.connect(self.toggle_expand)
        self.close_btn = QPushButton("x")
        self.close_btn.setObjectName("smallbtn")
        self.close_btn.clicked.connect(self.close)
        header.addWidget(self.title)
        header.addStretch(1)
        header.addWidget(self.collapse_btn)
        header.addWidget(self.close_btn)
        root.addLayout(header)

        self.answers = QTextBrowser()
        self.answers.setObjectName("answers")
        self.answers.setOpenExternalLinks(True)
        self.answers.append(
            "<span style='color:#8b93a1'>Capturing the meeting. Ask about your "
            "documents or what has been said.</span>"
        )
        root.addWidget(self.answers, 1)

        self.status = QLabel("")
        self.status.setObjectName("status")
        root.addWidget(self.status)

        self.input = QLineEdit()
        self.input.setObjectName("input")
        self.input.setPlaceholderText("Ask a question and press Enter...")
        self.input.returnPressed.connect(self.submit_question)
        root.addWidget(self.input)

        self.note = QLabel(CAPTURE_NOTE)
        self.note.setObjectName("note")
        self.note.setWordWrap(True)
        root.addWidget(self.note)

        self._chat_widgets = [self.answers, self.status, self.input, self.note]

        self.panel.setStyleSheet(
            """
            #panel { background: #171c24; border: 1px solid #2a313c; border-radius: 12px; }
            #title { color: #e6e9ef; font-size: 13px; font-weight: 600; }
            #answers { background: #0b0f14; color: #e6e9ef; border: 1px solid #2a313c;
                       border-radius: 8px; font-size: 13px; }
            #input { background: #0b0f14; color: #e6e9ef; border: 1px solid #2a313c;
                     border-radius: 7px; padding: 7px 10px; font-size: 13px; }
            #status { color: #c8a93f; font-size: 12px; }
            #note { color: #6b7280; font-size: 10px; }
            #smallbtn { background: #232b35; color: #e6e9ef; border: 0; border-radius: 6px;
                        padding: 3px 9px; font-size: 12px; }
            """
        )

    def _set_expanded(self, expanded):
        self.expanded = expanded
        for w in self._chat_widgets:
            w.setVisible(expanded)
        self.collapse_btn.setText("collapse" if expanded else "expand")
        if expanded:
            self.title.setText("Grounded Cluely")
            self.resize(360, 470)
        else:
            self.title.setText("GC")
            self.resize(170, 48)

    # ----- asking questions ----------------------------------------------------

    def submit_question(self):
        question = self.input.text().strip()
        if not question:
            return
        self.input.clear()
        self.answers.append(f"<b style='color:#4f9dff'>You:</b> {html.escape(question)}")
        self.status.setText("Thinking...")
        # Run the network calls on a worker thread so the UI stays responsive and
        # the audio thread is never touched. The result comes back via a signal.
        threading.Thread(target=self._compute, args=(question,), daemon=True).start()

    def _compute(self, question):
        try:
            answer, sources = answer_for_ui(question, self.doc_path, self.memory)
            self.answer_ready.emit(answer, sources)
        except Exception as e:  # noqa: BLE001  report errors in the UI, never crash
            self.answer_failed.emit(str(e))

    def _show_answer(self, answer, sources):
        self.status.setText("")
        self.answers.append(f"<b style='color:#3fb950'>Assistant:</b> {html.escape(answer)}")
        for line in sources:
            self.answers.append(
                f"<span style='color:#8b93a1'>&nbsp;&nbsp;{html.escape(line)}</span>"
            )
        self.answers.append("")

    def _show_error(self, message):
        self.status.setText("")
        self.answers.append(
            f"<span style='color:#f85149'>could not answer: {html.escape(message)}</span>"
        )

    # ----- show, hide, expand, drag --------------------------------------------

    def toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()
            self.apply_native_window_options()

    def toggle_expand(self):
        self._set_expanded(not self.expanded)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_offset = None

    # ----- native macOS window options -----------------------------------------

    def apply_native_window_options(self):
        """
        Reach the real macOS window object behind this Qt widget through PyObjC,
        and set two things on it.

        self.winId() on macOS returns a pointer to the native NSView. PyObjC's
        objc.objc_object wraps that raw pointer back into a Python object we can
        call Objective C methods on. From the view we get its window, then:
          setSharingType_(none)  best effort hide from screen capture.
          setCollectionBehavior_ keep the panel visible across spaces.

        HONEST LIMITATION: on macOS 15 and newer, ScreenCaptureKit ignores the
        sharing type, so modern screen recording and apps like Zoom may still
        show this panel, sometimes as a black rectangle. This is best effort, not
        guaranteed invisibility. The same truth is stated in the UI note.
        """
        try:
            import objc
            from AppKit import (
                NSWindowCollectionBehaviorCanJoinAllSpaces,
                NSWindowCollectionBehaviorFullScreenAuxiliary,
                NSWindowSharingNone,
            )

            view = objc.objc_object(c_void_p=int(self.winId()))
            window = view.window()
            window.setSharingType_(NSWindowSharingNone)
            window.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorFullScreenAuxiliary
            )
        except Exception as e:  # noqa: BLE001  this is best effort
            print("note: could not set native window options:", e)

    # ----- global hotkeys ------------------------------------------------------

    def start_hotkeys(self):
        """
        Register system wide hotkeys with pynput.

        The callbacks fire on pynput's own background thread, so they must not
        touch Qt widgets directly. Each one just emits a Qt signal, which Qt
        delivers safely to the UI thread. This is the same cross thread rule the
        audio uses: background threads talk to the UI only through signals.

        macOS needs Accessibility or Input Monitoring permission to capture keys
        globally. The first run will prompt. Grant it in System Settings, Privacy
        and Security, then relaunch.
        """
        try:
            self._hotkeys = keyboard.GlobalHotKeys(
                {
                    "<cmd>+<shift>+j": self.hotkey_toggle_visibility.emit,
                    "<cmd>+<shift>+k": self.hotkey_toggle_expand.emit,
                }
            )
            self._hotkeys.start()
        except Exception as e:  # noqa: BLE001
            print("note: could not start global hotkeys:", e)
            self._hotkeys = None

    # ----- shutdown ------------------------------------------------------------

    def cleanup(self):
        """Stop capture and save the summary. Safe to call more than once."""
        if self._cleaned:
            return
        self._cleaned = True
        if self._hotkeys is not None:
            try:
                self._hotkeys.stop()
            except Exception:  # noqa: BLE001
                pass
        if self.capture is not None:
            print("\nStopping capture.")
            self.capture.stop()
            print("Writing the meeting summary...")
            path = save_meeting(self.memory)
            print(f"Meeting summary saved to: {path}")

    def closeEvent(self, event):
        self.cleanup()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Meeting assistant overlay")
    parser.add_argument("--doc", help="path to a document to ground answers in")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    memory = TranscriptMemory()
    doc_path = None
    if args.doc:
        doc_path = args.doc
        print(f"Indexing document: {doc_path}")
        ensure_doc_indexed(doc_path)

    capture = LiveCapture(on_line=lambda label, text: memory.add(label, text))
    status = capture.start()
    print(
        f"Capturing. Mic alive: {status['mic_alive']}, "
        f"system alive: {status['system_alive']} at {status['system_rate']} Hz."
    )

    overlay = Overlay(memory, doc_path)
    overlay.capture = capture
    overlay.move(80, 80)
    overlay.show()
    # The native window only exists after show(), so set its options now.
    overlay.apply_native_window_options()
    overlay.start_hotkeys()
    app.aboutToQuit.connect(overlay.cleanup)

    print("Overlay running. Cmd+Shift+J show/hide, Cmd+Shift+K expand/collapse.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
