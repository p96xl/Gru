#!/usr/bin/env python3
"""Live compaction and background review on a helper source (a mock server here)."""
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


class _Helper(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(404); self.end_headers()

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Helper.seen.append(req)
        prompt = req["messages"][-1]["content"]
        if "stuck in a rut" in prompt:
            text = ("1. Write the script to a file and run it instead of quoting it inline."
                    if prompt.count("Syntax error") >= 3 else "OK")
        elif "running summary" in prompt:
            n = sum(1 for r in _Helper.seen if "running summary" in r["messages"][-1]["content"])
            text = f"SUMMARY {n}" + (" (built on the last one)" if "Summary so far" in prompt else "")
        else:
            added = [ln for ln in prompt.splitlines() if ln.startswith("+") and "BUG" in ln]
            text = "calc.py line 2: BUG divides by zero" if added else "The change is correct.\n\n**LGTM**"
        body = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "helper",
                           "choices": [{"index": 0, "finish_reason": "stop",
                                        "message": {"role": "assistant", "content": text}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


_srv = ThreadingHTTPServer(("127.0.0.1", 0), _Helper)
threading.Thread(target=_srv.serve_forever, daemon=True).start()


class _Setup:
    """A helper source next to the active one, and a known context window."""

    def __enter__(self):
        self.saved = (m.HELPER_SOURCE, m.ACTIVE._context_window, dict(m.SOURCES), m.BG_CHUNK_CHARS, m.YOLO)
        m.YOLO = True  # edit_file without a Y/n prompt
        m.SOURCES["helper"] = m.Source("helper", f"http://127.0.0.1:{_srv.server_port}/v1", "sk-noop", "helper")
        m.HELPER_SOURCE = "helper"
        m.ACTIVE._context_window = 10000
        m.BG_CHUNK_CHARS = 20000
        m._BG_COMPACT.__init__()
        m._BG_REVIEW.__init__()
        _Helper.seen.clear()
        return self

    def __exit__(self, *a):
        m.HELPER_SOURCE, m.ACTIVE._context_window = self.saved[0], self.saved[1]
        m.SOURCES.clear(); m.SOURCES.update(self.saved[2])
        m.BG_CHUNK_CHARS, m.YOLO = self.saved[3], self.saved[4]


def _history(turns=30):
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(turns):
        msgs.append({"role": "user", "content": f"step {i}: " + "x" * 1500})
        msgs.append({"role": "assistant", "content": f"did step {i}"})
    return msgs


def _wait(bg):
    if bg.thread is not None:
        bg.thread.join(10)


def test_compaction_builds_a_running_summary_and_swaps_it_in():
    with _Setup():
        msgs = _history()
        m._BG_COMPACT.tick(msgs, 3000)  # 30 % of the window: too early
        assert m._BG_COMPACT.thread is None
        m._BG_COMPACT.tick(msgs, 6000)
        _wait(m._BG_COMPACT)
        m._BG_COMPACT.tick(msgs, 6000)  # takes the first chunk, starts the next
        _wait(m._BG_COMPACT)
        m._BG_COMPACT._harvest()
        assert m._BG_COMPACT.summary == "SUMMARY 2 (built on the last one)"
        assert _Helper.seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
        n = len(m._BG_COMPACT.covered)
        tail = msgs[1 + n:]
        before = len(msgs)
        removed = m._BG_COMPACT.apply(msgs)
        assert removed > 0 and len(msgs) == before - n + 1
        assert msgs[1]["content"].startswith("[Compressed context") and "SUMMARY 2" in msgs[1]["content"]
        assert msgs[2:] == tail and msgs[2]["role"] in ("user", "assistant")
        assert len(tail) >= (before - 1) // 3  # the recent third stays verbatim


def test_summary_is_dropped_when_the_history_changes():
    with _Setup():
        msgs = _history()
        m._BG_COMPACT.tick(msgs, 6000)
        _wait(m._BG_COMPACT)
        other = _history()  # e.g. a /compress or a loaded session: new message objects
        assert m._BG_COMPACT.apply(other) == 0
        m._BG_COMPACT.tick(other, 1000)
        assert m._BG_COMPACT.summary is None


def test_review_reports_problems_and_stays_quiet_on_lgtm():
    with _Setup():
        path = os.path.join(_tmp, "calc.py")
        with open(path, "w") as f:
            f.write("def f(x):\n    return 1\n")
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "make f divide"}]
        args = {"path": path, "old": "return 1", "new": "return 1 / 0  # BUG"}
        before = m._BG_REVIEW.before_tool("edit_file", args)
        result = m.edit_file(**args)
        m._BG_REVIEW.after_tool("edit_file", args, before, result)
        m._BG_REVIEW.step(msgs)  # sends the diff
        assert m._BG_REVIEW.thread is not None and not m._BG_REVIEW.pending
        assert m._BG_REVIEW.finish(msgs) is True  # waits for it, delivers the note
        assert "BUG divides by zero" in msgs[-1]["content"] and msgs[-1]["content"].count("[Review") == 1
        diff = _Helper.seen[-1]["messages"][-1]["content"]
        assert "-    return 1\n" in diff and "make f divide" in diff

        args = {"path": path, "old": "return 1 / 0  # BUG", "new": "return x / 2"}
        before = m._BG_REVIEW.before_tool("edit_file", args)
        m._BG_REVIEW.after_tool("edit_file", args, before, m.edit_file(**args))
        n = len(msgs)
        assert m._BG_REVIEW.finish(msgs) is False and len(msgs) == n


def test_nothing_runs_without_a_separate_helper():
    with _Setup():
        m.HELPER_SOURCE = m.ACTIVE.name  # the helper is the model in use
        assert not m._BG_REVIEW.enabled()
        m._BG_COMPACT.tick(_history(), 9000)
        assert m._BG_COMPACT.thread is None
    with _Setup():
        os.environ["MINION_BG_REVIEW"] = "0"
        try:
            assert not m._BG_REVIEW.enabled()
        finally:
            os.environ.pop("MINION_BG_REVIEW")


def test_session_description_goes_to_the_helper():
    """The one-line description is written by the helper in the background (so a one-slot
    main server keeps the conversation cached) and saved at the next refresh check."""
    class _NoMain:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise AssertionError("the main model must not write the description")
    saved_client, saved_dir = m.client, os.environ.get("MINION_SESSIONS_DIR")
    os.environ["MINION_SESSIONS_DIR"] = _tmp   # other test files change it; never write the real one
    with _Setup():
        m.client = _NoMain()
        try:
            sid = "20261109-120000-" + os.urandom(3).hex()
            msgs = [{"role": "system", "content": "sys"}]
            for i in range(m.SESSION_DESC_REFRESH):
                msgs += [{"role": "user", "content": f"fix bug {i}"}, {"role": "assistant", "content": "ok"}]
            m._write_session(sid, msgs, {})
            assert m._maybe_refresh_description(sid, msgs) is None   # started in the background
            assert sid in m._DESC_PENDING
            m._DESC_PENDING[sid][0].join(10)
            desc = m._maybe_refresh_description(sid, msgs)
            assert desc and desc.startswith("The change is correct"), desc
            assert m._load_session(sid)["description"] == desc
            assert any(r["messages"][0]["content"] == m.DESC_SYSTEM for r in _Helper.seen)
        finally:
            m.client = saved_client
            if saved_dir is None:
                os.environ.pop("MINION_SESSIONS_DIR", None)
            else:
                os.environ["MINION_SESSIONS_DIR"] = saved_dir



def _steps(n, error=False):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "build the emulator"}]
    for i in range(n):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "run_bash", "arguments": '{"command": "x"}'}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                     "content": "sh: 13: Syntax error: end of file unexpected" if error else f"ok step {i}"})
    return msgs


def _coach_round(msgs):
    m._BG_COACH.__init__()
    m._BG_COACH.step(msgs)
    assert m._BG_COACH.thread is not None, "the coach did not look"
    m._BG_COACH.thread.join(10)
    n = len(msgs)
    delivered = m._BG_COACH.deliver(msgs)
    return delivered, msgs[n:] if len(msgs) > n else [msgs[-1]]


def test_coach_stays_quiet_when_work_moves_on():
    with _Setup():
        msgs = _steps(m.COACH_EVERY)
        n = len(msgs)
        delivered, _ = _coach_round(msgs)
        assert not delivered and len(msgs) == n


def test_coach_sends_a_note_when_going_in_circles():
    with _Setup():
        msgs = _steps(m.COACH_EVERY, error=True)
        delivered, _ = _coach_round(msgs)
        assert delivered
        note = msgs[-1]
        assert note["role"] == "user" and note["content"].startswith("[Coach note")
        assert "Write the script to a file" in note["content"]
        assert any("stuck in a rut" in r["messages"][-1]["content"] for r in _Helper.seen)


def test_coach_waits_for_enough_steps_and_sees_the_goal():
    with _Setup():
        m._BG_COACH.__init__()
        msgs = _steps(m.COACH_EVERY - 1)
        m._BG_COACH.step(msgs)
        assert m._BG_COACH.thread is None
        msgs[1]["content"] = "[Goal set by the user: a signed USB updater]\nWork toward this goal."
        assert m._BG_COACH._aim(msgs) == "Goal: a signed USB updater"
