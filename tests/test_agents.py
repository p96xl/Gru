#!/usr/bin/env python3
"""The agent tool runs real minion child processes against a mock model server, in parallel."""
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp
for k in [k for k in os.environ if k.startswith("MINION_SOURCE")]:
    os.environ.pop(k)


class _Model(BaseHTTPRequestHandler):
    """Streams back 'REPORT: <last line of the task>' after a short delay."""
    active, peak, lock = 0, 0, threading.Lock()
    delay = 1.0

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(404); self.end_headers()

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        main = "tools" in req  # agent turns; side requests (titles) carry no tools
        with _Model.lock:
            _Model.active += main
            _Model.peak = max(_Model.peak, _Model.active)
        try:
            time.sleep(_Model.delay)
            task = req["messages"][-1]["content"].strip().splitlines()[-1]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in ({"content": "REPORT: "}, {"content": task}):
                ev = {"id": "x", "object": "chat.completion.chunk", "model": "mock",
                      "choices": [{"index": 0, "delta": chunk, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            ev = {"id": "x", "object": "chat.completion.chunk", "model": "mock",
                  "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            self.wfile.write(f"data: {json.dumps(ev)}\n\ndata: [DONE]\n\n".encode())
        finally:
            with _Model.lock:
                _Model.active -= main


_srv = ThreadingHTTPServer(("127.0.0.1", 0), _Model)
threading.Thread(target=_srv.serve_forever, daemon=True).start()
os.environ["MINION_BASE_URL"] = f"http://127.0.0.1:{_srv.server_port}/v1"
os.environ["MINION_MODEL"] = "mock"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


def _quiet(fn, background=False):
    """Run with YOLO and with this file's mock server in the environment the agents inherit
    (pytest imports every test file first, and later ones change MINION_BASE_URL).
    Agents wait for their reports unless background=True."""
    saved_env, saved_yolo, saved_home = dict(os.environ), m.YOLO, m.SESSION_HOME
    for k in [k for k in os.environ if k.startswith("MINION_SOURCE")]:
        os.environ.pop(k)
    os.environ["MINION_AGENT_BACKGROUND"] = "1" if background else "0"
    os.environ["MINION_BASE_URL"] = f"http://127.0.0.1:{_srv.server_port}/v1"
    os.environ["MINION_MODEL"] = "mock"
    m.YOLO, m.SESSION_HOME = True, _tmp
    try:
        return fn()
    finally:
        m.YOLO, m.SESSION_HOME = saved_yolo, saved_home
        os.environ.clear(); os.environ.update(saved_env)


def test_agent_tool_registered_for_top_level_only():
    assert "agent" in m.DISPATCH
    assert any(t["function"]["name"] == "agent" for t in m.TOOLS)


def test_parallel_agents_return_their_reports():
    _Model.peak = 0
    t0 = time.time()
    out = _quiet(lambda: m.agent(tasks=["count the apples", "count the pears", "count the plums"], source="local"))
    took = time.time() - t0
    for fruit in ("apples", "pears", "plums"):
        assert f": count the {fruit}" in out
        assert f"REPORT: count the {fruit}" in out
    assert out.count("finished in") == 3
    assert _Model.peak >= 2, "agents did not run at the same time"
    assert took < 3 * _Model.delay + 15
    log = out.split("full run: ")[1].split("]")[0]
    assert os.path.exists(log)


def test_parallel_limit_is_respected():
    _Model.peak = 0
    saved = m.AGENT_PARALLEL
    m.AGENT_PARALLEL = 1
    try:
        out = _quiet(lambda: m.agent(tasks=["one", "two"], source="local"))
    finally:
        m.AGENT_PARALLEL = saved
    assert out.count("finished in") == 2
    assert _Model.peak == 1, out


def test_children_cannot_start_agents():
    env = m._agent_child_env("local")
    assert env["MINION_AGENT_DEPTH"] == "1" and env["MINION_LIVE_INPUT"] == "0"
    import subprocess
    code = "import minion; print('agent' in minion.DISPATCH)"
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert res.stdout.strip().endswith("False"), res.stdout + res.stderr


def test_bad_input_and_denial():
    assert m.agent(tasks=[]).startswith("ERROR")
    assert m.agent(tasks=["x"] * 9).startswith("ERROR")
    assert m.agent(tasks=["x"], source="nope").startswith("ERROR: no source")
    real = m._confirm
    m._confirm = lambda action: False
    try:
        assert m.agent(tasks=["x"]) == "DENIED by user"
    finally:
        m._confirm = real


def test_approval_flags_follow_this_session():
    saved = (m.YOLO, m.APPROVE_LEVEL)
    try:
        m.YOLO, m.APPROVE_LEVEL = True, None
        assert m._agent_approval_args() == ["--yolo"]
        m.YOLO, m.APPROVE_LEVEL = False, "medium"
        assert m._agent_approval_args() == ["--approval", "medium"]
        m.YOLO, m.APPROVE_LEVEL = False, None
        assert m._agent_approval_args() == ["--approval", "all"]
    finally:
        m.YOLO, m.APPROVE_LEVEL = saved


def test_agent_source_setting_picks_the_default():
    saved = (m.AGENT_SOURCE, dict(m.SOURCES), list(m.SOURCE_ORDER))
    try:
        m.SOURCES["vega"] = m.Source("vega", "http://127.0.0.1:1/v1", "sk-noop", "small")
        m.SOURCE_ORDER.append("vega")
        m.AGENT_SOURCE = "vega"
        assert m._agent_default_source() == "vega"
        help_text = m._agent_source_help()
        assert "vega (small)" in help_text and help_text.endswith("Default: vega")
        m.AGENT_SOURCE = "missing"  # e.g. the second server did not start: fall back to the current one
        assert m._agent_default_source() == m.ACTIVE.name
    finally:
        m.AGENT_SOURCE = saved[0]
        m.SOURCES.clear(); m.SOURCES.update(saved[1])
        m.SOURCE_ORDER[:] = saved[2]


def test_shared_pool_split_between_agents_at_once():
    saved = (m.AGENT_SHARED_CONTEXT, m.AGENT_CONTEXT_RESERVE, m.SOURCES["local"]._context_window)
    m.AGENT_SHARED_CONTEXT, m.AGENT_CONTEXT_RESERVE = True, 4096
    m.SOURCES["local"]._context_window = 65536 + 4096
    try:
        _Model.peak = 0
        out = _quiet(lambda: m.agent(tasks=["one", "two"], source="local"))
        assert out.count("context 32768 tokens") == 2 and _Model.peak == 2, out
        _Model.peak = 0
        out = _quiet(lambda: m.agent(tasks=["one", "two"], source="local", at_once=False))
        assert out.count("context 65536 tokens") == 2 and _Model.peak == 1, out
    finally:
        m.AGENT_SHARED_CONTEXT, m.AGENT_CONTEXT_RESERVE, m.SOURCES["local"]._context_window = saved


def test_agent_works_to_its_share():
    env = dict(m._agent_child_env("local"), MINION_AGENT_CONTEXT="12345")
    import subprocess
    code = "import minion; print(minion.ACTIVE._context_window)"
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert res.stdout.strip().endswith("12345"), res.stdout + res.stderr


def _history():
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a1", "type": "function", "function": {"name": "agent", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "a1", "content": "Started"}]


def test_background_agents_return_at_once_and_report_later():
    t0 = time.time()
    out = _quiet(lambda: m.agent(tasks=["count the figs"], source="local"), background=True)
    assert out.startswith("Started 1 agent in the background on local"), out
    assert time.time() - t0 < _Model.delay, "the call waited for the agent"
    msgs = _history()
    deadline = time.time() + 60
    while not m._BG_AGENTS.deliver(msgs):
        assert time.time() < deadline, "no report arrived"
        time.sleep(0.2)
    note = msgs[-1]
    assert note["role"] == "user" and note["content"].startswith("[Background agent report")
    assert "REPORT: count the figs" in note["content"] and "finished in" in note["content"]
    assert not m._BG_AGENTS.deliver(msgs)   # delivered once


def test_finish_waits_for_agents_still_working():
    _quiet(lambda: m.agent(tasks=["count the kiwis", "count the limes"], source="local"), background=True)
    msgs = _history()
    assert m._BG_AGENTS.busy() == 2
    m._BG_AGENTS.nudged = False
    assert m._BG_AGENTS.finish(msgs)            # first: told to keep working, no waiting
    assert m._BG_AGENTS.busy() >= 1 and "don't sit idle" in msgs[-1]["content"]
    assert m._BG_AGENTS.finish(msgs)            # then: waits and hands over the reports
    assert m._BG_AGENTS.busy() == 0
    text = msgs[-1]["content"]
    assert "REPORT: count the kiwis" in text and "REPORT: count the limes" in text


def test_background_queue_respects_the_parallel_limit():
    _Model.peak = 0
    saved = m.AGENT_PARALLEL
    m.AGENT_PARALLEL = 1
    try:
        _quiet(lambda: m.agent(tasks=["a"], source="local"), background=True)
        out = _quiet(lambda: m.agent(tasks=["b"], source="local"), background=True)
        assert "1 other already working or queued" in out, out
        msgs = _history()
        m._BG_AGENTS.nudged = True
        assert m._BG_AGENTS.finish(msgs)
        while m._BG_AGENTS.busy():
            time.sleep(0.2)
        m._BG_AGENTS.deliver(msgs)
    finally:
        m.AGENT_PARALLEL = saved
    assert _Model.peak == 1
    text = "\n".join(x["content"] for x in msgs if x["role"] == "user")
    assert "REPORT: a" in text and "REPORT: b" in text, text


def test_wait_true_blocks_and_returns_reports():
    out = _quiet(lambda: m.agent(tasks=["count the dates"], source="local", wait=True), background=True)
    assert "REPORT: count the dates" in out and "finished in" in out
