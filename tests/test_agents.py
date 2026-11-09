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


def _quiet(fn):
    """Run with YOLO and with this file's mock server in the environment the agents inherit
    (pytest imports every test file first, and later ones change MINION_BASE_URL)."""
    saved_env, saved_yolo, saved_home = dict(os.environ), m.YOLO, m.SESSION_HOME
    for k in [k for k in os.environ if k.startswith("MINION_SOURCE")]:
        os.environ.pop(k)
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
    for i, fruit in enumerate(("apples", "pears", "plums"), 1):
        assert f"## Agent {i}: count the {fruit}" in out
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
