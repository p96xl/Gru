#!/usr/bin/env python3
"""Esc while the request is still waiting for the server.

llama.cpp sends no response headers for a streamed completion until the first
token, so on a long prompt `create()` blocks for minutes and Esc did nothing
until then. Now the request runs on a helper thread; Esc cuts the connection
(llama.cpp cancels the prompt) and the SDK's automatic retry is blocked."""
import builtins
import os
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402
from openai import OpenAI  # noqa: E402


class _Armed:
    """Pretend the Esc watcher is armed in a terminal."""
    def __enter__(self):
        self.saved = sys.stdin.isatty
        sys.stdin.isatty = lambda: True
        m._USER_INTERRUPTED.clear()
        m._INTERRUPT_ARMED.set()

    def __exit__(self, *a):
        m._INTERRUPT_ARMED.clear()
        m._USER_INTERRUPTED.clear()
        sys.stdin.isatty = self.saved


def _press_esc_after(seconds):
    threading.Timer(seconds, m._USER_INTERRUPTED.set).start()


def test_direct_call_when_not_armed():
    m._INTERRUPT_ARMED.clear()
    assert m._call_interruptibly(lambda: 42) == 42


def test_returns_value_and_reraises_errors_when_armed():
    with _Armed():
        assert m._call_interruptibly(lambda: "ok") == "ok"
        try:
            m._call_interruptibly(lambda: (_ for _ in ()).throw(ValueError("boom")))
        except ValueError as e:
            assert str(e) == "boom"
        else:
            raise AssertionError("error not re-raised")


class _Holding(BaseHTTPRequestHandler):
    """Like llama.cpp mid-prompt: reads the request, sends nothing back."""
    hits = 0
    disconnected = threading.Event()

    def log_message(self, *a):
        pass

    def do_POST(self):
        type(self).hits += 1
        self.rfile.read(int(self.headers["Content-Length"]))
        self.connection.settimeout(0.1)
        end = time.time() + 20
        while time.time() < end:
            try:
                if self.connection.recv(1, socket.MSG_PEEK) == b"":
                    type(self).disconnected.set()
                    return
            except socket.timeout:
                pass


def test_esc_cuts_the_connection_without_a_retry():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Holding)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    saved = m.client
    m.client = OpenAI(base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key="x")  # default retries on
    try:
        with _Armed():
            _press_esc_after(0.5)
            t0 = time.time()
            try:
                m._call_interruptibly(lambda: m.client.chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "hi"}], stream=True))
            except m._RequestInterrupted:
                pass
            else:
                raise AssertionError("expected _RequestInterrupted")
            assert time.time() - t0 < 2, "Esc should return within a fraction of a second"
        assert _Holding.disconnected.wait(3), "server never saw the disconnect"
        time.sleep(2.5)  # longer than the SDK's retry backoff
        assert _Holding.hits == 1, f"request was retried ({_Holding.hits} hits)"
    finally:
        m.client = saved
        srv.shutdown()


def test_model_turn_esc_before_first_token():
    def blocked(*a, **k):
        raise m._RequestInterrupted()

    saved = (m.open_stream, builtins.print, m._ensure_interrupt_watcher)
    m.open_stream, builtins.print = blocked, (lambda *a, **k: None)
    m._ensure_interrupt_watcher = lambda: None
    m._INTERRUPT_RESTORED.set()
    try:
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]
        assert m.model_turn(messages) == m.TURN_DONE
        assert "interrupted" in messages[-1]["content"].lower()
        assert not m._INTERRUPT_ARMED.is_set(), "watcher left armed"
    finally:
        m.open_stream, builtins.print, m._ensure_interrupt_watcher = saved


def test_model_turn_disarms_when_server_unreachable():
    saved = (m.open_stream, builtins.print, m._ensure_interrupt_watcher)
    m.open_stream, builtins.print = (lambda *a, **k: None), (lambda *a, **k: None)
    m._ensure_interrupt_watcher = lambda: None
    m._INTERRUPT_RESTORED.set()
    try:
        assert m.model_turn([{"role": "user", "content": "go"}]) == m.TURN_DONE
        assert not m._INTERRUPT_ARMED.is_set(), "watcher left armed"
    finally:
        m.open_stream, builtins.print, m._ensure_interrupt_watcher = saved
