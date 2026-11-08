#!/usr/bin/env python3
"""MCP client: a stdio server (child process) and a Streamable HTTP server, end to end."""
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import minion as m  # noqa: E402


class _Http(BaseHTTPRequestHandler):
    seen_sessions = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        msg = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Http.seen_sessions.append(self.headers.get("Mcp-Session-Id"))
        if "id" not in msg:  # a notification
            self.send_response(202); self.end_headers(); return
        if msg["method"] == "initialize":
            body = {"jsonrpc": "2.0", "id": msg["id"], "result": {"protocolVersion": "2025-06-18",
                                                                  "capabilities": {}, "serverInfo": {"name": "h"}}}
            self._json(body, {"Mcp-Session-Id": "sess-42"})
        elif msg["method"] == "tools/list":  # answered as an event stream
            body = {"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": [
                {"name": "add", "description": "Add two numbers",
                 "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}}]}}
            data = (": keep-alive\n\n" "event: message\ndata: " + json.dumps(body) + "\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)
        elif msg["method"] == "tools/call":
            a = msg["params"]["arguments"]
            self._json({"jsonrpc": "2.0", "id": msg["id"],
                        "result": {"content": [], "structuredContent": {"sum": a["a"] + a["b"]}}})

    def do_DELETE(self):
        self.send_response(200); self.end_headers()

    def _json(self, body, headers=None):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)


def _setup(config):
    path = os.path.join(_tmp, "mcp.json")
    with open(path, "w") as f:
        json.dump({"mcpServers": config}, f)
    saved = (dict(m.DISPATCH), list(m.TOOLS), m.YOLO, m.SESSION_HOME)
    m.SESSION_HOME = _tmp  # server stderr logs go to <home>/mcp-logs
    m.MCP_SERVERS.clear()
    os.environ["MINION_MCP"] = path
    m.YOLO = True
    m._mcp_setup()
    return saved


def _teardown(saved):
    for s in m.MCP_SERVERS.values():
        s.close()
    m.MCP_SERVERS.clear()
    m.DISPATCH.clear(); m.DISPATCH.update(saved[0])
    m.TOOLS[:] = saved[1]
    m.YOLO, m.SESSION_HOME = saved[2], saved[3]
    os.environ.pop("MINION_MCP", None)


def test_stdio_server_tools_calls_errors_and_roots():
    saved = _setup({"fake": {"command": sys.executable, "args": [os.path.join(HERE, "fake_mcp_server.py")]},
                    "off": {"command": "nonexistent-binary", "disabled": True}})
    try:
        names = [t["function"]["name"] for t in m.TOOLS]
        for t in ("mcp__fake__echo", "mcp__fake__fail", "mcp__fake__roots"):  # both tools/list pages
            assert t in names
        assert "off" not in m.MCP_SERVERS
        assert m.DISPATCH["mcp__fake__echo"](text="hi") == "echo: hi"
        assert m.DISPATCH["mcp__fake__fail"]() == "ERROR: it broke"
        roots = json.loads(m.DISPATCH["mcp__fake__roots"]())
        assert roots[0]["uri"] == "file://" + os.getcwd()
    finally:
        _teardown(saved)


def test_tool_filter_and_env_expansion():
    os.environ["FAKE_MCP_SCRIPT"] = os.path.join(HERE, "fake_mcp_server.py")
    saved = _setup({"f": {"command": sys.executable, "args": ["${FAKE_MCP_SCRIPT}"], "tools": ["echo"]}})
    try:
        names = [t["function"]["name"] for t in m.TOOLS if t["function"]["name"].startswith("mcp__")]
        assert names == ["mcp__f__echo"]
    finally:
        _teardown(saved)


def test_broken_server_does_not_stop_minion():
    saved = _setup({"bad": {"command": "/nonexistent/server"}})
    try:
        assert "bad" not in m.MCP_SERVERS
    finally:
        _teardown(saved)


def test_http_server_json_and_event_stream():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Http)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    saved = _setup({"h": {"type": "http", "url": f"http://127.0.0.1:{srv.server_port}/mcp"}})
    try:
        assert "mcp__h__add" in m.DISPATCH
        assert json.loads(m.DISPATCH["mcp__h__add"](a=2, b=3)) == {"sum": 5}
        assert _Http.seen_sessions[0] is None and _Http.seen_sessions[-1] == "sess-42"
    finally:
        _teardown(saved)
        srv.shutdown()


def test_switch_off_skips_servers():
    os.environ["MINION_MCP_TOOLS"] = "0"
    try:
        saved = _setup({"fake": {"command": sys.executable, "args": [os.path.join(HERE, "fake_mcp_server.py")]}})
        assert not m.MCP_SERVERS
        _teardown(saved)
    finally:
        os.environ.pop("MINION_MCP_TOOLS")


def test_denied_call_does_not_reach_the_server():
    saved = _setup({"fake": {"command": sys.executable, "args": [os.path.join(HERE, "fake_mcp_server.py")]}})
    real = m._confirm
    m._confirm = lambda action: False
    try:
        assert m.DISPATCH["mcp__fake__echo"](text="x") == "DENIED by user"
    finally:
        m._confirm = real
        _teardown(saved)
