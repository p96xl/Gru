#!/usr/bin/env python3
"""A tiny MCP server over stdio for the tests: tools echo, fail and roots."""
import json
import sys

TOOLS = [
    {"name": "echo", "description": "Echo the text back",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "fail", "description": "Always fails", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "roots", "description": "Ask the client for its roots", "inputSchema": {"type": "object"}},
]


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def read():
    line = sys.stdin.readline()
    return json.loads(line) if line else None


print("a stray line the client must ignore", flush=True)
while True:
    msg = read()
    if msg is None:
        break
    method, rid = msg.get("method"), msg.get("id")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": msg["params"]["protocolVersion"], "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "0"}}})
    elif method == "tools/list":
        cursor = (msg.get("params") or {}).get("cursor")
        page = TOOLS[:2] if not cursor else TOOLS[2:]
        res = {"tools": page}
        if not cursor:
            res["nextCursor"] = "p2"
        send({"jsonrpc": "2.0", "id": rid, "result": res})
    elif method == "tools/call":
        name, args = msg["params"]["name"], msg["params"].get("arguments") or {}
        if name == "echo":
            res = {"content": [{"type": "text", "text": "echo: " + args.get("text", "")}]}
        elif name == "fail":
            res = {"content": [{"type": "text", "text": "it broke"}], "isError": True}
        else:
            send({"jsonrpc": "2.0", "id": "srv-1", "method": "roots/list"})
            answer = read()
            res = {"content": [{"type": "text", "text": json.dumps(answer["result"]["roots"])}]}
        send({"jsonrpc": "2.0", "id": rid, "result": res})
    elif rid is not None:
        send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "nope"}})
