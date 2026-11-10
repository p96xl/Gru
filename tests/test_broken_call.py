#!/usr/bin/env python3
"""A reply that is only tool-call markup the server could not parse (seen live:
"<tool_call>\\n<function=\\n</parameter>\\n</function>\\n</tool_call>") used to end the
task as if it were the answer. It is now discarded and retried with a note."""
import builtins
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402

LIVE = "<tool_call>\n<function=\n</parameter>\n</function>\n</tool_call>"
BROKEN = [
    LIVE,
    "<tool_call>\n<function=run_bash>\n<parameter=command>\nls\n",   # cut off, no close
    "Checking the count.\n\n<tool_call>\n<function=run_bash>\n</function>\n</tool_call>",
    '<tool_call>{"name": "run_bash", "arguments": {"command": "ls"</tool_call>',  # bad JSON
]
NOT_BROKEN = [
    "The model wraps calls in <tool_call> tags; Qwen uses <function=name> inside them.",
    "Example:\n\n```\n<tool_call>\n<function=run_bash>\n</function>\n</tool_call>\n```",
    "All done. The parser now rejects empty names.",
    "",
]


def test_detects_broken_calls():
    for t in BROKEN:
        assert m._is_broken_call(t), t


def test_ignores_prose_and_code():
    for t in NOT_BROKEN:
        assert not m._is_broken_call(t), t


class _Delta:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None
        self.model_extra = {}
        self.reasoning_content = None


class _Choice:
    def __init__(self, delta, finish_reason):
        self.delta = delta
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, content, finish_reason="stop"):
        self.choices = [_Choice(_Delta(content), finish_reason)]
        self.usage = None
        self.model_extra = {}


def _turn(messages, text, **kw):
    saved_open, saved_print, saved_watch = m.open_stream, builtins.print, m._interrupt_watcher
    m.open_stream = lambda *a, **k: iter([_Chunk(text)])
    m._interrupt_watcher = lambda: None
    builtins.print = lambda *a, **k: None
    try:
        return m.model_turn(messages, **kw)
    finally:
        m.open_stream, builtins.print, m._interrupt_watcher = saved_open, saved_print, saved_watch


def _msgs():
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": "decode the firmware"}]


def test_broken_call_retries_without_keeping_it():
    msgs = _msgs()
    assert _turn(msgs, LIVE) == m.TURN_STREAM_CUT
    assert len(msgs) == 2, msgs
    assert "could not be parsed" in msgs[-1]["content"]


def test_gives_up_after_the_retry_limit():
    msgs = _msgs()
    status = _turn(msgs, LIVE, malformed_stream_cut_count=m.MALFORMED_STREAM_RETRY_LIMIT)
    assert status == m.TURN_DONE
    assert msgs[-1] == {"role": "assistant", "content": LIVE}


def test_switch_off_keeps_old_behaviour():
    old = os.environ.get("MINION_BROKEN_CALL")
    os.environ["MINION_BROKEN_CALL"] = "0"
    try:
        msgs = _msgs()
        assert _turn(msgs, LIVE) == m.TURN_DONE
    finally:
        if old is None:
            os.environ.pop("MINION_BROKEN_CALL", None)
        else:
            os.environ["MINION_BROKEN_CALL"] = old
