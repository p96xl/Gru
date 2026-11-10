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


# Seen live from QFNA on Strata: the right opening tags, then the tool's own name and an
# Anthropic-style tag as closers. The intent is clear, so it is repaired and run.
WRONG_CLOSERS = ("<tool_call>\n<function=bash>\n<parameter=command>\n"
                 "python3 tools/err_strings.py 2>&1 | head -50</bash>\n</invoke>")


def test_repairs_wrong_closing_tags():
    assert m._salvage_xml_calls(WRONG_CLOSERS) == [
        ("run_bash", {"command": "python3 tools/err_strings.py 2>&1 | head -50"})]
    ok = "<function=read_file>\n<parameter=path>\na.py\n</parameter>\n<parameter=offset>\n5\n</parameter>\n</function>"
    assert m._salvage_xml_calls(ok) == [("read_file", {"path": "a.py", "offset": 5})]
    html = "<function=write_file><parameter=path>x.html</parameter><parameter=content><p>hi</p></html>\n</write_file>"
    assert m._salvage_xml_calls(html) == [("write_file", {"path": "x.html", "content": "<p>hi</p></html>"})]


def test_does_not_repair_what_it_cannot_be_sure_of():
    for t in BROKEN + [
        "Let me look.\n<function=bash><parameter=command>ls</bash>",      # prose first
        "<function=nuke><parameter=x>1</nuke>",                            # no such tool
        "<function=bash><parameter=cmd>ls</bash>",                         # no such argument
        "<function=read_file><parameter=path>a</parameter><parameter=offset>five</parameter></function>",
    ]:
        assert m._salvage_xml_calls(t) == [], t


def test_repaired_call_runs_as_a_native_call():
    msgs = _msgs()
    saved = m.DISPATCH["run_bash"]
    ran = []
    m.DISPATCH["run_bash"] = lambda **a: ran.append(a) or "output"
    try:
        assert _turn(msgs, WRONG_CLOSERS) == m.TURN_TOOL
    finally:
        m.DISPATCH["run_bash"] = saved
    assert ran == [{"command": "python3 tools/err_strings.py 2>&1 | head -50"}]
    call = msgs[2]
    assert call["role"] == "assistant" and not call["content"]
    assert call["tool_calls"][0]["function"]["name"] == "run_bash"
    assert msgs[3]["role"] == "tool" and msgs[3]["tool_call_id"] == call["tool_calls"][0]["id"]


def test_retry_note_after_tool_results_does_not_edit_old_messages():
    msgs = _msgs() + [{"role": "assistant", "content": None, "tool_calls": []},
                      {"role": "tool", "tool_call_id": "a", "content": "ok"}]
    before = [dict(x) for x in msgs]
    assert _turn(msgs, LIVE) == m.TURN_STREAM_CUT
    assert msgs[:4] == before                         # the prompt's middle is unchanged
    assert msgs[-1]["role"] == "user" and "could not be parsed" in msgs[-1]["content"]
    assert _turn(msgs, LIVE, malformed_stream_cut_count=1) == m.TURN_STREAM_CUT
    assert len(msgs) == 5                             # a second retry replaces the note
