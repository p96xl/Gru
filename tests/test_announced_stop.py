#!/usr/bin/env python3
"""A turn that says what it will do next ("Let me check X:") but makes no tool
call used to end the task, leaving unattended runs idle at the prompt. These
tests pin the detection and the auto-continue in model_turn and the loop."""
import builtins
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


# Real endings of turns that stopped mid-task on a long local-model run.
STALLS = [
    "Good, the key matches. Fixing the ShiftRows (column-major) and key-expansion bugs before running:",
    "Instead of more theorizing: I'll brute-force the exact AES formulation empirically in one script instead of guessing further.",
    "Let me settle the exact gate semantics from `FUN_80001ff8`/`FUN_80002068` (the two states that touch the digest):",
    "Now I’ll run the full test suite.",
]
FINISHED = [
    "All tests pass. The parser now rejects empty names.",
    "Done. Let me know if you want the README updated too.",
    "I can also add a CLI flag. Should I?",
    "If you want, I'll add a test for the empty string.",
    "Summary:\n\n- fixed X\n- fixed Y",
    "",
]


def test_detects_announced_steps():
    for t in STALLS:
        assert m._announces_unrun_action(t), t


def test_ignores_finished_answers_and_questions():
    for t in FINISHED:
        assert not m._announces_unrun_action(t), t


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


def _after_tool():
    return [
        {"role": "user", "content": "decode the firmware"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc1", "type": "function",
             "function": {"name": "run_bash", "arguments": '{"command": "true"}'}}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "[exit 0]\n"},
    ]


def _turn(messages, text, **kw):
    saved_open, saved_print = m.open_stream, builtins.print
    m.open_stream = lambda *a, **k: iter([_Chunk(text)])
    builtins.print = lambda *a, **k: None
    try:
        return m.model_turn(messages, **kw)
    finally:
        m.open_stream, builtins.print = saved_open, saved_print


def test_model_turn_flags_announced_step():
    messages = _after_tool()
    assert _turn(messages, STALLS[2]) == m.TURN_ANNOUNCED
    assert messages[-1] == {"role": "assistant", "content": STALLS[2]}


def test_model_turn_done_when_not_after_tool():
    """A reply to the user's own message is a real answer, even if it says 'I'll'."""
    messages = [{"role": "user", "content": "plan it"}]
    assert _turn(messages, "I'll start with the parser:") == m.TURN_DONE


def test_model_turn_done_for_finished_answer_or_forced_final():
    assert _turn(_after_tool(), FINISHED[0]) == m.TURN_DONE
    assert _turn(_after_tool(), STALLS[0], forced_final=True) == m.TURN_DONE


def _loop(script, limit=None):
    calls, msgs = [], [{"role": "user", "content": "go"}]

    def fake(messages, *a, **k):
        calls.append(len(messages))
        return script[len(calls) - 1]

    saved = (m.model_turn, builtins.print, m.ANNOUNCED_STOP_RETRY_LIMIT)
    m.model_turn, builtins.print = fake, (lambda *a, **k: None)
    if limit is not None:
        m.ANNOUNCED_STOP_RETRY_LIMIT = limit
    try:
        m._run_model_turn_loop(msgs)
    finally:
        m.model_turn, builtins.print, m.ANNOUNCED_STOP_RETRY_LIMIT = saved
    notes = [x for x in msgs if "[Runtime note:" in str(x.get("content"))]
    return len(calls), len(notes)


def test_loop_nudges_until_limit_and_resets_on_tool():
    A, T = m.TURN_ANNOUNCED, m.TURN_TOOL
    # limit 2: two nudges, the third announce ends the turn
    assert _loop([A, A, A, m.TURN_DONE], limit=2) == (3, 2)
    # a real tool call refills the budget
    assert _loop([A, A, T, A, A, A], limit=2) == (6, 4)
    # 0 disables it entirely
    assert _loop([A, m.TURN_DONE], limit=0) == (1, 0)
