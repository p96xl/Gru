#!/usr/bin/env python3
"""/goal: how a run that ended is read (done / needs the user / Esc / just stopped)."""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


def _run(*tail):
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": "fix it"}, *tail]


def test_stopped_without_a_marker_is_not_done():
    assert m._goal_state(_run({"role": "assistant", "content": "Now the emulator core. My plan: ..."})) is None


def test_done_and_need_user_markers():
    assert m._goal_state(_run({"role": "assistant", "content": "GOAL COMPLETE: tests pass (12/12)"})) == "done"
    assert m._goal_state(_run({"role": "assistant", "content": "NEED USER: plug in the USB stick"})) == "user"


def test_esc_stops_the_goal_loop():
    msgs = _run({"role": "assistant", "content": None, "tool_calls": []},
                {"role": "user", "content": "[User pressed Esc while your tool call ran and returned to chat.]"})
    assert m._goal_state(msgs) == "esc"


def test_a_marker_from_an_earlier_user_turn_does_not_count():
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "GOAL COMPLETE: old"},
            {"role": "user", "content": "now something else"},
            {"role": "assistant", "content": "working on it"}]
    assert m._goal_state(msgs) is None


def test_runtime_notes_do_not_hide_the_last_reply():
    msgs = _run({"role": "assistant", "content": "GOAL COMPLETE: done"},
                {"role": "user", "content": "[Runtime note: something]"})
    assert m._goal_state(msgs) == "done"


def test_messages_carry_the_goal_and_the_markers():
    start = m._goal_start_message("make the updater")
    nudge = m._goal_nudge("make the updater", 3)
    for text in (start, nudge):
        assert "make the updater" in text and m.GOAL_DONE in text and m.GOAL_NEED_USER in text
    assert nudge.startswith("[Runtime note:") and "continuation 3" in nudge
