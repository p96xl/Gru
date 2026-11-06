#!/usr/bin/env python3
"""Live nudge box: typing while the agent works, delivered at the next step.

No terminal or model needed: the box's key handling is driven with raw bytes,
the turn loop with a fake model_turn, and escape parsing with a pipe.
"""
import codecs
import io
import os
import sys
import builtins
import tempfile

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m

_REAL_PRINT = builtins.print


def _fresh_live(active=False):
    li = m._LiveInput()
    li._dec = codecs.getincrementaldecoder("utf-8")("replace")
    li._real = io.StringIO()
    li.rows, li.cols = 24, 80
    li.active = active
    return li


def test_typing_and_enter_queue():
    li = _fresh_live()
    li.feed(b"use pytesx")
    li.feed(b"\x7f")          # backspace
    li.feed(b"t not unittest\r")
    assert li.queue == ["use pytest not unittest"], li.queue
    assert li.buf == ""
    li.feed(b"\r")            # empty Enter queues nothing
    assert len(li.queue) == 1


def test_line_editing_keys():
    li = _fresh_live()
    li.feed(b"hello big world")
    li.feed(b"\x17")          # Ctrl+W
    assert li.buf == "hello big ", repr(li.buf)
    li.feed(b"\x15")          # Ctrl+U
    assert li.buf == ""
    li.feed(b"a\tb")
    assert li.buf == "a b"
    li.feed(b"\x03")          # Ctrl+C with a draft clears it, no SIGINT
    assert li.buf == ""


def test_utf8_split_across_reads():
    li = _fresh_live()
    for b in "ünï".encode():  # the watcher reads one byte at a time
        li.feed(bytes([b]))
    assert li.buf == "ünï", repr(li.buf)


def test_bracketed_paste_is_one_nudge():
    li = _fresh_live()
    li.escape_seq(b"[200~")
    li.feed(b"line one\rline two")
    li.escape_seq(b"[201~")
    assert li.queue == [] and li.buf == "line one line two", (li.queue, li.buf)
    li.feed(b"\r")
    assert li.queue == ["line one line two"]


def test_draw_fits_width_and_restores_cursor():
    li = _fresh_live(active=True)
    li.cols = 30
    li.buf = "x" * 200 + "全角"
    li.draw()
    out = li._real.getvalue()
    assert out.startswith("\x1b7") and out.endswith("\x1b8")
    shown = out.split("› ")[1]
    shown = shown.replace(m.RESET, "").replace(m.DIM, "").replace("\x1b8", "")
    assert shown.startswith("…") and shown.endswith("全角")
    assert m._cells(shown) <= li.cols - 4, (shown, m._cells(shown))


def test_leftover_on_stop():
    li = _fresh_live(active=True)
    saved = sys.stdout
    try:
        li.queue = ["first"]
        li.buf = "half typed"
        li.stop()
    finally:
        sys.stdout = saved
    assert li.take_leftover() == "first\nhalf typed"
    assert li.take_leftover() == ""


def test_lines_env_set_and_restored():
    li = _fresh_live(active=True)
    saved_env, saved_out = os.environ.get("LINES"), sys.stdout
    try:
        os.environ["LINES"] = "77"
        li._old_lines = os.environ.get("LINES")
        li._set_lines(30)
        assert os.environ["LINES"] == "28"      # rows above the 2-line box
        li.stop()
        assert os.environ["LINES"] == "77"      # restored
        li2 = _fresh_live(active=True)
        os.environ.pop("LINES")
        li2._old_lines = None
        li2._set_lines(30)
        li2.stop()
        assert "LINES" not in os.environ        # removed if it wasn't set
    finally:
        sys.stdout = saved_out
        if saved_env is None:
            os.environ.pop("LINES", None)
        else:
            os.environ["LINES"] = saved_env


def test_read_esc_seq():
    r, w = os.pipe()
    try:
        assert m._read_esc_seq(r) == b""           # bare Esc
        os.write(w, b"[A")
        assert m._read_esc_seq(r) == b"[A"         # arrow key
        os.write(w, b"[200~text")
        assert m._read_esc_seq(r) == b"[200~"      # paste start, text left unread
        assert os.read(r, 4) == b"text"
        os.write(w, b"OP")
        assert m._read_esc_seq(r) == b"OP"         # SS3 (F1)
        os.write(w, b"x")
        assert m._read_esc_seq(r) == b"x"          # Alt+x
    finally:
        os.close(r); os.close(w)


def test_deliver_merges_or_appends():
    saved = m._LIVE
    m._LIVE = _fresh_live()
    builtins.print = lambda *a, **k: None
    try:
        msgs = [{"role": "assistant", "content": "ok"}]
        m._LIVE.queue = ["go faster"]
        assert m._deliver_nudges(msgs) is True
        assert msgs[-1]["role"] == "user" and msgs[-1]["content"].endswith("go faster")
        n = len(msgs)
        m._LIVE.queue = ["and add tests"]
        m._deliver_nudges(msgs)                    # last is user → merged
        assert len(msgs) == n and msgs[-1]["content"].endswith("and add tests")
        assert m._deliver_nudges(msgs) is False     # empty queue
    finally:
        builtins.print = _REAL_PRINT
        m._LIVE = saved


def _run_steps_with(fake_turn, live):
    saved_turn, saved_live = m.model_turn, m._LIVE
    m.model_turn, m._LIVE = fake_turn, live
    builtins.print = lambda *a, **k: None
    try:
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
        m._run_model_turn_steps(msgs)
        return msgs
    finally:
        builtins.print = _REAL_PRINT
        m.model_turn, m._LIVE = saved_turn, saved_live
        m._USER_INTERRUPTED.clear()


def test_nudge_during_tool_reaches_next_turn():
    live = _fresh_live(active=True)
    calls = []

    def fake_turn(messages, *a, **k):
        calls.append([dict(x) for x in messages])
        if len(calls) == 1:
            messages.append({"role": "assistant", "content": "", "tool_calls": []})
            messages.append({"role": "tool", "tool_call_id": "1", "content": "ran"})
            live.queue.append("skip the docs")  # typed while the tool ran
            return m.TURN_TOOL
        messages.append({"role": "assistant", "content": "done"})
        return m.TURN_DONE

    msgs = _run_steps_with(fake_turn, live)
    assert len(calls) == 2
    assert calls[1][-1]["role"] == "user" and "skip the docs" in calls[1][-1]["content"]
    assert msgs[-1]["content"] == "done"


def test_nudge_at_finish_keeps_going():
    live = _fresh_live(active=True)
    n = {"turns": 0}

    def fake_turn(messages, *a, **k):
        n["turns"] += 1
        messages.append({"role": "assistant", "content": f"answer {n['turns']}"})
        if n["turns"] == 1:
            live.queue.append("also handle empty input")
        return m.TURN_DONE

    msgs = _run_steps_with(fake_turn, live)
    assert n["turns"] == 2, n
    assert any("also handle empty input" in str(x.get("content")) for x in msgs)


def test_esc_cut_stream_does_not_restart_on_queued_nudge():
    live = _fresh_live(active=True)
    n = {"turns": 0}

    def fake_turn(messages, *a, **k):
        n["turns"] += 1
        live.queue.append("never mind")
        live.esc_ended = True
        return m.TURN_DONE

    _run_steps_with(fake_turn, live)
    assert n["turns"] == 1
    assert live.queue == ["never mind"]  # kept → becomes chat-input leftover


def test_esc_during_tool_stops_loop():
    live = _fresh_live(active=True)
    n = {"turns": 0}

    def fake_turn(messages, *a, **k):
        n["turns"] += 1
        messages.append({"role": "tool", "tool_call_id": "1", "content": "x"})
        m._USER_INTERRUPTED.set()
        return m.TURN_TOOL

    msgs = _run_steps_with(fake_turn, live)
    assert n["turns"] == 1
    assert "pressed Esc" in msgs[-1]["content"]


def test_approval_waits_for_typing_pause():
    import time
    saved = m._LIVE
    m._LIVE = _fresh_live(active=True)
    m._INTERRUPT_ARMED.set()
    m._INTERRUPT_RESTORED.set()
    try:
        m._LIVE.last_key = time.time()        # user is typing right now
        t0 = time.time()
        with m._live_paused() as p:
            waited = time.time() - t0
            assert p.on and not m._INTERRUPT_ARMED.is_set()
        assert waited >= m._live_paused.QUIET - 0.05, waited
        assert m._INTERRUPT_ARMED.is_set()     # re-armed for the box afterwards
    finally:
        m._INTERRUPT_ARMED.clear()
        m._INTERRUPT_RESTORED.set()
        m._LIVE = saved


def test_inactive_without_tty():
    # Under a test runner stdin isn't a TTY: the loop must behave as before.
    assert m._LiveInput.wanted() is False or not sys.stdin.isatty()
    with m._live_paused() as p:
        assert p.on is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            _REAL_PRINT(f"ok  {name}")
    _REAL_PRINT("\nALL LIVE-NUDGE TESTS PASSED")
