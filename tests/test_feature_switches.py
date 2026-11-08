#!/usr/bin/env python3
"""Each fork feature has a MINION_* switch; set to 0 it falls back to upstream behaviour."""
import io
import os
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import redirect_stdout

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402
from test_announced_stop import STALLS, _after_tool, _turn  # noqa: E402
from test_poll_pid_esc import _fake_tty  # noqa: E402
from test_autocompress_mid_run import _run  # noqa: E402


class _Env:
    """Set environment variables for a block, then put them back."""
    def __init__(self, **values):
        self.values, self.saved = values, {}

    def __enter__(self):
        for k, v in self.values.items():
            self.saved[k] = os.environ.get(k)
            os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_every_feature_defaults_on_and_reads_off_values():
    for name, default, _ in m.FEATURES:
        assert default == "1"
        os.environ.pop(name, None)
        assert m._feature(name), name
        for off in ("0", "off", "False", " no "):
            with _Env(**{name: off}):
                assert not m._feature(name), (name, off)
        with _Env(**{name: "1"}):
            assert m._feature(name)


def test_features_flag_lists_every_switch():
    out = io.StringIO()
    with _Env(MINION_ANNOUNCED_STOP="0"), redirect_stdout(out):
        m._print_features()
    text = out.getvalue()
    for name, _, _ in m.FEATURES:
        assert name in text
    assert "off  MINION_ANNOUNCED_STOP" in text


def test_announced_stop_off_ends_the_turn():
    with _Env(MINION_ANNOUNCED_STOP="0"):
        assert _turn(_after_tool(), STALLS[2]) == m.TURN_DONE
    assert _turn(_after_tool(), STALLS[2]) == m.TURN_ANNOUNCED


def test_esc_while_waiting_off_calls_directly():
    """Off: the call runs on this thread, as upstream, even while armed in a terminal."""
    seen = {}

    def call():
        seen["thread"] = threading.current_thread()
        return 7

    def run():
        m._INTERRUPT_ARMED.set()
        with _Env(MINION_ESC_WHILE_WAITING="0"):
            return m._call_interruptibly(call)

    assert _fake_tty(run) == 7
    assert seen["thread"] is threading.current_thread()


def test_esc_in_bash_off_does_not_arm():
    p = subprocess.Popen(["sleep", "0.5"])
    seen = {}

    def peek():
        time.sleep(0.2)
        seen["armed"] = m._INTERRUPT_ARMED.is_set()

    def run():
        threading.Thread(target=peek).start()
        with _Env(MINION_ESC_IN_BASH="0"):
            return m._poll_pid(p.pid, 1, check_esc=True)

    try:
        _fake_tty(run)  # an unreaped child still looks alive, so "timeout" is fine here
    finally:
        p.kill(); p.wait()
    assert seen["armed"] is False


def test_live_input_off_is_not_wanted():
    with _Env(MINION_LIVE_INPUT="0"):
        assert not m._LiveInput.wanted()


def test_midrun_compress_off_skips_the_check():
    T, D = m.TURN_TOOL, m.TURN_DONE
    with _Env(MINION_MIDRUN_COMPRESS="0"):
        assert _run([T, T, T, D], [500, 700, 900, 100]) == []
    assert len(_run([T, T, T, D], [500, 700, 900, 100])) == 1
