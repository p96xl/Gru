#!/usr/bin/env python3
"""_poll_pid arms the Esc watcher while it waits (so Esc during run_bash works),
but only on a terminal and only if nothing else already armed it."""
import os
import subprocess
import sys
import tempfile
import threading
import time

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m


class _Tty:
    """Stand-in stream that claims to be a terminal."""
    def __init__(self, real):
        self._real = real

    def isatty(self):
        return True

    def __getattr__(self, name):
        return getattr(self._real, name)


def _fake_tty(fn):
    saved = (sys.stdin, sys.stdout, m._ensure_interrupt_watcher)
    sys.stdin, sys.stdout = _Tty(sys.stdin), _Tty(sys.stdout)
    m._ensure_interrupt_watcher = lambda: None  # no real thread / termios in tests
    try:
        return fn()
    finally:
        sys.stdin, sys.stdout, m._ensure_interrupt_watcher = saved
        m._INTERRUPT_ARMED.clear()
        m._INTERRUPT_RESTORED.set()
        m._USER_INTERRUPTED.clear()


def test_no_tty_no_arming_no_delay():
    p = subprocess.Popen(["true"]); p.wait()
    t0 = time.time()
    assert m._poll_pid(p.pid, 5, check_esc=True) == "exited"
    assert time.time() - t0 < 0.3
    assert not m._INTERRUPT_ARMED.is_set()


def test_arms_during_wait_and_esc_interrupts():
    p = subprocess.Popen(["sleep", "5"])
    seen = {}

    def press_esc():
        time.sleep(0.3)
        seen["armed"] = m._INTERRUPT_ARMED.is_set()
        m._USER_INTERRUPTED.set()
        m._INTERRUPT_RESTORED.set()  # what the watcher does when disarmed

    def run():
        threading.Thread(target=press_esc).start()
        t0 = time.time()
        r = m._poll_pid(p.pid, 5, check_esc=True)
        return r, time.time() - t0

    try:
        r, dt = _fake_tty(run)
    finally:
        p.kill(); p.wait()
    assert seen["armed"], "watcher wasn't armed while waiting"
    assert r == "interrupted" and dt < 2, (r, dt)
    assert not m._INTERRUPT_ARMED.is_set(), "left armed after the wait"


def test_leaves_existing_arming_alone():
    p = subprocess.Popen(["true"]); p.wait()

    def run():
        m._INTERRUPT_ARMED.set()           # e.g. armed by model_turn
        m._poll_pid(p.pid, 1, check_esc=True)
        return m._INTERRUPT_ARMED.is_set()

    assert _fake_tty(run) is True


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nALL POLL-PID ESC TESTS PASSED")
