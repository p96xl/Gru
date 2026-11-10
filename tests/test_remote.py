#!/usr/bin/env python3
"""Messages from minion-watch's chat page arrive through MINION_REMOTE_DIR."""
import json
import os
import sys
import tempfile
import time

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402

R = os.path.join(_tmp, "remote")


def _send(num, text):
    with open(os.path.join(R, "inbox", f"{num}.msg"), "w") as f:
        f.write(text)


def _state():
    with open(os.path.join(R, "state")) as f:
        return json.load(f)


def _until(cond, t=5):
    end = time.time() + t
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_remote_messages_queue_and_are_taken_in_order():
    os.environ["MINION_REMOTE_DIR"] = R  # other test files change the environment
    os.makedirs(os.path.join(R, "inbox"), exist_ok=True)
    _send(1, "stale")  # from before minion started: dropped
    assert m._REMOTE.start()
    assert not m._Remote().start()  # a second minion doesn't serve the same folder
    assert os.listdir(os.path.join(R, "inbox")) == []
    m._REMOTE.set_waiting(True, "20261109-120000-abcdef")
    assert _state()["waiting"] is True
    assert _state()["session"] == "20261109-120000-abcdef"  # the chat page maps its chats to it
    m._REMOTE.set_waiting(True, "20261109-130000-fedcba")  # /resume or /new while waiting
    assert _state()["session"] == "20261109-130000-fedcba"
    _send(10, "first")
    _send(9, "second")
    _send(100, "third")
    assert _until(lambda: len(m._REMOTE.queue) == 3)
    assert m._REMOTE.take() == ["second", "first", "third"]
    st = _state()
    assert st["taken"] == 100 and st["waiting"] is False and st["pid"] == os.getpid()

    # while the agent works, a message turns into a nudge
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]
    _send(101, "use tabs, not spaces")
    assert _until(m._REMOTE.pending)
    assert m._deliver_nudges(msgs)
    assert "use tabs, not spaces" in msgs[-1]["content"] and _state()["taken"] == 101

    # /stop works like Esc while busy, and is not queued
    m._USER_INTERRUPTED.clear()
    _send(102, "/stop\n")
    assert _until(lambda: _state()["taken"] == 102)
    assert m._USER_INTERRUPTED.is_set() and not m._REMOTE.pending()
    m._USER_INTERRUPTED.clear()


def test_agents_log_to_their_own_file():
    import subprocess
    out = subprocess.run([sys.executable, "-c", "import minion; print(minion.LOG_PATH)"],
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         env=dict(os.environ, MINION_AGENT_DEPTH="1"), capture_output=True, text=True)
    assert out.stdout.strip().endswith("llamacpp-agents.log")
    assert m.LOG_PATH.endswith(os.sep + "llamacpp.log")
