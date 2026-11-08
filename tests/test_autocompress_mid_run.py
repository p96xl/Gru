#!/usr/bin/env python3
"""Auto-compress never fired on long llama.cpp runs, for two reasons:
  1. the context size came from timings.prompt_n, which excludes the tokens
     llama.cpp reused from its cache (a 130,948-token context read as 333);
  2. auto-compress only ran after the whole task returned to the REPL, so a
     multi-hour autonomous run filled the window and stalled.
"""
import builtins
import os
import sys
import tempfile
import types

_tmp = tempfile.mkdtemp(prefix="minion-test-")
os.environ["MINION_SESSIONS_DIR"] = _tmp
os.environ["MINION_HOME"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import minion as m  # noqa: E402


def test_context_tokens_counts_cached_prompt():
    # real final-chunk numbers from a llama.cpp run at a full 131K window
    timings = {"cache_n": 130615, "prompt_n": 333, "predicted_n": 75}
    usage = types.SimpleNamespace(prompt_tokens=130948, completion_tokens=75)
    assert m._context_tokens(usage, timings) == 130948
    assert m._context_tokens(None, timings) == 130948
    assert m._context_tokens(None, {"prompt_n": 900}) == 900


def test_estimate_context_adds_new_messages():
    new = [{"role": "assistant", "content": None, "tool_calls": [
               {"id": "a", "function": {"name": "run_bash", "arguments": "x" * 70}}]},
           {"role": "tool", "tool_call_id": "a", "content": "y" * 3430}]
    assert m._estimate_context(1000, new) == 1000 + 1000


class _Src:
    _context_window = 1000


def _run(script, prompt_tokens_per_step, compress_ok=True):
    """Drive the real loop with a fake model_turn; record compress calls."""
    calls, seen = [], []

    def fake_turn(messages, *a, **k):
        i = len(seen)
        seen.append(len(messages))
        m._LAST_PROMPT_TOKENS = prompt_tokens_per_step[i]
        status = script[i]
        if status == m.TURN_TOOL:
            messages.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": f"c{i}", "type": "function",
                 "function": {"name": "run_bash", "arguments": "{}"}}]})
            messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})
        return status

    def fake_auto(messages, est):
        calls.append(est)
        return compress_ok

    saved = (m.model_turn, m._maybe_autocompress, m.ACTIVE, m.AUTOCOMPRESS_PERCENT, builtins.print)
    m.model_turn, m._maybe_autocompress, m.ACTIVE = fake_turn, fake_auto, _Src()
    m.AUTOCOMPRESS_PERCENT, builtins.print = 85, (lambda *a, **k: None)
    try:
        m._run_model_turn_loop([{"role": "system", "content": "s"},
                                {"role": "user", "content": "go"}])
    finally:
        (m.model_turn, m._maybe_autocompress, m.ACTIVE,
         m.AUTOCOMPRESS_PERCENT, builtins.print) = saved
    return calls


def test_compresses_between_steps_once_over_threshold():
    T, D = m.TURN_TOOL, m.TURN_DONE
    calls = _run([T, T, T, D], [500, 700, 900, 100])
    assert len(calls) == 1 and calls[0] >= 900  # only the step past 85%


def test_no_mid_run_compress_below_threshold():
    T, D = m.TURN_TOOL, m.TURN_DONE
    assert _run([T, T, D], [100, 200, 300]) == []


def test_failed_compress_not_retried_every_step():
    T, D = m.TURN_TOOL, m.TURN_DONE
    calls = _run([T, T, T, D], [900, 920, 940, 950], compress_ok=False)
    assert len(calls) == 1


def test_disabled_autocompress_never_runs_mid_run():
    saved = m._over_autocompress_threshold
    m._over_autocompress_threshold = lambda n: False
    try:
        T, D = m.TURN_TOOL, m.TURN_DONE
        assert _run([T, T, D], [990, 995, 999]) == []
    finally:
        m._over_autocompress_threshold = saved
