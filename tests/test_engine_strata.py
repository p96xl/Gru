import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine_strata as es
import gru_config as mc


def make_home(tmp_path):
    home = tmp_path / "strata"
    root = home / "terminal" / "0.1.58"
    (root / "python" / "bin").mkdir(parents=True)
    (root / "python" / "bin" / "python3.12").write_text("")
    (home / "terminal" / "current").symlink_to(root)
    return home, root


def make_cfg(tmp_path, home):
    p = tmp_path / "gru.toml"
    p.write_text(f'state = "./state"\n[engines.strata]\nkind = "strata"\nhome = "{home}"\n'
                 f'[cards.main]\npci = "1002:73bf"\nengines = ["strata"]\nmodels = "models"\n')
    return mc.load(str(p))


def test_expand_and_collapse_round_trip(tmp_path):
    subs = {"strata": "/mnt/llm/strata", "model": "/mnt/llm/models/QFNA"}
    spec = {"args": ["--pack", "/mnt/llm/strata/packs/x", "--kv", "int8"],
            "tokenizer": "/mnt/llm/models/QFNA/tok", "other": "/mnt/llm/stratax/no"}
    c = es.collapse(spec, subs)
    assert c["args"][1] == "{strata}/packs/x" and c["tokenizer"] == "{model}/tok"
    assert c["other"] == "/mnt/llm/stratax/no"        # a prefix match must end at a path separator
    assert es.expand(c, subs) == spec


def test_build_runs_stratas_own_server_from_the_model_folder(tmp_path):
    home, root = make_home(tmp_path)
    cfg = make_cfg(tmp_path, home)
    model = tmp_path / "models" / "QFNA"
    model.mkdir(parents=True)
    (model / "strata.json").write_text(json.dumps({
        "exe": "component:strata-engine", "args": ["--pack", "{strata}/p", "--spec", "4"],
        "env": {"STRATA_HIP_PROMPT_F16": "1"}, "port": 8080,
        "serve": {"parallel": 2, "idle_unload": 600, "lazy": True}}))
    cmd, env, cwd, run, p = es.build(cfg, str(model), port=8181)
    assert cmd[0] == str(root / "python" / "bin" / "python3.12")
    assert cmd[1:5] == ["-m", "serve.server", "--engine", "strata"]
    assert cmd[cmd.index("--port") + 1] == "8181" and cmd[cmd.index("--parallel") + 1] == "2"
    assert "--lazy" in cmd and cmd[cmd.index("--idle-unload") + 1] == "600"
    assert cwd == str(root)
    assert env["STRATA_DATA_DIR"] == str(home) and env["STRATA_NO_ACQUIRE"] == "1"
    assert run["args"] == ["--pack", f"{home}/p", "--spec", "4"]
    assert run["port"] == 8181 and run["log"] == p["engine_log"] and "serve" not in run
    assert p["run"].startswith(cfg["state"])


def test_missing_strata_json_says_how_to_make_one(tmp_path):
    home, _ = make_home(tmp_path)
    cfg = make_cfg(tmp_path, home)
    (tmp_path / "models" / "X").mkdir(parents=True)
    with pytest.raises(mc.ConfigError, match="engine_strata.py import"):
        es.build(cfg, str(tmp_path / "models" / "X"))


def test_import_preset_drops_launch_only_keys(tmp_path):
    home, _ = make_home(tmp_path)
    cfg = make_cfg(tmp_path, home)
    model = tmp_path / "models" / "QFNA"
    model.mkdir(parents=True)
    launch = tmp_path / "coder.json"
    launch.write_text(json.dumps({"exe": "component:strata-engine", "args": ["--pack", f"{home}/p"],
                                  "log": "/home/x/engine.log", "args_derived": [1], "cwd": "/x", "port": 8080}))
    out, _ = es.import_preset(cfg, str(launch), str(model))
    spec = json.load(open(out))
    assert spec["args"] == ["--pack", "{strata}/p"]
    assert "log" not in spec and "args_derived" not in spec and "cwd" not in spec
    assert spec["serve"] == es.SERVE_DEFAULTS
    with pytest.raises(mc.ConfigError, match="already exists"):
        es.import_preset(cfg, str(launch), str(model))


LOG = """[strata] the engine is running again
[strata] done: 2165 tokens in 304 s (27.9 tok/s) (stop, cancel=False), expert cache 87.8% hit
[strata] the engine stopped unexpectedly (exit code -15). Unrecoverable native verification failure: verify: timed out at layer 11; its GPU waits were released. The next request starts the engine again.
[strata] done: 277 tokens in 47 s (7.5 tok/s) (error, cancel=False)
[strata] starting the engine: reading the model's weights ...
[strata] reading the prompt: 10,747 of 121,414 tokens, 55 s so far
"""


def test_summarize_log_finds_restarts_and_current_work():
    restarts, last_done, now = es.summarize_log(LOG.splitlines())
    assert len(restarts) == 1 and restarts[0]["exit"] == "-15"
    assert "timed out at layer 11" in restarts[0]["why"] and "next request" not in restarts[0]["why"]
    assert last_done.startswith("done: 277 tokens")
    assert now.startswith("reading the prompt: 10,747")
