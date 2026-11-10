import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gru_config as mc


def write(p, text):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)
    return p


GOOD = """
state = "./state"
[engines.vk]
kind = "llama.cpp"
server = "bin/llama-server"
[engines.strata]
kind = "strata"
home = "~/strata"
[cards.main]
pci = "1002:73BF"
engines = ["strata", "vk"]
models = "models"
[cards.assistant]
pci = "1002:6863"
engines = ["vk"]
models = "/abs/vega"
idle = 3600
"""


def test_load_resolves_paths_relative_to_the_file(tmp_path):
    cfg = mc.load(write(str(tmp_path / "gru.toml"), GOOD))
    assert cfg["state"] == str(tmp_path / "state")
    assert cfg["engines"]["vk"]["server"] == str(tmp_path / "bin/llama-server")
    assert cfg["engines"]["strata"]["home"] == os.path.expanduser("~/strata")
    assert cfg["cards"]["main"]["models"] == str(tmp_path / "models")
    assert cfg["cards"]["assistant"]["models"] == "/abs/vega"
    assert cfg["cards"]["main"]["pci"] == "1002:73bf"          # lower-cased
    assert mc.card_order(cfg) == ["main", "assistant"]          # file order = priority


def test_load_reports_every_problem(tmp_path):
    bad = """
[engines.x]
kind = "vllm"
[engines.y]
kind = "llama.cpp"
[cards.main]
pci = "73bf"
engines = ["nope"]
idle = 0
"""
    with pytest.raises(mc.ConfigError) as e:
        mc.load(write(str(tmp_path / "gru.toml"), bad))
    msg = str(e.value)
    for part in ("kind = 'vllm'", "needs server", "pci = '73bf'", "'nope'", "needs models", "idle = 0"):
        assert part in msg


def test_missing_file_says_what_to_do(tmp_path):
    with pytest.raises(mc.ConfigError, match="gru.example.toml"):
        mc.load(str(tmp_path / "gru.toml"))


def fake_card(sysfs, bus, vendor, device, vram=None, hwmon=True):
    d = sysfs / "bus/pci/devices" / bus
    write(str(d / "vendor"), vendor + "\n")
    write(str(d / "device"), device + "\n")
    if vram:
        write(str(d / "mem_info_vram_total"), str(vram))
    if hwmon:
        os.makedirs(d / "hwmon/hwmon3")
    os.makedirs(d / "drm/card2")


def test_find_card_by_id_not_slot(tmp_path):
    fake_card(tmp_path, "0000:06:00.0", "0x1002", "0x6863", vram=16 * 2**30)
    fake_card(tmp_path, "0000:2d:00.0", "0x1002", "0x73bf", vram=16 * 2**30)
    c = mc.find_card("1002:73bf", str(tmp_path))
    assert c["bus"] == "0000:2d:00.0"
    assert c["hwmon"].endswith("hwmon3") and c["drm"] == "card2"
    assert c["vram_total"] == 16 * 2**30
    assert mc.find_card("10de:2684", str(tmp_path)) is None


def test_model_settings_folder_wins_key_by_key(tmp_path):
    models = tmp_path / "models"
    write(str(models / "minion.conf"), "server = /a\ncontext = 64\n# comment\n")
    write(str(models / "Coder/minion.conf"), "context = 128\nengine = strata\n")
    s = mc.model_settings(str(models), str(models / "Coder"))
    assert s == {"server": "/a", "context": "128", "engine": "strata"}
    assert mc.model_settings(str(models), str(models)) == {"server": "/a", "context": "64"}


def test_bad_conf_line_names_file_and_line(tmp_path):
    p = write(str(tmp_path / "minion.conf"), "ok = 1\nthis is wrong\n")
    with pytest.raises(mc.ConfigError, match="line 2"):
        mc.read_conf(p)


def test_hold_shared_and_exclusive(tmp_path):
    cfg = {"state": str(tmp_path / "s")}
    a = mc.hold(cfg, "sessions", shared=True)
    b = mc.hold(cfg, "sessions", shared=True)
    assert a and b
    assert mc.hold(cfg, "sessions", wait=False) is None       # exclusive blocked by the shared holders
    a.close(); b.close()
    c = mc.hold(cfg, "sessions", wait=False)
    assert c is not None
    c.close()


def test_describe_flags_missing_things(tmp_path):
    cfg = mc.load(write(str(tmp_path / "gru.toml"), GOOD))
    fake_card(tmp_path / "sys", "0000:2d:00.0", "0x1002", "0x73bf", vram=16 * 2**30)
    lines, problems = mc.describe(cfg, str(tmp_path / "sys"))
    text = "\n".join(lines)
    assert "at 0000:2d:00.0, 16 GB" in text and "NOT FOUND" in text
    assert any("1002:6863" in p for p in problems)
    assert any("models folder" in p for p in problems)
