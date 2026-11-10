"""minion_config.py: the one shared file. Every other script in this folder reads minion.toml through it.

  python3 minion_config.py            show the resolved config: engines, cards (found or not), parts present
  python3 minion_config.py --check    same, and exit 1 if anything is wrong

What's here:
  load()            minion.toml (next to this file, or $MINION_CONFIG) -> a plain dict with paths resolved
  find_card()       a PCI id like "1002:73bf" -> its bus address, hwmon folder and VRAM, read from /sys
  model_settings()  a model folder's minion.conf ("key = value" lines), layered over the models folder's own
  part_present()    whether an optional script (fan_curves, assistant_gpu, ...) is still in this folder
  hold()            a file lock (fcntl.flock) under the state folder

Nothing in here starts a process or changes the system. Python 3.11+ (tomllib), standard library only.
"""
import fcntl
import glob
import os
import re
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))

# Optional parts: deleting the file is how you turn one off. Name -> what it does (shown when it's missing).
PARTS = {
    "engine_llamacpp": "runs models with llama.cpp's llama-server",
    "engine_strata": "runs models with Strata",
    "assistant_gpu": "a second card as assistant (review, rut detection, compaction, agents)",
    "sandbox": "bwrap sandbox for --yolo runs",
    "detached_runs": "minion runs that keep going with no terminal",
    "thermal_guard": "stops a card's engine when it gets too hot",
    "fan_curves": "fan curves that follow GPU and CPU temperatures (needs root)",
    "web_ui": "chat page for your phone or LAN",
    "setup_wizard": "one-time setup: cards, drivers, builds, minion.toml",
}

ENGINE_KINDS = ("llama.cpp", "strata")
ROLES = ("main", "review", "coach", "compact", "agents", "describe", "embed", "rerank")
PCI_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{4}$")


class ConfigError(Exception):
    pass


def config_path():
    return os.path.abspath(os.path.expanduser(os.environ.get("MINION_CONFIG") or os.path.join(HERE, "minion.toml")))


def _path(base, p):
    """~ expands to home; a relative path is relative to minion.toml's folder, so the SSD can mount anywhere."""
    p = os.path.expanduser(str(p))
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(base, p))


def load(path=None):
    """Read and check minion.toml. Raises ConfigError with every problem found, not just the first."""
    path = path or config_path()
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f"{path} doesn't exist. Copy minion.example.toml to minion.toml, or run setup_wizard.py")
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}")
    base = os.path.dirname(path)
    errors = []
    cfg = {k: v for k, v in raw.items() if k not in ("engines", "cards")}
    cfg["file"] = path
    cfg["state"] = _path(base, raw.get("state", "~/.minion"))
    cfg["agent"] = _path(base, raw.get("agent", "minion.py"))

    cfg["engines"] = {}
    for name, e in (raw.get("engines") or {}).items():
        if not isinstance(e, dict):
            errors.append(f"[engines.{name}] must be a table"); continue
        kind = e.get("kind")
        if kind not in ENGINE_KINDS:
            errors.append(f"[engines.{name}] kind = {kind!r}: use one of {', '.join(ENGINE_KINDS)}"); continue
        e = dict(e)
        for key in ("server", "libs", "home"):
            if key in e:
                e[key] = _path(base, e[key])
        if kind == "llama.cpp" and "server" not in e:
            errors.append(f"[engines.{name}] needs server = <path to llama-server>")
        if kind == "strata" and "home" not in e:
            errors.append(f"[engines.{name}] needs home = <Strata folder>")
        cfg["engines"][name] = e

    cfg["cards"] = {}
    for name, c in (raw.get("cards") or {}).items():
        if not isinstance(c, dict):
            errors.append(f"[cards.{name}] must be a table"); continue
        c = dict(c)
        pci = str(c.get("pci", "")).lower()
        if not PCI_RE.match(pci):
            errors.append(f"[cards.{name}] pci = {c.get('pci')!r}: use vendor:device, e.g. \"1002:73bf\"")
        c["pci"] = pci
        for en in c.get("engines", []):
            if en not in cfg["engines"]:
                errors.append(f"[cards.{name}] engines lists {en!r}, which has no [engines.{en}] table")
        if "models" in c:
            c["models"] = _path(base, c["models"])
        else:
            errors.append(f"[cards.{name}] needs models = <folder with this card's models>")
        if "idle" in c and not (isinstance(c["idle"], int) and c["idle"] > 0):
            errors.append(f"[cards.{name}] idle = {c['idle']!r}: seconds, a whole number above 0")
        cfg["cards"][name] = c
    if not cfg["cards"]:
        errors.append("no [cards.<name>] table: at least one card (the main one) is needed")
    if errors:
        raise ConfigError(f"{path}:\n  " + "\n  ".join(errors))
    return cfg


def card_order(cfg):
    """Card names in file order: the first is the main card, the next the assistant, and so on."""
    return list(cfg["cards"])


def find_card(pci, sysfs="/sys"):
    """PCI id -> dict(bus, hwmon, vram_total, vram_used, drm) for the first matching card, or None.
    Found by id, not bus number, so moving a card to another slot changes nothing."""
    vendor, device = pci.split(":")
    for d in sorted(glob.glob(os.path.join(sysfs, "bus/pci/devices/*"))):
        try:
            if _read(d, "vendor") != "0x" + vendor or _read(d, "device") != "0x" + device:
                continue
        except OSError:
            continue
        hw = sorted(glob.glob(os.path.join(d, "hwmon/hwmon*")))
        drm = sorted(glob.glob(os.path.join(d, "drm/card*")))
        return {
            "bus": os.path.basename(d),
            "path": d,
            "hwmon": hw[0] if hw else None,
            "drm": os.path.basename(drm[0]) if drm else None,
            "vram_total": _int(d, "mem_info_vram_total"),
            "vram_used": _int(d, "mem_info_vram_used"),
        }
    return None


def _read(d, name):
    with open(os.path.join(d, name)) as f:
        return f.read().strip()


def _int(d, name):
    try:
        return int(_read(d, name))
    except (OSError, ValueError):
        return None


def read_conf(path, into=None):
    """One minion.conf: "key = value" lines, # comments. Values are never run through a shell."""
    conf = {} if into is None else into
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = re.match(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$", line)
            if not m:
                raise ConfigError(f'{path} line {n} isn\'t "key = value": {line}')
            conf[m.group(1)] = m.group(2)
    return conf


def model_settings(models_dir, folder):
    """Settings for one model folder: <models>/minion.conf, then <folder>/minion.conf (wins key by key)."""
    conf = {}
    paths = [os.path.join(models_dir, "minion.conf")]
    if os.path.abspath(folder) != os.path.abspath(models_dir):
        paths.append(os.path.join(folder, "minion.conf"))
    for p in paths:
        if os.path.isfile(p):
            read_conf(p, conf)
    return conf


def part_present(name):
    return os.path.isfile(os.path.join(HERE, name + ".py"))


def hold(cfg, name, shared=False, wait=True):
    """Lock <state>/<name>.lock. Returns the open file (keep it: closing releases the lock), or None when
    wait=False and someone else holds it."""
    os.makedirs(cfg["state"], exist_ok=True)
    f = open(os.path.join(cfg["state"], name + ".lock"), "a+")
    op = (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | (0 if wait else fcntl.LOCK_NB)
    try:
        fcntl.flock(f, op)
    except BlockingIOError:
        f.close()
        return None
    return f


def describe(cfg, sysfs="/sys"):
    """Lines for a human: what the config says and what this PC actually has."""
    out = [f"config  {cfg['file']}", f"state   {cfg['state']}", "", "engines:"]
    problems = []
    for name, e in cfg["engines"].items():
        where = e.get("server") or e.get("home")
        ok = os.path.exists(where)
        out.append(f"  {name:16} {e['kind']:10} {where}{'' if ok else '   MISSING'}")
        if not ok:
            problems.append(f"engine {name}: {where} doesn't exist")
    out += ["", "cards (first = main):"]
    for i, name in enumerate(card_order(cfg)):
        c = cfg["cards"][name]
        found = find_card(c["pci"], sysfs)
        if found:
            vram = f"{found['vram_total'] / 2**30:.0f} GB" if found["vram_total"] else "? GB"
            seen = f"at {found['bus']}, {vram}"
        else:
            seen = "NOT FOUND on this PC"
            problems.append(f"card {name}: no {c['pci']} on this PC")
        md = "" if os.path.isdir(c["models"]) else "   MISSING"
        if md:
            problems.append(f"card {name}: models folder {c['models']} doesn't exist")
        out.append(f"  {i + 1}. {name:12} {c['pci']}  {seen}")
        out.append(f"     engines {', '.join(c.get('engines', [])) or '(none)'}   models {c['models']}{md}")
    out += ["", "parts (delete a file to turn it off):"]
    for p, what in PARTS.items():
        out.append(f"  {'on ' if part_present(p) else 'off'}  {p + '.py':22} {what}")
    return out, problems


if __name__ == "__main__":
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__.strip()); sys.exit(0)
    try:
        cfg = load()
    except ConfigError as e:
        print(f"minion_config: {e}", file=sys.stderr); sys.exit(1)
    lines, problems = describe(cfg)
    print("\n".join(lines))
    if problems:
        print("\nproblems:\n  " + "\n  ".join(problems))
    sys.exit(1 if problems and "--check" in sys.argv else 0)
