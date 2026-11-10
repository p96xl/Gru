#!/usr/bin/env python3
"""engine_strata.py: run a model with Strata, from a strata.json that lives in the model's folder.

  python3 engine_strata.py start  <model folder> [--port 8080]   serve it here; Ctrl+C stops it
  python3 engine_strata.py stop                                  stop the one this script started
  python3 engine_strata.py status                                what Strata is doing right now
  python3 engine_strata.py flags  <model folder>                 the engine's flags, each one explained
  python3 engine_strata.py import <preset name> <model folder>   one time: take a preset made by `strata launch`
                                                                 and write it as <model folder>/strata.json

strata.json is the whole engine setup, nothing hidden:
  "args"        the engine's command line (--pack, --native, --spec, --mtp, --kv, --max-context, --ple-io, ...)
  "env"         environment for the engine (e.g. STRATA_HIP_PROMPT_F16)
  "gpu"         which GPU (Strata's numbering)
  "serve"       options for Strata's API server: {"parallel": 1, "shutdown_timeout": 30}
  everything else is passed to Strata's server as it is (tokenizer, model_name, ...)
Inside any string, {strata} is the Strata folder from minion.toml and {model} is the model folder, so the
file keeps working when the drive is mounted somewhere else.

How it runs: Strata's own server (terminal/<version>/serve/server.py) with that file, using the Python that
ships in Strata's folder. `strata launch`, its presets in ~/.config/strata and its runtime are not used.
The server restarts the engine by itself after a crash; `status` shows each restart and why.
Logs: <state>/strata/<model>.server.log and <model>.engine.log (state is set in minion.toml).
"""
import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

import minion_config as mc

SERVE_DEFAULTS = {"parallel": 1, "shutdown_timeout": 30}


def strata_home(cfg, name=None):
    engines = {n: e for n, e in cfg["engines"].items() if e["kind"] == "strata"}
    if not engines:
        raise mc.ConfigError("minion.toml has no engine with kind = \"strata\"")
    if name:
        if name not in engines:
            raise mc.ConfigError(f"no strata engine named {name!r} in minion.toml")
        return engines[name]["home"]
    return next(iter(engines.values()))["home"]


def terminal_root(home):
    """terminal/current if it exists, else the newest terminal/<version>."""
    cur = os.path.join(home, "terminal", "current")
    if os.path.isdir(cur):
        return os.path.realpath(cur)
    versions = sorted(glob.glob(os.path.join(home, "terminal", "*.*.*")))
    if not versions:
        raise mc.ConfigError(f"no Strata terminal in {home}/terminal")
    return versions[-1]


def strata_python(root):
    found = sorted(glob.glob(os.path.join(root, "python", "bin", "python3*")))
    found = [p for p in found if re.search(r"python3(\.\d+)?$", p)]
    if not found:
        raise mc.ConfigError(f"no Python inside {root}/python/bin")
    return found[-1]


def expand(value, subs):
    """{strata} / {model} inside every string of a JSON value."""
    if isinstance(value, str):
        for k, v in subs.items():
            value = value.replace("{" + k + "}", v)
        return value
    if isinstance(value, list):
        return [expand(v, subs) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, subs) for k, v in value.items()}
    return value


def collapse(value, subs):
    """The reverse of expand, for import: absolute paths back into {strata} / {model}. Longest first."""
    if isinstance(value, str):
        for k, v in sorted(subs.items(), key=lambda kv: -len(kv[1])):
            if value == v or value.startswith(v + os.sep):
                return "{" + k + "}" + value[len(v):]
        return value
    if isinstance(value, list):
        return [collapse(v, subs) for v in value]
    if isinstance(value, dict):
        return {k: collapse(v, subs) for k, v in value.items()}
    return value


def model_key(model_dir):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(os.path.normpath(model_dir)))


def paths(cfg, model_dir):
    d = os.path.join(cfg["state"], "strata")
    k = model_key(model_dir)
    return {
        "dir": d,
        "run": os.path.join(d, k + ".run.json"),
        "server_log": os.path.join(d, k + ".server.log"),
        "engine_log": os.path.join(d, k + ".engine.log"),
        "running": os.path.join(d, "running.json"),
    }


def read_strata_json(model_dir):
    p = os.path.join(model_dir, "strata.json")
    try:
        with open(p) as f:
            return json.load(f)
    except FileNotFoundError:
        raise mc.ConfigError(f"{p} doesn't exist. Make one with: engine_strata.py import <preset> {model_dir}")
    except json.JSONDecodeError as e:
        raise mc.ConfigError(f"{p}: {e}")


def build(cfg, model_dir, port=None, engine=None):
    """-> (server command, env, cwd, run config dict, paths). Pure: starts nothing."""
    home = strata_home(cfg, engine)
    root = terminal_root(home)
    model_dir = os.path.abspath(model_dir)
    spec = expand(read_strata_json(model_dir), {"strata": home, "model": model_dir})
    serve = {**SERVE_DEFAULTS, **spec.pop("serve", {})}
    if not isinstance(spec.get("args"), list) or not spec["args"]:
        raise mc.ConfigError(f"{model_dir}/strata.json needs \"args\": the engine's command line as a list")
    p = paths(cfg, model_dir)
    port = int(port or spec.get("port") or 8080)
    run = dict(spec, port=port, host="127.0.0.1", log=p["engine_log"])
    run.setdefault("cwd", root)
    cmd = [strata_python(root), "-m", "serve.server", "--engine", "strata", "--config", p["run"],
           "--host", "127.0.0.1", "--port", str(port), "--parallel", str(serve["parallel"]),
           "--shutdown-timeout", str(serve["shutdown_timeout"])]
    for k, v in serve.items():
        if k in SERVE_DEFAULTS:
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            cmd.append(flag)
        elif v not in (False, None):
            cmd += [flag, str(v)]
    env = dict(os.environ, STRATA_DATA_DIR=home, STRATA_COMPONENTS_DIR=os.path.join(home, "components"),
               STRATA_NO_ACQUIRE="1")
    return cmd, env, root, run, p


def start(cfg, model_dir, port=None, engine=None):
    """Start Strata's server in its own session. Returns (Popen, owner_fd): keep owner_fd open while you want
    Strata up; when it closes (or the holder dies), the server shuts itself and the engine down."""
    cmd, env, cwd, run, p = build(cfg, model_dir, port, engine)
    os.makedirs(p["dir"], exist_ok=True)
    with open(p["run"], "w") as f:
        json.dump(run, f, indent=1)
    reader, writer = os.pipe()
    cmd += ["--owner-fd", str(reader)]
    with open(p["server_log"], "ab") as log:
        log.write(f"\n[engine_strata] {time.strftime('%F %T')} start {model_dir} port {run['port']}\n".encode())
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                pass_fds=(reader,), start_new_session=True)
    os.close(reader)
    with open(p["running"], "w") as f:
        json.dump({"pid": proc.pid, "model": os.path.abspath(model_dir), "port": run["port"],
                   "started": time.time(), "run": p["run"], "server_log": p["server_log"],
                   "engine_log": p["engine_log"]}, f, indent=1)
    return proc, writer


def wait_ready(port, proc=None, timeout=900, tick=None):
    """Poll /health until the model answers. False if the server exits or time runs out."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200 and json.load(r).get("status", "ok") in ("ok", "ready"):
                    return True
        except Exception:
            pass
        if tick:
            tick(time.time() - t0)
        time.sleep(1)
    return False


def running(cfg):
    p = os.path.join(cfg["state"], "strata", "running.json")
    try:
        with open(p) as f:
            r = json.load(f)
    except (OSError, ValueError):
        return None
    try:
        with open(f"/proc/{r['pid']}/cmdline", "rb") as f:
            if b"serve.server" not in f.read():
                return None
    except OSError:
        return None
    return r


def stop(cfg, timeout=60):
    """Stop the server this script started (SIGTERM; it stops the engine first). Then any strata-host left."""
    r = running(cfg)
    if r:
        try:
            os.kill(r["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        t0 = time.time()
        while time.time() - t0 < timeout and os.path.exists(f"/proc/{r['pid']}"):
            time.sleep(0.5)
        if os.path.exists(f"/proc/{r['pid']}"):
            os.kill(r["pid"], signal.SIGKILL)
        # The engine is a child of our server; one left behind (server killed hard) still holds the card
        for pid in engine_pids():
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    return bool(r)


def engine_pids():
    out = []
    for d in glob.glob("/proc/[0-9]*"):
        try:
            with open(d + "/comm") as f:
                if f.read().strip() == "strata-host":
                    out.append(int(d[6:]))
        except OSError:
            pass
    return out


# --- status -----------------------------------------------------------------------------------------------
def _tail(path, n=4000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n * 200))
            return f.read().decode("utf-8", "replace").splitlines()[-n:]
    except OSError:
        return []


def _meminfo():
    m = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                m[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return m


def _proc_started(pid):
    try:
        st = os.stat(f"/proc/{pid}")
        return st.st_mtime
    except OSError:
        return None


def summarize_log(lines):
    """Server log lines -> restarts (with reasons), the last finished request, and what it's doing now."""
    restarts, last_done, now = [], None, None
    for line in lines:
        if "the engine stopped unexpectedly" in line or "the GPU reported a fatal error" in line:
            m = re.search(r"\(exit code (-?\d+)\)", line)
            why = re.sub(r"^\[strata\]\s*", "", line)
            why = why.split(". The next request")[0][:220]
            restarts.append({"exit": m.group(1) if m else "?", "why": why})
        elif line.startswith("[strata] done:"):
            last_done = line[len("[strata] "):][:160]
            now = None
        elif line.startswith("[strata] reading the prompt") or line.startswith("[strata] still starting") \
                or line.startswith("[strata] starting the engine") or "engine is running again" in line:
            now = line[len("[strata] "):][:160]
    return restarts, last_done, now


def status(cfg, sysfs="/sys"):
    out = []
    r = running(cfg)
    pids = engine_pids()
    if r:
        age = time.time() - r["started"]
        out.append(f"server   pid {r['pid']}  model {os.path.basename(r['model'])}  port {r['port']}  "
                   f"up {age / 60:.0f} min")
    else:
        out.append("server   not started by engine_strata.py"
                   + (" (but a strata-host engine is running: started by something else)" if pids else ""))
    for pid in pids:
        t = _proc_started(pid)
        since = time.strftime("%H:%M:%S", time.localtime(t)) if t else "?"
        try:
            with open(f"/proc/{pid}/status") as f:
                rss = next(int(x.split()[1]) * 1024 for x in f if x.startswith("VmRSS"))
        except (OSError, StopIteration):
            rss = 0
        out.append(f"engine   pid {pid}  since {since}  RAM {rss / 2**30:.1f} GB")
    if r:
        restarts, last_done, now = summarize_log(_tail(r["server_log"]))
        out.append(f"now      {now or 'idle'}")
        if last_done:
            out.append(f"last     {last_done}")
        out.append(f"restarts {len(restarts)} in this log")
        for x in restarts[-5:]:
            out.append(f"  exit {x['exit']:>3}  {x['why']}")
        out.append(f"logs     {r['server_log']}")
        out.append(f"         {r['engine_log']}")
    mem = _meminfo()
    if mem:
        out.append(f"RAM      {mem.get('MemAvailable', 0) / 2**30:.1f} GB available of "
                   f"{mem.get('MemTotal', 0) / 2**30:.0f} GB, swap free {mem.get('SwapFree', 0) / 2**30:.1f} GB")
    for name, c in cfg["cards"].items():
        card = mc.find_card(c["pci"], sysfs)
        if card and card["vram_total"]:
            used = mc._int(card["path"], "mem_info_vram_used") or 0
            out.append(f"VRAM     {name}: {used / 2**30:.1f} of {card['vram_total'] / 2**30:.0f} GB used")
    return out


# --- flags: explain each engine flag from the engine's own --help ------------------------------------------
def engine_help(home):
    exe = sorted(glob.glob(os.path.join(home, "components", "engine", "*", "strata")))
    if not exe:
        return {}
    try:
        r = subprocess.run([exe[-1], "--help"], capture_output=True, text=True, timeout=20)
        text = r.stdout + r.stderr   # the engine prints its help on stderr
    except (OSError, subprocess.TimeoutExpired):
        return {}
    helps, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"^\s{1,4}(--[a-z0-9-]+)\s+(?:\S+\s+)?(.*)$", line)
        if m and line.lstrip().startswith("--"):
            cur = m.group(1)
            helps.setdefault(cur, []).append(line.strip())
        elif cur and line.startswith(" " * 10) and line.strip():
            helps[cur].append(line.strip())
        else:
            cur = None
    return {k: " ".join(v) for k, v in helps.items()}


def flags(cfg, model_dir, engine=None):
    home = strata_home(cfg, engine)
    spec = read_strata_json(model_dir)
    helps = engine_help(home)
    out, args = [], spec.get("args", [])
    i = 0
    while i < len(args):
        a = args[i]
        val = args[i + 1] if i + 1 < len(args) and not str(args[i + 1]).startswith("--") else None
        out.append(f"{a} {val if val is not None else ''}".rstrip())
        h = helps.get(a)
        out.append("    " + (h[:300] if h else "(not in the engine's --help)"))
        i += 2 if val is not None else 1
    if spec.get("env"):
        out.append("env: " + " ".join(f"{k}={v}" for k, v in spec["env"].items()))
    return out


def import_preset(cfg, preset, model_dir, engine=None):
    """Take ~/.local/state/strata/launch/<preset>.json (written by `strata launch`) as <model>/strata.json."""
    home = strata_home(cfg, engine)
    src = preset if os.path.isfile(preset) else os.path.expanduser(f"~/.local/state/strata/launch/{preset}.json")
    with open(src) as f:
        launch = json.load(f)
    drop = {"log", "host", "open_browser", "gpus_asked", "args_written", "args_derived", "engine_ready_s", "cwd"}
    spec = {k: v for k, v in launch.items() if k not in drop}
    spec["serve"] = dict(SERVE_DEFAULTS)
    spec = collapse(spec, {"strata": os.path.realpath(home), "model": os.path.abspath(model_dir)})
    spec = collapse(spec, {"strata": home})
    out = os.path.join(model_dir, "strata.json")
    if os.path.exists(out):
        raise mc.ConfigError(f"{out} already exists; move it away first")
    with open(out, "w") as f:
        json.dump(spec, f, indent=1)
        f.write("\n")
    return out, src


def main(argv=None):
    ap = argparse.ArgumentParser(prog="engine_strata.py", description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start"); s.add_argument("model"); s.add_argument("--port", type=int)
    s.add_argument("--dry-run", action="store_true")
    sub.add_parser("stop")
    sub.add_parser("status")
    f = sub.add_parser("flags"); f.add_argument("model")
    i = sub.add_parser("import"); i.add_argument("preset"); i.add_argument("model")
    for p in (s, f, i):
        p.add_argument("--engine", help="which [engines.<name>] of kind strata (default: the first)")
    a = ap.parse_args(argv)
    try:
        cfg = mc.load()
        if a.cmd == "status":
            print("\n".join(status(cfg)))
        elif a.cmd == "flags":
            print("\n".join(flags(cfg, a.model, a.engine)))
        elif a.cmd == "import":
            out, src = import_preset(cfg, a.preset, a.model, a.engine)
            print(f"wrote {out} from {src}\nCheck it with: engine_strata.py flags {a.model}")
        elif a.cmd == "stop":
            print("stopped" if stop(cfg) else "nothing started by engine_strata.py was running")
        elif a.cmd == "start":
            if a.dry_run:
                cmd, env, cwd, run, p = build(cfg, a.model, a.port, a.engine)
                print("cwd  ", cwd)
                print("cmd  ", " ".join(cmd + ["--owner-fd", "N"]))
                print("env   STRATA_DATA_DIR=" + env["STRATA_DATA_DIR"], "STRATA_COMPONENTS_DIR="
                      + env["STRATA_COMPONENTS_DIR"], "STRATA_NO_ACQUIRE=1")
                print("engine", " ".join(run["args"]))
                print("engine env", run.get("env", {}))
                return 0
            if running(cfg) or engine_pids():
                print("engine_strata: Strata is already running (engine_strata.py status)", file=sys.stderr)
                return 1
            proc, owner = start(cfg, a.model, a.port, a.engine)
            p = paths(cfg, a.model)
            print(f"engine_strata: starting {a.model} (log: {p['server_log']}); Ctrl+C stops it")
            try:
                ok = wait_ready(json.load(open(p["run"]))["port"], proc)
                print("engine_strata: ready" if ok else "engine_strata: the server stopped; see the log")
                proc.wait()
            except KeyboardInterrupt:
                print("\nengine_strata: stopping")
                stop(cfg)
            finally:
                os.close(owner)
    except mc.ConfigError as e:
        print(f"engine_strata: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
