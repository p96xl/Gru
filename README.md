# Gru

Gru runs the minions. Carry a drive with your models from PC to PC: Gru finds the cards, starts the
engines on them, puts a second card to work as an assistant, and hands you
[minion](https://github.com/Sentdex/minion), a small coding agent for self-hosted models.

**Work in progress.** What works today:

- `gru_config.py`: reads `gru.toml`, the one config file. It finds cards by PCI id, not slot, and
  resolves paths relative to the file, so the drive can mount anywhere. Run it to see what this PC has.
- `engine_strata.py`: runs Strata straight from the model folder's `strata.json`, so every engine
  flag is in one file you can read and edit. Also `status`, `flags` and `import`.
- `minion.py`: the agent, from sentdex/minion with our additions. Each one is behind a `MINION_*`
  switch, so with them off it behaves like stock.

Coming: `start.py` (menu: pick a model, it starts everything and stops everything after),
`engine_llamacpp.py`, `assistant_gpu.py` (review, rut detection, compaction and sub-agents on a
second card), `sandbox.py`, `detached_runs.py`, `thermal_guard.py`, `fan_curves.py`, `web_ui.py`,
and an optional `setup_wizard.py`.

Each part is its own script. Delete a script and that part is off; nothing else breaks.

Linux first. Python 3.11+, standard library only (plus `openai` for the agent).

See [CREDITS.md](CREDITS.md) for where each part comes from.
