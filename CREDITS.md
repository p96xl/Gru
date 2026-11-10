# Credits

Gru runs minions: it starts the engines, the cards and the helpers, then hands you a coding agent.
Most of what it runs was made by other people.

| Part | From | License | In this repo |
|---|---|---|---|
| `minion.py`, the coding agent | [sentdex/minion](https://github.com/Sentdex/minion) by Harrison Kinsley, with our changes (every one behind a `MINION_*` switch) | MIT, [licenses/minion-MIT.txt](licenses/minion-MIT.txt) | yes, modified |
| The chat page `web_ui.py` serves (`webui/`) | the web UI built into [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`, by the ggml authors | MIT, [licenses/llama.cpp-MIT.txt](licenses/llama.cpp-MIT.txt) | yes, unmodified; `web_ui.py` adds its sync and status scripts on top when serving |
| `llama-server` | [llama.cpp](https://github.com/ggml-org/llama.cpp), by the ggml authors | MIT | no: `setup_wizard.py` downloads release builds, or point `gru.toml` at your own |
| Strata | Strata's own authors | Strata's own terms | no: install it yourself and point `gru.toml` at its folder |
| `openai` (Python) | [openai/openai-python](https://github.com/openai/openai-python) | Apache-2.0 | no: installed into `vendor/` by `setup_wizard.py` |

The name: Gru, and his Minions.
