# GalacticStochastic

## Bash command rules

- **Use `/private/tmp/claude-501/` for all temp files**, not bare `/tmp/`: `/private/tmp/claude-501/` is the Claude sandbox area, is stable across sessions, and has broader allowlist coverage. Bare `/tmp/` is a system-wide directory with tighter permissions.
- **Never prefix `mamba run` with `cd`**: Claude Code already runs in the project root. A `cd ...\nmamba run ...` compound command starts with `cd` and bypasses all `mamba run` allowlist patterns, causing unnecessary permission prompts.
- **Use `jq` for JSON parsing**, not `python -c "import json..."`
- **Use shell (`cat`, `printf`, heredocs) for file concatenation and prompt assembly**, not `python3 - <<'PY'`: reading files and joining them with headers is pure shell — `cat file >> out`, `printf '=== HEADER ===\n' >> out` — and those tools are allowlisted while Python is not.: `jq` is a read-only allowlisted tool; invoking Python solely to parse JSON generates an unnecessary permission prompt.
- **Use direct tool executables inside `mamba run -n DTMCMC-dev`**, not `python -m <tool>`: use `ruff check` not `python -m ruff check`, `mypy` not `python -m mypy`, `pyright` not `python -m pyright`, `prek` not `python -m prek`. The `python -m` form bypasses allowlist patterns for those tools.
- **Use `prek` instead of `pre-commit`**: `prek` is a faster drop-in replacement; prefer `mamba run -n gstoch-dev prek run --files ...` over the `pre-commit` equivalent.
