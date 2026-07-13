# CLAUDE.md

This file provides agent-memory aid for Claude Code (claude.ai/code) when
working in this repository. **Naming, code style, the CuPy/GPU convention,
testing, and the commit convention live in [`CONVENTIONS.md`](CONVENTIONS.md)**
— read that first and follow it. This file only holds project-specific context
that does not belong in the conventions doc.

## Project

SAR (synthetic-aperture radar) **Frozen Background Reference (FBR)** library
— a pure-NumPy implementation of the FBR image-computation step (iterative
CV-based stable-pixel selection + MP averaging) from Taillade, Thirion-Lefevre
& Guinvarc'h, *Remote Sensing* 2020, 12(11), 1720. The downstream
likelihood-ratio change-detection test from the paper is out of scope. The
package is `sarFbr` (import name), distributed as `sar-fbr`. Optional CuPy GPU
acceleration via the array-module dispatch convention (see `CONVENTIONS.md`
"CuPy / dual CPU-GPU backend convention"); NumPy fallback when no CUDA device.

Single-module library: everything lives in `src/sarFbr/fbr.py`; `__init__.py`
re-exports `computeFbr` and `theoreticalCv` as the public API.

## Commands

This project is [uv](https://docs.astral.sh/uv/)-managed.

```powershell
uv sync --extra dev                          # .venv + runtime deps (numpy) + dev tooling (pytest, ruff, mypy)
uv run python scripts/check.py                # GATE: ruff + format + mypy + tests + >=90% branch coverage
uv run pytest                                 # dev loop (no coverage gate)
uv run python -m sarFbr.fbr                   # __main__ self-check (cupy-vs-NumPy; skips GPU part on CPU hosts)
uv run python scripts/install_cupy.py        # detect GPU/CUDA and install the matching CuPy wheel (dev only; not shipped in the wheel)
```

## Notes for the agent

- The `__main__` self-check and `_computeFbr_cupy_selfcheck()` are intentionally
  kept runnable for GPU developers but are excluded from the coverage gate
  (`# pragma: no cover`) — see `CONVENTIONS.md` "GPU coverage rule". Do not
  remove them; do not try to push them under the ≥90% gate.
- `requires-python = ">=3.12"`. The package targets Python 3.12+.
- `computeFbr` and `theoreticalCv` are the public API. `mode="rp"` is not
  implemented — it falls back to `"mp"` with a printed warning (see
  `CONVENTIONS.md` "Important divergences").