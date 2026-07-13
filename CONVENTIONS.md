# Conventions

This document is the single source of truth for naming, code style, and the
domain-specific patterns used to maintain this project. It is intentionally
tool-agnostic — it does not assume any particular editor, linter UI, or AI
assistant. `CLAUDE.md` (when present) points here and holds only agent-memory
aid, not conventions.

## Naming conventions

This project uses **camelCase for the name and a snake_case underscore as a
separator for different aspects** of an identifier. The underscore-separated tail
carries a unit, coordinate frame, or other aspect; the camelCase part carries
the quantity's name.

- **Variables:** `camelCase` + `_aspect` suffix.
  - Units: `heightOfX_m`, `targetBeamAngle_rad`, `radarCenterFrequency_Hz`,
    `aperturePowerBound_dB`, `fastTimeAxis_s`, `wavePropogationSpeed_mPerSec`,
    `pulseRepFreq_Hz`.
  - Coordinate frames as a composite suffix: `apcStartPos_XYZ_m`,
    `apcVelVec_XYZ_mPerSec`, `sceneCenter_ENU_m`.
  - Unitless quantities are plain camelCase: `pixelWiseDelay`, `interpValues`,
    `phaseCorrection`.
- **Modules / files & the import package:** camelCase — `normalizeSignal.py`,
  `computeStats.py`, the package `sarFbr/`. Private modules lead with an
  underscore: `_chunking.py`, `_phaseCorrelationCore.py`.
- **Functions:** camelCase public — `formImage`, `computeFbr`, `getTxPos`.
  Private helpers are `_camelCase` — `_robustMode`, `_selfcheck`. Backend-dispatch
  compute cores use the `_<func>_core(..., xp)` pattern (one code path for
  numpy/cupy, no mirrored `_<func>_cupy` duplicates): `_formImage_core`,
  `_computeFbr_core`.
- **Classes:** PascalCase — `Transaction`, `Statement`, `Level`.
- **Constants:** UPPER_SNAKE. Private constants keep a single leading underscore
  — `_HAVE_CUPY_GPU`, `_STMT_DATE_RE`, `_SKIP`.

## Code rigor

- Start every module with `from __future__ import annotations` (deferred
  evaluation, enables PEP 604 on older targets).
- **Type hints are required** on public functions. Use PEP 604/585 style:
  `x | None`, `list[int]`, `dict[str, float]`, `tuple[float, float] | None`. Do
  not use `typing.Optional` / `typing.List`. Enforced by **mypy**
  (`uv run mypy src/sarFbr`, and via the gate) — the gate fails on an untyped
  public function.
- **Group imports** in order: future → stdlib → third-party → local, with a
  blank line between groups. Let `ruff` (rule `I`) own the ordering.
- **Google-style docstrings** on every public function:
  ```python
  def formImage(signal, rangeAxis_m):
      """Form a focused image.

      Args:
          signal: Range-compressed signal, shape (nRange, nPulse).
          rangeAxis_m: Output range axis, in meters.

      Returns:
          Focused image as a 2-D array.
      """
  ```
- Declare the public API with `__all__` in `__init__.py` and in tool modules
  that export helpers.
- **ruff** selects `E, F, W, I, UP, B` and ignores `E501`. `N` (pep8-naming) is
  deliberately excluded so camelCase identifiers are not flagged — do not add it.

## Testing / Test-driven development

This project is test-driven. Features land with tests, and a coverage gate
keeps it that way.

- **Test-first (red-green-refactor):** before writing or changing behavior,
  write or extend the relevant `tests/test_<module>.py` test and run it red
  (`uv run pytest tests/test_<module>.py`). Then write the minimal
  implementation to turn it green, then refactor.
- **Dev loop:** `uv run pytest` runs the full suite fast, with **no coverage
  enforcement** — use it during red/green work.
- **Gate:** `uv run python scripts/check.py` runs ruff check, ruff format
  `--check`, **mypy**, and pytest with branch coverage. Branch coverage on the
  package must stay **≥ 90%**. Run it (and it must pass) before pushing or opening
  a PR. CI (`.github/workflows/ci.yml`) runs the same gate on every push and PR,
  so a red PR cannot merge.
- **Test naming:** `test` + camelCase name, optionally a `_aspect` suffix —
  `testVersionIsSet`, `testClampValue_clampsBelowLow`. (Same camelCase rule as
  the rest of the codebase; `N`/pep8-naming stays excluded in ruff.)
- **Layout:** one `tests/test_<module>.py` per source module
  (`example.py` → `tests/test_example.py`). `test_smoke.py` is the lone
  cross-cutting smoke test.

## Commit convention

Use conventional commit types: `feat`, `fix`, `docs`, `style`, `refactor`,
`perf`, `test`, `build`, `ci`, `chore`, `revert`.

## CuPy / dual CPU-GPU backend convention

This package follows the **canonical CuPy array-module convention** defined in
the `sarChangeDetection` repo (the cross-repo reference — see its
`CONVENTIONS.md` "CuPy / dual CPU-GPU backend convention"). The rule: one code
path parameterized by the array module, with CuPy as an optional, on-demand
dependency.

- `cupy` is **never** a declared runtime dependency — the correct wheel depends
  on the host GPU/CUDA and cannot be chosen statically (see
  `scripts/install_cupy.py`). It is imported eagerly at module load under
  `try/except`; `_HAVE_CUPY_GPU = cp.cuda.runtime.getDeviceCount() > 0` decides
  at import time (CPU-only hosts fall back to NumPy, never hard-fail). A
  *usable* CUDA device is required, not just an importable cupy.
- Compute helpers are `xp`-parameterized (`_<func>_core(..., xp)`): the same
  code runs on numpy or cupy — no `if gpu:`/`else:` mirror branches, no
  `_<func>_cupy` duplicate. `_computeFbr_core` is the single core.
- The public function auto-dispatches on `if _HAVE_CUPY_GPU:` — **cupy in →
  cupy out** (kept resident, detected with `isinstance(x, cp.ndarray)`) and
  **numpy in → numpy out** (auto round-tripped via `cp.asnumpy`). A caller that
  wants to keep arrays on device passes cupy arrays through.
- **Gap rule:** when a NumPy op has no clean cupy vectorized equivalent, leave a
  documented host round-trip inside the `xp` core (`cp.asnumpy` → NumPy op →
  `cp.asarray` back) rather than mirroring a GPU branch.
- The array-module file ships a `_computeFbr_cupy_selfcheck()` (cupy path vs a
  NumPy reference produced by calling `_core(..., np)` directly, bypassing
  auto-dispatch), run from `if __name__ == "__main__":` and gated on
  `_HAVE_CUPY_GPU` (`skipped (no GPU available …)` otherwise). Run it directly,
  e.g. `uv run python -m sarFbr.fbr`.

### GPU coverage rule

The gate (`scripts/check.py`) runs on CPU CI (no GPU), so the GPU dispatch
branch (`if _HAVE_CUPY_GPU:`), the cupy import try-body, the
`_computeFbr_cupy_selfcheck()` GPU body, and the `__main__` selfcheck entry are
marked `# pragma: no cover` to keep them out of the coverage denominator. **The
≥90% gate therefore applies to the CPU-reachable (NumPy) path only.** This is
the documented accommodation for GPU-only code, not a way to skip real coverage
— the NumPy path must be fully tested. A future GPU CI job would lift the
pragma.

## FBR algorithm and numerics that are easy to break

This package implements only the **FBR image-computation** step (iterative
CV-based stable-pixel selection + MP combination) from Taillade, Thirion-Lefevre
& Guinvarc'h, *Remote Sensing* 2020, 12(11), 1720. The downstream
likelihood-ratio change-detection test from the paper is out of scope. Inputs
are assumed to be SAR **amplitude** images; the theoretical speckle CV therefore
follows the Nakagami distribution with shape `m = L` (`L` = equivalent number of
looks), reducing to the Rayleigh case for `L = 1`.

`computeFbr(stack, mode="mp", enl=1.0, a=0.0)` — `stack` is `(D, H, W)`
amplitude, axis 0 is time. Per spatial pixel, NaN-aware so rejected dates drop
out of later iterations:

1. Compute temporal CV `= std / mean` over the D dates.
2. While any pixel's CV exceeds the acceptance threshold
   `Psi(k) = CV_speckle + a/sqrt(k)` — with `CV_speckle = theoreticalCv(enl)`
   and `k` the per-pixel surviving-date count — reject the single date whose
   value departs most from that pixel's temporal mean (the putative ephemeral
   target) by setting its validity entry to NaN, then recompute. The loop is
   bounded by `fbrIteration < D - 2` (leaves at least 2 surviving dates). The
   `a/sqrt(k)` margin (the paper's false-alarm term, `a = alpha`) cushions the
   threshold as dates are rejected; with `a = 0` it reduces to the plain
   `CV_speckle` threshold. `fbrThreshold` is recomputed each iteration as dates
   drop out.
3. Combine survivors into the FBR value: MP = `sqrt(nanmean(x**2, axis=0))`, the
   multi-look average of intensity (amplitude²) over each pixel's surviving
   stable dates, rooted back to amplitude — returning an `(H, W)` array.

### Important divergences from the README / docstrings

- **RP mode is not implemented.** `mode="rp"` falls back to `"mp"` with a
  `print(...)` warning (captured in tests via `capsys`). The README describes
  true random-pick behavior that does not exist yet.
- **FBR is a per-pixel `(H, W)` image.** MP mode computes
  `fbr = np.sqrt(np.nanmean(x**2, axis=0))` — incoherent averaging of intensities,
  not amplitudes.
- **Mask dtype/values.** `mask` is a float array of `1.0` (stable) and `NaN`
  (rejected), not the boolean `(D, H, W)` mask the README once described. The
  shape `(D, H, W)` is correct.
- **Complex SLC inputs.** A complex `stack` is reduced to amplitude `|z|` before
  the float32 cast so the imaginary part is not dropped (a bare
  `.astype(float32)` would keep only the real component — `ComplexWarning`).
- **Loop iteration bound.** The rejection loop runs while
  `cvX > threshold AND fbrIteration < D - 2`; it can exit either because the CV
  converged (CV condition False) or because the iteration limit was reached
  (`fbrIteration < D - 2` False) with CV still above threshold. Both exit
  branches are tested.

## Extensibility

- The library is single-module (`src/sarFbr/fbr.py`); `__init__.py` re-exports
  `computeFbr` and `theoreticalCv` as the public API. Keep public functions
  lowercase camelCase.
- Runtime deps stay limited to `numpy`. CuPy must never be added to
  `[project.dependencies]`. A future RP combination mode would extend
  `_computeFbr_core` (keeping the `xp`-parameterized single-core pattern) and
  drop the `mode == "rp"` fallback in `computeFbr`.