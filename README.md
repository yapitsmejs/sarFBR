# sar-fbr

SAR Frozen Background Reference (FBR) method for ephemeral-object change
detection in SAR time-series — a pure-NumPy implementation of the FBR
image-computation step from Taillade, Thirion-Lefevre & Guinvarc'h, *Remote
Sensing* 2020, 12(11), 1720 (<https://doi.org/10.3390/rs12111720>). It builds a
target-free reference image of a scene's stable background by iteratively
rejecting dates whose temporal coefficient of variation exceeds the theoretical
speckle CV, then multi-looks the survivors. `numpy` is required; `cupy` is an
**optional** GPU fast-path with a NumPy fallback when it (or a CUDA device) is
absent.

This package implements only the FBR image-computation step (iterative CV-based
stable-pixel selection + MP combination). The downstream likelihood-ratio
change-detection test from the paper is out of scope. Inputs are assumed to be
SAR **amplitude** images.

## Install

This project is managed with [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev          # create .venv, install runtime deps (numpy) + dev tooling (pytest, ruff, mypy)
```

### As a dependency from git

The package is **not on PyPI**; install it directly from GitHub. `numpy` is
pulled automatically as the required dependency; `cupy` is intentionally **not**
installed here (see [Optional: GPU acceleration with CuPy](#optional-gpu-acceleration-with-cupy) below).

```bash
pip install git+https://github.com/yapitsmejs/sarFBR.git
# or with uv:
uv add "sar-fbr @ git+https://github.com/yapitsmejs/sarFBR.git"
```

## Optional: GPU acceleration with CuPy

CuPy is **not** a declared dependency, because the correct wheel depends on the
host's NVIDIA GPU and CUDA Toolkit version and cannot be chosen statically.
Install it manually for your platform.

**With a CUDA Toolkit installed** (uses system CUDA, no bundled libs):

| CUDA major | Wheel |
|---|---|
| 11 | `cupy-cuda11x` |
| 12 | `cupy-cuda12x` |
| 13 | `cupy-cuda13x` |

```bash
pip install cupy-cuda12x   # match your CUDA Toolkit major version (11 / 12 / 13)
```

**With an NVIDIA GPU but no CUDA Toolkit** (bundles CUDA libraries via PyPI):

```bash
pip install "cupy-cuda12x[ctk]"   # use the major your driver supports; default 12
```

Determine your CUDA major from `nvcc --version` (Toolkit) or the "CUDA Version"
line of `nvidia-smi` (driver-supported runtime). Only one cupy distribution may
be installed at a time — uninstall any existing `cupy` / `cupy-cuda*` first.

When developing from a clone, the bundled `scripts/install_cupy.py` automates
this detection and installation:

```bash
uv run python scripts/install_cupy.py
```

(That script is not shipped in the wheel and is therefore unavailable to
git-installed consumers — use the manual `pip install` steps above.)

## Usage

`computeFbr` dispatches on the **input array type**, not a flag: a `cupy.ndarray`
runs on the GPU and returns CuPy arrays; anything else uses the NumPy path. Move
the stack to the GPU once and bring the results back with `cupy.asnumpy`.

```python
import numpy as np
import sarFbr

# stack: time-series of SAR amplitude images, shape (D, H, W); axis 0 is time.
stack = np.load("stack.npy")                       # (D, H, W)

# NumPy path (default; works with or without cupy installed)
fbr, mask = sarFbr.computeFbr(stack, mode="mp", enl=16)

# GPU path: move the stack up once, run on the GPU, bring the results back once
import cupy as cp
g = cp.asarray(stack)
fbr_g, mask_g = sarFbr.computeFbr(g, mode="mp", enl=16)
fbr, mask = cp.asnumpy(fbr_g), cp.asnumpy(mask_g)
```

### API

| Function | Description |
| --- | --- |
| `computeFbr(stack, mode="mp", enl=1.0, a=0.0)` | Compute the FBR image from a `(D, H, W)` amplitude stack. `mode` is `"mp"` (multi-look the surviving stable dates — intensity averaged then rooted back to amplitude, variance reduced by ~√k) or `"rp"` (not implemented; falls back to `"mp"` with a warning). `enl` is the equivalent number of looks *L*; `a` is the false-alarm margin in the acceptance threshold `Ψ(k) = CV_speckle + a/√k`. Returns `(fbr, mask)`, where `mask` is a `(D, H, W)` float array — `1.0` where the date was stable, `NaN` where rejected. |
| `theoreticalCv(enl)` | Theoretical speckle CV for amplitude SAR with *L* = `enl` looks (Nakagami, `sqrt(L·Γ(L)²/Γ(L+1/2)² − 1)`); reduces to the Rayleigh `sqrt(4/π − 1)` at `L = 1`. |

## Development

```bash
uv run pytest                   # run tests (dev loop, no coverage gate)
uv run ruff check .             # lint
uv run ruff format .            # format
uv run mypy src/sarFbr           # type check
uv run python scripts/check.py  # GATE: lint + format + types + tests + branch coverage ≥ 90%
```

Run `scripts/check.py` before pushing or opening a PR — it fails if ruff is
unhappy, formatting would change, **mypy** finds a type error, or branch
coverage on the package drops below 90%. See [`CONVENTIONS.md`](CONVENTIONS.md)
for the full naming, code-style, CuPy/GPU, and test-driven conventions.

### Automation

These run the same gate as you do locally, so a red PR cannot merge:

- **CI** (`.github/workflows/ci.yml`) runs `scripts/check.py` on every push and
  PR.
- **Git hooks** (`.githooks/`) are opt-in per clone — fast `ruff` checks on
  commit, the full gate on push. One-time setup:
  ```bash
  git config core.hooksPath .githooks
  git update-index --chmod=+x .githooks/pre-commit .githooks/pre-push
  ```
  Use `git commit --no-verify` during active red-phase work; the pre-push hook
  and CI still enforce before anything lands.
- **Dependabot** (`.github/dependabot.yml`) opens weekly dependency PRs (uv +
  GitHub Actions) that must pass the gate like any other.
- **PR template** (`.github/PULL_REQUEST_TEMPLATE.md`) prompts contributors for
  the red-green workflow and the gate checklist.

### GPU self-check

The `__main__` self-check (`uv run python -m sarFbr.fbr`) compares the CuPy path
against the NumPy reference. The CuPy check is skipped with `no GPU` on CPU-only
hosts. Run it after touching the FBR code on a GPU machine. (This is excluded
from the coverage gate — see `CONVENTIONS.md` "GPU coverage rule".)

## License

MIT.