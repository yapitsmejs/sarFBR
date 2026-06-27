# sarFBR

A Python implementation of the **Frozen Background Reference (FBR)** method for
detecting ephemeral objects in SAR (Synthetic Aperture Radar) time-series, based on
Taillade, Thirion-Lefevre & Guinvarc'h, *Remote Sensing* 2020, 12(11), 1720
(<https://doi.org/10.3390/rs12111720>). It builds a target-free reference image of a
scene's stable background by iteratively rejecting dates whose temporal coefficient
of variation exceeds the theoretical speckle CV, then multi-looks the survivors.
`numpy` is required; `cupy` is an **optional** GPU fast-path with a NumPy fallback
when it (or a CUDA device) is absent.

This package implements only the FBR image-computation step (iterative CV-based
stable-pixel selection + MP/RP combination). The downstream likelihood-ratio
change-detection test from the paper is out of scope. Inputs are assumed to be SAR
**amplitude** images.

## Install — pull from GitHub into an external repo

The package is **not on PyPI**; install it directly from GitHub. `numpy` is pulled
automatically as a required dependency. `cupy` is intentionally **not** installed
here — it is a machine-specific CUDA wheel (see
[Set up cupy](#set-up-cupy-optional-gpu-fast-path) below).

### uv (this project's own toolchain)

New project:

```bash
uv init myrepo && cd myrepo
uv add "git+https://github.com/yapitsmejs/sarFBR.git"
```

Pin to a specific commit (or a tag, once one is cut — same `@<ref>` syntax):

```bash
uv add "git+https://github.com/yapitsmejs/sarFBR.git@<commit-sha>"
```

### pip (any virtualenv)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      | macOS/Linux:  source .venv/bin/activate
python -m pip install "git+https://github.com/yapitsmejs/sarFBR.git"
```

Pin to a commit/tag:

```bash
python -m pip install "git+https://github.com/yapitsmejs/sarFBR.git@<commit-sha>"
```

## Set up cupy (optional GPU fast-path)

A usable GPU is **not** required — `sarfbr` runs on the NumPy fallback when `cupy`
is absent or no CUDA device is detected. Install cupy only if you want the GPU
path, and install it into the **same venv** that holds `sarfbr`.

### Option A — manual (recommended, self-contained)

1. Detect your CUDA major version:

   ```bash
   nvcc --version        # CUDA Toolkit present -> "release X.Y" gives the major
   nvidia-smi           # no toolkit -> read the "CUDA Version:" the driver supports
   ```

2. Install the matching wheel. Pick `cupy-cuda11x`, `cupy-cuda12x`, or
   `cupy-cuda13x` by your CUDA major (11, 12, or 13). If you have a GPU but **no**
   CUDA Toolkit, add the `[ctk]` extra so the wheel bundles CUDA libraries via PyPI.

   uv:

   ```bash
   uv pip install cupy-cuda13x                       # toolkit present -> system CUDA
   uv pip install "cupy-cuda12x[ctk]"                # GPU, no toolkit -> bundled CUDA
   ```

   pip:

   ```bash
   python -m pip install cupy-cuda13x
   python -m pip install "cupy-cuda12x[ctk]"
   ```

   Only **one** `cupy-cuda*x` distribution may be installed at a time — if you are
   upgrading or switching CUDA versions, uninstall the others first:

   ```bash
   uv pip uninstall cupy cupy-cuda11x cupy-cuda12x cupy-cuda13x
   # or:  python -m pip uninstall -y cupy cupy-cuda11x cupy-cuda12x cupy-cuda13x
   ```

3. Verify a device is visible:

   ```bash
   python -c "import cupy; print(cupy.__version__, cupy.cuda.runtime.getDeviceCount())"
   ```

### Option B — reuse this repo's auto-detecting installer (convenience)

This repo ships `scripts/install_cupy.py`, which auto-detects the GPU + CUDA Toolkit
and installs the right wheel (no GPU → installs nothing; GPU + toolkit →
`cupy-cuda{MAJOR}x`; GPU, no toolkit → `cupy-cuda{MAJOR}x[ctk]`). It installs into its
**own** repo's `.venv`, so to use it for your external repo, fetch just the script and
run it from your repo root (it auto-detects `REPO_ROOT/.venv`):

```bash
# macOS/Linux
curl -fsSLO https://raw.githubusercontent.com/yapitsmejs/sarFBR/main/scripts/install_cupy.py
python install_cupy.py

# Windows PowerShell
Invoke-WebRequest -UseBasicParsing -OutFile install_cupy.py `
  https://raw.githubusercontent.com/yapitsmejs/sarFBR/main/scripts/install_cupy.py
python install_cupy.py
```

(Alternatively, `git clone` this repo and run `uv run python scripts/install_cupy.py`
inside it, then point that venv at your project.)

## Usage

`computeFbr` dispatches on the **input array type**, not a flag: a `cupy.ndarray`
runs on the GPU and returns CuPy arrays; anything else uses the NumPy path. Move the
stack to the GPU once and bring the results back with `cupy.asnumpy`.

```python
import numpy as np
import sarfbr

# stack: time-series of SAR amplitude images, shape (D, H, W); axis 0 is time.
stack = np.load("stack.npy")                       # (D, H, W)

# NumPy path (default; works with or without cupy installed)
fbr, mask = sarfbr.computeFbr(stack, mode="mp", enl=16)

# GPU path: move the stack up once, run on the GPU, bring the results back once
import cupy as cp
g = cp.asarray(stack)
fbr_g, mask_g = sarfbr.computeFbr(g, mode="mp", enl=16)
fbr, mask = cp.asnumpy(fbr_g), cp.asnumpy(mask_g)
```

### API

| Function | Description |
| --- | --- |
| `computeFbr(stack, mode="mp", enl=1.0, a=0.0)` | Compute the FBR image from a `(D, H, W)` amplitude stack. `mode` is `"mp"` (multi-look the surviving stable dates — intensity averaged then rooted back to amplitude, variance reduced by ~√k) or `"rp"` (not implemented; falls back to `"mp"` with a warning). `enl` is the equivalent number of looks *L*; `a` is the false-alarm margin in the acceptance threshold `Ψ(k) = CV_speckle + a/√k`. Returns `(fbr, mask)`, where `mask` is a `(D, H, W)` float array — `1.0` where the date was stable, `NaN` where rejected. |
| `theoreticalCv(enl)` | Theoretical speckle CV for amplitude SAR with *L* = `enl` looks (Nakagami, `sqrt(L·Γ(L)²/Γ(L+1/2)² − 1)`); reduces to the Rayleigh `sqrt(4/π − 1)` at `L = 1`. |

## License

MIT.