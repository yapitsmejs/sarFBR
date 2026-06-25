# sarFBR

A Python implementation of the **Frozen Background Reference (FBR)** method for
detecting ephemeral objects in SAR (Synthetic Aperture Radar) time-series, based on

> Taillade, T.; Thirion-Lefevre, L.; Guinvarc'h, R.
> *Detecting Ephemeral Objects in SAR Time-Series Using Frozen Background-Based
> Change Detection.* **Remote Sensing**, 2020, 12(11), 1720.
> https://doi.org/10.3390/rs12111720

The FBR method builds a target-free reference image of a scene's stable background
from a stack of SAR acquisitions. Each pixel's temporal coefficient of variation (CV)
is compared against the theoretical speckle CV for amplitude imagery
(Rayleigh–Nakagami, parameterised by the equivalent number of looks). Dates whose CV
exceeds the threshold Ψ = CV_speckle + α/√D are rejected as temporally unstable
(i.e. occupied by an ephemeral target such as a car or ship). The surviving "stable
candidate" dates are then combined into a single FBR image in one of two modes:

- **RP (Random Pixel)** — pick a single random stable date per pixel; preserves the
  original speckle statistics.
- **MP (Multi-Temporal Pixels)** — coherently average all stable dates per pixel;
  reduces variance by √D, improving low-SNR target detection.

This package implements only the **FBR image computation** step (RP + MP, iterative
CV-based stable-pixel selection). The downstream likelihood-ratio change-detection
test from the paper is out of scope. Inputs are assumed to be SAR **amplitude**
images. The core runs on NumPy; optional CuPy GPU acceleration is planned.

## Installation

Install directly from git (the intended distribution channel):

```bash
pip install git+https://github.com/yapitsmejs/sarFBR.git
```

## Quick start

```python
import numpy as np
import sarfbr

# stack: time-series of SAR amplitude images, shape (D, H, W), D = number of dates.
stack = np.load("stack.npy")                       # (D, H, W)
fbr, mask = sarfbr.computeFbr(
    stack,
    mode="mp",                                     # "mp" (average) or "rp" (random pick)
    enl=16,                                         # equivalent number of looks L
)
# fbr : (H, W) frozen background reference image
# mask: (D, H, W) bool — True where the stack entry was used (temporally stable)
```

## API

### `sarfbr.computeFbr(stack, mode="mp", enl=1.0)`

Compute the Frozen Background Reference image from a SAR time-series stack.

| Parameter | Description |
|-----------|-------------|
| `stack` | 3-D array `(D, H, W)`; axis 0 is time (D dates). Assumed to be **amplitude**. |
| `mode` | `"mp"` (average the stable dates — variance reduced by √k) or `"rp"` (draw one random stable date per pixel, preserving speckle statistics). |
| `enl` | Equivalent number of looks *L*, used to derive the theoretical speckle CV via `theoreticalCv`. |

Returns a tuple `(fbr, mask)`:

| Return | Description |
|--------|-------------|
| `fbr` | `(H, W)` frozen background reference image. |
| `mask` | `(D, H, W)` boolean valid mask: `mask[d, h, w]` is `True` when the stack entry at date `d`, pixel `(h, w)` was judged temporally stable and contributed to the FBR image. |

The acceptance threshold is `Ψ(k) = CV_speckle + α/√k`, where `CV_speckle =
theoreticalCv(enl)` and α is a fixed internal false-alarm margin (~3-sigma). For each
pixel the largest prefix of the temporally-sorted dates whose CV falls below Ψ is
accepted as the stable set; everything above it is treated as a target date.

### `sarfbr.theoreticalCv(enl)`

Theoretical speckle coefficient of variation for amplitude SAR data with *L* = `enl`
equivalent looks. Follows the Nakagami distribution with shape *m = L*:

    CV(L) = sqrt(L · Γ(L)² / Γ(L + 1/2)² − 1)

which reduces to `sqrt(4/π − 1)` (the Rayleigh case) for `L = 1`.

## References

- Primary paper: <https://doi.org/10.3390/rs12111720>
- SONDRA group write-up: <https://sondra.fr/detecting-ephemeral-objects-in-sar-time-series-using-frozen-background-based-change-detection/>

## License

MIT.