"""Frozen Background Reference (FBR) image computation.

Implements the iterative CV-based stable-pixel selection described in

    Taillade, T.; Thirion-Lefevre, L.; Guinvarc'h, R.
    "Detecting Ephemeral Objects in SAR Time-Series Using Frozen Background-Based
    Change Detection." Remote Sensing, 2020, 12(11), 1720.
    https://doi.org/10.3390/rs12111720

Inputs are assumed to be SAR **amplitude** images. The theoretical speckle
coefficient of variation therefore follows the Nakagami distribution with shape
``m = L`` (``L`` being the equivalent number of looks), reducing to the Rayleigh
case for ``L = 1``.

Per spatial pixel the stable background set is built by iteratively rejecting
the worst date:

1. Compute the temporal coefficient of variation ``CV = std / mean`` over the
   *D* dates. The statistic is NaN-aware, so dates rejected in earlier
   iterations drop out of the mean/std of later ones.
2. While any pixel's CV exceeds the speckle threshold ``CV_speckle =
   theoreticalCv(enl)``, flag as invalid the single date whose value shows the
   largest absolute deviation from that pixel's temporal mean (the likely
   ephemeral target), set the matching entry of the validity mask to ``NaN``,
   and recompute the CV. Dates are removed one at a time until every surviving
   pixel's CV is at or below the speckle level.
3. Combine the surviving dates into a single FBR value by averaging them (MP
   mode). RP mode (random pick) is not implemented and currently falls back to
   MP with a warning.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import cupy as cp
    try:
        _HAVE_CUPY_GPU = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        _HAVE_CUPY_GPU = False
except ImportError:
    cp = None
    _HAVE_CUPY_GPU = False

__all__ = ["computeFbr", "theoreticalCv"]

_EPS = 1e-12

# False-alarm margin for the paper's acceptance threshold
# ``Psi(k) = CV_speckle + alpha / sqrt(k)``. The current iterative implementation
# rejects dates against a plain ``CV_speckle`` threshold (no margin), so
# ``_ALPHA`` is currently unused; it is kept as a module constant so a
# margin-based threshold can be reintroduced without widening the public API.
_ALPHA = 3.0


def theoreticalCv(enl: float) -> float:
    """Theoretical speckle coefficient of variation for amplitude SAR data.

    The speckle in an amplitude image follows a Nakagami distribution with shape
    ``m = enl`` (the equivalent number of looks *L*). The exact CV is

        CV(L) = sqrt(L * Gamma(L)^2 / Gamma(L + 1/2)^2 - 1)

    which reduces to ``sqrt(4/pi - 1)`` for ``L = 1`` (the single-look Rayleigh
    case).

    Parameters
    ----------
    enl : float
        Equivalent number of looks *L* (>= 1).
    """
    if enl <= 0:
        raise ValueError(f"enl must be > 0, got {enl}")

    theoreticalCv = np.sqrt(
        math.gamma(enl)
        * math.gamma(enl + 1)
        / (math.gamma(enl + 0.5)**2)
    )

    return theoreticalCv

def _computeFbr_cupy(stack, mode: str, enl: float):
    """cupy backend for computeFbr (`stack` is a cupy.ndarray; returns cupy).

    Same iterative CV-based stable-pixel selection as the NumPy path, but the
    per-iteration nanmean/nanstd/nanmax reductions and the rejection mask run on
    the GPU. The loop guard still syncs to the host once per iteration
    (``bool(cp.any(...))`` reads the any-reduction back to a Python bool), so this
    is most worthwhile when D is large and H*W is large enough that the reduction
    kernels dominate over the host/device round-trips. ``stack`` stays on the GPU
    for the whole run; the caller brings ``fbr``/``mask`` back with ``cp.asnumpy``.
    """
    x = stack.astype(cp.float32)
    if x.ndim != 3:
        raise ValueError(f"stack must be 3-D (D, H, W); got shape {x.shape}")
    if x.size == 0:
        raise ValueError("stack is empty")

    D, H, W = x.shape
    cvSpeckle = float(theoreticalCv(enl))
    # Iteratively reject the worst date per pixel until every pixel's temporal CV
    # is at or below the speckle level. `cvX` is the per-pixel CV over dates
    # (shape (H, W)); `validMask` carries NaN where a date has been rejected.
    validMask = cp.ones_like(x)
    cvX = cp.nanstd(x, axis=0) / cp.nanmean(x, axis=0)
    fbrIteration = 0
    while bool(cp.any(cvX > cvSpeckle)) and fbrIteration < D - 2:
        # For pixels still above the threshold, reject the single date whose
        # value departs most from that pixel's temporal mean (the putative
        # ephemeral target): set its validity entry to NaN.
        xMeanDeviation = cp.abs(x - cp.nanmean(x, axis=0)[None, ...])
        xMaxMeanDeviation = cp.nanmax(xMeanDeviation, axis=0)
        reject = (cvX > cvSpeckle)[None, ...] & (xMeanDeviation == xMaxMeanDeviation[None, ...])
        # Cast to float32 explicitly: cupy's scalar promotion for `cp.nan` is
        # version-dependent, and the mask must stay float32 to match the NumPy path.
        validMask = cp.where(reject, cp.nan, validMask).astype(cp.float32, copy=False)
        x *= validMask
        cvX = cp.nanstd(x, axis=0) / cp.nanmean(x, axis=0)
        fbrIteration += 1

    if mode == "mp":
        # MP: average the surviving stable dates into the FBR image.
        fbr = cp.sqrt(cp.nanmean(x**2))

    return fbr, validMask


def computeFbr(stack, mode: str = "mp", enl: float = 1.0):
    """Compute the Frozen Background Reference image from a SAR time-series stack.

    Parameters
    ----------
    stack : array, shape (D, H, W)
        Time-series of SAR **amplitude** images; axis 0 is time (D dates).
    mode : {"mp", "rp"}, default "mp"
        ``"mp"`` averages the surviving stable dates per pixel (variance reduced
        by roughly sqrt(k)). ``"rp"`` would draw a single random stable date per
        pixel to preserve the original speckle statistics, but it is not
        implemented: an ``"rp"`` request currently falls back to ``"mp"`` with a
        warning.
    enl : float, default 1.0
        Equivalent number of looks *L*, used to derive the theoretical speckle CV
        via :func:`theoreticalCv`.

    Returns
    -------
    fbr : ndarray, shape (H, W)
        The frozen background reference image (MP: the per-pixel mean over the
        surviving stable dates).
    mask : ndarray, shape (D, H, W)
        Validity mask matching ``stack``. Entries are ``1`` where the date was
        judged temporally stable and contributed to the FBR image, and ``NaN``
        where the date was rejected as an ephemeral target. ``mask[d, h, w]``
        corresponds to the stack entry at date ``d`` and pixel ``(h, w)``.
    """
    if mode not in ("mp", "rp"):
        raise ValueError(f"mode must be 'mp' or 'rp', got {mode!r}")
    if mode == "rp":
        print("rp is not implemented, defaulting to mp")
        mode = "mp"

    # Dispatch on the input array type: a cupy.ndarray is processed on the GPU
    # (cupy reductions — nanmean/nanstd/nanmax) and returned as cupy arrays;
    # anything else uses the NumPy path and returns NumPy arrays. A caller that
    # wants GPU acceleration moves the stack to the GPU once with cupy.asarray
    # and brings `fbr`/`mask` back with cupy.asnumpy. (GPU use is gated on a
    # device actually being present; without one the cupy branch is never taken.)
    if _HAVE_CUPY_GPU and isinstance(stack, cp.ndarray):
        print("computeFbr: using cupy GPU acceleration")
        return _computeFbr_cupy(stack, mode, enl)

    x = stack.astype(np.float32)
    if x.ndim != 3:
        raise ValueError(f"stack must be 3-D (D, H, W); got shape {x.shape}")
    if x.size == 0:
        raise ValueError("stack is empty")

    D, H, W = x.shape
    cvSpeckle = float(theoreticalCv(enl))
    # Iteratively reject the worst date per pixel until every pixel's temporal CV
    # is at or below the speckle level. `cvX` is the per-pixel CV over dates
    # (shape (H, W)); `validMask` carries NaN where a date has been rejected.
    validMask = np.ones_like(x)
    cvX = np.nanstd(x, axis=0) / np.nanmean(x, axis=0)
    fbrIteration = 0
    while np.any(cvX > cvSpeckle) and fbrIteration < D - 2:
        # For pixels still above the threshold, reject the single date whose
        # value departs most from that pixel's temporal mean (the putative
        # ephemeral target): set its validity entry to NaN.
        xMeanDeviation = np.abs(x - np.nanmean(x,axis = 0)[np.newaxis,...])
        xMaxMeanDeviation = np.nanmax(xMeanDeviation,axis = 0)
        validMask = np.where((cvX > cvSpeckle)[np.newaxis,...] & (xMeanDeviation == xMaxMeanDeviation[np.newaxis,...]), np.nan, validMask)
        x *= validMask
        cvX = np.nanstd(x, axis=0) / np.nanmean(x, axis=0)
        fbrIteration += 1

    if mode == "mp":
        # MP: average the surviving stable dates into the FBR image.
        fbr = np.sqrt(np.nanmean(x**2))

    return fbr, validMask


def _computeFbr_selfcheck():
    """Equivalence check: cupy computeFbr vs the NumPy reference.

    Both paths run the rejection loop in float32, so the reductions agree to
    ~1e-5 relative but not bit-for-bit; we check allclose, not array_equal. The
    input is built so clean pixels are constant in time (CV = 0, never rejected)
    and only the planted target dates are rejected — the argmax selection is
    unambiguous, so the two paths reject the same dates and the masks match
    exactly (allclose with equal_nan). Skipped (and reported as such) when no
    GPU is available.
    """
    if not _HAVE_CUPY_GPU:
        print("computeFbr self-check: skipped (no GPU available — NumPy path only).")
        return True

    rng = np.random.default_rng(0)
    D, H, W = 10, 32, 32
    # Per-pixel constant background (constant in time -> CV = 0, never rejected).
    bg = rng.uniform(0.5, 2.0, size=(H, W)).astype(np.float32)
    stack = np.broadcast_to(bg[None, :, :], (D, H, W)).copy().astype(np.float32)
    # Plant well-separated ephemeral targets on a few pixels/dates.
    stack[5, 0, 1] = 20.0
    stack[2, 1, 0] = 20.0
    stack[7, 3, 3] = 20.0
    stack[4, 5, 5] = 20.0

    # Reference: NumPy path.
    fbr_ref, mask_ref = computeFbr(stack, mode="mp", enl=1.0)
    # GPU: move up once, run on GPU, bring back once.
    g = cp.asarray(stack)
    fbr_g, mask_g = computeFbr(g, mode="mp", enl=1.0)
    fbr_g = cp.asnumpy(fbr_g)
    mask_g = cp.asnumpy(mask_g)

    shape_ok = mask_g.shape == mask_ref.shape == (D, H, W)
    dtype_ok = mask_g.dtype == mask_ref.dtype == np.float32
    fbr_close = np.allclose(np.asarray(fbr_g), np.asarray(fbr_ref), rtol=1e-4, atol=1e-5)
    mask_close = np.allclose(mask_g, mask_ref, rtol=0.0, atol=0.0, equal_nan=True)
    max_fbr_diff = float(np.nanmax(np.abs(np.asarray(fbr_g) - np.asarray(fbr_ref))))

    ok = bool(shape_ok and dtype_ok and fbr_close and mask_close)
    print(f"computeFbr self-check: {'PASS' if ok else 'FAIL'} "
          f"(shape={shape_ok}, dtype={dtype_ok}, fbr_close={fbr_close}, "
          f"mask_close={mask_close}, max_fbr_diff={max_fbr_diff:.3e})")
    if not ok:
        print("ref fbr", np.asarray(fbr_ref), "got fbr", np.asarray(fbr_g))
    return ok


if __name__ == "__main__":
    _computeFbr_selfcheck()