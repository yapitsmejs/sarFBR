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
2. While any pixel's CV exceeds the acceptance threshold
   ``Psi(k) = CV_speckle + a / sqrt(k)`` — with ``CV_speckle =
   theoreticalCv(enl)`` and ``k`` the per-pixel count of surviving dates — flag
   as invalid the single date whose value shows the largest absolute deviation
   from that pixel's temporal mean (the likely ephemeral target), set the
   matching entry of the validity mask to ``NaN``, and recompute the CV. The
   ``a / sqrt(k)`` margin (the paper's false-alarm term, ``a = alpha``) cushions
   the threshold as dates are rejected; with ``a = 0`` it reduces to the plain
   speckle level. Dates are removed one at a time until every surviving pixel's
   CV is at or below the threshold.
3. Combine the surviving dates into a per-pixel FBR image. In MP mode this is
   the multi-look average of each pixel's surviving stable dates: the intensity
   (amplitude**2) is averaged over the survivors and square-rooted back to
   amplitude, i.e. ``fbr = sqrt(nanmean(x**2, axis=0))`` — incoherent averaging
   of intensities, not amplitudes, yielding an ``(H, W)`` image. RP mode (random
   pick) is not implemented and currently falls back to MP with a warning.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# cupy is intentionally NOT a declared dependency (see scripts/install_cupy.py
# and README.md). It is imported eagerly at module load via try/except; a usable
# GPU is not required: we detect at import whether a CUDA device is actually
# present (_HAVE_CUPY_GPU = cp.cuda.runtime.getDeviceCount() > 0, treating
# 0/exception as "no GPU") and fall back to the NumPy path if not. The module
# never hard-fails on a CPU-only host.
#
# `cp` is typed `Any` because cupy mirrors numpy's API but is not stub-typed; the
# array-module dispatch (`xp` param) is likewise `Any` -- see CONVENTIONS.md
# "GPU coverage rule".
cp: Any = None
_HAVE_CUPY_GPU: bool = False
try:  # pragma: no cover  -- cupy not installed on CPU CI; try body is un-runnable there
    import cupy as _cp

    cp = _cp
    _HAVE_CUPY_GPU = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAVE_CUPY_GPU = False


__all__ = ["computeFbr", "theoreticalCv"]


def theoreticalCv(enl: float) -> float:
    """Theoretical speckle coefficient of variation for amplitude SAR data.

    The speckle in an amplitude image follows a Nakagami distribution with shape
    ``m = enl`` (the equivalent number of looks *L*). The exact CV is

        CV(L) = sqrt(L * Gamma(L)^2 / Gamma(L + 1/2)^2 - 1)

    which reduces to ``sqrt(4/pi - 1)`` for ``L = 1`` (the single-look Rayleigh
    case).

    Args:
        enl: Equivalent number of looks *L* (>= 1).

    Raises:
        ValueError: if ``enl <= 0``.
    """
    if enl <= 0:
        raise ValueError(f"enl must be > 0, got {enl}")

    cv: float = float(
        np.sqrt(math.gamma(enl) * math.gamma(enl + 1) / (math.gamma(enl + 0.5) ** 2) - 1)
    )
    return cv


def _computeFbr_core(
    stack: Any,
    mode: str,
    enl: float,
    a: float,
    xp: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Single ``xp``-parameterized core for computeFbr (numpy or cupy).

    Runs the iterative CV-based stable-pixel selection on whichever array module
    ``xp`` is (``np`` or ``cp``). The per-iteration nanmean/nanstd/nanmax
    reductions and the rejection mask run through ``xp.*`` — one code path on
    both backends, no mirrored cupy/NumPy functions. On the cupy path the loop
    guard still syncs to the host once per iteration (``bool(xp.any(...))`` reads
    the any-reduction back to a Python bool), so the GPU path is most worthwhile
    when D is large and H*W is large enough that the reduction kernels dominate
    over the host/device round-trips. The caller (``computeFbr``) handles
    device placement: it calls this core with ``cp`` on the GPU branch (having
    moved the stack up with ``cp.asarray``) and with ``np`` on the CPU branch.

    Args:
        stack: array-like, shape (D, H, W). Time-series of SAR amplitude images;
            axis 0 is time. Real or complex (complex is reduced to amplitude via
            ``|z|`` before the float32 cast so the imaginary part is not dropped).
        mode: combination mode. Only ``"mp"`` is implemented (the caller converts
            ``"rp"`` to ``"mp"`` before reaching here).
        enl: equivalent number of looks *L*, used to derive the speckle CV.
        a: false-alarm margin coefficient in the acceptance threshold.
        xp: the array module to compute on (``np`` or ``cp``).

    Returns:
        ``(fbr, validMask)``: ``fbr`` is the ``(H, W)`` multi-look FBR image
        (``sqrt(nanmean(intensity))`` over surviving stable dates); ``validMask``
        is a ``(D, H, W)`` float array with ``1.0`` where a date was stable and
        ``NaN`` where it was rejected.

    Raises:
        ValueError: if ``stack`` is not 3-D or is empty.
    """
    # Move to the target backend (no-op on NumPy; host->device on cupy) and, if
    # `stack` carries complex SLC data, take the magnitude first so amplitude is
    # preserved: a bare `.astype(xp.float32)` on a complex array would drop the
    # imaginary part (ComplexWarning) and keep only the real component.
    x: np.ndarray = (
        xp.abs(xp.asarray(stack)) if xp.iscomplexobj(stack) else xp.asarray(stack)
    ).astype(xp.float32)
    if x.ndim != 3:
        raise ValueError(f"stack must be 3-D (D, H, W); got shape {x.shape}")
    if x.size == 0:
        raise ValueError("stack is empty")

    D, H, W = x.shape
    cvSpeckle = float(theoreticalCv(enl))
    # Per-pixel acceptance threshold Psi(k) = CV_speckle + a/sqrt(k), where k is
    # the per-pixel count of surviving (non-rejected) dates. With a=0 this
    # reduces to the plain CV_speckle threshold. Recomputed each iteration as
    # dates are rejected (k shrinks -> margin grows).
    fbrThreshold = cvSpeckle + a / xp.sqrt(xp.sum(~xp.isnan(x), axis=0))
    # Iteratively reject the worst date per pixel until every pixel's temporal CV
    # is at or below the threshold. `cvX` is the per-pixel CV over dates
    # (shape (H, W)); `validMask` carries NaN where a date has been rejected.
    validMask: np.ndarray = xp.ones_like(x)
    cvX = xp.nanstd(x, axis=0) / xp.nanmean(x, axis=0)
    fbrIteration = 0
    while bool(xp.any(cvX > fbrThreshold)) and fbrIteration < D - 2:
        # For pixels still above the threshold, reject the single date whose
        # value departs most from that pixel's temporal mean (the putative
        # ephemeral target): set its validity entry to NaN.
        xMeanDeviation = xp.abs(x - xp.nanmean(x, axis=0)[None, ...])
        xMaxMeanDeviation = xp.nanmax(xMeanDeviation, axis=0)
        reject = (cvX > fbrThreshold)[None, ...] & (xMeanDeviation == xMaxMeanDeviation[None, ...])
        # Cast to float32 explicitly: cupy's scalar promotion for `cp.nan` is
        # version-dependent, and the mask must stay float32. On the NumPy path
        # this is a no-op — the float32 array already absorbs the python-float
        # nan under NEP 50 weak promotion (so the mask never upcasts to float64).
        validMask = xp.where(reject, xp.nan, validMask).astype(xp.float32, copy=False)
        x *= validMask
        cvX = xp.nanstd(x, axis=0) / xp.nanmean(x, axis=0)
        fbrThreshold = cvSpeckle + a / xp.sqrt(xp.sum(~xp.isnan(x), axis=0))
        fbrIteration += 1

    # MP: multi-look the surviving stable dates per pixel — average the intensity
    # (amplitude**2) over survivors along the time axis and root back to
    # amplitude, yielding an (H, W) image. (The caller converts "rp" to "mp"
    # before reaching here, so `mode` is always "mp" at this point; the parameter
    # is retained for the future RP implementation.)
    fbr: np.ndarray = xp.sqrt(xp.nanmean(x**2, axis=0))
    return fbr, validMask


def computeFbr(
    stack: Any,
    mode: str = "mp",
    enl: float = 1.0,
    a: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Frozen Background Reference image from a SAR time-series stack.

    Args:
        stack: array-like, shape (D, H, W). Time-series of SAR **amplitude**
            images; axis 0 is time (D dates). Real or complex (complex SLC data
            is reduced to amplitude ``|z|`` before the float32 cast so the
            imaginary part is not dropped). May be a NumPy or CuPy array; when a
            CUDA device is available, NumPy inputs are auto-moved to the GPU and
            returned as NumPy (numpy in → numpy out), and CuPy inputs stay on the
            GPU and return CuPy arrays (cupy in → cupy out). On a CPU-only host
            the NumPy path is used regardless.
        mode: ``{""mp", "rp"}``, default ``"mp"``. ``"mp"`` averages the intensity
            of surviving stable dates per pixel (variance reduced by roughly
            sqrt(k)). ``"rp"`` would draw a single random stable date per pixel to
            preserve the original speckle statistics, but it is not implemented:
            an ``"rp"`` request currently falls back to ``"mp"`` with a warning.
        enl: equivalent number of looks *L*, used to derive the theoretical speckle
            CV via :func:`theoreticalCv`.
        a: false-alarm margin coefficient ``alpha`` in the paper's acceptance
            threshold ``Psi(k) = CV_speckle + a / sqrt(k)``, with ``k`` the
            per-pixel count of surviving dates. ``a = 0`` (the default) recovers
            the plain ``CV_speckle`` threshold; a larger ``a`` cushions the
            threshold as dates are rejected (the margin grows when few dates
            remain), reducing over-rejection.

    Returns:
        ``(fbr, mask)``: ``fbr`` is the ``(H, W)`` frozen background reference
        image (MP: the multi-look average over the surviving stable dates —
        ``sqrt(mean(intensity))``, i.e. the intensity-averaged value rooted back
        to amplitude). ``mask`` is a ``(D, H, W)`` float array matching ``stack``
        — ``1.0`` where the date was judged temporally stable and contributed to
        the FBR image, and ``NaN`` where the date was rejected as an ephemeral
        target. ``mask[d, h, w]`` corresponds to the stack entry at date ``d`` and
        pixel ``(h, w)``.

    Raises:
        ValueError: if ``mode`` is not ``"mp"`` or ``"rp"``, or if ``stack`` is not
            3-D / is empty (raised by the core).
    """
    if mode not in ("mp", "rp"):
        raise ValueError(f"mode must be 'mp' or 'rp', got {mode!r}")
    if mode == "rp":
        print("rp is not implemented, defaulting to mp")
        mode = "mp"

    # Auto-accelerate when a CUDA device is present (the convention shared with
    # the sister SAR repos): a cupy input stays on the GPU and returns cupy
    # arrays (cupy in -> cupy out); a NumPy input is computed on the GPU and
    # returned as NumPy (numpy in -> numpy out, auto round-tripped via
    # cp.asnumpy). A caller that wants to keep the stack resident across calls
    # passes a cupy array; one that wants plain NumPy semantics passes a NumPy
    # array and the device transfer is handled here. Without a GPU the NumPy
    # path is used regardless.
    # Excluded from coverage: the gate runs on CPU CI (no GPU); the NumPy
    # fallback below is the tested path. See CONVENTIONS.md "GPU coverage rule".
    if _HAVE_CUPY_GPU:  # pragma: no cover
        print("computeFbr: using cupy GPU acceleration")
        fbr, mask = _computeFbr_core(stack, mode, enl, a, cp)
        if isinstance(stack, cp.ndarray):
            return fbr, mask  # cupy in -> cupy out
        fbrHost: np.ndarray = cp.asnumpy(fbr)
        maskHost: np.ndarray = cp.asnumpy(mask)
        return fbrHost, maskHost  # numpy in -> numpy out

    # --- CPU / NumPy fallback ---
    return _computeFbr_core(stack, mode, enl, a, np)


def _computeFbr_cupy_selfcheck() -> bool:
    """Equivalence check: cupy computeFbr vs the NumPy reference.

    The NumPy reference is produced by calling the ``xp``-parameterized
    ``_computeFbr_core`` with ``np`` directly (bypassing ``computeFbr``'s
    auto-dispatch, which uses the GPU whenever one is present). The GPU result
    comes from the public ``computeFbr`` with a cupy input (cupy in -> cupy out),
    brought back with ``cp.asnumpy``. The numpy-in -> numpy-out contract is also
    exercised: ``computeFbr(stack, ...)`` with a NumPy input must return NumPy
    arrays (auto round-tripped through the GPU). Both paths run the rejection
    loop in float32, so the reductions agree to ~1e-5 relative but not
    bit-for-bit; we check allclose, not array_equal. The input is built so clean
    pixels are constant in time (CV = 0, never rejected) and only the planted
    target dates are rejected — the argmax selection is unambiguous, so the two
    paths reject the same dates and the masks match exactly (allclose with
    equal_nan). Skipped (and reported as such) when no GPU is available.
    """
    if _HAVE_CUPY_GPU:  # pragma: no cover  -- GPU path, excluded from CPU CI coverage
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

        # Exercise both the plain threshold (a=0) and the paper's margin threshold
        # (a=3): the cupy path must match the NumPy reference for each.
        g = cp.asarray(stack)
        ok_all = True
        for a in (0.0, 3.0):
            # NumPy reference: the core directly with np (bypasses auto-dispatch,
            # which would otherwise compute on the GPU when one is present).
            fbr_ref, mask_ref = _computeFbr_core(stack, "mp", 1.0, a, np)
            # GPU via the public function: cupy in -> cupy out, then bring back once.
            fbr_g, mask_g = computeFbr(g, mode="mp", enl=1.0, a=a)
            fbr_g = cp.asnumpy(fbr_g)
            mask_g = cp.asnumpy(mask_g)
            # numpy in -> numpy out contract: a NumPy input must return NumPy arrays.
            fbr_nio, mask_nio = computeFbr(stack, mode="mp", enl=1.0, a=a)
            nio_ok = isinstance(fbr_nio, np.ndarray) and isinstance(mask_nio, np.ndarray)

            shape_ok = mask_g.shape == mask_ref.shape == (D, H, W)
            dtype_ok = mask_g.dtype == mask_ref.dtype == np.float32
            fbr_close = np.allclose(np.asarray(fbr_g), np.asarray(fbr_ref), rtol=1e-4, atol=1e-5)
            mask_close = np.allclose(mask_g, mask_ref, rtol=0.0, atol=0.0, equal_nan=True)
            max_fbr_diff = float(np.nanmax(np.abs(np.asarray(fbr_g) - np.asarray(fbr_ref))))

            ok = bool(shape_ok and dtype_ok and fbr_close and mask_close and nio_ok)
            ok_all = ok_all and ok
            print(
                f"computeFbr self-check (a={a}): {'PASS' if ok else 'FAIL'} "
                f"(shape={shape_ok}, dtype={dtype_ok}, fbr_close={fbr_close}, "
                f"mask_close={mask_close}, nio={nio_ok}, max_fbr_diff={max_fbr_diff:.3e})"
            )
            if not ok:
                print(f"  ref fbr {np.asarray(fbr_ref)} got fbr {np.asarray(fbr_g)}")
        return ok_all

    print("computeFbr self-check: skipped (no GPU available — NumPy path only).")
    return True


if __name__ == "__main__":  # pragma: no cover
    _computeFbr_cupy_selfcheck()
