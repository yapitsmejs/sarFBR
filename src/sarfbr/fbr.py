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
    m = float(enl)
    ratio = math.gamma(m) / math.gamma(m + 0.5)
    return math.sqrt(m * ratio * ratio - 1.0)


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
    while np.any(cvX > cvSpeckle):
        # For pixels still above the threshold, reject the single date whose
        # value departs most from that pixel's temporal mean (the putative
        # ephemeral target): set its validity entry to NaN.
        xMeanDeviation = np.abs(x - np.nanmean(x,axis = 0)[np.newaxis,...])
        xMaxMeanDeviation = np.nanmax(xMeanDeviation,axis = 0)
        validMask = np.where((cvX > cvSpeckle)[np.newaxis,...] & (xMeanDeviation == xMaxMeanDeviation[np.newaxis,...]), np.nan, validMask)
        x *= validMask
        cvX = np.nanstd(x, axis=0) / np.nanmean(x, axis=0)

    # Combine the surviving (non-NaN) dates per pixel.
    maskedStack = validMask * stack

    if mode == "mp":
        # MP: average the surviving stable dates into the FBR image.
        fbr = np.nanmean(maskedStack)

    return fbr, validMask