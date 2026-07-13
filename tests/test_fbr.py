"""Tests for the FBR image computation (pure-NumPy / CPU path).

See ``src/sarFbr/fbr.py`` and the "Testing" section of ``CONVENTIONS.md``. The
CuPy/GPU dispatch branch is excluded from coverage (``# pragma: no cover``) --
it cannot run on CPU CI.
"""

from __future__ import annotations

import math
from importlib import import_module

import numpy as np
import pytest

import sarFbr
from sarFbr import computeFbr, theoreticalCv

# Import the *module* (not the re-exported function of the same name) so the
# private helpers and self-checks are reachable. The package __init__ re-exports
# `computeFbr`/`theoreticalCv`, which would shadow a `from sarFbr import fbr`
# style access to the submodule; import_module reaches the module directly.
fbr = import_module("sarFbr.fbr")


# ---------------------------------------------------------------------------
# theoreticalCv
# ---------------------------------------------------------------------------


def testTheoreticalCv_singleLookIsRayleigh():
    # L = 1 -> Rayleigh amplitude -> sqrt(4/pi - 1) ~= 0.5227.
    assert theoreticalCv(1.0) == pytest.approx(math.sqrt(4.0 / math.pi - 1.0), rel=1e-9)


def testTheoreticalCv_decreasesWithMoreLooks():
    # Multi-looking reduces the speckle CV.
    assert theoreticalCv(1.0) > theoreticalCv(4.0) > theoreticalCv(16.0)


def testTheoreticalCv_rejectsNonPositive():
    with pytest.raises(ValueError):
        theoreticalCv(0.0)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _syntheticStack(D: int = 10, targetValue: float = 20.0) -> tuple[np.ndarray, int]:
    """A 2x2 amplitude stack with two clean and two target pixels.

    - (0,0): always background (1.0)
    - (0,1): target on date 5
    - (1,0): target on date 2
    - (1,1): always background (1.0)
    Background amplitude = 1.0; target amplitude = ``targetValue`` (well above
    the speckle threshold so the target date is rejected).
    """
    bg = 1.0
    stack = np.full((D, 2, 2), bg, dtype=np.float64)
    stack[5, 0, 1] = targetValue
    stack[2, 1, 0] = targetValue
    return stack, D


# ---------------------------------------------------------------------------
# computeFbr
# ---------------------------------------------------------------------------


def testMp_recoversBackgroundEverywhere():
    stack, D = _syntheticStack()
    fbrImg, mask = computeFbr(stack, mode="mp", enl=1.0)
    # MP returns a per-pixel (H, W) image: each pixel is the multi-look
    # intensity average of its own surviving stable dates, rooted back to
    # amplitude (sqrt(nanmean(x**2, axis=0))). For a constant background of
    # amplitude 1.0 every pixel is exactly 1.0.
    assert fbrImg.shape == (2, 2)
    np.testing.assert_allclose(fbrImg, 1.0, atol=1e-12)


def testMask_marksTargetDatesAsInvalid():
    stack, D = _syntheticStack()
    fbrImg, mask = computeFbr(stack, mode="mp", enl=1.0)

    assert mask.shape == (D, 2, 2)
    # The mask is a float array: 1.0 where a date is stable, NaN where rejected.
    assert np.issubdtype(mask.dtype, np.floating)

    # Clean pixels use all dates (all 1.0, none rejected).
    np.testing.assert_array_equal(mask[:, 0, 0], np.ones(D))
    np.testing.assert_array_equal(mask[:, 1, 1], np.ones(D))

    # Target pixel (0,1): date 5 is rejected (NaN), all others 1.0.
    assert np.isnan(mask[5, 0, 1])
    assert np.count_nonzero(~np.isnan(mask[:, 0, 1])) == D - 1
    used = np.where(~np.isnan(mask[:, 0, 1]))[0]
    assert set(used.tolist()) == set(range(D)) - {5}

    # Target pixel (1,0): date 2 is rejected (NaN), all others 1.0.
    assert np.isnan(mask[2, 1, 0])
    assert np.count_nonzero(~np.isnan(mask[:, 1, 0])) == D - 1


def testRp_fallsBackToMp(capsys):
    # RP is not implemented: the request prints a warning and falls back to MP,
    # so it must produce identical outputs to mode="mp".
    stack, D = _syntheticStack()
    fbrRp, maskRp = computeFbr(stack, mode="rp", enl=1.0)
    captured = capsys.readouterr()
    assert "not implemented" in captured.out

    np.testing.assert_allclose(fbrRp, 1.0, atol=1e-12)
    # The mask is identical to the MP mask (same selection / fallback path).
    fbrMp, maskMp = computeFbr(stack, mode="mp", enl=1.0)
    np.testing.assert_allclose(fbrRp, fbrMp, atol=1e-12)
    assert np.array_equal(maskRp, maskMp, equal_nan=True)


def testConstantStack_usesAllDates():
    D = 8
    stack = np.full((D, 3, 3), 3.5, dtype=np.float64)
    fbrImg, mask = computeFbr(stack, mode="mp", enl=4.0)
    np.testing.assert_allclose(fbrImg, 3.5, atol=1e-12)
    assert mask.all()


def testRejectionLoop_hitsIterationLimit():
    # D = 3 => the loop guard `fbrIteration < D - 2` bounds the loop to at most
    # one rejection. With a pixel whose CV stays above the threshold even after
    # that single rejection, the loop exits via the iteration-limit branch (the
    # `cvX > threshold` condition is still True but `fbrIteration < D - 2` is
    # False), not via CV convergence. Values [1, 50, 100]: reject 100 (max dev),
    # remaining [1, 50] has cv = 49/51 ~= 0.96 > cvSpeckle(1) ~= 0.5227, so the
    # CV condition stays True while the iteration bound ends the loop.
    stack = np.array([[[1.0]], [[50.0]], [[100.0]]], dtype=np.float32)
    fbrImg, mask = computeFbr(stack, mode="mp", enl=1.0)
    # One date rejected (the 100, date 2); the other two survive.
    assert np.isnan(mask[2, 0, 0])
    assert np.count_nonzero(~np.isnan(mask[:, 0, 0])) == 2
    # fbr = sqrt(nanmean([1, 2500, NaN])) = sqrt(1250.5).
    np.testing.assert_allclose(fbrImg[0, 0], math.sqrt(1250.5), rtol=1e-5)


def testInvalidMode_raises():
    with pytest.raises(ValueError):
        computeFbr(np.zeros((4, 2, 2)), mode="xx")


def testNonThreeD_raises():
    with pytest.raises(ValueError):
        computeFbr(np.zeros((4, 2)), mode="mp")


def testEmptyStack_raises():
    with pytest.raises(ValueError):
        computeFbr(np.zeros((0, 2, 2)), mode="mp")


def testIntInput_isCoerced():
    # Integer stacks should be accepted (coerced to float) without error.
    stack, _ = _syntheticStack()
    fbrImg, mask = computeFbr(stack.astype(np.int32), mode="mp", enl=1.0)
    np.testing.assert_allclose(fbrImg, 1.0, atol=1e-9)


def testComplexInput_usesAmplitudeNotRealPart():
    # Complex SLC data must be reduced to amplitude (|z| = sqrt(re^2 + im^2))
    # before the float32 cast, NOT projected onto its real part (which is what a
    # bare `.astype(np.float32)` does, dropping the imaginary component).
    stackReal, _ = _syntheticStack()
    # Build a complex stack whose magnitude equals the real-amplitude stack but
    # whose real part is *not* the amplitude: put energy in the imaginary part.
    re = np.full_like(stackReal, 1.0)
    im = np.sqrt(np.maximum(stackReal**2 - re**2, 0.0))
    stackComplex = (re + 1j * im).astype(np.complex64)

    # Sanity: the complex stack's magnitude recovers the real-amplitude stack.
    np.testing.assert_allclose(np.abs(stackComplex), stackReal, atol=1e-6)

    fbrC, maskC = computeFbr(stackComplex, mode="mp", enl=1.0)
    fbrR, maskR = computeFbr(stackReal, mode="mp", enl=1.0)

    # Amplitude preservation: the complex stack must give the same FBR value and
    # the same rejection mask as the equivalent real-amplitude stack. If the cast
    # had dropped the imaginary part, the target pixels (whose real part is 1.0
    # for every date) would look like pure background and nothing would be rejected.
    np.testing.assert_allclose(fbrC, fbrR, atol=1e-5)
    assert np.array_equal(maskC, maskR, equal_nan=True)


# ---------------------------------------------------------------------------
# self-checks (kept runnable from __main__ for GPU devs; the skip path covered)
# ---------------------------------------------------------------------------


def testComputeFbrCupySelfcheck_runsOnCpuAndReportsSkip(capsys):
    # On a CPU-only host the cupy self-check reports a skip and returns True.
    assert fbr._computeFbr_cupy_selfcheck() is True
    captured = capsys.readouterr()
    assert "skipped" in captured.out


def testPackageReexportsPublicApi():
    # The package __init__ re-exports computeFbr / theoreticalCv unchanged.
    assert sarFbr.computeFbr is fbr.computeFbr
    assert sarFbr.theoreticalCv is fbr.theoreticalCv
