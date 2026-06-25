"""Tests for the FBR image computation (pure-NumPy)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import sarfbr
from sarfbr import computeFbr, theoreticalCv


# ---------------------------------------------------------------------------
# theoreticalCv
# ---------------------------------------------------------------------------

def test_theoretical_cv_single_look_is_rayleigh():
    # L = 1 -> Rayleigh amplitude -> sqrt(4/pi - 1) ~= 0.5227.
    assert theoreticalCv(1.0) == pytest.approx(math.sqrt(4.0 / math.pi - 1.0), rel=1e-9)


def test_theoretical_cv_decreases_with_more_looks():
    # Multi-looking reduces the speckle CV.
    assert theoreticalCv(1.0) > theoreticalCv(4.0) > theoreticalCv(16.0)


def test_theoretical_cv_rejects_non_positive():
    with pytest.raises(ValueError):
        theoreticalCv(0.0)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _synthetic_stack(D=10, target_value=20.0):
    """A 2x2 amplitude stack with two clean and two target pixels.

    - (0,0): always background (1.0)
    - (0,1): target on date 5
    - (1,0): target on date 2
    - (1,1): always background (1.0)
    Background amplitude = 1.0; target amplitude = ``target_value`` (well above
    the speckle threshold so the target date is rejected).
    """
    bg = 1.0
    stack = np.full((D, 2, 2), bg, dtype=np.float64)
    stack[5, 0, 1] = target_value
    stack[2, 1, 0] = target_value
    return stack, D


# ---------------------------------------------------------------------------
# computeFbr
# ---------------------------------------------------------------------------

def test_mp_recovers_background_everywhere():
    stack, D = _synthetic_stack()
    fbr, mask = computeFbr(stack, mode="mp", enl=1.0)
    # The current implementation collapses the surviving stack to a single
    # scalar FBR value (np.nanmean over the whole masked stack).
    assert np.ndim(fbr) == 0
    np.testing.assert_allclose(fbr, 1.0, atol=1e-12)


def test_mask_marks_target_dates_as_invalid():
    stack, D = _synthetic_stack()
    fbr, mask = computeFbr(stack, mode="mp", enl=1.0)

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


def test_rp_falls_back_to_mp(capsys):
    # RP is not implemented: the request prints a warning and falls back to MP,
    # so it must produce identical outputs to mode="mp".
    stack, D = _synthetic_stack()
    fbr_rp, mask_rp = computeFbr(stack, mode="rp", enl=1.0)
    captured = capsys.readouterr()
    assert "not implemented" in captured.out

    np.testing.assert_allclose(fbr_rp, 1.0, atol=1e-12)
    # The mask is identical to the MP mask (same selection / fallback path).
    fbr_mp, mask_mp = computeFbr(stack, mode="mp", enl=1.0)
    np.testing.assert_allclose(fbr_rp, fbr_mp, atol=1e-12)
    assert np.array_equal(mask_rp, mask_mp, equal_nan=True)


def test_constant_stack_uses_all_dates():
    D = 8
    stack = np.full((D, 3, 3), 3.5, dtype=np.float64)
    fbr, mask = computeFbr(stack, mode="mp", enl=4.0)
    np.testing.assert_allclose(fbr, 3.5, atol=1e-12)
    assert mask.all()


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        computeFbr(np.zeros((4, 2, 2)), mode="xx")


def test_non_3d_raises():
    with pytest.raises(ValueError):
        computeFbr(np.zeros((4, 2)), mode="mp")


def test_empty_stack_raises():
    with pytest.raises(ValueError):
        computeFbr(np.zeros((0, 2, 2)), mode="mp")


def test_int_input_is_coerced():
    # Integer stacks should be accepted (coerced to float) without error.
    stack, _ = _synthetic_stack()
    fbr, mask = computeFbr(stack.astype(np.int32), mode="mp", enl=1.0)
    np.testing.assert_allclose(fbr, 1.0, atol=1e-9)