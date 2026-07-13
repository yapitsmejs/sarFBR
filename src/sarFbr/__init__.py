"""sarFbr: Frozen Background Reference (FBR) method for SAR time-series."""

from __future__ import annotations

from .fbr import computeFbr, theoreticalCv

__version__ = "0.2.0"

__all__ = ["computeFbr", "theoreticalCv", "__version__"]
