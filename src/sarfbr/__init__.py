"""sarfbr — Frozen Background Reference (FBR) method for SAR time-series.

A pure-NumPy implementation of the FBR image computation step from
Taillade, Thirion-Lefevre & Guinvarc'h (Remote Sensing, 2020, 12(11), 1720).

Install from git:

    pip install git+https://github.com/yapitsmejs/sarFBR.git
"""

from __future__ import annotations

from .fbr import computeFbr, theoreticalCv

__version__ = "0.1.3"

__all__ = ["computeFbr", "theoreticalCv", "__version__"]