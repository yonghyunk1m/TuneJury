"""Anchor calibration for TuneJury under generator drift.

Public API:
    AnchorCalibrator — fit + predict per-system Bradley--Terry intercepts.

See README.md for context and the paper (§A.D) for empirical results.
"""
from .fit import AnchorCalibrator

__all__ = ["AnchorCalibrator"]
