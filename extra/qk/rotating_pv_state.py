"""Compatibility export for the verifier-safe rotating-PV state descriptor.

No backend lowering is selected here.  This only gives the experimental
scheduler a typed, exact LDS ownership map for eight float8 accumulator windows.
"""
from __future__ import annotations

from tinygrad.uop.ops import RotatingPVStateSpec

RotatingPVStateHandle = RotatingPVStateSpec
