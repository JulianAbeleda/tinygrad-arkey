#!/usr/bin/env python3
"""Minimal AMD correctness probe used by the eGPU qualification gates."""
from __future__ import annotations

import sys
from typing import Iterable

EXPECTED = [2.0, 5.0, 10.0, 17.0]


def validate_result(values: Iterable[object]) -> None:
  got = [float(value) for value in values]
  if got != EXPECTED:
    raise ValueError(f"AMD result mismatch: expected {EXPECTED}, got {got}")


def run() -> None:
  # Keep tinygrad imports here: importing this module for CPU-only validation is inert.
  from tinygrad import Device, Tensor

  x = Tensor([1, 2, 3, 4], device=Device["AMD"])
  result = (x * x + 1).realize().tolist()
  validate_result(result)


def main() -> int:
  try:
    run()
  except Exception as exc:
    print(f"minimal AMD compute failed: {exc}", file=sys.stderr)
    return 1
  print(EXPECTED)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
