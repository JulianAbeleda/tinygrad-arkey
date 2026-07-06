#!/usr/bin/env python3
"""Shared timing controls for QK probes and benchmarks."""
from __future__ import annotations

import argparse
import os
from collections.abc import MutableMapping
from typing import Any

PIN_CLOCK_ENV = "PREFILL_PIN_CLOCK"


def env_wants_clock_pin(env: MutableMapping[str, str] | None = None) -> bool:
  """Return whether the benchmark/probe should pin GPU clocks for its timing window."""
  return (env or os.environ).get(PIN_CLOCK_ENV, "0") == "1"


def set_clock_pin_env(env: dict[str, str], enabled: bool) -> dict[str, str]:
  """Mutate and return env with the canonical clock-pin flag set or cleared."""
  if enabled: env[PIN_CLOCK_ENV] = "1"
  else: env.pop(PIN_CLOCK_ENV, None)
  return env


def add_clock_pin_arg(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("--pin-clock", action="store_true",
                      help="pin AMD clocks during timing windows via extra.qk.clock_pin")


def pinned_peak_from_env() -> Any:
  """Return the canonical clock-pin context manager controlled by PREFILL_PIN_CLOCK."""
  from extra.qk.clock_pin import pinned_peak
  return pinned_peak(enabled=env_wants_clock_pin())
