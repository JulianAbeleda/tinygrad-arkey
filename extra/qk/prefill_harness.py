#!/usr/bin/env python3
"""Central prefill harness policy.

This module is intentionally import-light: no tinygrad import, no GPU access. It owns the run profiles and subprocess
shape for the canonical prefill authority so callers do not clone smoke/authority defaults or PREFILL_V2 env setup.
The timing implementation stays in extra/qk/prefill_whole_synced.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import pathlib

from extra.qk.harness_contract import csv_ints

ROOT = pathlib.Path(__file__).resolve().parents[2]

AUTHORITY_START_POSITIONS = (0, 512, 1024, 2048, 3584)
AUTHORITY_WHOLE_LENGTHS = (512, 1024, 2048, 4096)
SMOKE_START_POSITIONS = (0,)
SMOKE_WHOLE_LENGTHS = (512,)
PREFILL_MODES = ("authority", "smoke")


@dataclass(frozen=True)
class PrefillRunProfile:
  mode: str
  K: int
  warmups: int
  rounds: int
  start_positions: tuple[int, ...]
  whole_lengths: tuple[int, ...]
  chunk_n: int = 512
  max_context: int = 4608

  def validate(self) -> None:
    if self.mode not in PREFILL_MODES: raise ValueError(f"prefill mode must be one of {PREFILL_MODES}, got {self.mode!r}")
    if self.K < 1 or self.warmups < 0 or self.rounds < 1:
      raise ValueError("K >= 1, warmups >= 0, and rounds >= 1 are required")
    if self.chunk_n <= 0 or self.max_context <= 0: raise ValueError("chunk_n and max_context must be positive")
    if not self.start_positions or not self.whole_lengths:
      raise ValueError("start_positions and whole_lengths must be non-empty")
    if any(sp < 0 for sp in self.start_positions): raise ValueError("start_positions must be non-negative")
    if any(L <= 0 for L in self.whole_lengths): raise ValueError("whole_lengths must be positive")
    max_start = max(self.start_positions)
    if max_start + self.chunk_n > self.max_context:
      raise ValueError(f"max start_pos {max_start} + chunk_n {self.chunk_n} exceeds max_context {self.max_context}")


def prefill_run_profile(mode:str="authority", *, K:int|None=None, warmups:int|None=None, rounds:int|None=None,
                        start_positions:tuple[int, ...]|None=None, whole_lengths:tuple[int, ...]|None=None,
                        chunk_n:int=512, max_context:int=4608) -> PrefillRunProfile:
  mode = mode.strip().lower()
  smoke = mode == "smoke"
  prof = PrefillRunProfile(
    mode=mode,
    K=K if K is not None else (1 if smoke else 8),
    warmups=warmups if warmups is not None else (1 if smoke else 4),
    rounds=rounds if rounds is not None else (1 if smoke else 3),
    start_positions=start_positions if start_positions is not None else (SMOKE_START_POSITIONS if smoke else AUTHORITY_START_POSITIONS),
    whole_lengths=whole_lengths if whole_lengths is not None else (SMOKE_WHOLE_LENGTHS if smoke else AUTHORITY_WHOLE_LENGTHS),
    chunk_n=chunk_n,
    max_context=max_context,
  )
  prof.validate()
  return prof


def prefill_authority_argv(model_path:str, profile:PrefillRunProfile) -> list[str]:
  profile.validate()
  return ["extra/qk/prefill_whole_synced.py", "--model", model_path, "--mode", profile.mode,
          "-K", str(profile.K), "--warmups", str(profile.warmups), "--rounds", str(profile.rounds),
          "--start-positions", ",".join(str(x) for x in profile.start_positions),
          "--whole-lengths", ",".join(str(x) for x in profile.whole_lengths)]


def prefill_subprocess_env(extra:dict|None=None) -> dict[str, str]:
  env = {"PYTHONPATH": str(ROOT), "PREFILL_V2": "1"}
  for k, v in (extra or {}).items(): env[str(k)] = str(v)
  return env
