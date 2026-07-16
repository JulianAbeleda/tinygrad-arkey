#!/usr/bin/env python3
"""Central decode authority process policy.

This import-light module owns decode benchmark defaults and subprocess
construction. The W==D timing method lives in decode_runtime_overhead.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]

DEFAULT_MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
DEFAULT_CKPTS = (128, 512, 1024, 4096)
DEFAULT_MAX_CONTEXT = 4608
DEFAULT_NMEAS = 40
DEFAULT_REPS = 5


@dataclass(frozen=True)
class DecodeRunProfile:
  ckpts: tuple[int, ...] = DEFAULT_CKPTS
  max_context: int = DEFAULT_MAX_CONTEXT
  nmeas: int = DEFAULT_NMEAS

  def validate(self) -> None:
    if not self.ckpts: raise ValueError("decode ckpts must be non-empty")
    if self.max_context <= 1: raise ValueError("decode max_context must be greater than 1")
    if self.nmeas < 1: raise ValueError("decode nmeas must be positive")
    if any(c <= 0 for c in self.ckpts): raise ValueError("fixed decode depths must be positive")
    if max(self.ckpts) >= self.max_context:
      raise ValueError(f"max decode ckpt {max(self.ckpts)} must be < max_context {self.max_context}")
    if max(self.ckpts) + self.nmeas >= self.max_context:
      raise ValueError(f"max decode ckpt {max(self.ckpts)} + nmeas {self.nmeas} must be < max_context {self.max_context}")


def csv_ints(raw: str) -> tuple[int, ...]:
  vals = tuple(int(x) for x in raw.replace(" ", "").split(",") if x)
  if not vals: raise ValueError("expected at least one comma-separated integer")
  return vals


def decode_run_profile(*, ckpts: tuple[int, ...] | None = None, max_context: int | None = None,
                       nmeas: int | None = None) -> DecodeRunProfile:
  prof = DecodeRunProfile(
    ckpts=ckpts if ckpts is not None else DEFAULT_CKPTS,
    max_context=max_context if max_context is not None else DEFAULT_MAX_CONTEXT,
    nmeas=nmeas if nmeas is not None else DEFAULT_NMEAS,
  )
  prof.validate()
  return prof


def decode_authority_argv(model_path: str, profile: DecodeRunProfile, *, out_path: str|pathlib.Path,
                          reps:int=DEFAULT_REPS) -> list[str]:
  profile.validate()
  if reps < 1: raise ValueError("decode reps must be positive")
  return ["extra/qk/decode_runtime_overhead.py", "--model", model_path,
          "--ckpts", ",".join(str(x) for x in profile.ckpts),
          "--max-context", str(profile.max_context), "--nmeas", str(profile.nmeas),
          "--reps", str(reps), "--out", str(out_path)]


def decode_subprocess_env(model_path: str, extra: dict | None = None) -> dict[str, str]:
  env = {"PYTHONPATH": str(ROOT), "QK_MODEL": model_path}
  for k, v in (extra or {}).items(): env[str(k)] = str(v)
  return env
