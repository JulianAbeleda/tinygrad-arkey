#!/usr/bin/env python3
"""Shared LLM generation core for the flywheel rollout/eval tooling.

This module is the single home for the policy-mode environment setup and the
model+tokenizer load used across the QK eval/bench tooling.

The AMD env-ordering invariant is sacred: `DEV`/`JIT`/`QK_PRIMITIVE_STORAGE` plus
the Q4K/Q6K primitive flags must be set *before* `from tinygrad import ...`. This
module therefore imports tinygrad lazily, *inside* `load_model_and_tokenizer`, so
importing it never pulls in tinygrad and never freezes the env. Subprocess callers
build a full child env via `child_env`, derived from the `policy_overrides` mapping.
"""
from __future__ import annotations

import os, pathlib
from typing import Any

from extra.qk.modes import PolicyMode, validate_policy_mode

# Stale policy keys cleared before every run so a previous mode can't leak in.
_CLEAR_KEYS = ("QK_GENERATED_POLICY", "QK_GENERATED_POLICY_DEBUG", "Q4K_PRIMITIVE_DEBUG", "Q6K_PRIMITIVE_DEBUG")


def policy_overrides(mode:str, *, device:str, storage:str, policy:Any=None, policy_debug:bool=False) -> dict[str, str]:
  """The environment variables to *set* for a given policy mode.

  Covers the device/JIT/storage base plus the Q4K/Q6K primitive flags (and the
  generated-policy pointer for generated mode). Callers are responsible for
  clearing `_CLEAR_KEYS` first (`child_env` does).
  """
  env = {"DEV": device, "JIT": "1", "PYTHONPATH": ".", "QK_PRIMITIVE_STORAGE": storage}
  m = validate_policy_mode(mode)
  if m == PolicyMode.GENERATED:
    if policy is None: raise ValueError("--policy is required for generated mode")
    env["Q4K_PRIMITIVE"] = "0"
    env["Q6K_PRIMITIVE"] = "0"
    env["QK_GENERATED_POLICY"] = str(policy)
    if policy_debug: env["QK_GENERATED_POLICY_DEBUG"] = "1"
  elif m == PolicyMode.EXPLICIT:
    env["Q4K_PRIMITIVE"] = "1"
    env["Q6K_PRIMITIVE"] = "1"
  elif m == PolicyMode.BASELINE:
    env["Q4K_PRIMITIVE"] = "0"
    env["Q6K_PRIMITIVE"] = "0"
  return env


def child_env(mode:str, *, device:str, storage:str, policy:Any=None, policy_debug:bool=False) -> dict[str, str]:
  """Subprocess: a full env dict (inherit `os.environ`, clear stale keys, apply overrides)."""
  env = {**os.environ}
  for key in _CLEAR_KEYS: env.pop(key, None)
  env.update(policy_overrides(mode, device=device, storage=storage, policy=policy, policy_debug=policy_debug))
  return env


def load_model_and_tokenizer(model:Any, max_context:int, *, seed:int, adapter:Any=None) -> tuple[Any, Any]:
  """Seed, load the gguf Transformer (+ optional adapter), return (model, tokenizer).

  tinygrad is imported here, lazily, so the env-ordering invariant holds: callers
  must have configured the environment before calling this.
  """
  from tinygrad import Tensor
  from tinygrad.llm.cli import SimpleTokenizer
  from tinygrad.llm.model import Transformer

  Tensor.manual_seed(seed)
  model_obj, kv = Transformer.from_gguf(pathlib.Path(model).expanduser(), max_context)
  if adapter is not None:
    from extra.llm.adapter import load_adapter
    load_adapter(model_obj, pathlib.Path(adapter).expanduser())
  return model_obj, SimpleTokenizer.from_gguf_kv(kv)
