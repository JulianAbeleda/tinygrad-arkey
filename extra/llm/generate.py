#!/usr/bin/env python3
"""Shared LLM generation core for the flywheel rollout/eval tooling.

This module is the single home for the three pieces that `llm_rollout.py`
(in-process loop) and `llm_eval_harness.py` (subprocess-isolated child) used to
duplicate: the policy-mode environment setup, the model+tokenizer load, and the
per-prompt greedy/temperature generation loop.

Two irreducible invariants are preserved exactly:

- **AMD env-ordering is sacred.** `DEV`/`JIT`/`QK_PRIMITIVE_STORAGE` plus the
  Q4K/Q6K primitive flags must be set *before* `from tinygrad import ...`. This
  module therefore imports tinygrad lazily, *inside* `load_model_and_tokenizer`,
  so importing `llm_generate` never pulls in tinygrad and never freezes the env.
- **Two entry points, one core.** In-process callers mutate `os.environ` via
  `configure_process_env`; subprocess callers build a full child env via
  `child_env`. Both derive from the same `policy_overrides` mapping.
"""
from __future__ import annotations

import os, pathlib, time
from typing import Any

from extra.llm.eval_common import build_prompt_ids
from extra.qk.modes import PolicyMode, validate_policy_mode

# Stale policy keys cleared before every run so a previous mode can't leak in.
_CLEAR_KEYS = ("QK_GENERATED_POLICY", "QK_GENERATED_POLICY_DEBUG", "Q4K_PRIMITIVE_DEBUG", "Q6K_PRIMITIVE_DEBUG")


def policy_overrides(mode:str, *, device:str, storage:str, policy:Any=None, policy_debug:bool=False) -> dict[str, str]:
  """The environment variables to *set* for a given policy mode.

  Covers the device/JIT/storage base plus the Q4K/Q6K primitive flags (and the
  generated-policy pointer for generated mode). Callers are responsible for
  clearing `_CLEAR_KEYS` first; both `configure_process_env` and `child_env` do.
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


def configure_process_env(mode:str, *, device:str, storage:str, policy:Any=None, policy_debug:bool=False) -> None:
  """In-process: clear stale policy keys, then apply the mode overrides to `os.environ`."""
  for key in _CLEAR_KEYS: os.environ.pop(key, None)
  os.environ.update(policy_overrides(mode, device=device, storage=storage, policy=policy, policy_debug=policy_debug))


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


def generate_one(model:Any, tok:Any, prompt:str, prompt_format:str, *, temperature:float, max_tokens:int) -> dict[str, Any]:
  """Generate one prompt and return the core (timing-independent) result fields.

  Returns prompt_len/tokens/text/generated plus elapsed_s/tok_s. Callers layer
  their own metadata (id, mode, tags, score, ...) on top.
  """
  ids = build_prompt_ids(tok, prompt, prompt_format)
  out: list[int] = []
  st = time.perf_counter()
  for tid in model.generate(ids, temperature=temperature):
    if tok.is_end(tid): break
    out.append(tid)
    if len(out) >= max_tokens: break
  elapsed = time.perf_counter() - st
  return {
    "prompt_len": len(ids), "tokens": out, "text": tok.decode(out),
    "generated": len(out), "elapsed_s": round(elapsed, 6),
    "tok_s": 0.0 if elapsed == 0 else len(out) / elapsed,
  }
