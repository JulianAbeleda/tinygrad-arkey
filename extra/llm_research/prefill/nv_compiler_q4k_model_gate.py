#!/usr/bin/env python3
"""Shared execution helpers for the compiler-owned pp512 Q4_K model gate.

Candidate and control must run in separate processes. tinygrad caches
environment reads, so switching the route variable inside one process cannot
produce an authoritative control. The executable entry point therefore fails
closed; use ``nv_compiler_q4k_model_arm.py``.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
import numpy as np

from tinygrad import Device, Tensor, TinyJit
from tinygrad.llm.prefill_route_observer import prefill_route_scope
from tinygrad.uop.ops import Ops


@contextmanager
def _candidate_env(enabled:bool):
  keys = ("NV_COMPILER_Q4_IMMA_PP512", "NV_Q4_IMMA_PP512")
  old = {key:os.environ.get(key) for key in keys}
  try:
    os.environ.pop("NV_Q4_IMMA_PP512", None)
    if enabled: os.environ["NV_COMPILER_Q4_IMMA_PP512"] = "1"
    else: os.environ.pop("NV_COMPILER_Q4_IMMA_PP512", None)
    yield
  finally:
    for key, value in old.items():
      if value is None: os.environ.pop(key, None)
      else: os.environ[key] = value


@contextmanager
def _compile_scope(model):
  import tinygrad.codegen.opt.postrange as pr
  merged = {**(model._pf16_warmstart or {}), **(model._packed_wmma_warmstart or {})}
  with prefill_route_scope(True), pr.warmstart_candidate_state(merged, model._packed_wmma_warmstart_contexts): yield


def _configure(model, binding=None):
  for linear in model._q4k_linears.linears: linear.decode_enabled = False
  for block in model.blk:
    block._use_flash, block._prefill_v2, block._is_prefill = True, True, True
    block._ring_freqs, block._ring_full = None, False
    if binding is not None: block._nv_q4_imma_pp512_binding = binding


def _program_calls(linear):
  return [u for u in linear.toposort() if u.op is Ops.CALL and u.src and u.src[0].op is Ops.PROGRAM]


def _call_name(call) -> str: return call.src[0].arg.name


def _call_and_sync(jit, chunk, temp):
  out = jit(chunk, temp)
  Tensor.realize(*out)
  Device[Device.DEFAULT].synchronize()
  return out


def _capture(model, binding, chunk, temp, *, candidate:bool):
  _configure(model, binding if candidate else None)
  @TinyJit
  def run(tokens, temperature): return model.forward_greedy_with_logits(tokens, 0, temperature)
  with _candidate_env(candidate), _compile_scope(model):
    for _ in range(3):
      if candidate: binding.begin_trace()
      _call_and_sync(run, chunk, temp)
  if run.captured is None: raise RuntimeError("whole-model arm did not capture")
  return run


def _numpy_output(out):
  token, logits = out
  return int(token.numpy().reshape(-1)[0]), logits.numpy().astype(np.float32, copy=True)


def main():
  raise SystemExit("disabled: run nv_compiler_q4k_model_arm.py in separate candidate/fp16 processes")


if __name__ == "__main__": main()
