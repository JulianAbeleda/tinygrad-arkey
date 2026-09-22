#!/usr/bin/env python3
"""Qualify native-NV decode predispatch against full logits, then time A/B/A.

This is deliberately a small, fixed-depth d512 harness rather than a new
measurement authority.  It owns only the two reversible JIT predispatch
switches: descriptor construction memoization and written-input shadow reuse.
Each arm is a fresh process in normal use, so its JIT capture and allocator
state cannot leak across arms.  ``--mode logits`` serializes the full decode
logits; ``--mode timing`` returns W timing only.  The caller compares logits
from an off/on/off triplet and books a timing result only when both off arms
bound the on arm.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, statistics, sys, time
import numpy as np

from extra.llm_research.decode.decode_harness import DEFAULT_MODEL


def _prompt(model_path:str, depth:int) -> list[int]:
  # Exact decode-authority d512 corpus.  Repeated token id 1 is useful for a
  # timing-only smoke test but is not a numerical decode oracle here.
  from tinygrad.llm.gguf import gguf_load_metadata
  from tinygrad.llm.runtime_state import SimpleTokenizer
  kv, _ = gguf_load_metadata(model_path)
  tok = SimpleTokenizer.from_gguf_kv(kv)
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  return (ids * (1 + depth // len(ids)))[:depth]


def _digest(arrays:list[np.ndarray]) -> str:
  digest = hashlib.sha256()
  for array in arrays:
    digest.update(np.ascontiguousarray(array).view(np.uint8))
  return digest.hexdigest()


def _load(model_path:str, max_context:int):
  # Native d512 authority uses this same setup-only guard: the promoted fused
  # prefill route is presently not compilable on sm_120, while the measured
  # decode route is ordinary SDPA.  It changes only prompt construction, never
  # the d512 decode graph being qualified here.
  import tinygrad.llm.model as model_module
  model_module._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()
  # Keep this construction byte-for-byte aligned with the prior native A/B
  # authority.  In particular, tokenizer-derived prompt setup can select a
  # different prefill shape before the d512 decode graph is ever reached.
  from tinygrad.llm.model import Transformer
  model = Transformer.from_gguf(model_path, max_context)[0]
  # The authority's ``--no-fused-prefill`` excluded both independent fused
  # attention routes. Clearing only the custom-kernel promotion above is not
  # enough: an immutable selected policy may still set ``prefill_tc_attn`` and
  # compile the unrelated packed-fragment prefill path on NV.
  object.__setattr__(model.config, "prefill_custom_kernel_attn", False)
  object.__setattr__(model.config, "prefill_tc_attn", False)
  return model


def logits_run(model_path:str, depth:int, count:int, max_context:int) -> dict:
  from tinygrad import Tensor, UOp
  from tinygrad.helpers import Context
  model = _load(model_path, max_context)
  # Native NV's sampled-token scalar boundary is independently known to
  # return the sentinel vocab id at this revision.  Feeding that sentinel back
  # would poison every later logit with NaNs, so build the production KV prefix
  # normally, then exercise the *same one-token decode JIT contract* with a
  # fixed valid feedback token.  This makes full-logit equality meaningful
  # without pretending to qualify the separate sampler blocker.
  gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0)
  try:
    prelude = int(next(gen))
  finally: gen.close()
  token, temp = Tensor([[1]], dtype="int32").contiguous(), Tensor([0.0])
  start_pos = UOp.variable("start_pos", 0, max_context - 1)
  # First establish that the retained value is finite before a TinyJit has any
  # opportunity to affect its output lifetime.  This differentiates a model
  # numerics issue from a captured-return binding issue.
  with Context(JIT=0): _, eager_logits = model.forward_with_logits(token, start_pos.bind(depth), temp)
  eager = eager_logits.numpy()
  if not np.isfinite(eager).all(): raise RuntimeError("eager full-logit oracle produced non-finite values for valid fixed feedback")
  sampled, logits, returned_argmax = [], [], []
  for index in range(count):
    sample, full_logits = model.decode_with_logits(token, start_pos.bind(depth + 1 + index), temp)
    sample_value, logit_value = int(sample.item()), full_logits.numpy()
    argmax_value = int(logit_value.argmax(axis=-1).item())
    if sample_value != argmax_value:
      raise RuntimeError(f"diagnostic replay return mismatch at token {index}: sample={sample_value} logits.argmax={argmax_value}")
    sampled.append(sample_value)
    returned_argmax.append(argmax_value)
    logits.append(logit_value)
  stacked = np.stack(logits)
  if not np.isfinite(stacked).all(): raise RuntimeError("full-logit diagnostic produced non-finite values for valid fixed feedback")
  # Position advances on each fixed-token replay, so all-equal full-logit
  # snapshots are evidence of a stale return binding rather than stable model
  # behavior. The caller separately compares fresh-process runs for stability.
  changed = any(not np.array_equal(stacked[0], row) for row in stacked[1:])
  if count > 1 and not changed: raise RuntimeError("full-logit diagnostic snapshots are stale across advancing decode positions")
  return {"prelude_token": prelude, "tokens": sampled, "returned_argmax": returned_argmax, "sample_argmax_match": True,
          "logits_change_across_positions": changed, "logits": stacked, "logits_sha256": _digest(logits),
          "shape": list(stacked.shape), "dtype": str(stacked.dtype)}


def timing_run(model_path:str, depth:int, count:int, max_context:int, reps:int) -> dict:
  from tinygrad import Device
  model = _load(model_path, max_context)
  prompt = _prompt(model_path, depth)
  dev, samples, hashes = Device[Device.DEFAULT], [], []
  # Separate warmup to ensure every recorded token is capture/replay steady state.
  warm = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
  try:
    for _ in range(3): next(warm)
  finally: warm.close()
  for _ in range(reps):
    model.reset_generation_state()
    gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
    out = []
    try:
      next(gen)
      dev.synchronize()
      started = time.perf_counter_ns()
      for _ in range(count): out.append(int(next(gen)))
      dev.synchronize()
      samples.append((time.perf_counter_ns() - started) / count / 1e6)
    finally: gen.close()
    hashes.append(hashlib.sha256(",".join(map(str, out)).encode()).hexdigest())
  return {"per_token_ms": samples, "median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples),
          "spread_pct": (max(samples)-min(samples))/statistics.median(samples)*100, "generated_sha256": hashes,
          "generated_identical": len(set(hashes)) == 1}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default=os.environ.get("QK_MODEL", DEFAULT_MODEL))
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--count", type=int, default=8)
  ap.add_argument("--max-context", type=int, default=1024)
  ap.add_argument("--reps", type=int, default=5)
  ap.add_argument("--mode", choices=("logits", "timing"), required=True)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  if args.depth < 1 or args.count < 1 or args.max_context <= args.depth + args.count: raise ValueError("invalid depth/count/context")
  enabled = {key: os.environ.get(key, "1") for key in ("JIT_INPUT_DESCRIPTOR_CACHE", "JIT_REUSE_WRITTEN_INPUT_SHADOWS")}
  result = logits_run(args.model, args.depth, args.count, args.max_context) if args.mode == "logits" else \
           timing_run(args.model, args.depth, args.count, args.max_context, args.reps)
  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  if args.mode == "logits":
    np.savez_compressed(out.with_suffix(".npz"), logits=result.pop("logits"))
  result.update({"schema": "tinygrad.nv_predispatch_qualification.v1", "mode": args.mode, "depth": args.depth,
                 "count": args.count, "max_context": args.max_context, "switches": enabled})
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, sort_keys=True))
  return 0


if __name__ == "__main__": raise SystemExit(main())
