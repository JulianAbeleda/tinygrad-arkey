#!/usr/bin/env python3
"""Fresh-process full-logit gate for the decode-only REDUCE_OUTPUT RMSNorm lease.

This intentionally reuses the fixed-feedback d512 oracle rather than the native
sampled-token boundary: full finite logits and an in-range argmax are the gate;
wall timing is forbidden until a control/candidate comparison passes.
"""
from __future__ import annotations

import argparse, json, pathlib
import numpy as np

from extra.llm_research.decode.nv_predispatch_full_logits_qualification import DEFAULT_MODEL, _load, _prompt


def run(mode:str, model_path:str, depth:int, count:int, max_context:int) -> tuple[dict, np.ndarray]:
  from tinygrad import Tensor, UOp
  from tinygrad.helpers import Context

  model = _load(model_path, max_context)
  if mode == "candidate":
    model._decode_reduce_output_rmsnorm_promoted = True
    for block in model.blk: block._decode_reduce_output_rmsnorm_promoted = True

  # Populate the exact production KV prefix.  The first yielded sampler value
  # is deliberately not used as feedback; this diagnostic owns a valid token.
  gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0)
  try: prelude = int(next(gen))
  finally: gen.close()

  token, temp = Tensor([[1]], dtype="int32").contiguous(), Tensor([0.0])
  start_pos = UOp.variable("start_pos", 0, max_context - 1)
  with Context(JIT=0): _, eager_logits = model.forward_greedy_with_logits(token, start_pos.bind(depth), temp)
  eager = eager_logits.numpy()
  if not np.isfinite(eager).all(): raise RuntimeError("eager logits are non-finite")

  logits, tokens = [], []
  for index in range(count):
    sample, full_logits = model.decode_with_logits(token, start_pos.bind(depth + 1 + index), temp)
    array, sampled = full_logits.numpy(), int(sample.item())
    if not np.isfinite(array).all(): raise RuntimeError(f"decode logits {index} are non-finite")
    if not 0 <= sampled < array.shape[-1]: raise RuntimeError(f"decode token {sampled} is outside [0,{array.shape[-1]})")
    if sampled != int(array.argmax(-1).reshape(-1)[0]): raise RuntimeError("sample does not equal full-logit argmax")
    logits.append(array); tokens.append(sampled)

  stacked = np.stack(logits)
  return ({"schema":"tinygrad.nv_reduce_output_rmsnorm_logits_gate.v1", "mode":mode, "depth":depth,
           "count":count, "max_context":max_context, "prelude_token_diagnostic_only":prelude,
           "tokens":tokens, "shape":list(stacked.shape), "dtype":str(stacked.dtype),
           "finite":bool(np.isfinite(stacked).all()), "token_range_valid":True}, stacked)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", choices=("control", "candidate"), required=True)
  ap.add_argument("--model", default=DEFAULT_MODEL)
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--count", type=int, default=2)
  ap.add_argument("--max-context", type=int, default=1024)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  result, logits = run(args.mode, args.model, args.depth, args.count, args.max_context)
  out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(out.with_suffix(".npz"), logits=logits)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, sort_keys=True))
  return 0


if __name__ == "__main__": raise SystemExit(main())
