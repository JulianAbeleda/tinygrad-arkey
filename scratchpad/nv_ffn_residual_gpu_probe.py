#!/usr/bin/env python3
"""GPU correctness probe for the ffn-norm residual bind (1_4096 site).

The fp32 q/k harness candidate arm already promotes the ffn-norm site through
the shared global flag, but its control arm does not close the per-site ffn
knob, so it cannot isolate this change.  This probe gates BOTH flags per arm:

  control    global False + ffn False   (closed route-less graph)
  candidate  global True  + ffn True    (q/k + attn + ffn promoted)

``--mode smoke`` renders one decode token and reports the compiled
``reduce_output_rmsnorm_1_4096`` body count plus the q/k body counts.  On the
RTX 5090 at ``--depth 512`` the control arm emits no promoted q/k/ffn bodies
while the candidate arm emits one q and one k body per block and one ffn body
for each block whose o-proj residual add was not absorbed into the o-proj
epilogue (19 of 36 blocks in the default Qwen3-8B graph).  ``--mode logits``
runs the eager JIT=0 finite check plus the production decode logits and
returns the stacked-row SHA-256; the control/candidate hashes are the actual
correctness gate and must match bit-exactly.  Run each arm as a fresh process
under the GPU bench lock.
"""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np

from extra.llm_research.decode.nv_predispatch_full_logits_qualification import DEFAULT_MODEL, _digest, _load, _prompt


def _configure(model, arm: str) -> dict:
  model._decode_direct_greedy_promoted = True
  promoted = arm == "candidate"
  model._decode_reduce_output_rmsnorm_promoted = promoted
  model._decode_reduce_output_ffn_rmsnorm_promoted = promoted
  for block in model.blk:
    block._decode_reduce_output_rmsnorm_promoted = promoted
    block._decode_reduce_output_ffn_rmsnorm_promoted = promoted
  return {
    "reduce_output_rmsnorm_promoted": bool(getattr(model, "_decode_reduce_output_rmsnorm_promoted", False)),
    "ffn_rmsnorm_promoted": bool(getattr(model, "_decode_reduce_output_ffn_rmsnorm_promoted", False)),
    "blocks_ffn_promoted": sum(bool(getattr(b, "_decode_reduce_output_ffn_rmsnorm_promoted", False)) for b in model.blk),
  }


def _arm_context(arm: str):
  from contextlib import contextmanager
  @contextmanager
  def _ctx():
    if arm == "candidate":
      from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
      from tinygrad.helpers import Context
      with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
        yield
    else:
      yield
  return _ctx()


def smoke(arm: str, model_path: str, depth: int, max_context: int) -> dict:
  from tinygrad import Device
  from tinygrad.engine.jit import GraphAdmissionCensus, observe_graph_admissions
  from tinygrad.helpers import Context
  with _arm_context(arm):
    model = _load(model_path, max_context)
    gates = _configure(model, arm)
    model.reset_generation_state()
    gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0)
    admission = GraphAdmissionCensus()
    try:
      prelude = int(next(gen))
      token = None
      for index in range(3):
        if index == 1:
          with Context(TRACEMETA=1), observe_graph_admissions(admission):
            token = int(next(gen))
        else:
          next(gen)
      Device[Device.DEFAULT].synchronize()
      programs = [r.program_name for r in admission.records if r.program_name]
      c6 = sum(1 for n in programs if n.startswith("reduce_output_rmsnorm_1_4096"))
      q = sum(1 for n in programs if n.startswith("reduce_output_rmsnorm_32_128"))
      k = sum(1 for n in programs if n.startswith("reduce_output_rmsnorm_8_128"))
      return {"arm": arm, "mode": "smoke", "gates": gates, "survive": True,
              "prelude_token": prelude, "token": token,
              "c6_bodies": c6, "q_bodies": q, "k_bodies": k,
              "program_count": len(programs), "program_names": programs}
    finally:
      gen.close()


def logits(arm: str, model_path: str, depth: int, count: int, max_context: int) -> dict:
  from tinygrad import Tensor, UOp
  from tinygrad.helpers import Context
  with _arm_context(arm):
    model = _load(model_path, max_context)
    gates = _configure(model, arm)
    gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0)
    try:
      prelude = int(next(gen))
    finally:
      gen.close()
    token, temp = Tensor([[1]], dtype="int32").contiguous(), Tensor([0.0])
    start_pos = UOp.variable("start_pos", 0, max_context - 1)
    with Context(JIT=0):
      _, eager_logits = model.forward_with_logits(token, start_pos.bind(depth), temp)
    if not np.isfinite(eager_logits.numpy()).all():
      raise RuntimeError("non-finite eager logits")
    samples, rows = [], []
    for idx in range(count):
      sample, full = model.decode_with_logits(token, start_pos.bind(depth + 1 + idx), temp)
      row = full.numpy()
      if not np.isfinite(row).all() or int(row.argmax(axis=-1).item()) != int(sample.item()):
        raise RuntimeError(f"invalid diagnostic output at row {idx}")
      samples.append(int(sample.item()))
      rows.append(row)
    stacked = np.stack(rows)
    return {"arm": arm, "mode": "logits", "gates": gates, "prelude_token": prelude,
            "tokens": samples, "shape": list(stacked.shape), "dtype": str(stacked.dtype),
            "logits_sha256": _digest([stacked])}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", choices=("smoke", "logits"), required=True)
  ap.add_argument("--arm", choices=("control", "candidate"), required=True)
  ap.add_argument("--model", default=DEFAULT_MODEL)
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--count", type=int, default=32)
  ap.add_argument("--max-context", type=int, default=1024)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  result = (smoke(args.arm, args.model, args.depth, args.max_context) if args.mode == "smoke"
            else logits(args.arm, args.model, args.depth, args.count, args.max_context))
  path = pathlib.Path(args.out)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"arm": args.arm, "mode": args.mode,
                    **({k: result[k] for k in ("c6_bodies", "q_bodies", "k_bodies", "survive")} if args.mode == "smoke"
                       else {"logits_sha256": result["logits_sha256"], "tokens": result["tokens"]})}, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
