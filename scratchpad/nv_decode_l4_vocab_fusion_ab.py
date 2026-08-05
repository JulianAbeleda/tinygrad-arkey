#!/usr/bin/env python3
"""Default-off isolated same-session d512 wall A/B for the L4 vocab substrate fusion.

The vocab head is fusion-admitted on NV sm_120 (decode-epilogue-fusion-route-policy.json)
and currently renders as the single fused `q6k_gen_coop_151936_4096_inkernel`.  This
harness toggles ONLY the vocab head's route admission (model.output) at runtime:

  control   epilogue_fusion_promoted=False -> external_sum chain
            (q6k_gen_coop_151936_4096 + q6k_vocab_scalar_reduce + scatter)
  candidate epilogue_fusion_promoted=True  -> fused in_kernel
            (single q6k_gen_coop_151936_4096_inkernel)

The FFN-down M2 in-kernel merge (q6k_gen_coop_4096_12288_inkernel) is untouched in both
arms; every other admission stays at its loaded value.  Exact token-stream equality
across arms is the correctness gate (the fused shape is bit-identical, measurement
record 05b1e9774).

This is a runtime diagnostic: importing it changes nothing.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, subprocess, sys, time
from dataclasses import replace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def median(xs): return statistics.median(xs) if xs else None


def summarize(rows):
  steady = rows[3:]
  return {"samples": len(steady), "wall_us_median": median([x["wall_us"] for x in steady]),
          "wall_us_mad": median([abs(x["wall_us"] - median([y["wall_us"] for y in steady])) for x in steady]),
          "tokens_sha256": hashlib.sha256(json.dumps([x["token"] for x in steady]).encode()).hexdigest()}


def captured_program_names(model) -> list[str]:
  from tinygrad.uop.ops import Ops
  names: set[str] = set()
  for jit_name in ("rollout_greedy_jit", "rollout_greedy_jit_flash", "rollout_jit", "rollout_jit_flash"):
    captured = getattr(getattr(model, jit_name, None), "captured", None)
    if captured is None: continue
    def collect(target):
      if target.op is Ops.PROGRAM: names.add(target.arg.name)
      elif target.op is Ops.CUSTOM_FUNCTION and target.arg == "graph" and target.src:
        for call in target.src[0].src: collect(call.src[0])
    for call in captured.linear.src: collect(call.src[0])
  return sorted(names)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--samples", type=int, default=30)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  import tinygrad.llm.model as tgm
  from tinygrad.device import Device
  from tinygrad.helpers import Context
  from tinygrad.llm.model import Transformer

  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()
  model, _ = Transformer.from_gguf(args.model, 4608)
  head = model.output
  if not hasattr(head, "q6k_storage"): raise RuntimeError(f"vocab head is not a Q6K primitive: {type(head).__name__}")
  if head.out_features != 151936: raise RuntimeError(f"unexpected vocab head rows {head.out_features}")
  if not head.route_admission.fusion_admitted: raise RuntimeError("vocab head is not fusion-admitted at load")
  default_admission = head.route_admission

  def arm(name: str, fused: bool):
    head.route_admission = default_admission if fused else replace(default_admission, epilogue_fusion_promoted=False)
    model.reset_generation_state()
    gen = model.generate([1] * args.depth, chunk_size=32, temperature=0.0)
    rows = []
    try:
      with Context(DEBUG=0): next(gen); next(gen); next(gen)
      Device["NV"].synchronize()
      for _ in range(args.samples + 3):
        st = time.perf_counter_ns()
        with Context(DEBUG=0): token = int(next(gen))
        Device["NV"].synchronize(); en = time.perf_counter_ns()
        rows.append({"token": token, "wall_us": (en - st) / 1e3})
    finally:
      gen.close()
    return {"name": name, "fused": fused, "rows": rows, "summary": summarize(rows),
            "vocab_programs": [n for n in captured_program_names(model) if "151936" in n or "1187" in n]}

  spec = [("control_a0", False), ("candidate_a", True), ("control_a1", False)]
  arms = [arm(*x) for x in spec]
  hashes = {x["summary"]["tokens_sha256"] for x in arms}
  if len(hashes) != 1: raise RuntimeError(f"arms diverged token stream: {hashes}")
  payload = {"schema": "tinygrad.nv_decode_l4_vocab_fusion_ab.v1", "evidence": "NATIVE_NV_REVERSE_BRACKET",
             "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "route": "DEV=NV", "depth": args.depth, "steady_samples": args.samples, "arms": arms,
             "exact_token_stream": True}
  pathlib.Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
  print(json.dumps({x["name"]: x["summary"] for x in arms}, indent=2))


if __name__ == "__main__": main()
