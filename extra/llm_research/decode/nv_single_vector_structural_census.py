#!/usr/bin/env python3
"""Structural census for the Q4_K single-projection vector-load candidate.

Loads the production model and runs a short decode under ``observe_graph_admissions``
so the exact installed program-name set is retained. Run twice (TINYGRAD_Q4K_SCALAR_LOAD=1
vs unset) to prove whether the vector spelling actually engages on Q/K/O and how many
kernels it replaces. No production file is modified.
"""
from __future__ import annotations

import argparse, json, os, pathlib, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--depth", type=int, default=128)
  ap.add_argument("--count", type=int, default=4)
  ap.add_argument("--max-context", type=int, default=512)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  from tinygrad import Tensor, UOp
  from tinygrad.engine.jit import GraphAdmissionCensus, observe_graph_admissions
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt

  model = _load("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", args.max_context)
  model._decode_direct_greedy_promoted = False
  model._decode_feedback_pingpong_promoted = False
  gen = model.generate(_prompt("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", args.depth), chunk_size=32, temperature=0.0)
  try: prelude = int(next(gen))
  finally: gen.close()

  token, temp = Tensor([[1]], dtype="int32").contiguous(), Tensor([0.0])
  start_pos = UOp.variable("start_pos", 0, args.max_context - 1)
  census = GraphAdmissionCensus()
  for i in range(args.count):
    with observe_graph_admissions(census):
      sampled, full = model.decode_with_logits(token, start_pos.bind(args.depth + 1 + i), temp)
      token = Tensor([[int(sampled.item())]], dtype="int32").contiguous()

  programs = Counter(r.program_name for r in census.records if r.program_name)
  vec = {n: c for n, c in programs.items() if n.startswith("q4k_g3_lanemap_gemv_vec")}
  scalar = {n: c for n, c in programs.items() if n.startswith("q4k_g3_lanemap_gemv_")
            and not n.startswith("q4k_g3_lanemap_gemv_vec")
            and not n.startswith("q4k_g3_lanemap_gemv_w1w3")
            and not n.startswith("q4k_g3_lanemap_gemv_quad")}
  w1w3_vec = {n: c for n, c in programs.items() if n.startswith("q4k_g3_lanemap_gemv_w1w3vec")}
  w1w3_scalar = {n: c for n, c in programs.items() if n.startswith("q4k_g3_lanemap_gemv_w1w3fused")}
  down_vec = {n: c for n, c in programs.items() if n.startswith("q4k_fp16_mmvq_direct_vec_")}
  down_scalar = {n: c for n, c in programs.items() if n.startswith("q4k_fp16_mmvq_direct_")
                 and not n.startswith("q4k_fp16_mmvq_direct_vec_")}
  coop_direct = {n: c for n, c in programs.items() if n.startswith("q4k_warp_coop_q8_dp4a_direct_")}
  coop_partial = {n: c for n, c in programs.items() if n.startswith("q4k_warp_coop_q8_dp4a_partial_")}
  # Scheduled reduction names carry a source hash suffix.  These two prefixes are
  # the only completion shapes emitted by the cooperative Q4 partial-output route.
  coop_completion = {n: c for n, c in programs.items()
                     if n.startswith("r_32_32_4_4_") or n.startswith("r_8_32_4_4_")}
  coop = {n: c for n, c in programs.items() if n.startswith("q4k_warp_coop_q8_dp4a_partial_")}
  legacy = {n: c for n, c in programs.items() if n.startswith("q4k_q8_dp4a_")}

  result = {
    "schema": "tinygrad.nv_single_vector_structural_census.v1",
    "load_style": "scalar" if os.environ.get("TINYGRAD_Q4K_SCALAR_LOAD") else "vector",
    "depth": args.depth, "count": args.count,
    "total_programs": len(programs),
    "single_proj_vector": dict(sorted(vec.items())),
    "single_proj_vector_count": sum(vec.values()),
    "single_proj_scalar": dict(sorted(scalar.items())),
    "single_proj_scalar_count": sum(scalar.values()),
    "w1w3_vector": dict(sorted(w1w3_vec.items())),
    "w1w3_scalar": dict(sorted(w1w3_scalar.items())),
    "ffn_down_vector": dict(sorted(down_vec.items())),
    "ffn_down_vector_count": sum(down_vec.values()),
    "ffn_down_scalar": dict(sorted(down_scalar.items())),
    "ffn_down_scalar_count": sum(down_scalar.values()),
    "cooperative_q4_direct": dict(sorted(coop_direct.items())),
    "cooperative_q4_direct_count": sum(coop_direct.values()),
    "cooperative_q4_partial": dict(sorted(coop_partial.items())),
    "cooperative_q4_partial_count": sum(coop_partial.values()),
    "cooperative_q4_completion": dict(sorted(coop_completion.items())),
    "cooperative_q4_completion_count": sum(coop_completion.values()),
    "coop_q4_count": sum(coop.values()),
    "legacy_shared_q4_count": sum(legacy.values()),
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
