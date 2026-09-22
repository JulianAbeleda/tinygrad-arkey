#!/usr/bin/env python3
"""Wall + token-SHA A/B for the warp-reduce native RMSNorm route (Path 3).

Each arm is a fresh process: control runs the production graph; candidate sets
``_rmsnorm_native_promoted=True`` on the selected norm sites so ``nn.RMSNorm``
emits the ``_semantic_rmsnorm`` marker, which ``lower_rmsnorm_semantic`` lowers
to the warp-reduce ``rmsnorm_native_*`` kernel (with its per-call input copies
on lazy producer views). The caller runs a reverse control/candidate/control
bracket and compares median wall plus the byte-identical token-stream gate.
"""
from __future__ import annotations

import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from extra.llm_research.decode.nv_predispatch_full_logits_qualification import DEFAULT_MODEL, _load, _prompt
from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows

SITES = ("attn", "ffn", "q", "k", "output")


def _configure_native(model, sites: tuple[str, ...], enabled: bool = True) -> None:
  from tinygrad.dtype import dtypes
  for block in model.blk:
    if "attn" in sites:
      block.attn_norm._rmsnorm_native_promoted = enabled
      block.attn_norm._rmsnorm_native_output_dtype = dtypes.float16
    if "ffn" in sites:
      block.ffn_norm._rmsnorm_native_promoted = enabled
      block.ffn_norm._rmsnorm_native_output_dtype = dtypes.float16
    if "q" in sites and getattr(block, "attn_q_norm", None) is not None:
      block.attn_q_norm._rmsnorm_native_promoted = enabled
    if "k" in sites and getattr(block, "attn_k_norm", None) is not None:
      block.attn_k_norm._rmsnorm_native_promoted = enabled
  if "output" in sites: model.output_norm._rmsnorm_native_promoted = enabled


def run(arm: str, sites: tuple[str, ...], model_path: str, depth: int, count: int,
        max_context: int, reps: int) -> dict:
  from tinygrad import Device
  model = _load(model_path, max_context)
  if arm == "candidate":
    _configure_native(model, sites)
  elif arm == "control":
    _configure_native(model, sites, False)
  else:
    raise ValueError(f"unknown arm {arm!r}")
  gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0)
  try:
    row = _settled_continuous_windows(gen, Device[Device.DEFAULT], count, reps)
  finally:
    gen.close()
  gates = {
    "arm": arm, "sites": list(sites),
    "native_promoted_sample": {
      "attn": bool(getattr(model.blk[0].attn_norm, "_rmsnorm_native_promoted", False)),
      "ffn": bool(getattr(model.blk[0].ffn_norm, "_rmsnorm_native_promoted", False)),
      "q": bool(getattr(getattr(model.blk[0], "attn_q_norm", None), "_rmsnorm_native_promoted", False)),
      "k": bool(getattr(getattr(model.blk[0], "attn_k_norm", None), "_rmsnorm_native_promoted", False)),
      "output": bool(getattr(model.output_norm, "_rmsnorm_native_promoted", False)),
    },
  }
  return {"schema": "tinygrad.nv_norm_native_wall_ab.v1", **gates,
          "median_ms_per_token": row["median_ms_per_token"],
          "token_stream_hash": row["token_stream_hash"],
          "samples_ms_per_token": row["samples_ms_per_token"],
          "rejected_high_samples_ms_per_token": row["rejected_high_samples_ms_per_token"],
          "timed_token_count": row["timed_token_count"]}


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--arm", required=True, choices=("control", "candidate"))
  ap.add_argument("--sites", default="ffn", help="comma-separated subset of attn,ffn,q,k,output")
  ap.add_argument("--model", default=DEFAULT_MODEL)
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--count", type=int, default=24)
  ap.add_argument("--max-context", type=int, default=1024)
  ap.add_argument("--reps", type=int, default=4)
  ap.add_argument("--out", type=pathlib.Path)
  args = ap.parse_args()
  sites = tuple(s for s in args.sites.split(",") if s)
  for s in sites:
    if s not in SITES:
      raise SystemExit(f"unknown site {s!r}; choose from {SITES}")
  got = run(args.arm, sites, args.model, args.depth, args.count, args.max_context, args.reps)
  text = json.dumps(got, indent=2, sort_keys=True)
  if args.out:
    args.out.write_text(text + "\n")
  print(text)
