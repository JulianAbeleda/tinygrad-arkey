#!/usr/bin/env python3
"""In-loop census for the Q6_K attention-V four-warp fp16 direct route.

The graph-replay microgate is launch-bound, so the standing gate for a Q6 V
geometry change is the same DEBUG=2 prime-token census the production route
uses.  This harness installs the closed-default ``Q6KVFourWarpAdmission`` on
every exact Q6_K 1024x4096 attn_kv linear, runs the census, and reports the
new kernel's in-loop median against the installed ``q6k_gen_partial_1024_4096_4``
(17.94 us at HEAD) plus the token-stream pins.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, io, json, re, statistics, time

from tinygrad.helpers import Context
from tinygrad.llm.model import Transformer

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LAUNCH = re.compile(
  r"\*\*\* \w+\s+(\d+)\s+(\S+)\s+arg\s+\d+\s+mem\s+([\d.]+) GB\s+tm\s+([\d.]+)(us|ms)/")
HASH64 = re.compile(r"_[0-9a-f]{64}\b")


def canonical_name(name:str) -> str:
  return HASH64.sub("", name).strip()


def install_admission(model, enabled:bool) -> list[int]:
  from tinygrad.llm.q6k_v_mmvq import Q6KVFourWarpAdmission
  from tinygrad.llm.qk_primitives import Q6KPrimitiveLinear
  out = []
  for idx, block in enumerate(model.blk):
    v = getattr(block, "attn_v", None)
    if not isinstance(v, Q6KPrimitiveLinear) or getattr(v, "route_role", "") != "attn_kv":
      continue
    if (getattr(v, "out_features", None), getattr(v, "in_features", None)) != (1024, 4096):
      continue
    if enabled:
      v._q6k_v_four_warp_admission = Q6KVFourWarpAdmission(idx)
    elif hasattr(v, "_q6k_v_four_warp_admission"):
      delattr(v, "_q6k_v_four_warp_admission")
    out.append(idx)
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--nmeas", type=int, default=20)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--out", required=True)
  ap.add_argument("--disable", action="store_true", help="run the installed control route (no admission)")
  args = ap.parse_args()

  from tinygrad import Device
  dev = Device[Device.DEFAULT]
  model, kv = Transformer.from_gguf(MODEL, 4608)
  admitted = install_admission(model, not args.disable)
  prompt = [1] * args.depth
  gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
  with Context(DEBUG=0):
    next(gen)
  buf = io.StringIO()
  marks = []
  with contextlib.redirect_stdout(buf):
    with Context(DEBUG=2):
      next(gen); marks.append(len(buf.getvalue().splitlines()))
      next(gen); marks.append(len(buf.getvalue().splitlines()))
      next(gen); marks.append(len(buf.getvalue().splitlines()))
  gen.close()
  lines = buf.getvalue().splitlines()
  prime_end = marks[0]

  per_kernel: dict[str, list[float]] = {}
  for raw in lines[:prime_end]:
    m = LAUNCH.search(ANSI.sub("", raw))
    if not m: continue
    name = canonical_name(m.group(2))
    if name.startswith("copy"): continue
    us = float(m.group(4)) * (1000.0 if m.group(5) == "ms" else 1.0)
    per_kernel.setdefault(name, []).append(us)

  groups = 0
  for raw in lines[marks[1]:marks[2]]:
    if "batched " in ANSI.sub("", raw):
      groups += 1

  tok_s, shas, firsts = [], [], []
  for _ in range(args.reps):
    model.reset_generation_state()
    gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
    next(gen)
    dev.synchronize()
    lat, toks = [], []
    for _ in range(args.nmeas):
      t0 = time.perf_counter()
      toks.append(int(next(gen)))
      lat.append(time.perf_counter() - t0)
    gen.close()
    tok_s.append(args.nmeas / sum(lat))
    shas.append(hashlib.sha256(",".join(map(str, toks)).encode()).hexdigest())
    firsts.append(toks[0])

  direct = per_kernel.get("q6k_v_four_warp_fp16_direct_1024_4096", [])
  partial = per_kernel.get("q6k_gen_partial_1024_4096_4", [])
  def _med(values:list[float]) -> float|None:
    return round(statistics.median(values), 2) if values else None
  out = {
    "route": "NV",
    "admission_enabled": not args.disable,
    "admitted_block_indices": admitted,
    "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"],
      cwd="/home/ubuntu/tinygrad-arkey").decode().strip(),
    "depth": args.depth,
    "graph_groups_per_token_replay": groups,
    "per_kernel_us": {name: {"count": len(v), "median_us": round(statistics.median(v), 2),
                             "total_us": round(sum(v), 2)} for name, v in per_kernel.items()},
    "q6k_v_four_warp_direct": {"count": len(direct), "median_us": _med(direct),
      "total_us": round(sum(direct), 2)},
    "q6k_gen_partial_control": {"count": len(partial), "median_us": _med(partial),
      "total_us": round(sum(partial), 2)},
    "tok_s_median": round(statistics.median(tok_s), 3),
    "token_sha_reps": shas,
    "first_token_reps": firsts,
  }
  print(json.dumps(out, indent=1))
  with open(args.out, "w") as f:
    json.dump(out, f, indent=1)


if __name__ == "__main__":
  main()
