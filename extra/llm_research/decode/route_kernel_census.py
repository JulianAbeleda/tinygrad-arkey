#!/usr/bin/env python3
"""Route-generic per-token kernel-class census for the B3 numerics census (G-B3-C).

Captures the DEBUG=2 prime token at a fixed decode depth (d512 default) on the
selected route (DEV=NV native or DEV=CUDA with CUDA_GRAPH_STREAMS=1) and
classifies every kernel launch into a class (flash decode attention, q4k gemv,
q6k gemv, norms/rmsnorm, elementwise/fusions, copies, kv-store, vocab, scatter,
other) using the same kernel-name vocabulary the NV-side census tools use
(kernel_log_diff.py / gemv_class_census_nv.py). Reports per-class counts, the
graph-group count per token, and the house pins (first token, token sha).

Run: PYTHONPATH=. DEV=NV .venv/bin/python extra/llm_research/decode/route_kernel_census.py --depth 512 --out /tmp/census_nv.json
     PYTHONPATH=. DEV=CUDA CUDA_GRAPH_STREAMS=1 .venv/bin/python extra/llm_research/decode/route_kernel_census.py --depth 512 --out /tmp/census_cuda.json
"""
from __future__ import annotations

import argparse, contextlib, hashlib, io, json, re, statistics, time

from tinygrad.helpers import Context
from tinygrad.llm.model import Transformer

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"

# Route-generic kernel launch line: `*** {dev:7} {count:4d} {name} arg {n} mem {g} GB tm {us}/{ms}/...`
# The device token is any of NV/CUDA; names carry a trailing 64-hex content hash stripped by canonical_name.
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LAUNCH = re.compile(
  r"\*\*\* \w+\s+(\d+)\s+(\S+)\s+arg\s+\d+\s+mem\s+([\d.]+) GB\s+tm\s+([\d.]+)(us|ms)/")
HASH64 = re.compile(r"_[0-9a-f]{64}\b")


def canonical_name(name:str) -> str:
  return HASH64.sub("", name).strip()


def is_copy(name:str) -> bool:
  return name.startswith("copy")


def classify(name:str) -> str:
  """Per-class rule, NV/CUDA-agnostic. Order matters: vocab before q6k, scatter before E_/r_."""
  if name.startswith("flash_"):
    return "flash_decode_attention"
  if name.startswith("q4k_g3_lanemap_gemv"):
    return "q4k_gemv"
  if "151936" in name or name.startswith("q6k_vocab_scalar_reduce"):
    return "vocab_head"
  if name.startswith("q6k_gen_coop") or name.startswith("q6k_gen_partial"):
    return "q6k_gemv"
  if name.startswith("decode_kv_rope_store"):
    return "kv_store"
  if "1187" in name:
    return "scatter"
  if name.startswith("E_") or name.startswith("r_"):
    return "elementwise_fusion"
  return "other"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--nmeas", type=int, default=20)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  from tinygrad import Device
  dev = Device[Device.DEFAULT]
  route = Device.DEFAULT
  model, kv = Transformer.from_gguf(MODEL, 4608)
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

  # Prime token (direct exec, per-kernel lines): the census basis.
  per_kernel: dict[str, list[float]] = {}
  prime_kernel_lines = 0
  for raw in lines[:prime_end]:
    m = LAUNCH.search(ANSI.sub("", raw))
    if not m: continue
    name = m.group(2)
    if is_copy(name): continue
    us = float(m.group(4)) * (1000.0 if m.group(5) == "ms" else 1.0)
    per_kernel.setdefault(canonical_name(name), []).append(us)
    prime_kernel_lines += 1
  class_counts: dict[str, int] = {}
  class_us: dict[str, float] = {}
  for name, us_list in per_kernel.items():
    cls = classify(name)
    class_counts[cls] = class_counts.get(cls, 0) + len(us_list)
    class_us[cls] = class_us.get(cls, 0.0) + statistics.median(us_list) * len(us_list)

  # Replay token (token 3 window): count graph-group launches per token.
  groups = 0
  for raw in lines[marks[1]:marks[2]]:
    s = ANSI.sub("", raw)
    if "batched " in s:
      groups += 1

  # Pins: first token + token sha over the nmeas decode stream, same as the census harness.
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

  out = {
    "route": route,
    "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd="/home/ubuntu/tinygrad-arkey").decode().strip(),
    "depth": args.depth,
    "kernels_per_token_prime": sum(class_counts.values()),
    "kernel_launch_lines_prime": prime_kernel_lines,
    "graph_groups_per_token_replay": groups,
    "class_counts": class_counts,
    "class_us_sum": {k: round(v, 1) for k, v in class_us.items()},
    "per_kernel_us": {k: {"count": len(v), "median_us": round(statistics.median(v), 2),
                          "total_us": round(sum(v), 2), "min_us": round(min(v), 2),
                          "max_us": round(max(v), 2)} for k, v in per_kernel.items()},
    "tok_s_median": round(statistics.median(tok_s), 3),
    "token_sha_reps": shas,
    "first_token_reps": firsts,
  }
  print(json.dumps(out, indent=1))
  with open(args.out, "w") as f:
    json.dump(out, f, indent=1)


if __name__ == "__main__":
  main()
