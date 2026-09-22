#!/usr/bin/env python3
"""DEV=NV same-session A/B: promoted S=48 flash decode vs env-gated coarse splits S=4 / S=2.

The llama d512 source-derived flash install is a 4-way KV split (~3.14 us/launch body);
tinygrad's promoted G4 route uses S=48 (~4.16 us/launch) and 36 score launches/token. This
harness runs the PRODUCTION decode route (flash_decode_attention_route -> the generated
live-split tile) on DEV=NV with ``FLASH_DECODE_COARSE_SPLIT`` set only in the candidate
windows, interleaving control / candidate / control / ... under one flock. Each window is a
fresh process because tinygrad's ``getenv`` is functools-cache'd per key: the env must be
present at process start for the route override to take effect. Byte-identical token sha
between arms is the correctness gate; wall delta must clear +50 us/token to PROMOTE.

Run (from the repo root):

  flock -w 600 /tmp/gpu-bench.lock env PYTHONPATH=. DEV=NV \
    python3 extra/llm_research/decode/nv_flash_coarse_split_ab.py --out <evidence.json>

``--child`` mode runs one arm window in THIS process and prints a JSON window record; the
default (parent) mode spawns children interleaved, aggregates, and writes the evidence file.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, io, json, os, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
MAX_CONTEXT = 4608
TILE_PREFIX = "flash_block_tiled_xlane_score_pv_tile_whole_cache_"
ENV_KEY = "FLASH_DECODE_COARSE_SPLIT"
CONTROL_SPLIT = 48
SCHEMA = "tinygrad.nv_flash_coarse_split_ab.v1"
PROMOTION_BAR_US = 50.0


def _tile_names(model, depth: int) -> list[str]:
  """Capture the flash tile kernel names rendered during the first decode tokens."""
  import tinygrad.llm.model as tgm
  from tinygrad.helpers import Context
  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()
  gen = model.generate([1] * depth, chunk_size=32, temperature=0.0)
  with Context(DEBUG=0):
    next(gen)
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    with Context(DEBUG=2):
      next(gen)
  gen.close()
  return sorted({line.split()[3] for line in buf.getvalue().splitlines() if TILE_PREFIX in line})


def run_window(split: int, depth: int, nmeas: int, reps: int, settle: int) -> dict:
  """Run one arm window: fresh process, env already set by the parent. Returns a JSON-able record."""
  from tinygrad import Device
  from tinygrad.helpers import Context
  from tinygrad.llm.model import Transformer
  import tinygrad.llm.model as tgm
  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()
  model, _kv = Transformer.from_gguf(MODEL, MAX_CONTEXT)
  tile_names = _tile_names(model, depth)
  dev = Device[Device.DEFAULT]
  # Warm the decode graph once before any timed window so JIT/capture cost is not charged.
  gen = model.generate([1] * depth, chunk_size=32, temperature=0.0)
  with Context(DEBUG=0):
    next(gen)
  dev.synchronize()
  gen.close()
  tok_s, wall_us, shas, firsts, all_tokens = [], [], [], [], []
  for _ in range(reps):
    model.reset_generation_state()
    gen = model.generate([1] * depth, chunk_size=32, temperature=0.0)
    for _ in range(1 + settle):  # first token after prefill + settle tokens, not charged
      next(gen)
    dev.synchronize()
    lat, toks = [], []
    for _ in range(nmeas):
      t0 = time.perf_counter()
      toks.append(int(next(gen)))
      lat.append(time.perf_counter() - t0)
    gen.close()
    tok_s.append(nmeas / sum(lat))
    wall_us.append(sum(lat) / nmeas * 1e6)
    shas.append(hashlib.sha256(",".join(map(str, toks)).encode()).hexdigest())
    firsts.append(toks[0])
    all_tokens.extend(toks)
  return {
    "split": split, "depth": depth, "nmeas": nmeas, "reps": reps, "settle": settle,
    "tile_names": tile_names,
    "tok_s_samples": tok_s, "wall_us_per_token_samples": wall_us,
    "token_sha_reps": shas, "first_token_reps": firsts,
    "token_sha": hashlib.sha256(",".join(map(str, all_tokens)).encode()).hexdigest(),
    "tokens": all_tokens,
  }


def _arm_split(arm: str) -> int:
  if arm == "control": return CONTROL_SPLIT
  return int(arm[len("s"):])


def _child_env(arm: str) -> dict:
  env = dict(os.environ)
  if arm == "control":
    env.pop(ENV_KEY, None)  # unset env keeps production behavior byte-identical
  else:
    env[ENV_KEY] = str(_arm_split(arm))
  return env


def _spawn_window(arm: str, args) -> dict:
  cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--arm", arm,
         "--depth", str(args.depth), "--nmeas", str(args.nmeas), "--reps", str(args.reps),
         "--settle", str(args.settle)]
  proc = subprocess.run(cmd, env=_child_env(arm), capture_output=True, text=True, check=True)
  return json.loads(proc.stdout.strip().splitlines()[-1])


def _window_plan(args) -> list[str]:
  """Bracketed interleave: each round brackets s4 and s2 with control on both sides."""
  plan: list[str] = []
  for _ in range(args.rounds):
    plan += ["control", "s4", "control", "s2", "control"]
  return plan


def _arm_records(plan: list[str], args) -> dict[str, list[dict]]:
  arms: dict[str, list[dict]] = {}
  for window_index, arm in enumerate(plan):
    rec = _spawn_window(arm, args)
    rec["window_index"] = window_index
    arms.setdefault(arm, []).append(rec)
    print(f"[window {window_index}] arm={arm} split={rec['split']} "
          f"tok_s={statistics.median(rec['tok_s_samples']):.2f} "
          f"wall_us={statistics.median(rec['wall_us_per_token_samples']):.1f} "
          f"sha={rec['token_sha'][:12]}", flush=True)
  return arms


def _arm_summary(records: list[dict], arm: str) -> dict:
  tok_s = [s for r in records for s in r["tok_s_samples"]]
  wall_us = [s for r in records for s in r["wall_us_per_token_samples"]]
  # The arm token sha is the sha256 of the full measured token stream (all reps of one window;
  # every window in the same session must be byte-identical, asserted below). Hashing the rep-sha
  # list would make the arm hash depend on window count, so we hash the tokens themselves.
  window_shas = [r["token_sha"] for r in records]
  identical_windows = len(set(window_shas)) == 1
  stream = records[0]["tokens"]
  aggregate = hashlib.sha256(",".join(map(str, stream)).encode()).hexdigest()
  return {
    "arm": arm, "split": _arm_split(arm), "n_windows": len(records),
    "n_measured_tokens": sum(r["nmeas"] * r["reps"] for r in records),
    "tok_s_median": statistics.median(tok_s), "tok_s_samples": tok_s,
    "wall_us_per_token_median": statistics.median(wall_us), "wall_us_per_token_samples": wall_us,
    "token_sha": aggregate, "token_sha_windows": window_shas, "token_sha_identical_windows": identical_windows,
    "first_tokens": [t for r in records for t in r["first_token_reps"]],
    "tile_names": sorted({n for r in records for n in r["tile_names"]}),
  }


def _git_commit() -> str:
  try:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=str(ROOT), check=True).stdout.strip()
    return out
  except Exception:
    return "unknown"


def _gpu_state() -> str:
  try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
    return out.replace("\n", "; ")
  except Exception:
    return "nvidia-smi unavailable"


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--child", action="store_true", help="run one arm window in this process")
  ap.add_argument("--arm", choices=("control", "s2", "s4"), default="control")
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--nmeas", type=int, default=8)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--settle", type=int, default=3)
  ap.add_argument("--rounds", type=int, default=2, help="bracket rounds in parent mode")
  ap.add_argument("--out", type=Path, help="evidence JSON path (parent mode)")
  args = ap.parse_args(argv)

  if args.child:
    if os.getenv("DEV", "") != "NV":
      raise RuntimeError("child must run with DEV=NV")
    print(json.dumps(run_window(_arm_split(args.arm), args.depth, args.nmeas, args.reps, args.settle),
                     sort_keys=True))
    return 0

  if os.getenv("DEV", "") != "NV":
    raise RuntimeError("this harness measures DEV=NV only; set DEV=NV")
  plan = _window_plan(args)
  arms = _arm_records(plan, args)
  summaries = {arm: _arm_summary(recs, arm) for arm, recs in arms.items()}
  control = summaries["control"]
  shas = [control["token_sha"]] + [summaries[a]["token_sha"] for a in summaries if a != "control"]
  sha_identical = len(set(shas)) == 1 and all(s["token_sha_identical_windows"] for s in summaries.values())

  deltas: dict[str, dict] = {}
  for arm in ("s4", "s2"):
    if arm not in summaries: continue
    cand = summaries[arm]
    deltas[arm] = {
      "wall_delta_us_vs_control": control["wall_us_per_token_median"] - cand["wall_us_per_token_median"],
      "tok_s_delta_vs_control": cand["tok_s_median"] - control["tok_s_median"],
      "sha_identical_to_control": cand["token_sha"] == control["token_sha"],
    }
  cleared = {arm: d["wall_delta_us_vs_control"] >= PROMOTION_BAR_US for arm, d in deltas.items()}
  verdict = "PROMOTE" if (sha_identical and any(cleared.values())) else "NO-GO"
  if not sha_identical:
    reason = "token sha differs between arms (correctness NO-GO)"
  elif any(cleared.values()):
    best = max(cleared, key=lambda a: deltas[a]["wall_delta_us_vs_control"])
    reason = f"{best} clears the +{PROMOTION_BAR_US:g} us/token bar"
  else:
    reason = f"no candidate clears the +{PROMOTION_BAR_US:g} us/token bar"

  evidence = {
    "schema": SCHEMA,
    "date": "2026-08-19",
    "git_commit": _git_commit(),
    "gpu": "RTX 5090 sm_120",
    "gpu_state": _gpu_state(),
    "model": MODEL,
    "backend": "DEV=NV (production native route)",
    "method": (f"fresh process per arm window under flock -w 600 /tmp/gpu-bench.lock; depth {args.depth}; "
               f"{args.settle} settle tokens; {args.nmeas} measured tokens x {args.reps} reps; "
               f"interleaved window plan {plan}; candidate runs with {ENV_KEY} set at process start; "
               "token stream sha256 is the correctness gate; wall bar +50 us/token vs S=48 control"),
    "promotion_bar_us": PROMOTION_BAR_US,
    "arms": summaries,
    "delta_median": deltas,
    "token_sha_identical_across_arms": sha_identical,
    "verdict": verdict,
    "conclusion": reason,
  }
  encoded = json.dumps(evidence, indent=2, sort_keys=True)
  print(encoded)
  if args.out:
    args.out.write_text(encoded + "\n")
  return 0


if __name__ == "__main__":
  sys.exit(main())
