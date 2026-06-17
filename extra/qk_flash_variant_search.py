#!/usr/bin/env python3
"""Track 3 of the machine-search scaffold: the decode flash-attention PRIMITIVE FAMILY search.

The flash-decode primitive is now a parameterized family (extra/qk_flash_decode.flash_decode_attention):
  variant in {v1, hoisted}  x  KV-split L in {64, 128, 256, 512}
'hoisted' computes the softmax probability once per key (flash_prob kernel) instead of recomputing exp
per output-dim lane (v1 = Hd+1 = 129x redundant exp); it is byte-identical to v1 and to SDPA.

This module searches that family IN-MODEL with the carried-forward measurement discipline -- the W==D
warm device-token-feed method (NOT eager DEBUG=2, NOT per-step Tensor creation) -- and emits the
best-(variant,L)-per-KV policy plus a durable frontier table. FLASH_VARIANT / FLASH_L are read per
process (getenv is cached), so each grid cell runs as a subprocess (self-invoked with --worker), exactly
as qk_flash_search subprocesses qk_flash_sweep per mode.

  candidate grid -> per-cell W==D runner (subprocess) -> scorer (best tok/s per ctx, exactness gate)
  -> AcceptedPolicy + frontier table

Worker:       DEV=AMD JIT=1 FLASH_DECODE=1 FLASH_VARIANT=hoisted FLASH_L=256 PYTHONPATH=. \
                .venv/bin/python -m extra.qk_flash_variant_search --worker
Orchestrator: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python -m extra.qk_flash_variant_search
"""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys, time

from extra.llm_eval_common import write_json
from extra.qk_demote_search import _git_commit, _model_id
from extra.qk_search_spec import AcceptedPolicy, Constraints, assemble_search_row, baseline as model_baseline

HARDWARE = "RX 7900 GRE / gfx1100"   # measured on this host (campaign docs assume XTX; same gfx1100 arch)
DEFAULT_CKPTS = (512, 1024, 2048, 4096)
# baseline first (v1@L256), then the hoisted L sweep. v1 is the pre-arc default; hoisted is the candidate.
DEFAULT_GRID = (("v1", 256), ("hoisted", 256), ("hoisted", 128), ("hoisted", 64))
MAXC = 4608


def _worker(model:str, ckpts:tuple[int, ...], nmeas:int) -> dict:
  """One grid cell: measure W (real decode tok/s) + D (dispatch ceiling) + a deterministic greedy
  sequence at each ctx, for the FLASH_VARIANT/FLASH_L set in env. Forces flash on at every ctx."""
  variant, L = os.environ.get("FLASH_VARIANT", "v1"), int(os.environ.get("FLASH_L", "256"))
  from tinygrad import Tensor, UOp, TinyJit, Device
  from extra.llm_generate import load_model_and_tokenizer
  dev = Device[Device.DEFAULT]
  m, tok = load_model_and_tokenizer(model, MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  ids = (ids * (1 + MAXC // max(1, len(ids))))[:MAXC]
  v_sp = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0])   # temp=0 -> greedy/argmax
  for b in m.blk: b._use_flash, b._prefill_v2 = True, False
  step = TinyJit(m.forward)
  rows = []
  for ck in ckpts:
    tokid = int(ids[ck])
    out = Tensor([[tokid]], dtype="int32").contiguous()
    for i in range(8): out = step(out, v_sp.bind(ck + i), temp).realize()      # warm: compile + clock ramp
    seq, out = [], Tensor([[tokid]], dtype="int32").contiguous()
    for i in range(24):
      out = step(out, v_sp.bind(ck + i), temp); seq.append(int(out.item()))    # deterministic greedy trace
    out, W = Tensor([[tokid]], dtype="int32").contiguous(), []
    for i in range(nmeas):
      t0 = time.perf_counter(); out = step(out, v_sp.bind(ck + i), temp); _ = int(out.item())
      W.append(time.perf_counter() - t0)
    out = Tensor([[tokid]], dtype="int32").contiguous(); dev.synchronize(); t0 = time.perf_counter()
    for i in range(nmeas): out = step(out, v_sp.bind(ck + i), temp)
    dev.synchronize(); D = (time.perf_counter() - t0) / nmeas
    w_ms = statistics.median(W) * 1e3
    rows.append({"ctx": ck, "tok_s_W": round(1000 / w_ms, 1), "tok_s_D_ceiling": round(1 / D, 1),
                 "wall_ms": round(w_ms, 3), "greedy": seq})
  return {"variant": variant, "L": L, "rows": rows}


def _run_cell(model:str, variant:str, L:int, ckpts:tuple[int, ...], nmeas:int, timeout:int) -> dict:
  env = {**os.environ, "DEV": "AMD", "JIT": "1", "FLASH_DECODE": "1", "FLASH_VARIANT": variant,
         "FLASH_L": str(L), "PYTHONPATH": ".", "QK_CKPTS": ",".join(map(str, ckpts)), "QK_NMEAS": str(nmeas)}
  cmd = [sys.executable, "-m", "extra.qk_flash_variant_search", "--worker", "--model", model]
  p = subprocess.run(cmd, cwd=str(pathlib.Path(__file__).resolve().parents[1]), env=env, text=True,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
  if p.returncode != 0: raise RuntimeError(f"cell {variant}/L{L} failed:\n{p.stdout[-1200:]}")
  return json.loads(p.stdout.strip().splitlines()[-1])


def run_search(model:str, *, grid=DEFAULT_GRID, ckpts=DEFAULT_CKPTS, nmeas:int, timeout:int,
               out_dir:pathlib.Path) -> dict:
  model_id, commit = _model_id(model), _git_commit()
  llama = model_baseline("qwen3_8b")["llama_tok_s"]
  cells = {}
  for variant, L in grid:
    print(f"  measuring {variant} L={L} ...", file=sys.stderr)
    cells[(variant, L)] = _run_cell(model, variant, L, ckpts, nmeas, timeout)

  # exactness gate: every cell's greedy trace must be identical to the v1 baseline at each ctx (flash is exact).
  base = next(c for (v, _), c in cells.items() if v == "v1")
  base_greedy = {r["ctx"]: r["greedy"] for r in base["rows"]}
  exact_ok = all(r["greedy"] == base_greedy[r["ctx"]] for c in cells.values() for r in c["rows"])

  # per-ctx frontier + winner (best tok_s_W); baseline = v1 at the same L=256.
  base_tok = {r["ctx"]: r["tok_s_W"] for r in cells[("v1", 256)]["rows"]}
  frontier = []
  policy_by_ctx = {}
  for ck in ckpts:
    cell_scores = {(v, L): next(r["tok_s_W"] for r in c["rows"] if r["ctx"] == ck) for (v, L), c in cells.items()}
    (bv, bL), btok = max(cell_scores.items(), key=lambda kv: kv[1])
    policy_by_ctx[ck] = {"variant": bv, "L": bL, "tok_s": btok}
    frontier.append({"ctx": ck, "baseline_v1_tok_s": base_tok[ck], "best_variant": bv, "best_L": bL,
                     "best_tok_s": btok, "speedup": round(btok / base_tok[ck], 3),
                     "cells": {f"{v}/L{L}": s for (v, L), s in cell_scores.items()}})

  # the dominant lever is the variant (hoisted), monotone across ctx; L is a marginal per-ctx refinement.
  # AcceptedPolicy: flash-active range, hoisted vs the v1 baseline, measured at the longest ctx.
  top = max(ckpts)
  ap = AcceptedPolicy(model="qwen3_8b", phase="long_context_decode", backend="AMD",
                      ctx_range=(512, MAXC), objective="tok_s",
                      baseline_tok_s=base_tok[top], accepted_tok_s=policy_by_ctx[top]["tok_s"],
                      quality_gate="byte-identical greedy (flash variant is exact vs SDPA up to fp reassociation)",
                      exactness="byte-identical", commit=commit, hardware=HARDWARE)
  spec = assemble_search_row(row_id=f"flash_variant:{model_id}", phase="long_context_decode",
                             model="qwen3_8b", op_scope="attention", backend="AMD",
                             search_space="flash_variant", objective="tok_s",
                             constraints=Constraints(exact_required=True, ctx_range=(512, MAXC)))
  summary = {"model_id": model_id, "hardware": HARDWARE, "commit": commit, "llama_tok_s": llama,
             "exact_gate_passed": exact_ok, "grid": [f"{v}/L{L}" for v, L in grid], "ckpts": list(ckpts),
             "frontier": frontier, "policy_by_ctx": policy_by_ctx, "spec": spec,
             "accepted_policy": ap.to_dict()}
  write_json(out_dir / "flash-variant-search.json", summary)
  write_json(out_dir / "accepted-flash-variant.json", {**ap.to_dict(), "model_id": model_id,
                                                       "policy_by_ctx": policy_by_ctx})
  return summary


def frontier_md(s:dict) -> str:
  lines = [f"# Flash-variant frontier — best (variant, L) per KV", "",
           f"llama.cpp ref = {s['llama_tok_s']} tok/s (campaign baseline; measured on XTX, this host is GRE).",
           f"exactness gate (greedy identical to v1 at every ctx): {'PASS' if s['exact_gate_passed'] else 'FAIL'}", "",
           "| ctx | v1 (baseline) | best variant | best L | best tok/s | speedup |",
           "| ---: | ---: | :-- | ---: | ---: | ---: |"]
  for r in s["frontier"]:
    lines.append(f"| {r['ctx']} | {r['baseline_v1_tok_s']} | {r['best_variant']} | {r['best_L']} | "
                 f"{r['best_tok_s']} | {r['speedup']}x |")
  return "\n".join(lines)


def main():
  ap = argparse.ArgumentParser(description="flash-variant primitive-family search (Track 3)")
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--worker", action="store_true", help="internal: measure one grid cell (env-configured)")
  ap.add_argument("--ckpts", type=int, nargs="*", default=list(DEFAULT_CKPTS))
  ap.add_argument("--nmeas", type=int, default=int(os.environ.get("QK_NMEAS", "30")))
  ap.add_argument("--timeout", type=int, default=600)
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/qk-flash-variant-search"))
  args = ap.parse_args()
  if args.worker:
    ckpts = tuple(int(x) for x in os.environ.get("QK_CKPTS", "").split(",") if x) or DEFAULT_CKPTS
    print(json.dumps(_worker(args.model, ckpts, args.nmeas)))
    return
  summary = run_search(args.model, ckpts=tuple(args.ckpts), nmeas=args.nmeas, timeout=args.timeout, out_dir=args.out)
  print(frontier_md(summary))
  print(f"\nartifacts: {args.out}/flash-variant-search.json , {args.out}/accepted-flash-variant.json")


if __name__ == "__main__":
  main()
