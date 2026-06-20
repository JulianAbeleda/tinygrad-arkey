#!/usr/bin/env python3
"""Branch B: does explicit TC attention (PREFILL_TC_ATTN) beat SDPA on the CONCRETE first prefill chunk?

The prior probe (qk_prefill_tc_attention_measure.py) is INVALID: it set env `PREFILL_TC_ATTENTION` but the model
reads `PREFILL_TC_ATTN` (typo -> both arms ran SDPA), AND it bound a symbolic start_pos (a UOp) which fails the
`isinstance(start_pos,int)` guard so the TC branch could never fire. This harness measures the path in its valid
regime: a CONCRETE int start_pos=0 (KV=512), graph route ON in both arms (isolate the attention delta).

Method (iron law): SAME-process interleaved A/B. Capture the OFF jit fully (PREFILL_TC_ATTN=False), THEN set True
and capture the ON jit fully -> no TinyJit flag-leak. Kernel-identity assert (ON graph must contain wmma attn
kernels OFF lacks) guards the leak. Correctness rel-RMSE(off,on)<1e-2. Synced arbiter (K forwards/one sync,
clock pinned high, interleaved best-of-N) is the perf metric.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_tc_attn_concrete_gate.py [model.gguf]
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys, time
from collections import defaultdict

os.environ["PREFILL_V2"] = "1"
os.environ.setdefault("PREFILL_GRAPH_GEMM", "1")   # promoted baseline ON in BOTH arms
os.environ.setdefault("PROFILE", "1")              # for the kernel-identity assert


def perflevel(x): subprocess.run(["rocm-smi", "--setperflevel", x], capture_output=True, text=True)


def kernels_of(evs):
  """name -> (us, launches) from ProfileGraphEvent ents (proven by qk_prefill_inmodel_attribution)."""
  ku = defaultdict(float); kn = defaultdict(int)
  for e in evs:
    if type(e).__name__ == "ProfileGraphEvent":
      sigs = e.sigs
      for ent in e.ents:
        nm = str(getattr(ent.name, "display_name", ent.name))
        ku[nm] += (float(sigs[ent.en_id]) - float(sigs[ent.st_id])) / 1000.0; kn[nm] += 1
  return ku, kn


def main() -> int:
  model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
  from tinygrad import Tensor, Device
  from tinygrad.engine.jit import TinyJit
  import tinygrad.codegen.opt.postrange as pr
  import tinygrad.llm.model as M
  from tinygrad.llm.model import Transformer, PREFILL_UBATCH
  from tinygrad.device import Compiled

  dev = Device["AMD"]; Tensor.manual_seed(0)
  model, _ = Transformer.from_gguf(pathlib.Path(model_path).expanduser(), 1024)
  N = PREFILL_UBATCH; sp = 0                              # CONCRETE int start_pos -> TC branch eligible
  temp = Tensor([0.0])
  toks = Tensor([5, 6, 7, 8, 9, 10] * 170 + [0] * (1024 - 1020), dtype="int32").reshape(1, 1024)
  chunk = toks[:, sp:sp + N].contiguous().realize()
  for b in model.blk: b._use_flash, b._prefill_v2 = False, True

  def warm(j, calls=4):
    pr._WARMSTART_OPTS = model._pf16_warmstart
    try:
      out = None
      for _ in range(calls): out = j(chunk, sp, temp).realize(); dev.synchronize()
      return out
    finally:
      pr._WARMSTART_OPTS = None

  def profile_once(j):
    pr._WARMSTART_OPTS = model._pf16_warmstart
    try:
      base = len(Compiled.profile_events)
      j(chunk, sp, temp).realize(); dev.synchronize(); dev._at_profile_finalize()
      return kernels_of(Compiled.profile_events[base:])
    finally:
      pr._WARMSTART_OPTS = None

  def burst(j, K=8):
    pr._WARMSTART_OPTS = model._pf16_warmstart
    try:
      dev.synchronize(); t0 = time.perf_counter()
      for _ in range(K): j(chunk, sp, temp).realize()
      dev.synchronize(); return (time.perf_counter() - t0) / K * 1e3
    finally:
      pr._WARMSTART_OPTS = None

  perflevel("high")
  try:
    # --- capture OFF (SDPA) fully, BEFORE flipping the flag (flag-leak rule) ---
    M.PREFILL_TC_ATTN = False
    jit_off = TinyJit(model.forward); out_off = warm(jit_off)
    ku_off, kn_off = profile_once(jit_off)
    # --- now flip and capture ON (TC) fully ---
    M.PREFILL_TC_ATTN = True
    jit_on = TinyJit(model.forward); out_on = warm(jit_on)
    ku_on, kn_on = profile_once(jit_on)

    # correctness
    a, b = out_off.float().numpy().ravel(), out_on.float().numpy().ravel()
    import numpy as np
    rel_rmse = float(np.sqrt(((a - b) ** 2).mean()) / (np.sqrt((a ** 2).mean()) + 1e-12))

    # kernel-identity (flag-leak guard): ON must contain wmma kernels OFF lacks
    wmma_off = sorted(n for n in ku_off if "wmma" in n.lower())
    wmma_on = sorted(n for n in ku_on if "wmma" in n.lower())
    new_wmma = sorted(set(wmma_on) - set(wmma_off))
    graphs_differ = set(ku_off.keys()) != set(ku_on.keys())

    # synced interleaved arbiter
    REPS = int(os.environ.get("REPS", "6"))
    off_ms, on_ms, ratios = [], [], []
    for _ in range(REPS):
      o = burst(jit_off); n = burst(jit_on); off_ms.append(o); on_ms.append(n); ratios.append(o / n)
  finally:
    perflevel("auto")

  import statistics
  off_p50 = round(statistics.median(off_ms), 2); on_p50 = round(statistics.median(on_ms), 2)
  speedup = round(statistics.median(ratios), 3)
  LLAMA_MS = 170.0
  result = {
    "date": "2026-06-20", "phase": "BRANCH_B_TC_ATTN_CONCRETE", "model": pathlib.Path(model_path).name,
    "regime": "concrete int start_pos=0, KV=512, graph route ON both arms", "N": N,
    "correctness_rel_rmse": round(rel_rmse, 6), "correctness_pass": rel_rmse < 1e-2,
    "kernel_identity": {"graphs_differ": graphs_differ, "wmma_off": len(wmma_off), "wmma_on": len(wmma_on),
                        "new_wmma_in_on": new_wmma[:6], "tc_fired": len(new_wmma) > 0},
    "perf_synced": {"off_sdpa_p50_ms": off_p50, "on_tc_p50_ms": on_p50, "speedup_median": speedup,
                    "off_toks": round(N / (off_p50 / 1e3), 1), "on_toks": round(N / (on_p50 / 1e3), 1),
                    "off_pct_llama": round(100 * LLAMA_MS / off_p50, 1), "on_pct_llama": round(100 * LLAMA_MS / on_p50, 1),
                    "reps": REPS, "off_ms_all": [round(x, 2) for x in off_ms], "on_ms_all": [round(x, 2) for x in on_ms]},
  }
  win = result["correctness_pass"] and result["kernel_identity"]["tc_fired"] and speedup >= 1.05
  result["verdict"] = ("WIN" if win else "NO_WIN") + (
    f": TC {speedup}x over SDPA (rel_rmse {rel_rmse:.2e}, tc_fired={result['kernel_identity']['tc_fired']})")
  # attention bucket top kernels per arm (for the writeup)
  def top(ku, kn):
    tot = sum(ku.values()) or 1
    return [{"name": n[:42], "launches": kn[n], "pct": round(100 * u / tot, 1)} for n, u in
            sorted(ku.items(), key=lambda kv: -kv[1])[:8]]
  result["top_kernels_off"] = top(ku_off, kn_off); result["top_kernels_on"] = top(ku_on, kn_on)

  out = pathlib.Path("bench/qk-prefill-tc-attention"); out.mkdir(parents=True, exist_ok=True)
  (out / "concrete_gate_result.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({k: result[k] for k in ("correctness_rel_rmse", "kernel_identity", "perf_synced", "verdict")}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
