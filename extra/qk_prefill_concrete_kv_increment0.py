#!/usr/bin/env python3
"""Increment 0: does forcing CONCRETE chunks (+ the shipped fusion attention) capture the SYMBOLIC-regime prize?

Today only the first prefill chunk (start_pos=0) is concrete -> fusion attention (~5%). Chunks 2+ use a SYMBOLIC
(bound) start_pos -> SDPA, where attention is ~47% (and the symbolic codegen is ~3x costlier). PREFILL_CONCRETE_KV=1
makes every chunk concrete int -> the default-on fusion path fires on all of them. This probe measures, per
start_pos, the SYMBOLIC forward (today) vs the CONCRETE forward (Increment 0), with NO new kernel code.

No flag toggle / no flag-leak risk: the attention path is chosen by `isinstance(start_pos,int)` baked per-jit at
capture (symbolic UOp -> SDPA, int -> fusion). We build both jits and measure them interleaved (iron-law synced
arbiter), report attention-share per arm + correctness + the concrete-jit CAPTURE (compile) cost (the real tax).

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_concrete_kv_increment0.py [model.gguf]
     (orchestrates a subprocess per start_pos; --worker <sp> runs one)
"""
from __future__ import annotations
import json, os, pathlib, statistics, subprocess, sys, time
from collections import defaultdict

os.environ.setdefault("PREFILL_V2", "1")
os.environ.setdefault("PREFILL_GRAPH_GEMM", "1")     # promoted baseline ON in BOTH arms
os.environ.setdefault("PREFILL_TC_ATTN", "1")        # shipped fusion ON (fires only on the concrete arm)

SPS = [0, 512, 1536, 3072]
LLAMA_MS = 170.0


def perflevel(x): subprocess.run(["rocm-smi", "--setperflevel", x], capture_output=True, text=True)


def _bucket(ku, kn):
  """attention-share from per-kernel us; attention = r_ reduces that aren't the FFN graph gemm / norm."""
  tot = sum(ku.values()) or 1.0
  attn = ffn = other = 0.0
  for n, u in ku.items():
    low = n.lower()
    if "prefill_graph_gemm" in low: ffn += u
    elif "start_pos" in low: attn += u                       # symbolic attention kernels
    elif low.startswith("r_") and kn[n] >= 30 and "32_3" not in low: attn += u  # concrete attention reduces (36/layer)
    else: other += u
  return {"attn_pct": round(100 * attn / tot, 1), "ffn_pct": round(100 * ffn / tot, 1),
          "other_pct": round(100 * other / tot, 1)}


def worker(model_path: str, start_pos: int):
  from tinygrad import Tensor, UOp, Device
  from tinygrad.engine.jit import TinyJit
  import tinygrad.codegen.opt.postrange as pr
  from tinygrad.llm.model import Transformer, PREFILL_UBATCH
  from tinygrad.device import Compiled
  import numpy as np
  os.environ.setdefault("PROFILE", "1")
  dev = Device["AMD"]; Tensor.manual_seed(0)
  N = PREFILL_UBATCH
  mc = max(1024, ((start_pos + N + 511) // 512) * 512 + 512)   # KV headroom for start_pos+T
  model, _ = Transformer.from_gguf(pathlib.Path(model_path).expanduser(), mc)
  temp = Tensor([0.0])
  t = Tensor(([5, 6, 7, 8, 9, 10] * (mc // 6 + 1))[:mc], dtype="int32").reshape(1, mc)
  for b in model.blk: b._use_flash, b._prefill_v2 = False, True
  vsp = UOp.variable("start_pos", 0, mc - 1); sp_sym = vsp.bind(start_pos)

  def install(): pr._WARMSTART_OPTS = model._pf16_warmstart
  def restore(): pr._WARMSTART_OPTS = None

  def warm(j, tok, spv, calls=4):
    install()
    try:
      out = None
      for _ in range(calls): out = j(tok.contiguous(), spv, temp).realize(); dev.synchronize()
      return out
    finally: restore()

  def profile_once(j, tok, spv):
    install()
    try:
      base = len(Compiled.profile_events)
      j(tok.contiguous(), spv, temp).realize(); dev.synchronize(); dev._at_profile_finalize()
      ku, kn = defaultdict(float), defaultdict(int)
      for e in Compiled.profile_events[base:]:
        if type(e).__name__ == "ProfileGraphEvent":
          for ent in e.ents:
            nm = str(getattr(ent.name, "display_name", ent.name))
            ku[nm] += (float(e.sigs[ent.en_id]) - float(e.sigs[ent.st_id])) / 1000.0; kn[nm] += 1
      return ku, kn
    finally: restore()

  def burst(j, tok, spv, K=8):
    install()
    try:
      dev.synchronize(); t0 = time.perf_counter()
      for _ in range(K): j(tok.contiguous(), spv, temp).realize()
      dev.synchronize(); return (time.perf_counter() - t0) / K * 1e3
    finally: restore()

  perflevel("high")
  try:
    sym_tok = t[:, sp_sym:sp_sym + N]
    con_tok = t[:, start_pos:start_pos + N]
    # SYMBOLIC arm (today's chunk>0 behavior: SDPA)
    jit_sym = TinyJit(model.forward); out_sym = warm(jit_sym, sym_tok, sp_sym)
    ku_s, kn_s = profile_once(jit_sym, sym_tok, sp_sym)
    # CONCRETE arm (Increment 0: int start_pos -> fusion); time the CAPTURE cost (the K-jit tax)
    jit_con = TinyJit(model.forward)
    t_cap = time.perf_counter(); out_con = warm(jit_con, con_tok, start_pos); capture_s = time.perf_counter() - t_cap
    ku_c, kn_c = profile_once(jit_con, con_tok, start_pos)

    a, b = out_sym.float().numpy().ravel(), out_con.float().numpy().ravel()
    rel_rmse = float(np.sqrt(((a - b) ** 2).mean()) / (np.sqrt((a ** 2).mean()) + 1e-12))

    REPS = int(os.environ.get("REPS", "6"))
    sym_ms, con_ms, ratios = [], [], []
    for _ in range(REPS):
      s = burst(jit_sym, sym_tok, sp_sym); c = burst(jit_con, con_tok, start_pos)
      sym_ms.append(s); con_ms.append(c); ratios.append(s / c)
  finally:
    perflevel("auto")

  print("@@R@@" + json.dumps({
    "start_pos": start_pos, "KV": start_pos + N, "max_context": mc,
    "sym_p50_ms": round(statistics.median(sym_ms), 2), "con_p50_ms": round(statistics.median(con_ms), 2),
    "speedup_con_over_sym": round(statistics.median(ratios), 3),
    "sym_pct_llama": round(100 * LLAMA_MS / statistics.median(sym_ms), 1),
    "con_pct_llama": round(100 * LLAMA_MS / statistics.median(con_ms), 1),
    "sym_buckets": _bucket(ku_s, kn_s), "con_buckets": _bucket(ku_c, kn_c),
    "rel_rmse": round(rel_rmse, 6), "concrete_capture_s": round(capture_s, 1),
    "sym_ms_all": [round(x, 2) for x in sym_ms], "con_ms_all": [round(x, 2) for x in con_ms]}))


def main() -> int:
  if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
    model_path = sys.argv[3] if len(sys.argv) > 3 else "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
    worker(model_path, int(sys.argv[2])); return 0
  model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
  rows = []
  for sp in SPS:
    p = subprocess.run([sys.executable, __file__, "--worker", str(sp), model_path],
                       env={**os.environ, "PYTHONPATH": "."}, capture_output=True, text=True, timeout=1800)
    line = next((l for l in p.stdout.splitlines() if l.startswith("@@R@@")), None)
    if line is None:
      print(f"start_pos={sp} FAILED:\n{p.stdout[-400:]}\n{p.stderr[-600:]}"); continue
    r = json.loads(line[5:]); rows.append(r)
    print(f"  sp={r['start_pos']:5d} KV={r['KV']:5d}: sym {r['sym_p50_ms']:7.2f}ms (attn {r['sym_buckets']['attn_pct']:4.1f}%) "
          f"| con {r['con_p50_ms']:7.2f}ms (attn {r['con_buckets']['attn_pct']:4.1f}%) -> {r['speedup_con_over_sym']}x "
          f"| %llama {r['sym_pct_llama']:.0f}->{r['con_pct_llama']:.0f} | cap {r['concrete_capture_s']}s rmse {r['rel_rmse']}")
  result = {"date": "2026-06-20", "phase": "PREFILL_CONCRETE_KV_INCREMENT_0",
            "regime": "per start_pos: symbolic (SDPA, today) vs concrete (fusion, Increment 0); graph route ON both",
            "llama_pp512_ms": LLAMA_MS, "rows": rows}
  out = pathlib.Path("bench/qk-prefill-tc-attention"); out.mkdir(parents=True, exist_ok=True)
  (out / "concrete_kv_increment0_result.json").write_text(json.dumps(result, indent=2) + "\n")
  print(f"\nartifact: {out / 'concrete_kv_increment0_result.json'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
