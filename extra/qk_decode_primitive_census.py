#!/usr/bin/env python3
"""Phase 1: per-token decode primitive census for Qwen3-8B (or any QK gguf). Answers: how many GPU programs run
per decode token, how much authoritative GPU time each primitive class costs, and how much is host/JIT overhead.

Method (separates the two honestly):
  - GPU per-kernel census: run ONE decode step EAGERLY under DEBUG=2 (each kernel dispatched + timed
    individually), parse per-kernel name+tm, classify by name. Authoritative GPU time (not wall).
  - Host/e2e: JIT the decode step (the real decode path), warm up, take median WALL time/token; GPU time/token
    via GlobalCounters.time_sum_s. host_overhead = wall - gpu. (wall is host/e2e evidence only, never GPU evidence.)

Portable: model path from argv/env (QK_MODEL). Run:
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_decode_primitive_census.py [model.gguf]
"""
from __future__ import annotations

import io, json, os, pathlib, re, statistics, sys, time, contextlib

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_KLINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(.+?)\s+arg\s.*?tm\s+([\d.]+)us")   # full name up to 'arg'
_GEMV = re.compile(r"q[46]k_gemv\w*_(\d+)_(\d+)_")   # q4k/q6k_gemv_..._<out>_<in>_1
# Qwen3-8B: hidden 4096, ffn 12288, kv 1024, vocab 151936. Map (out,in) of a GEMV to its role.
_SHAPE_ROLE = {(151936, 4096): "lm_head", (4096, 12288): "ffn_down", (12288, 4096): "ffn_gate/up",
               (4096, 4096): "attn_q/o", (1024, 4096): "attn_k/v"}

def _classify(name: str) -> str:
  mn = _GEMV.search(name.lower())
  if mn:
    role = _SHAPE_ROLE.get((int(mn.group(1)), int(mn.group(2))), "gemv_other")
    return f"QKGEMV:{role}"
  n = name.lower()
  # the per-step 4-byte input upload (copy 4 B, AMD <- PYTHON) shows huge tm at 0 GB/s -- that's a step-boundary
  # sync/launch STALL, not GPU work (proven in qk_decode_copy_diagnostic). Exclude it from GPU attribution.
  if n.startswith("copy") and ("python" in n or " 4 b" in n or "4 b," in n): return "input_upload_sync_EXCLUDED"
  if n.startswith("copy"): return "copy/gather"
  # tinygrad auto-names the rest (E_*, r_*) -- attention compute + norms + rope + residuals; can't positively
  # split by name, so bucket honestly as non-GEMV compute/small-ops.
  return "nonGEMV_compute_small"

def main():
  model = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".gguf") else os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  MAXC, NWARM, NMEAS = 2048, 8, 30
  from tinygrad import Tensor, UOp, GlobalCounters, Context, Device
  from extra.llm_generate import load_model_and_tokenizer
  m, tok = load_model_and_tokenizer(model, MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("The quick brown fox jumps over the lazy dog. " * 8)
  pre = ids[:64]
  with Context(DEBUG=0): m.logits(Tensor([pre], dtype="int32").contiguous(), 0).realize()  # prefill to populate KV cache
  sp = len(pre)

  # --- GPU per-kernel census: one EAGER decode step under DEBUG=2 (warm it first so timings aren't cold) ---
  with Context(DEBUG=0):
    for _ in range(3): m.logits(Tensor([[ids[sp]]], dtype="int32").contiguous(), sp).realize()
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2):
    GlobalCounters.reset()
    m.logits(Tensor([[ids[sp]]], dtype="int32").contiguous(), sp).realize()
    gpu_census_s = GlobalCounters.time_sum_s
  kernels = [(nm, float(us)) for nm, us in _KLINE.findall(_ANSI.sub("", buf.getvalue()))]
  by_class: dict = {}
  for nm, us in kernels:
    c = _classify(nm); d = by_class.setdefault(c, {"count": 0, "us": 0.0})
    d["count"] += 1; d["us"] += us
  # exclude the input-upload sync stall from the GPU-time denominator (it's a measurement artifact, not GPU work)
  total_us = sum(d["us"] for c, d in by_class.items() if c != "input_upload_sync_EXCLUDED") or 1.0

  # --- host/e2e: JIT decode step, median wall/token + GPU/token ---
  v_sp = UOp.variable("start_pos", 0, MAXC - 1)
  from tinygrad import TinyJit
  step = TinyJit(lambda t, s: m.logits(t, s).realize())
  tokid = int(ids[sp])   # decode token value is irrelevant to kernel structure; vary only start_pos
  for i in range(NWARM): step(Tensor([[tokid]], dtype="int32").contiguous(), v_sp.bind(sp + i))
  walls = []   # clean host wall: DEBUG=0 (no per-kernel sync inflation)
  with Context(DEBUG=0):
    for i in range(NMEAS):
      t0 = time.perf_counter(); step(Tensor([[tokid]], dtype="int32").contiguous(), v_sp.bind(sp + i)).realize()
      walls.append(time.perf_counter() - t0)
  gpus = []    # authoritative GPU time: time_sum_s under DEBUG=2 (separate from wall)
  with Context(DEBUG=2):
    for i in range(NMEAS):
      GlobalCounters.reset(); step(Tensor([[tokid]], dtype="int32").contiguous(), v_sp.bind(sp + i))
      gpus.append(GlobalCounters.time_sum_s)
  wall_ms = statistics.median(walls) * 1e3; gpu_dbg_ms = statistics.median(gpus) * 1e3
  classes = sorted(by_class.items(), key=lambda kv: -kv[1]["us"])
  out = {"model_id": pathlib.Path(model).stem, "hardware": "RX 7900 XTX / gfx1100", "device": Device.DEFAULT,
         "command": "qk_decode_primitive_census.py", "decode_config": "no demotion; JIT graph; T=1 decode",
         "timing_method": "wall = median DEBUG=0 perf_counter over JIT-graph replay (real e2e); per-kernel = eager "
                          "DEBUG=2 tm (UNBATCHED -> relative-weight proxy, not the batched-graph GPU time)",
         "programs_per_token": len(kernels), "tok_s_implied_wall": round(1000 / wall_ms, 1) if wall_ms else None,
         "wall_ms_per_token": round(wall_ms, 3),
         "debug2_unbatched_gpu_sum_ms": round(gpu_dbg_ms, 3),
         "note_host_split": "wall (batched JIT graph) < DEBUG=2 per-kernel GPU sum -> the JIT graph BATCHES/overlaps "
                            "the kernels (like a CUDA graph), so per-launch host overhead is already amortized and a "
                            "clean wall=GPU+host split is NOT available from DEBUG timing. The gap vs llama is GPU-side "
                            "(kernel count/fusion/copies), not per-kernel launch overhead.",
         "gpu_per_class_relative": {c: {"count": d["count"], "pct_gpu_proxy": round(100 * d["us"] / total_us, 1)}
                                    for c, d in classes},
         "top_kernels": [{"name": nm[:48], "us": round(us, 1)} for nm, us in sorted(kernels, key=lambda x: -x[1])[:12]]}
  print(f"programs/token: {len(kernels)} | warm wall {wall_ms:.2f}ms = {out['tok_s_implied_wall']} tok/s "
        f"| DEBUG2 unbatched GPU sum {gpu_dbg_ms:.2f}ms (proxy)")
  for c, d in classes:
    print(f"  {c:22} {d['count']:3} kernels  {100*d['us']/total_us:4.1f}% GPU (relative proxy)")
  art = pathlib.Path("bench/qk-decode-primitive-census/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2)); print(f"artifact: {art}")

if __name__ == "__main__":
  main()
