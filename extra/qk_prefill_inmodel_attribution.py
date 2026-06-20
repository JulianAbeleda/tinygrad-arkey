#!/usr/bin/env python3
"""In-model PREFILL attribution: where does the warm 512-tok forward actually spend GPU time?

Profiles the warm PREFILL_V2 forward (Qwen3-8B) at pinned clock, reads per-kernel GPU device timestamps,
and buckets them: ffn_gate/up GEMM, ffn_down, attention, norm/elementwise, other. This tells us the biggest
*integration* cost to attack (not another GEMM microkernel).

Run: DEV=AMD PREFILL_V2=1 PROFILE=1 PYTHONPATH=. python3 extra/qk_prefill_inmodel_attribution.py <model.gguf>
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys
from collections import defaultdict


def perflevel(x): subprocess.run(["rocm-smi", "--setperflevel", x], capture_output=True, text=True)


def main() -> int:
  model_path = sys.argv[1] if len(sys.argv) > 1 else (os.environ.get("QK_MODEL") or os.environ.get("MODEL"))
  os.environ.setdefault("PROFILE", "1"); os.environ.setdefault("PREFILL_V2", "1")
  from tinygrad import Tensor, UOp, Device
  import tinygrad.codegen.opt.postrange as pr
  from tinygrad.llm.model import Transformer, PREFILL_UBATCH
  from tinygrad.device import Compiled
  dev = Device["AMD"]
  Tensor.manual_seed(0)
  model, _ = Transformer.from_gguf(pathlib.Path(model_path).expanduser(), 768)  # small KV: prefill is start_pos=0, ctx only needs >=512
  N = PREFILL_UBATCH; maxc = model.max_context
  vsp = UOp.variable("start_pos", 0, maxc - 1)
  temp = Tensor([0.0])
  t = Tensor((([5, 6, 7, 8, 9, 10] * (maxc // 6 + 1))[:maxc]), dtype="int32").reshape(1, maxc)
  sp = vsp.bind(0)
  v2_chunk = t[:, sp:sp + N]
  pr._warmstart_stats = {"match": 0, "apply": 0, "error": 0}
  fwd = lambda: model(v2_chunk, sp, temp)

  import time
  perflevel("high")
  try:
    for _ in range(3): fwd().realize(); dev.synchronize()  # warm + JIT-capture (TinyJit captures by ~2nd call)
    base = len(Compiled.profile_events)
    dev.synchronize(); t0 = time.perf_counter(); fwd().realize(); dev.synchronize(); wall_ms = (time.perf_counter() - t0) * 1e3
    dev._at_profile_finalize()
  finally:
    perflevel("auto")

  evs = Compiled.profile_events[base:]
  from collections import Counter
  types = Counter(type(e).__name__ for e in evs)
  # JIT-captured forward => ProfileGraphEvent(ents=[ProfileGraphEntry(name, st_id, en_id)], sigs=[ts...]).
  # per-kernel GPU time = sigs[en_id]-sigs[st_id]. Also fold in any loose ProfileRangeEvent (uncaptured leaks).
  kern_us = defaultdict(float); kern_n = defaultdict(int)
  for e in evs:
    if type(e).__name__ == "ProfileGraphEvent":
      sigs = e.sigs
      for ent in e.ents:
        nm = str(getattr(ent.name, "display_name", ent.name))
        dt = float(sigs[ent.en_id]) - float(sigs[ent.st_id])
        kern_us[nm] += dt / 1000.0; kern_n[nm] += 1
    elif type(e).__name__ == "ProfileRangeEvent" and getattr(e, "st", None) is not None and getattr(e, "en", None) is not None:
      nm = str(getattr(e, "name", "?"))
      if "TINY" in nm: continue
      kern_us[nm] += (float(e.en) - float(e.st)) / 1000.0; kern_n[nm] += 1

  def bucket(nm: str) -> str:
    n = nm.lower()
    # tinygrad kernel names encode the reduce/elementwise shape; classify by launches + shape signature below
    return n
  # classify by launch count + the known atlas signatures (gate/up = 72 launches/2x36 layers, down=36, attn=36)
  rows = sorted(kern_us.items(), key=lambda kv: -kv[1])
  total = sum(kern_us.values())
  buckets = defaultdict(float); bn = defaultdict(int)
  for nm, us in kern_us.items():
    n = kern_n[nm]
    low = nm.lower()
    if "start_pos" in low: b = "attention(start_pos)"      # attention kernels depend on the symbolic context
    elif "wmma" in low or (low.startswith("r_") and n >= 60): b = "ffn_gate_up_GEMM"
    elif low.startswith("r_"): b = "matmul_other(down/qkv/o)"
    elif low.startswith("e_") or "norm" in low or "cast" in low: b = "elementwise_norm_cast"
    else: b = "other"
    buckets[b] += us; bn[b] += n
  # absolute estimate via wall calibration (GPU device-timestamp unit is uncalibrated; relative % is robust)
  scale = wall_ms / (total + 1e-9)

  result = {"date": "2026-06-20", "phase": "PREFILL_INMODEL_ATTRIBUTION", "model": pathlib.Path(model_path).name,
            "N": N, "wall_ms": round(wall_ms, 1), "raw_gpu_us_unscaled": round(total, 1), "event_types": dict(types),
            "buckets_pct": {k: round(100 * v / total, 1) for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])},
            "buckets_ms_wallcalib": {k: round(v * scale, 2) for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])},
            "top_kernels": [{"name": nm[:46], "launches": kern_n[nm], "pct": round(100 * us / total, 1), "ms_wallcalib": round(us * scale, 3)} for nm, us in rows[:14]]}
  out = pathlib.Path("bench/amd-broad-backend-roadmap"); out.mkdir(parents=True, exist_ok=True)
  (out / "prefill_inmodel_attribution_result.json").write_text(json.dumps(result, indent=2) + "\n")
  print(f"wall (profiled forward): {wall_ms:.1f} ms | raw GPU-accounted {total/1000:.2f} ms over {sum(kern_n.values())} dispatches | scale x{scale:.0f}")
  print("=== buckets (% of GPU time, wall-calibrated ms) ===")
  for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]): print(f"  {k:26} {100*v/total:5.1f}%  ~{v*scale:6.1f} ms  ({bn[k]} dispatches)")
  print("=== top kernels ===")
  for nm, us in rows[:12]: print(f"  {kern_n[nm]:3}x  {100*us/total:5.1f}%  ~{us*scale:6.1f}ms  {nm[:46]}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
