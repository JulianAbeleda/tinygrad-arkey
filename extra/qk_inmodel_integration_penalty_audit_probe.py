#!/usr/bin/env python3
from __future__ import annotations

import collections
import json
import os
import pathlib
import re
import statistics
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-inmodel-integration-penalty/inmodel_integration_penalty_audit_result.json"
MODEL = pathlib.Path(os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))


def classify(name: str) -> str:
  nm = str(name)
  if re.search(r"r_(16_192|8_64|8_16|16_64)", nm): return "ffn_proj_matmul"
  if "start_pos" in nm or re.search(r"(_512_|_128_4|2_8_16_4_4_128|512_16_256)", nm): return "attention_kv"
  if nm.startswith("E_"): return "elementwise_glue"
  if "1187" in nm: return "lm_head"
  return "other"


def pct(x: float, total: float) -> float:
  return round(100.0 * x / total, 2) if total else 0.0


def am_dahl(component_share: float, component_speedup: float) -> float:
  if component_speedup <= 0: return 1.0
  return 1.0 / ((1.0 - component_share) + component_share / component_speedup)


def profile_one(model: Any, chunk: Any, temp: Any, sp: Any, label: str, warmups: int, iters: int) -> dict[str, Any]:
  from tinygrad import Device, TinyJit
  from tinygrad.device import Compiled

  for b in model.blk:
    b._use_flash, b._prefill_v2 = False, True
  model.prefill_v2_jit = TinyJit(model.forward)

  for _ in range(warmups):
    model(chunk, sp, temp).realize()
    Device["AMD"].synchronize()

  walls = []
  for _ in range(iters):
    t0 = time.perf_counter()
    model(chunk, sp, temp).realize()
    Device["AMD"].synchronize()
    walls.append(time.perf_counter() - t0)

  base = len(Compiled.profile_events)
  model(chunk, sp, temp).realize()
  Device["AMD"].synchronize()
  Device["AMD"]._at_profile_finalize()
  evs = [e for e in Compiled.profile_events[base:] if type(e).__name__ == "ProfileGraphEvent"]
  if not evs:
    return {"label": label, "profile_available": False, "wall_ms": {"median": statistics.median(walls) * 1000.0}}

  e = evs[-1]
  sigs = [float(s) for s in e.sigs]
  per, cnt = collections.defaultdict(float), collections.defaultdict(int)
  rows = []
  for ent in e.ents:
    dur = sigs[ent.en_id] - sigs[ent.st_id]
    cat = classify(ent.name)
    per[cat] += dur
    cnt[cat] += 1
    rows.append({"name": str(ent.name), "category": cat, "start_us": sigs[ent.st_id], "end_us": sigs[ent.en_id], "dur_us": dur})

  busy_us = sum(r["dur_us"] for r in rows)
  span_us = max(r["end_us"] for r in rows) - min(r["start_us"] for r in rows)
  gap_us = span_us - busy_us
  components = {
    k: {
      "busy_us": round(v, 3),
      "count": cnt[k],
      "busy_share": round(v / busy_us, 6) if busy_us else 0.0,
      "span_share": round(v / span_us, 6) if span_us else 0.0,
    }
    for k, v in sorted(per.items(), key=lambda x: -x[1])
  }
  top_kernels = [
    {"name": r["name"], "category": r["category"], "dur_us": round(r["dur_us"], 3)}
    for r in sorted(rows, key=lambda x: -x["dur_us"])[:20]
  ]
  return {
    "label": label,
    "profile_available": True,
    "kernel_count": len(rows),
    "wall_ms": {
      "median": round(statistics.median(walls) * 1000.0, 3),
      "min": round(min(walls) * 1000.0, 3),
      "max": round(max(walls) * 1000.0, 3),
      "samples": [round(x * 1000.0, 3) for x in walls],
    },
    "graph": {
      "span_us": round(span_us, 3),
      "busy_us": round(busy_us, 3),
      "gap_us": round(gap_us, 3),
      "gap_span_share": round(gap_us / span_us, 6) if span_us else 0.0,
    },
    "components": components,
    "top_kernels": top_kernels,
  }


def main() -> int:
  if not os.environ.get("PREFILL_V2"):
    print("ERROR: run with PREFILL_V2=1")
    return 2
  if not os.environ.get("PROFILE"):
    print("ERROR: run with PROFILE=1 so HCQ graph timestamps are collected")
    return 2

  from tinygrad import Tensor, UOp
  from tinygrad.llm.model import PREFILL_UBATCH, Transformer

  Tensor.manual_seed(0)
  model, _ = Transformer.from_gguf(MODEL.expanduser(), 2048)
  n = PREFILL_UBATCH
  t = Tensor([5, 6, 7, 8, 9, 10] * 200 + [0] * (2048 - 1200), dtype="int32").reshape(1, 2048)
  chunk = t[:, 0:n]
  temp = Tensor([0.0])
  vsp = UOp.variable("start_pos", 0, 2047)

  concrete = profile_one(model, chunk, temp, 0, "concrete_start_pos_0", warmups=5, iters=5)
  symbolic = profile_one(model, chunk, temp, vsp.bind(0), "symbolic_start_pos_bound_0", warmups=5, iters=5)
  authority = symbolic if symbolic.get("profile_available") else concrete
  comps = authority.get("components") or {}
  matmul_share = float((comps.get("ffn_proj_matmul") or {}).get("span_share") or 0.0)
  attention_share = float((comps.get("attention_kv") or {}).get("span_share") or 0.0)
  glue_share = sum(float((comps.get(k) or {}).get("span_share") or 0.0) for k in ("elementwise_glue", "lm_head", "other"))

  source = {
    "prefill_inmodel_tok_s": 2797,
    "prefill_inmodel_wall_ms_per_512": 183.0,
    "prefill_inmodel_effective_tflops": 45.0,
    "isolated_ours_gemm_tflops": 78.6,
    "isolated_tensile_gemm_tflops": 70.9,
    "llama_pp512_tok_s": 3020,
  }
  isolated_to_inmodel = source["isolated_ours_gemm_tflops"] / source["prefill_inmodel_effective_tflops"]
  amdahl = {
    "matmul_share_of_span": round(matmul_share, 6),
    "attention_share_of_span": round(attention_share, 6),
    "glue_other_share_of_span": round(glue_share, 6),
    "if_matmul_gets_1p10x": round(am_dahl(matmul_share, 1.10), 4),
    "if_matmul_gets_1p25x": round(am_dahl(matmul_share, 1.25), 4),
    "if_matmul_gets_isolated_78p6_over_45": round(am_dahl(matmul_share, isolated_to_inmodel), 4),
    "if_attention_gets_1p25x": round(am_dahl(attention_share, 1.25), 4),
    "if_non_matmul_removed": round(am_dahl(1.0 - matmul_share, 10**9), 4),
  }
  graph_span_ms = float(((authority.get("graph") or {}).get("span_us") or 0.0)) / 1000.0
  gates = {
    "profile_available": bool(authority.get("profile_available")),
    "authority_is_symbolic_measure_path": authority.get("label") == "symbolic_start_pos_bound_0",
    "matmul_share_measured": matmul_share > 0.0,
    "graph_span_close_to_banked_prefill": 120.0 <= graph_span_ms <= 260.0,
  }
  if not gates["profile_available"]:
    verdict = "BLOCKED_INMODEL_INTEGRATION_AUDIT_NO_PROFILE"
  elif amdahl["if_matmul_gets_1p10x"] < 1.05:
    verdict = "PASS_INMODEL_INTEGRATION_AUDIT_GEMM_ONLY_LOW_LEVERAGE"
  else:
    verdict = "PASS_INMODEL_INTEGRATION_AUDIT_AMDAHL_LEDGER"

  result = {
    "date": "2026-06-20",
    "phase": "INMODEL_INTEGRATION_PENALTY_AUDIT",
    "schema": "inmodel_integration_penalty_audit_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_") and all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": True,
    "source": source,
    "profiles": {"concrete_start_pos_0": concrete, "symbolic_start_pos_bound_0": symbolic},
    "authority_profile": authority.get("label"),
    "amdahl": amdahl,
    "gates": gates,
    "decision": {
      "prefill": "Use the role split and Amdahl ledger before starting more GEMM microkernel work.",
      "decode": "Apply the same component-vs-lifecycle timing standard to q8 producer/consumer promotion.",
    },
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "gate_pass": result["gate_pass"],
    "authority_profile": result["authority_profile"],
    "wall_ms": authority.get("wall_ms"),
    "graph": authority.get("graph"),
    "components": authority.get("components"),
    "amdahl": amdahl,
    "gates": gates,
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
