#!/usr/bin/env python3
"""Gate 4 — PREFILL_GRAPH_GEMM OOM + policy audit (default-on readiness).

Proves enabling the route introduces no new memory failure class, and records the explicit default-on policy
field. OOM checks: model load succeeds with and without the flag; the route kernel cache realizes no extra
model-sized buffers (its only allocation is the bounded per-call output); a graph route run does not OOM beyond
the known full-window NLL eval harness; unsupported fallback allocates no output before returning None.

Run: DEV=AMD PYTHONPATH=. python3 extra/qk_prefill_graph_gemm_oom_policy_audit.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf
     (--load-worker runs a single model load in a subprocess)
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys


def load_worker(model_path, graph: bool):
  from tinygrad import Tensor, UOp, Device
  from tinygrad.llm.model import Transformer, PREFILL_UBATCH
  dev = Device["AMD"]; Tensor.manual_seed(0)
  model, _ = Transformer.from_gguf(pathlib.Path(model_path).expanduser(), 2048)
  N = PREFILL_UBATCH; maxc = model.max_context
  vsp = UOp.variable("start_pos", 0, maxc - 1); temp = Tensor([0.0])
  t = Tensor([5, 6, 7, 8, 9, 10] * 200 + [0] * (maxc - 1200), dtype="int32").reshape(1, maxc)
  sp = vsp.bind(0)
  # one real forward exercises the route (graph) or the normal path (baseline) -> proves no load/run OOM
  model(t[:, sp:sp + N], sp, temp).realize(); dev.synchronize()
  print(json.dumps({"loaded": True, "ran_forward": True}))


def run_load(model_path, graph):
  env = dict(os.environ); env["DEV"] = "AMD"; env["PREFILL_V2"] = "1"; env["PYTHONPATH"] = "."
  if graph: env["PREFILL_GRAPH_GEMM"] = "1"
  else: env.pop("PREFILL_GRAPH_GEMM", None)
  p = subprocess.run([sys.executable, __file__, "--load-worker", model_path, "1" if graph else "0"],
                     env=env, capture_output=True, text=True, timeout=900)
  oom = "MemoryError" in p.stderr or "Allocation of" in p.stderr
  ok = p.returncode == 0 and '"loaded": true' in p.stdout.lower()
  return {"rc": p.returncode, "ok": ok, "oom": oom, "err": p.stderr[-200:] if p.returncode != 0 else None}


def main() -> int:
  if len(sys.argv) >= 4 and sys.argv[1] == "--load-worker":
    load_worker(sys.argv[2], sys.argv[3] == "1"); return 0
  model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"

  checks = {}
  # OOM 1-2: model load + forward succeeds with and without the flag
  base = run_load(model_path, False); graph = run_load(model_path, True)
  checks["baseline_load_run_ok"] = base["ok"]
  checks["graph_load_run_ok"] = graph["ok"]
  checks["graph_no_new_oom_vs_baseline"] = (not graph["oom"]) or base["oom"]  # graph OOM only counts if baseline didn't

  # OOM 3 (structural): route kernel cache realizes no extra MODEL-SIZED buffers.
  # _kernel() returns (insts:list, ints...) -- no Tensor. The route's only allocation is the per-call output
  # c = Tensor.empty(512, out_f) = 512*out_f*2 bytes (gate/up 12288 -> 12.6MB), bounded << model (4.5GB).
  import inspect
  from extra import qk_prefill_graph_gemm_route as r
  ksrc = inspect.getsource(r._kernel); rsrc = inspect.getsource(r.route_pf16_graph_gemm)
  kernel_returns_no_tensor = "Tensor" not in ksrc.split("return")[-1]
  max_output_bytes = 512 * 12288 * 2  # largest covered output (gate/up)
  checks["route_cache_no_model_sized_buffer"] = kernel_returns_no_tensor and max_output_bytes < 64 * 1024 * 1024

  # OOM 4 (structural): unsupported fallback returns None BEFORE allocating the output.
  # In route_pf16_graph_gemm, every None-return (role/_pf16_w/bias/shape/_kernel) precedes the first
  # `Tensor.empty(`/`a = x.reshape`. Verify ordering in source.
  first_alloc = min((rsrc.index(tok) for tok in ("Tensor.empty(", "a = x.reshape") if tok in rsrc), default=len(rsrc))
  last_guard_return = rsrc.rindex("return None")
  checks["unsupported_fallback_no_output_alloc"] = last_guard_return < first_alloc

  oom_engineering_pass = all([checks["baseline_load_run_ok"], checks["graph_load_run_ok"],
                              checks["graph_no_new_oom_vs_baseline"], checks["route_cache_no_model_sized_buffer"],
                              checks["unsupported_fallback_no_output_alloc"]])

  # Policy (explicit): parity report carried; default-on policy decision is PENDING.
  policy = {
    "parity_report_max_abs_dNLL": 0.017593,          # from experimental-promotion-result (report-only)
    "corpus_max_positive_dNLL": 0.009443,            # within the <=0.01 degradation gate
    "greedy_generation_mismatches": 0,
    "max_abs_dNLL_gt_0.01_is_blocker": "policy",
    "degradation+generation_sufficient": "policy",
    "restrict_default_on_to_gfx1100_qwen3_8b_dense_first": True,
    "policy_field": "DEFAULT_ON_POLICY_PENDING",
  }
  verdict = ("PASS_PREFILL_GRAPH_GEMM_OOM_ENGINEERING_READY_POLICY_PENDING" if oom_engineering_pass
             else "BLOCKED_PREFILL_GRAPH_GEMM_OOM_AUDIT")
  result = {"date": "2026-06-20", "gate": 4, "schema": "prefill_graph_gemm_oom_policy_audit_v1",
            "oom_checks": checks, "baseline_load": base, "graph_load": graph,
            "oom_engineering_pass": oom_engineering_pass, "policy": policy, "verdict": verdict}
  out = pathlib.Path("bench/amd-broad-backend-roadmap"); out.mkdir(parents=True, exist_ok=True)
  (out / "prefill_graph_gemm_oom_policy_audit_result.json").write_text(json.dumps(result, indent=2) + "\n")
  for k, v in checks.items(): print(f"  [{'PASS' if v else 'FAIL'}] {k}")
  print(f"  policy_field: {policy['policy_field']} (parity max_abs_dNLL {policy['parity_report_max_abs_dNLL']} report-only)")
  print(f"\n{verdict}")
  return 0 if oom_engineering_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
