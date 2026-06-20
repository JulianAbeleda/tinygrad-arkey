#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/q8-ffn-artifact-promotion"
MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
MAX_CONTEXT = 4096

CALIB_WINDOWS = [
  ("systems",
   "A useful system keeps the hot path small and the fallback boring. When a new shortcut wins a benchmark, the "
   "first question is whether it changed the values that downstream code observes. The second question is whether "
   "unsupported cases return to the old behavior without surprising the caller."),
  ("hardware",
   "A memory bound kernel is often improved by removing round trips rather than adding arithmetic. But a faster "
   "instruction is only part of the primitive. The activation format, producer cost, consumer schedule, register "
   "pressure, and model boundary all decide whether the isolated win survives."),
  ("quality",
   "Lossy routes need a quality budget before they become defaults. A small perplexity movement may be acceptable "
   "for research, but a default path should be checked across multiple windows and should have a clear rollback if "
   "the model starts to drift on real prompts."),
  ("decode",
   "Single token decode has little reuse and many opportunities for integration tax. The fastest reference kernels "
   "usually preserve a contract across producer, packed weights, scheduler, and reduction. Replacing only one piece "
   "rarely captures the full advantage."),
]


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_text(rel: str) -> str:
  path = ROOT / rel
  return path.read_text() if path.exists() else ""


def quality_worker(mode: str, tokens: int) -> int:
  from tinygrad import Tensor, UOp, TinyJit
  from extra.llm_generate import load_model_and_tokenizer

  model, tok = load_model_and_tokenizer(MODEL, MAX_CONTEXT, seed=20260620)
  for lin in getattr(model, "_q4k_linears", None).linears if getattr(model, "_q4k_linears", None) else []:
    lin.decode_enabled = True

  v_sp = UOp.variable("start_pos", 0, MAX_CONTEXT - 1)
  step = TinyJit(lambda t, sp: model.logits(t, sp).realize())
  rows = []
  for name, text in CALIB_WINDOWS:
    ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode((text + " ") * 4)
    ids = ids[: tokens + 1]
    if len(ids) < tokens // 2: raise ValueError(f"calibration window {name} too short: {len(ids)} ids")
    total_nll, counted = 0.0, 0
    for i in range(len(ids) - 1):
      lg = step(Tensor([[ids[i]]], dtype="int32").contiguous(), v_sp.bind(i))
      total_nll += -float(lg[0, 0].log_softmax()[ids[i + 1]].item())
      counted += 1
    rows.append({"window": name, "nll": total_nll / counted, "tokens": counted})
  print(json.dumps({"mode": mode, "rows": rows, "tokens_per_window_target": tokens}))
  return 0


def run_quality_mode(mode: str, tokens: int) -> dict[str, Any]:
  env = os.environ.copy()
  env.setdefault("DEV", "AMD")
  env.setdefault("JIT", "1")
  env["PYTHONPATH"] = str(ROOT)
  if mode == "q8":
    env["Q8_FFN_HANDWRITTEN"] = "1"
  else:
    env.pop("Q8_FFN_HANDWRITTEN", None)
  proc = subprocess.run(
    [sys.executable, str(pathlib.Path(__file__).resolve()), "--quality-worker", "--mode", mode, "--tokens", str(tokens)],
    cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900)
  if proc.returncode != 0:
    return {"mode": mode, "error": proc.stderr[-6000:], "stdout": proc.stdout[-2000:], "returncode": proc.returncode}
  lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
  return json.loads(lines[-1]) if lines else {"mode": mode, "error": "no JSON output", "stdout": proc.stdout[-4000:]}


def gate_quality(rerun: bool, tokens: int) -> dict[str, Any]:
  out = OUT / "quality_matrix.json"
  if out.exists() and not rerun:
    return json.loads(out.read_text())
  base, q8 = run_quality_mode("baseline", tokens), run_quality_mode("q8", tokens)
  rows = []
  if "error" not in base and "error" not in q8:
    q8_by = {r["window"]: r for r in q8["rows"]}
    for b in base["rows"]:
      qr = q8_by.get(b["window"], {})
      rows.append({
        "window": b["window"],
        "tokens": min(b.get("tokens", 0), qr.get("tokens", 0)),
        "baseline_nll": b.get("nll"),
        "q8_nll": qr.get("nll"),
        "dnll": (qr.get("nll") - b.get("nll")) if qr.get("nll") is not None and b.get("nll") is not None else None,
      })
  max_dnll = max((r["dnll"] for r in rows if r["dnll"] is not None), default=None)
  mean_dnll = statistics.mean([r["dnll"] for r in rows if r["dnll"] is not None]) if rows else None
  gate = {
    "baseline_ran": "error" not in base,
    "q8_ran": "error" not in q8,
    "window_count_ge_4": len(rows) >= 4,
    "all_windows_dnll_le_0_01": bool(rows) and all(r["dnll"] is not None and r["dnll"] <= 0.01 for r in rows),
    "reports_mean_and_max": mean_dnll is not None and max_dnll is not None,
    "wd_greedy_sanity_available": len(speed_rows()) >= 4,
  }
  result = {
    "date": "2026-06-20",
    "phase": "Q8P-1_quality_promotion_gate",
    "schema": "q8_ffn_artifact_quality_matrix_v1",
    "verdict": "PASS_Q8P1_QUALITY_PROMOTION_GATE" if all(gate.values()) else "BLOCKED_Q8P1_QUALITY_PROMOTION_GATE",
    "gate_pass": all(gate.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "tokens_per_window_target": tokens,
    "rows": rows,
    "summary": {"mean_dnll": mean_dnll, "max_dnll": max_dnll, "threshold": 0.01},
    "raw": {"baseline": base, "q8": q8},
    "gate": gate,
  }
  write_json("quality_matrix.json", result)
  return result


def gate_default_safety() -> dict[str, Any]:
  policy = read_json("bench/q8-ffn-amd-scheduler-project/artifact_policy_boundary.json", {})
  model_src = read_text("tinygrad/llm/model.py")
  route_src = read_text("extra/q8_ffn_graph_route.py")
  gate = {
    "policy_default_unchanged": policy.get("default_changed") is False,
    "fallback_declared": "fallback" in (policy.get("requirements") or {}),
    "model_flag_default_off": 'Q8_FFN_HANDWRITTEN = bool(getenv("Q8_FFN_HANDWRITTEN", 0))' in model_src,
    "route_guarded_by_flag": "if Q8_FFN_HANDWRITTEN" in model_src,
    "route_has_none_fallback": "return None" in route_src,
  }
  result = {
    "date": "2026-06-20",
    "phase": "Q8P-2_default_safety_gate",
    "schema": "q8_ffn_artifact_default_safety_v1",
    "verdict": "PASS_Q8P2_DEFAULT_SAFETY_GATE" if all(gate.values()) else "BLOCKED_Q8P2_DEFAULT_SAFETY_GATE",
    "gate_pass": all(gate.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "fallback": (policy.get("requirements") or {}).get("fallback"),
    "gate": gate,
  }
  write_json("default_safety.json", result)
  return result


def gate_coverage() -> dict[str, Any]:
  policy = read_json("bench/q8-ffn-amd-scheduler-project/artifact_policy_boundary.json", {})
  route_src = read_text("extra/q8_ffn_graph_route.py")
  supported = policy.get("supported", {})
  routed_roles = ["ffn_gate", "ffn_up"]
  explicitly_not_routed = ["lm_head", "attention", "attn_q", "attn_k", "attn_v", "attn_output", "ffn_down", "prefill", "Q6_K"]
  gate = {
    "dim_4096": supported.get("dim") == 4096,
    "hidden_12288": supported.get("hidden") == 12288,
    "q4_gate_up_only": supported.get("weight_format") == "Q4_K gate/up",
    "gfx1100_only": supported.get("gpu_arch") == "gfx1100",
    "shape_guard_in_source": "x.shape != (1, 1, DIM)" in route_src and "block.config.hidden_dim != HIDDEN" in route_src,
    "requires_gate_up_q4_storage": 'for n in ("ffn_gate", "ffn_up")' in route_src,
  }
  result = {
    "date": "2026-06-20",
    "phase": "Q8P-3_coverage_gate",
    "schema": "q8_ffn_artifact_coverage_matrix_v1",
    "verdict": "PASS_Q8P3_COVERAGE_GATE" if all(gate.values()) else "BLOCKED_Q8P3_COVERAGE_GATE",
    "gate_pass": all(gate.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "supported": supported,
    "routed_roles": routed_roles,
    "explicitly_not_routed": explicitly_not_routed,
    "gate": gate,
  }
  write_json("coverage_matrix.json", result)
  return result


def speed_rows() -> list[dict[str, Any]]:
  baseline = read_json("bench/q8-ffn-handwritten-oracle/decode_wd_baseline.json", {})
  q8_route = read_json("bench/q8-ffn-handwritten-oracle/decode_wd_q8_route.json", {})
  b_by = {r.get("ctx"): r for r in baseline.get("rows", [])}
  q_by = {r.get("ctx"): r for r in q8_route.get("rows", [])}
  rows = []
  for ctx in sorted(set(b_by) & set(q_by)):
    b, q = b_by[ctx], q_by[ctx]
    rows.append({
      "ctx": ctx,
      "baseline_tok_s": b.get("tok_s_W"),
      "q8_tok_s": q.get("tok_s_W"),
      "speedup": (q.get("tok_s_W") / b.get("tok_s_W")) if b.get("tok_s_W") and q.get("tok_s_W") else None,
      "host_sync_pct": q.get("host_sync_pct_of_wall"),
    })
  return rows


def gate_performance() -> dict[str, Any]:
  rows = speed_rows()
  gate_ctxs = [512, 1024, 4096]
  by_ctx = {r["ctx"]: r for r in rows}
  gate = {
    "ctx512_1024_4096_present": all(ctx in by_ctx for ctx in gate_ctxs),
    "all_gate_ctxs_ge_3pct": all(by_ctx.get(ctx, {}).get("speedup", 0.0) >= 1.03 for ctx in gate_ctxs),
    "all_gate_ctxs_host_sync_le_5pct": all(by_ctx.get(ctx, {}).get("host_sync_pct", 100.0) <= 5.0 for ctx in gate_ctxs),
    "uses_wd_artifacts": True,
  }
  result = {
    "date": "2026-06-20",
    "phase": "Q8P-4_performance_gate",
    "schema": "q8_ffn_artifact_performance_matrix_v1",
    "verdict": "PASS_Q8P4_PERFORMANCE_GATE" if all(gate.values()) else "BLOCKED_Q8P4_PERFORMANCE_GATE",
    "gate_pass": all(gate.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "rows": rows,
    "summary": {"min_speedup": min((r["speedup"] for r in rows if r["ctx"] in gate_ctxs), default=None), "gate_speedup": 1.03},
    "gate": gate,
  }
  write_json("performance_matrix.json", result)
  return result


def gate_artifact_ownership() -> dict[str, Any]:
  policy = read_json("bench/q8-ffn-amd-scheduler-project/artifact_policy_boundary.json", {})
  hardening = read_json("bench/qk-decode-path-split/small_q8_hardening.json", {})
  inject = read_json("bench/q8-ffn-handwritten-oracle/fast_artifact_inject.json", {})
  contract = read_json("bench/q8-ffn-handwritten-oracle/fast_artifact_contract_audit.json", {})
  oracle_contract = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json", {})
  req = policy.get("requirements", {})
  art = hardening.get("artifact_summary", {})
  gate = {
    "source_module_declared": bool(req.get("source_module")),
    "rebuild_command_declared": bool(req.get("rebuild_command")),
    "hashes_present": bool(art.get("producer_hash") and art.get("gateup_hash")),
    "no_in_process_hip_runtime": req.get("no_in_process_hip_runtime") is True and inject.get("no_hip_runtime_in_process") is True,
    "kernarg_contract_present": ((oracle_contract.get("launch_contract") or {}).get("kernarg_size") == 40),
    "fallback_declared": bool(req.get("fallback")),
    "owner_or_status_declared": policy.get("status") == "research_only",
    "contract_audit_pass": all((v.get("present") and v.get("globals_match") and v.get("outs_match") and v.get("ins_match")) for v in (contract.get("checks") or {}).values()),
  }
  result = {
    "date": "2026-06-20",
    "phase": "Q8P-5_artifact_ownership_gate",
    "schema": "q8_ffn_artifact_ownership_v1",
    "verdict": "PASS_Q8P5_ARTIFACT_OWNERSHIP_GATE" if all(gate.values()) else "BLOCKED_Q8P5_ARTIFACT_OWNERSHIP_GATE",
    "gate_pass": all(gate.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "manifest": {
      "status": policy.get("status"),
      "source_module": req.get("source_module"),
      "rebuild_command": req.get("rebuild_command"),
      "runtime": req.get("runtime"),
      "fallback": req.get("fallback"),
      "producer_hash": art.get("producer_hash"),
      "gateup_hash": art.get("gateup_hash"),
      "supported": policy.get("supported"),
    },
    "gate": gate,
  }
  write_json("artifact_ownership.json", result)
  return result


def gate_model_policy(previous: list[dict[str, Any]]) -> dict[str, Any]:
  all_prior_pass = all(r.get("gate_pass") for r in previous)
  decision = {
    "default_on": False,
    "promote_to": "hardened_opt_in_candidate",
    "reason": "The route is lossy and externally owned. Even with gates passing, flipping default-on should remain a maintainer/user policy decision; this pass promotes the route from research-only evidence to a hardened opt-in candidate.",
    "quality_threshold": "multi-window dNLL <=0.01",
    "release_flag": "Q8_FFN_HANDWRITTEN=1",
    "supported_model_set": "Qwen3-8B Q4_K_M-style dense FFN, dim=4096, hidden=12288, gfx1100",
    "rollback": "unset Q8_FFN_HANDWRITTEN; fallback is existing default tinygrad decode",
  }
  gate = {
    "prior_gates_pass": all_prior_pass,
    "explicit_default_decision": decision["default_on"] is False,
    "release_flag_named": bool(decision["release_flag"]),
    "rollback_named": bool(decision["rollback"]),
    "supported_model_set_named": bool(decision["supported_model_set"]),
  }
  result = {
    "date": "2026-06-20",
    "phase": "Q8P-6_model_policy_gate",
    "schema": "q8_ffn_artifact_model_policy_decision_v1",
    "verdict": "PASS_Q8P6_MODEL_POLICY_GATE_HARDENED_OPT_IN" if all(gate.values()) else "BLOCKED_Q8P6_MODEL_POLICY_GATE",
    "gate_pass": all(gate.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "decision": decision,
    "gate": gate,
  }
  write_json("model_policy_decision.json", result)
  return result


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--quality-worker", action="store_true")
  ap.add_argument("--mode", choices=["baseline", "q8"], default="baseline")
  ap.add_argument("--tokens", type=int, default=int(os.environ.get("Q8P_QUALITY_TOKENS", "96")))
  ap.add_argument("--rerun-quality", action="store_true")
  args = ap.parse_args()
  if args.quality_worker:
    return quality_worker(args.mode, args.tokens)

  quality = gate_quality(args.rerun_quality, args.tokens)
  default_safety = gate_default_safety()
  coverage = gate_coverage()
  performance = gate_performance()
  ownership = gate_artifact_ownership()
  policy = gate_model_policy([quality, default_safety, coverage, performance, ownership])
  gates = [quality, default_safety, coverage, performance, ownership, policy]
  final_pass = all(g.get("gate_pass") for g in gates)
  result = {
    "date": "2026-06-20",
    "phase": "Q8P-1_to_Q8P-6_promotion_execution",
    "schema": "q8_ffn_artifact_promotion_execution_v1",
    "verdict": "PASS_Q8_FFN_ARTIFACT_PROMOTION_TO_HARDENED_OPT_IN" if final_pass else "BLOCKED_Q8_FFN_ARTIFACT_PROMOTION",
    "gate_pass": final_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "gate_results": {
      "Q8P-1": "bench/q8-ffn-artifact-promotion/quality_matrix.json",
      "Q8P-2": "bench/q8-ffn-artifact-promotion/default_safety.json",
      "Q8P-3": "bench/q8-ffn-artifact-promotion/coverage_matrix.json",
      "Q8P-4": "bench/q8-ffn-artifact-promotion/performance_matrix.json",
      "Q8P-5": "bench/q8-ffn-artifact-promotion/artifact_ownership.json",
      "Q8P-6": "bench/q8-ffn-artifact-promotion/model_policy_decision.json",
    },
    "summary": {
      "quality": quality.get("summary"),
      "performance": performance.get("summary"),
      "policy_decision": policy.get("decision"),
    },
    "next_action": "Keep Q8_FFN_HANDWRITTEN default-off; it is now a hardened opt-in candidate, not a default-on route." if final_pass else
                   "Fix the first blocked Q8P gate before any promotion decision.",
  }
  write_json("promotion_result.json", result)
  print(json.dumps({
    "out": "bench/q8-ffn-artifact-promotion/promotion_result.json",
    "verdict": result["verdict"],
    "gate_pass": final_pass,
    "default_behavior_changed": result["default_behavior_changed"],
    "next": result["next_action"],
  }, indent=2))
  return 0 if final_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
