#!/usr/bin/env python3
"""Generated decode-attention PV kernel wall audit.

This is a focused follow-up to qk_decode_attention_generated_wall_audit.py.  It
answers whether the current generated split x-lane route is slow because of a
bad harness/flag, or because the generated PV kernel exposes the wrong physical
primitive shape relative to the owned tile.

The audit is intentionally conservative:
  - dynamic capture proves which route programs are in the JIT graph,
  - source inspection proves the generated PV axis/lifecycle shape,
  - existing owned ISA facts prove the oracle primitive set,
  - W==D/wall artifacts anchor the performance consequence.
"""
from __future__ import annotations

import contextlib, inspect, io, json, os, pathlib, re, subprocess, sys, time
from collections import Counter
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-generated-pv-kernel-audit"
MAXC = 4608
CTX = 512
Hq, Hkv, Hd, L = 32, 8, 128, 128

ANSI = re.compile(r"\x1b\[[0-9;]*m")
DEBUG_LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(.+?)\s+arg\s+\d+\s+mem")

SPLIT_ENV = {
  "DECODE_ATTN_GENERATED_WHOLECACHE": "1",
  "DECODE_ATTN_ONLINE_STATE_SPLIT_XLANE": "1",
}
CLEAR_FLAGS = (
  "DECODE_ATTN_GENERATED_SKELETON", "DECODE_ATTN_ONLINE_STATE_PV_TILE_XLANE",
  "DECODE_ATTN_ONLINE_STATE_PV_TILE", "DECODE_ATTN_ONLINE_PV_TILE",
  "DECODE_ATTN_TILE_PROB_PARTIAL_PV", "DECODE_ATTN_TILE_PROB", "DECODE_ATTN_TILE_SCORE_MAX",
  "DECODE_ATTN_SCORE_XLANE", "DECODE_ATTN_SCORE_VDOT2", "WARP_REDUCE_LOWERING", "V_DOT2_LOWERING",
)


def _env(child: str) -> dict[str, str]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_PV_KERNEL_AUDIT_CHILD": child}
  for k in CLEAR_FLAGS: env[k] = "0"
  if child == "split": env.update(SPLIT_ENV)
  return env


def _program_names(captured) -> list[str]:
  from tinygrad.uop.ops import Ops
  if captured is None: return []
  return [str(getattr(u.arg, "name", "")) for u in captured.linear.toposort() if u.op is Ops.PROGRAM]


def _child_capture(mode: str) -> dict[str, Any]:
  from tinygrad import Tensor, UOp, TinyJit, Context, GlobalCounters
  from extra.llm_generate import load_model_and_tokenizer
  from extra.qk_harness_contract import DEFAULT_MODEL

  m, _tok = load_model_and_tokenizer(os.environ.get("QK_MODEL", DEFAULT_MODEL), MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  for b in m.blk:
    b._use_flash, b._prefill_v2 = True, False

  v = UOp.variable("start_pos", 0, MAXC - 1)
  temp = Tensor([0.0])
  step = TinyJit(m.forward)
  tk = Tensor([[100]], dtype="int32").contiguous()
  for _ in range(8):
    out = step(tk, v.bind(CTX), temp).realize()
    out.item()

  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2):
    GlobalCounters.reset()
    step(tk, v.bind(CTX + 1), temp).realize()
    gpu_ms = GlobalCounters.time_sum_s * 1e3
  debug_lines = [ANSI.sub("", l) for l in buf.getvalue().splitlines() if "***" in l]
  debug_programs = []
  for line in debug_lines:
    mline = DEBUG_LINE.search(line)
    if mline: debug_programs.append(mline.group(1).strip())
  names = _program_names(step.captured)
  return {
    "mode": mode,
    "ctx": CTX,
    "gpu_ms_debug2": round(gpu_ms, 3),
    "captured_program_counts": Counter(names).most_common(80),
    "generated_attention_programs": [n for n in names if n.startswith("flash_")],
    "owned_attention_programs": [n for n in names if n.startswith("owned_flash_")],
    "debug_program_counts": Counter(debug_programs).most_common(40),
    "debug_line_samples": debug_lines[:12],
  }


def _run_child(mode: str) -> dict[str, Any]:
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=_env(mode),
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
  if p.returncode != 0:
    return {"mode": mode, "failed": True, "returncode": p.returncode, "output_tail": (p.stdout or "")[-8000:]}
  for line in reversed((p.stdout or "").splitlines()):
    try: return json.loads(line)
    except Exception: pass
  return {"mode": mode, "failed": True, "returncode": 0, "error": "no json", "output_tail": (p.stdout or "")[-8000:]}


def _source_features(name: str, src: str) -> dict[str, Any]:
  markers = {
    "axis_global": "AxisType.GLOBAL",
    "axis_local": "AxisType.LOCAL",
    "axis_reduce": "AxisType.REDUCE",
    "special_lane": "UOp.special",
    "warp_reduce_sum": "_warp_reduce_sum_staged",
    "warp_reduce_max": "warp_reduce_max",
    "reg_placeholder": "AddrSpace.REG",
    "local_placeholder": "AddrSpace.LOCAL",
    "exp2": ".exp2()",
    "global_d_axis": "d = UOp.range(W, 2, AxisType.GLOBAL)" if "flash_xlane_pv_from_m" in name else "d = UOp.range(Hd, 1, AxisType.GLOBAL)",
    "local_d_axis": "d = UOp.range(W, 2, AxisType.LOCAL)",
    "program_sink": ".sink(",
  }
  return {
    "name": name,
    "line_count": len(src.splitlines()),
    "marker_counts": {k: src.count(v) for k, v in markers.items()},
    "selected_source_lines": _selected_lines(src, ("def ", "UOp.range", "UOp.special", "AddrSpace.", "_warp_reduce", "warp_reduce", ".store", "sink("), 80),
  }


def _selected_lines(src: str, needles: tuple[str, ...], limit: int) -> list[str]:
  rows = []
  for i, line in enumerate(src.splitlines(), 1):
    if any(n in line for n in needles):
      rows.append(f"{i}: {line.rstrip()}")
    if len(rows) >= limit: break
  return rows


def _owned_isa_facts() -> dict[str, Any]:
  path = ROOT / "bench/qk-isa-primitive-audit/owned_decode_attention.json"
  if not path.exists(): return {"available": False, "path": str(path.relative_to(ROOT))}
  data = json.loads(path.read_text())
  return {
    "available": True,
    "path": str(path.relative_to(ROOT)),
    "verdict": data.get("verdict"),
    "resources": data.get("resources", {}),
    "instruction_flags": data.get("instruction_flags", {}),
    "instr_counts": data.get("instr_counts", {}),
    "wd_delta_pct": data.get("wd", {}).get("delta_pct", {}),
  }


def _wall_artifact_summary() -> dict[str, Any]:
  path = ROOT / "bench/qk-decode-attention-generated-wall-audit/latest.json"
  if not path.exists(): return {"available": False, "path": str(path.relative_to(ROOT))}
  data = json.loads(path.read_text())
  diag = data.get("diagnosis", {})
  rows = {}
  for arm, arm_data in data.get("arms", {}).items():
    rows[arm] = {str(r.get("ctx")): r.get("debug2_unbatched_gpu_ms") for r in arm_data.get("rows", [])}
  return {
    "available": True,
    "path": str(path.relative_to(ROOT)),
    "verdict": data.get("verdict"),
    "diagnosis_reason": diag.get("reason"),
    "debug_gpu_ms_by_ctx": rows,
    "estimated_generated_workgroups": diag.get("estimated_generated_workgroups", {}),
  }


def _wd_summary() -> dict[str, Any]:
  path = ROOT / "bench/qk-decode-eval/runs/20260626T001220-decode_attention_split_xlane.json"
  if not path.exists(): return {"available": False, "path": str(path.relative_to(ROOT))}
  data = json.loads(path.read_text()).get("wd", {})
  rows = []
  if "rows" in data:
    for r in data.get("rows", []):
      rows.append({"ctx": r.get("ctx"), "baseline_tok_s": r.get("baseline_tok_s"), "candidate_tok_s": r.get("candidate_tok_s"), "delta_pct": r.get("delta_pct")})
  else:
    base, cand, delta = data.get("baseline_per_ctx", {}), data.get("per_ctx", {}), data.get("delta_pct", {})
    for ctx in sorted(set(base) | set(cand) | set(delta), key=lambda x: int(x)):
      rows.append({"ctx": int(ctx), "baseline_tok_s": base.get(ctx), "candidate_tok_s": cand.get(ctx), "delta_pct": delta.get(ctx)})
  return {"available": True, "path": str(path.relative_to(ROOT)), "rows": rows, "promotion_gate_passed": data.get("promotion_gate_passed"), "authority": data.get("authority")}


def _static_kernel_analysis() -> dict[str, Any]:
  import extra.qk_flash_decode as qfd
  import extra.qk_owned_flash_decode_graph_node as owned

  generated_pv_src = inspect.getsource(qfd.flash_xlane_pv_from_m_kernel)
  generated_score_src = inspect.getsource(qfd.flash_score_whole_cache_kernel)
  generated_combine_src = inspect.getsource(qfd.flash_combine_kernel)
  generated_local_reference_src = inspect.getsource(qfd.flash_partial_coop_vec_kernel)
  owned_make_program_src = inspect.getsource(owned._make_program)
  owned_specialize_src = inspect.getsource(owned._specialize_tile)

  owned_hip = ""
  if owned.SRC.exists():
    hip = owned.SRC.read_text()
    # Keep the artifact useful without embedding the full kernel source.
    lines = [l for l in hip.splitlines() if any(n in l for n in (
      "__shared__", "ds_bpermute", "v_dot2", "TK", "WCVEC", "WCUNROLL", "owned_flash_tile_gqa_whole",
      "global_load", "__builtin_amdgcn", "for (int", "threadIdx", "blockIdx"))]
    owned_hip = "\n".join(lines[:160])

  return {
    "generated_pv": _source_features("flash_xlane_pv_from_m_kernel", generated_pv_src),
    "generated_score": _source_features("flash_score_whole_cache_kernel", generated_score_src),
    "generated_combine": _source_features("flash_combine_kernel", generated_combine_src),
    "generated_local_d_reference": _source_features("flash_partial_coop_vec_kernel", generated_local_reference_src),
    "owned_graph_node": {
      "make_program": _source_features("owned._make_program", owned_make_program_src),
      "specialize_tile": _source_features("owned._specialize_tile", owned_specialize_src),
      "hip_selected_lines": owned_hip.splitlines(),
    },
  }


def _classify(captures: dict[str, Any], static: dict[str, Any], owned_isa: dict[str, Any], wall: dict[str, Any], wd: dict[str, Any]) -> dict[str, Any]:
  split_programs = captures.get("split", {}).get("generated_attention_programs", [])
  baseline_owned = captures.get("baseline", {}).get("owned_attention_programs", [])
  pv_present = any(n.startswith("flash_xlane_pv_from_m") for n in split_programs)
  owned_present = any(n.startswith("owned_flash_tile_gqa_whole") for n in baseline_owned)
  pv_markers = static["generated_pv"]["marker_counts"]
  local_ref_markers = static["generated_local_d_reference"]["marker_counts"]
  isa_flags = owned_isa.get("instruction_flags", {})

  generated_has_global_d = pv_markers.get("global_d_axis", 0) > 0 and pv_markers.get("axis_global", 0) >= 3
  generated_lacks_local_d = pv_markers.get("axis_local", 0) == 0
  generated_uses_xlane_sum = pv_markers.get("warp_reduce_sum", 0) > 0 and pv_markers.get("special_lane", 0) > 0
  local_d_known_expressible = local_ref_markers.get("local_d_axis", 0) > 0
  owned_has_tile_primitives = all(bool(isa_flags.get(k)) for k in ("has_vector_dot", "has_lds", "has_cross_lane", "has_vector_global_load"))

  if pv_present and owned_present and generated_has_global_d and generated_lacks_local_d and generated_uses_xlane_sum and owned_has_tile_primitives:
    verdict = "PV_WALL_CONFIRMED__GENERATED_GLOBAL_COLUMN_SCALAR_CODEGEN"
    confidence = "high"
  elif pv_present and generated_has_global_d:
    verdict = "PV_WALL_PROBABLE__GENERATED_GLOBAL_COLUMN_CODEGEN"
    confidence = "medium"
  else:
    verdict = "PV_WALL_UNKNOWN__NEEDS_BINARY_ISA_CAPTURE"
    confidence = "low"

  s512 = (CTX + L - 1) // L
  generated_work = {
    "ctx": CTX,
    "S": s512,
    "physical_workgroups_current_pv": Hkv * s512 * (Hd + 1),
    "logical_outputs_current_pv": Hq * s512 * (Hd + 1),
    "owned_tile_workgroups": Hkv * 48,
    "note": "current generated PV maps (kvh, split, d) to grid and loops G=4 query heads; owned tile maps (kvh, fixed S) to a fused tile and combines separately.",
  }

  blockers = []
  if generated_has_global_d: blockers.append("PV output column d is GLOBAL, not LOCAL/coalesced tile ownership")
  if generated_uses_xlane_sum: blockers.append("token lanes are reduced, but PV still materializes per-column partials instead of a fused LDS/register tile")
  if owned_has_tile_primitives: blockers.append("owned oracle uses LDS + vector global loads + cross-lane + v_dot2 confirmed by ISA artifact")
  if wall.get("available"): blockers.append("whole-decode wall artifact shows GPU time collapse with normal route firing")
  if wd.get("available"): blockers.append("W==D artifact shows catastrophic candidate tok/s collapse, so this is not async timing noise")

  return {
    "verdict": verdict,
    "confidence": confidence,
    "pv_program_present": pv_present,
    "owned_tile_program_present": owned_present,
    "generated_pv_shape_flags": {
      "global_d_axis": generated_has_global_d,
      "local_d_axis_absent": generated_lacks_local_d,
      "xlane_sum_present": generated_uses_xlane_sum,
      "local_d_reference_is_expressible_elsewhere": local_d_known_expressible,
    },
    "owned_oracle_primitive_flags": isa_flags,
    "work_shape": generated_work,
    "blockers": blockers,
    "interpretation": (
      "The current generated split x-lane route is correctness-clean but physically wrong for speed. "
      "It exposes PV as a global output-column reduction with cross-lane summation, while the owned route's winning "
      "primitive set is a fused tile with LDS, vectorized loads, cross-lane exchange, and v_dot2. Search can only pick "
      "among represented programs; this generated PV representation does not contain the owned tile economics."
    ),
    "next_required_work": [
      "Add a generated fused PV tile lowering where d is locally/cooperatively owned and V reuse is tile-local, not one global program per output column.",
      "Expose LDS/register tile layout plus vector-load/v_dot2/cross-lane primitives to BubbleBeam/FutureSight as searchable decisions.",
      "Only after that, rerun W==D; pure search cannot recover this route by toggling existing flags.",
    ],
  }


def build() -> dict[str, Any]:
  captures = {"baseline": _run_child("baseline"), "split": _run_child("split")}
  if any(v.get("failed") for v in captures.values()):
    verdict = "PV_KERNEL_AUDIT_FAIL__CAPTURE"
    diagnosis = {"verdict": verdict, "captures": captures}
    static = {}
  else:
    static = _static_kernel_analysis()
    diagnosis = _classify(captures, static, _owned_isa_facts(), _wall_artifact_summary(), _wd_summary())
    verdict = diagnosis["verdict"]
  return {
    "date": "2026-06-26",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "captures": captures,
    "static_kernel_analysis": static,
    "owned_isa_facts": _owned_isa_facts(),
    "wall_artifact_summary": _wall_artifact_summary(),
    "wd_summary": _wd_summary(),
    "diagnosis": diagnosis,
  }


def main() -> int:
  os.chdir(ROOT)
  child = os.environ.get("QK_PV_KERNEL_AUDIT_CHILD")
  if child:
    print(json.dumps(_child_capture(child)))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-generated-pv-kernel-audit-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if not out["verdict"].endswith("__CAPTURE") else 1


if __name__ == "__main__":
  raise SystemExit(main())
