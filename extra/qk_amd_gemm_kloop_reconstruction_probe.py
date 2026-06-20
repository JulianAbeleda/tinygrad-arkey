#!/usr/bin/env python3
# AMD GEMM K-loop reconstruction probe (no GPU, no timing, no performance claim, no routing change).
#
# Moves the selected rocBLAS ffn_gate/up Tensile contract from "schedule object exists" to "the exact
# K-loop / buffer-swap schedule is reconstructed enough to lower." It parses the selected function out of the
# captured disassembly, recovers the control-flow regions from labels + a resolved backward branch (instead of
# the prior first-WMMA/last-WMMA heuristic), and extracts a SYMBOLIC repeated K-loop template: per
# sub-iteration global-load -> wait -> LDS-store(other slot) -> barrier -> LDS-read(this slot) -> wait ->
# WMMA -> swap. It explains why v_wmma=80 != 256 K-slices, ties each phase to opcode evidence, and emits a
# lowering-readiness gate.
#
# Answers exactly one question: can we turn the structural GEMM object into a lowerable repeated K-loop
# template? PASS_KLOOP_TEMPLATE_RECONSTRUCTED_FOR_LOWERING if yes; else
# BLOCKED_KLOOP_TEMPLATE_NEEDS_DISASM_OR_CFG with the exact missing artifact/tool.
from __future__ import annotations

import json, pathlib, re
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
CONTRACT = "bench/qk-tensile-extraction/ffn_gate_up_contract.json"
TEMPLATE = "bench/qk-tensile-extraction/ffn_gate_up_schedule_template.json"
AUDIT = "bench/amd-broad-backend-roadmap/bb5a10_tensile_layout_audit_result.json"
DISASM = pathlib.Path("/tmp/td_all.txt")

LABEL_RE = re.compile(r"^([0-9a-fA-F]{16}) <([^>]+)>:")
ADDR_RE = re.compile(r"//\s*([0-9A-Fa-f]+):")
OFFSET_RE = re.compile(r"offset:(\d+)")
LDS_SLOT_THRESHOLD = 8192   # < gap-base => low buffer (slot 0); >= 2nd-buffer base => slot 1


def read_json(rel: str) -> dict[str, Any]:
  path = ROOT / rel
  if not path.exists(): raise FileNotFoundError(f"required artifact missing: {rel}")
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def mnemonic(text: str) -> str:
  m = re.match(r"\s*([a-zA-Z0-9_]+)", text)
  return m.group(1) if m else ""


def load_function(symbol: str, line_start: int, line_end: int) -> list[dict[str, Any]]:
  """Read the selected function body; each row is {local, mnem, addr, text}."""
  rows: list[dict[str, Any]] = []
  with DISASM.open("r", errors="replace") as f:
    for no, line in enumerate(f, 1):
      if no < line_start: continue
      if no > line_end: break
      text = line.rstrip("\n")
      am = ADDR_RE.search(text)
      lm = LABEL_RE.match(text)
      rows.append({
        "local": no - line_start, "file_line": no, "mnem": "" if lm else mnemonic(text),
        "label": lm.group(2) if lm else None,
        "addr": int(am.group(1), 16) if am else (int(lm.group(1), 16) if lm else None),
        "text": text,
      })
  return rows


def resolve_backward_branch(rows: list[dict[str, Any]], target_symbol: str) -> dict[str, Any] | None:
  """Find `s_cbranch* <target_symbol>` and resolve its target file-line via the encoded simm16 offset."""
  addr_to_idx = {r["addr"]: i for i, r in enumerate(rows) if r["addr"] is not None}
  for i, r in enumerate(rows):
    if r["mnem"].startswith("s_cbranch") and target_symbol in r["text"]:
      # encoding is the last hex word in the trailing comment, e.g. "BFA1FE28"
      words = re.findall(r"\b([0-9A-Fa-f]{8})\b", r["text"].split("//", 1)[-1])
      if not words or r["addr"] is None: continue
      simm16 = int(words[-1][-4:], 16)
      if simm16 >= 0x8000: simm16 -= 0x10000
      target_addr = r["addr"] + 4 + simm16 * 4
      tgt = addr_to_idx.get(target_addr)
      if tgt is not None:
        return {"branch_idx": i, "branch_line": r["file_line"], "target_idx": tgt,
                "target_line": rows[tgt]["file_line"], "target_addr": hex(target_addr),
                "target_label": rows[tgt]["label"]}
  return None


def slot_of(offset: int) -> int: return 0 if offset < LDS_SLOT_THRESHOLD else 1


def phase_tags(seg: list[dict[str, Any]]) -> dict[str, Any]:
  """Ordered, run-compressed phases + slot usage + opcode evidence for one loop sub-iteration."""
  global_loads = 0
  ds_load_off: list[int] = []
  ds_store_off: list[int] = []
  wmma = 0
  order: list[str] = []
  def push(tag: str):
    if not order or order[-1] != tag: order.append(tag)
  for r in seg:
    m = r["mnem"]
    if m.startswith(("buffer_load_b64", "buffer_load_d16", "global_load")):
      global_loads += 1; push("global_load")
    elif m == "ds_load_b128":
      mo = OFFSET_RE.search(r["text"]); ds_load_off.append(int(mo.group(1)) if mo else 0); push("lds_read")
    elif m.startswith("ds_store"):
      mo = OFFSET_RE.search(r["text"]); ds_store_off.append(int(mo.group(1)) if mo else 0); push("lds_store")
    elif m.startswith("v_wmma"):
      wmma += 1; push("wmma")
    elif m == "s_barrier": push("barrier")
    elif m.startswith("s_waitcnt"):
      push("wait_vmcnt" if "vmcnt" in r["text"] else "wait_lgkmcnt" if "lgkmcnt" in r["text"] else "wait")
    elif m.startswith("s_sub") and " s5" in r["text"]: push("kcounter_dec")
    elif m.startswith(("s_cbranch", "s_branch")): push("branch")
    elif "v_xor_b32" in r["text"] and "0x4000" in r["text"]: push("slot_swap_xor")
  read_slots = sorted({slot_of(o) for o in ds_load_off})
  write_slots = sorted({slot_of(o) for o in ds_store_off})
  return {
    "global_loads": global_loads, "ds_load_count": len(ds_load_off), "ds_store_count": len(ds_store_off),
    "wmma": wmma, "read_slots": read_slots, "write_slots": write_slots,
    "read_slot": read_slots[0] if len(read_slots) == 1 else None,
    "write_slot": write_slots[0] if len(write_slots) == 1 else None,
    "ds_load_offsets": sorted(set(ds_load_off)), "ds_store_offsets": sorted(set(ds_store_off)),
    "ordered_phases": order,
  }


def wmma_clusters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  lines = [r["file_line"] for r in rows if r["mnem"].startswith("v_wmma")]
  clusters: list[dict[str, Any]] = []
  if not lines: return clusters
  start = prev = lines[0]; n = 1
  for ln in lines[1:]:
    if ln - prev > 3:
      clusters.append({"start": start, "end": prev, "n": n}); start = ln; n = 1
    else: n += 1
    prev = ln
  clusters.append({"start": start, "end": prev, "n": n})
  return clusters


def enclosing_label(rows: list[dict[str, Any]], file_line: int) -> str | None:
  cur = None
  for r in rows:
    if r["label"] is not None and r["file_line"] <= file_line: cur = r["label"]
    if r["file_line"] > file_line: break
  return cur


def blocked(missing: str, detail: str, extra: dict[str, Any] | None = None) -> int:
  result = {
    "date": "2026-06-20", "phase": "AMD_GEMM_KLOOP_RECONSTRUCTION",
    "schema": "amd_gemm_kloop_reconstruction_v1",
    "verdict": "BLOCKED_KLOOP_TEMPLATE_NEEDS_DISASM_OR_CFG", "gate_pass": False,
    "default_behavior_changed": False, "performance_claim": False,
    "missing_artifact_or_tool": missing, "detail": detail, **(extra or {}),
  }
  write_json("amd_gemm_kloop_reconstruction_result.json", result)
  print(json.dumps({"verdict": result["verdict"], "missing": missing, "detail": detail}, indent=2))
  return 1


def main() -> int:
  contract = read_json(CONTRACT)
  template = read_json(TEMPLATE)
  audit = read_json(AUDIT)
  symbol = contract["kernel_symbol"]
  sm = template["solution_params"]["sizeMapping"]
  depth_u, k = sm["depthU"], contract["shape"]["K"]
  ic = audit["disasm_summary"]["instruction_counts"]
  total_wmma = ic.get("v_wmma", 0)

  if not DISASM.exists():
    return blocked("selected disassembly /tmp/td_all.txt",
                   "live disasm absent; cannot resolve CFG/branch targets. Re-capture objdump of the selected "
                   ".co (TensileLibrary_..._gfx1100.co) to /tmp/td_all.txt, then re-run.",
                   {"have_static_segmentation": True, "static_segmentation": template["schedule_segments"]})

  sf = audit["selected_function"]
  rows = load_function(symbol, sf["line_start"], sf["line_end"])
  if not any(r["mnem"].startswith("v_wmma") for r in rows):
    return blocked("selected function body", f"no v_wmma in lines {sf['line_start']}..{sf['line_end']}")

  # ---- CFG: resolve the main summation loop via the backward branch to LoopBeginL_1 ----
  back = resolve_backward_branch(rows, "LoopBeginL_1")
  if back is None:
    return blocked("CFG branch-target resolver",
                   "could not resolve the LoopBeginL_1 backward branch; need a CFG/branch-target extraction "
                   "tool to label the loop head.")
  head_local = rows[back["target_idx"]]["local"]
  branch_local = next(r["local"] for r in rows if r["file_line"] == back["branch_line"])
  body = [r for r in rows if head_local <= r["local"] <= branch_local]

  # ---- split the unrolled body into sub-iterations at each s5 decrement ----
  sub_bounds: list[int] = [i for i, r in enumerate(body) if r["mnem"].startswith("s_sub") and " s5" in r["text"]]
  segments, prev = [], 0
  for b in sub_bounds:
    segments.append(body[prev:b+1]); prev = b+1
  if prev < len(body): segments.append(body[prev:])
  sub_iters = [phase_tags(seg) for seg in segments if any(r["mnem"].startswith("v_wmma") for r in seg)]
  unroll = len(sub_iters)

  # ---- whole-function WMMA cluster map (explains 80 vs 256) ----
  clusters = wmma_clusters(rows)
  for c in clusters: c["region"] = enclosing_label(rows, c["start"])
  per_wave_wmma = clusters[0]["n"] if clusters else None   # 16 fragments = one wave's output for one DepthU slice
  symbolic_k_slices = k // depth_u

  # ---- structural assessment of lowerability ----
  alternation_ok = (unroll >= 2
                    and all(s["read_slot"] is not None and s["write_slot"] is not None for s in sub_iters)
                    and [s["read_slot"] for s in sub_iters] == [0, 1][:unroll]
                    and all(s["read_slot"] != s["write_slot"] for s in sub_iters))
  required_phase_ops = {"global_load", "lds_store", "lds_read", "wmma", "barrier"}
  edges_ok = all(required_phase_ops.issubset(set(s["ordered_phases"])) for s in sub_iters)
  has_kcounter = len(sub_bounds) >= 1
  clusters_sum_ok = sum(c["n"] for c in clusters) == total_wmma

  lowerable = bool(alternation_ok and edges_ok and has_kcounter and clusters_sum_ok and unroll >= 2)

  # ---- ordered dependency edges (canonical, one sub-iteration) ----
  dependency_edges = [
    {"from": "lds_read[prev]", "to": "barrier", "via": "lgkmcnt", "why": "all waves finish reading a buffer before it is overwritten"},
    {"from": "global_load[next_k]", "to": "lds_store[other_slot]", "via": "vmcnt", "why": "store only after the global load lands"},
    {"from": "lds_store[other_slot]", "to": "lds_read[other_slot]@next_iter", "via": "barrier", "why": "next iteration reads only after the store completes"},
    {"from": "lds_read[this_slot]", "to": "wmma", "via": "lgkmcnt", "why": "WMMA operands must be LDS-loaded VGPRs"},
    {"from": "wmma", "to": "kcounter_dec/branch", "via": "wmma_dependency", "why": "loop control after compute issue"},
  ]

  regions = {
    "kernel_prologue": {"role": "args/alpha/address setup", "to_label": "ShadowInitStart_10"},
    "shadow_init_prefetch": {"role": "PGR fill: prefetch first K tiles into LDS slot 0", "label": "ShadowInitStart_10"},
    "main_k_loop_body": {"role": f"steady state, unrolled x{unroll}, hardware loop on s5",
                          "head_line": back["target_line"], "head_addr": back["target_addr"],
                          "branch_line": back["branch_line"], "wmma_clusters": 2},
    "steady_drain": {"role": "final-iteration drain after even/odd exit", "label": "LoopEndL_2/Summation_End_OptNLL_16", "wmma_clusters": 1},
    "optnll": {"role": "optimized no-load-loop last iteration (no global prefetch)", "label": "OptNLL_End_15", "wmma_clusters": 1},
    "tail_loop": {"role": "K-remainder loop, s5 -= 16 per MI-K step", "label": "TailLoopBeginL_6..TailLoopEndL_7", "wmma_clusters": 1},
    "global_write_epilogue": {"role": "beta*C load + store output", "label": "GW_*/label_GW_End_21"},
  }

  verdict = "PASS_KLOOP_TEMPLATE_RECONSTRUCTED_FOR_LOWERING" if lowerable else "BLOCKED_KLOOP_TEMPLATE_NEEDS_DISASM_OR_CFG"
  result = {
    "date": "2026-06-20", "phase": "AMD_GEMM_KLOOP_RECONSTRUCTION",
    "schema": "amd_gemm_kloop_reconstruction_v1", "role": "ffn_gate/up",
    "verdict": verdict, "gate_pass": lowerable,
    "default_behavior_changed": False, "performance_claim": False,
    "shape": {"M": contract["shape"]["M"], "N": contract["shape"]["N"], "K": k},
    "loop_counter": {"register": "s5", "init_formula": "s5 = SizesSum(K) >> log2(DepthU) = K // DepthU",
                     "init_value": symbolic_k_slices, "unrolled_by": unroll,
                     "decrement": "s5 -= 1 per sub-iteration; exit when s5 == 1; tail uses s5 -= 16"},
    "regions": regions,
    "symbolic_kloop_template": {
      "unroll": unroll,
      "sub_iterations": sub_iters,
      "slot_alternation": [{"sub": chr(65+i), "read_slot": s["read_slot"], "write_slot": s["write_slot"]}
                           for i, s in enumerate(sub_iters)],
      "dependency_edges": dependency_edges,
      "lds_slot_threshold_bytes": LDS_SLOT_THRESHOLD,
    },
    "wmma_accounting": {
      "total_v_wmma_emitted": total_wmma,
      "clusters": clusters,
      "per_wave_wmma_per_kslice": per_wave_wmma,
      "symbolic_k_slices": symbolic_k_slices,
      "explanation": (
        f"v_wmma={total_wmma} is STATIC code size across {len(clusters)} distinct scheduled regions "
        f"(main-loop x2, steady-drain, OptNLL, tail), each emitting {per_wave_wmma} WMMA = one wave's full "
        f"{per_wave_wmma}-fragment 128x128 output update for ONE DepthU={depth_u} K-slice. The symbolic K-loop "
        f"DYNAMICALLY executes {symbolic_k_slices} slices (K//DepthU) via the s5 hardware loop (unrolled x{unroll}, "
        f"~{symbolic_k_slices//max(unroll,1)} iterations of the main body) + pipeline fill/drain. So "
        f"{total_wmma} != {symbolic_k_slices}: it is an emitted-region footprint, not the trip count."),
    },
    "opcode_evidence": {
      "global_load": "buffer_load_b64 (GLVW4 fp16 pairs)", "lds_store": "ds_store_b64 (write next-K into opposite slot)",
      "wait_global_before_store": "s_waitcnt vmcnt", "barrier": "s_barrier (buffer-reuse protection)",
      "lds_read": "ds_load_b128 (LRVW16 operand fragments)", "wait_lds_before_wmma": "s_waitcnt lgkmcnt",
      "wmma": "v_wmma_f32_16x16x16_f16", "slot_swap": "ds offset alternation (slot0 <8192B vs slot1 >=16384B); v_xor 0x4000 in drain",
    },
    "remaining_unknown": [
      "exact per-element address VGPR evolution: A/B base offsets carried in address VGPRs, not reconstructed; "
      "lowering can use the structural slot model (offsets) instead, but a bit-identical clone cannot.",
      "exact branch/loop-counter micro-structure beyond s5 (even/odd PLR exits, OptNLL guard predicates) is "
      "labeled but not symbolically modeled; a lowering can roll its own counter.",
      "exact bank-conflict-avoidance rationale beyond the known LdsPadB=8/128B padding is not replayed from the "
      "Tensile generator.",
      "tail-loop dynamic entry condition for K=4096 (clean multiple) not evaluated; emitted but may be skipped.",
    ],
    "lowering_assessment": {
      "alternating_lds_slots": alternation_ok, "ordered_dependency_edges": edges_ok,
      "hardware_loop_counter_recovered": has_kcounter, "wmma_clusters_sum_matches_audit": clusters_sum_ok,
      "ready_for_lowering": lowerable,
    },
    "input_artifacts": [CONTRACT, TEMPLATE, AUDIT, str(DISASM)],
    "next_action": (
      "Lower the symbolic template: emit the unrolled-by-2 body with alternating LDS slot offsets and the "
      "ordered vmcnt/lgkmcnt/barrier dependency edges, gated by the existing AMDGemmScheduleObject structural "
      "gate; then (and only then) time vs the >=60 TFLOPS authority. No BEAM/search until lowering exists."
      if lowerable else
      "Need CFG/branch-target + address-VGPR symbolic tracking to recover the loop head and slot model."),
  }
  write_json("amd_gemm_kloop_reconstruction_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/amd_gemm_kloop_reconstruction_result.json",
    "verdict": verdict, "gate_pass": lowerable,
    "unroll": unroll, "slot_alternation": result["symbolic_kloop_template"]["slot_alternation"],
    "wmma_total": total_wmma, "wmma_clusters": [(c["region"], c["n"]) for c in clusters],
    "symbolic_k_slices": symbolic_k_slices, "per_wave_wmma_per_kslice": per_wave_wmma,
    "lowering_assessment": result["lowering_assessment"],
  }, indent=2))
  return 0 if lowerable else 1


if __name__ == "__main__":
  raise SystemExit(main())
