#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, re
from collections import Counter, deque
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
DISASM = pathlib.Path("/tmp/td_all.txt")

FUNC_RE = re.compile(r"^[0-9a-fA-F]+ <([^>]+)>:")
OFFSET_RE = re.compile(r"offset:([0-9]+)")
VGPR_RE = re.compile(r"v\[(\d+):(\d+)\]|v(\d+)")
MNEMONIC_RE = re.compile(r"^\s*([a-zA-Z0-9_]+)")


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def regs(line: str) -> list[set[int]]:
  out: list[set[int]] = []
  for m in VGPR_RE.finditer(line.split("//", 1)[0]):
    if m.group(3) is not None:
      v = int(m.group(3))
      out.append({v})
    else:
      lo, hi = int(m.group(1)), int(m.group(2))
      out.append(set(range(lo, hi + 1)))
  return out


def mnemonic(line: str) -> str | None:
  m = MNEMONIC_RE.match(line)
  return m.group(1) if m else None


def parse_offset(line: str) -> int:
  m = OFFSET_RE.search(line)
  return int(m.group(1)) if m else 0


def find_selected_body(symbol: str) -> tuple[bool, str | None, int | None, int | None, list[tuple[int, str]]]:
  if not DISASM.exists():
    return False, None, None, None, []
  selected_label = symbol + "_preloaded"
  active = False
  found_label: str | None = None
  start: int | None = None
  body: list[tuple[int, str]] = []
  with DISASM.open("r", errors="replace") as f:
    for no, line in enumerate(f, 1):
      fm = FUNC_RE.match(line)
      if fm:
        label = fm.group(1)
        if active and label.startswith("Cijk_"): break
        if label == selected_label or label == symbol or label.startswith(symbol):
          active = True
          found_label = label
          start = no
      if active:
        body.append((no, line.rstrip("\n")))
  end = body[-1][0] if body else None
  return bool(body), found_label, start, end, body


def summarize_body(body: list[tuple[int, str]]) -> dict[str, Any]:
  interesting = {"ds_load_b128", "ds_store_b128", "ds_store_b64", "v_wmma_f32_16x16x16_f16", "s_waitcnt", "s_barrier",
                 "global_load_b128", "global_load_dwordx4", "buffer_load_dwordx4", "buffer_load_b64"}
  counts: Counter[str] = Counter()
  offsets: dict[str, Counter[int]] = {"ds_load_b128": Counter(), "ds_store_b128": Counter(), "ds_store_b64": Counter()}
  samples: dict[str, list[dict[str, Any]]] = {k: [] for k in ["ds_load_b128", "ds_store_b128", "ds_store_b64", "v_wmma", "s_waitcnt", "s_barrier"]}
  windows: dict[str, list[dict[str, Any]]] = {k: [] for k in ["ds_load_b128", "ds_store_b128", "ds_store_b64", "v_wmma"]}
  last_lines: deque[tuple[int, str]] = deque(maxlen=16)
  recent_ds_loads: deque[tuple[int, set[int], str]] = deque(maxlen=48)
  recent_global_loads: deque[tuple[int, set[int], str]] = deque(maxlen=96)
  wmma_with_recent_ds_src_overlap = 0
  wmma_examined = 0
  ds_store_with_recent_global_data_overlap = 0
  ds_store_examined = 0

  for idx, (no, line) in enumerate(body):
    mnem = mnemonic(line)
    if not mnem:
      last_lines.append((no, line))
      continue
    if mnem.startswith("v_wmma"):
      key = "v_wmma"
      counts[key] += 1
    elif mnem in interesting:
      key = mnem
      counts[key] += 1
    elif mnem.startswith("global_load") or mnem.startswith("buffer_load"):
      key = mnem
      counts[key] += 1
    else:
      key = ""

    if key in offsets:
      offsets[key][parse_offset(line)] += 1
    if key in samples and len(samples[key]) < 16:
      samples[key].append({"line": no, "text": line.strip(), "offset": parse_offset(line) if key in offsets else None})
    if key in windows and len(windows[key]) < 3:
      post = body[idx + 1:idx + 17]
      windows[key].append({
        "line": no,
        "pre": [x[1].strip() for x in last_lines],
        "hit": line.strip(),
        "post": [x[1].strip() for x in post],
      })

    line_regs = regs(line)
    if key == "ds_load_b128" and line_regs:
      recent_ds_loads.append((no, set(line_regs[0]), line.strip()))
    if key.startswith("global_load") or key.startswith("buffer_load"):
      if line_regs:
        recent_global_loads.append((no, set(line_regs[0]), line.strip()))
    if key in {"ds_store_b128", "ds_store_b64"} and len(line_regs) >= 2:
      ds_store_examined += 1
      data_regs = set().union(*line_regs[1:])
      if any(data_regs & loaded for _, loaded, _ in recent_global_loads):
        ds_store_with_recent_global_data_overlap += 1
    if key == "v_wmma":
      wmma_examined += 1
      src_regs = set().union(*line_regs[1:3]) if len(line_regs) >= 3 else set()
      if any(src_regs & loaded for _, loaded, _ in recent_ds_loads):
        wmma_with_recent_ds_src_overlap += 1
    last_lines.append((no, line))

  offset_summary = {}
  for key, ctr in offsets.items():
    offset_summary[key] = {
      "unique_count": len(ctr),
      "min": min(ctr) if ctr else None,
      "max": max(ctr) if ctr else None,
      "most_common": [{"offset": k, "count": v} for k, v in ctr.most_common(24)],
      "first_64_sorted": sorted(ctr)[:64],
    }
  return {
    "line_count": len(body),
    "instruction_counts": dict(sorted(counts.items())),
    "offset_summary": offset_summary,
    "samples": samples,
    "windows": windows,
    "handoff_inference": {
      "wmma_examined": wmma_examined,
      "wmma_with_recent_ds_load_source_register_overlap": wmma_with_recent_ds_src_overlap,
      "ds_store_examined": ds_store_examined,
      "ds_store_with_recent_global_load_data_register_overlap": ds_store_with_recent_global_data_overlap,
    },
  }


def main() -> int:
  selection = read_json("bench/qk-tensile-extraction/selection.json", {})
  codegen = read_json("bench/qk-tensile-extraction/codegen_oracle.json", {})
  capture = read_json("bench/amd-broad-backend-roadmap/bb5a8_authority_kernel_capture_result.json", {})
  causal = read_json("bench/amd-broad-backend-roadmap/bb5a9_causal_delta_package_result.json", {})
  symbol = selection.get("selected", {}).get("rocblas", {}).get("kernel_symbol", "")
  found, label, start, end, body = find_selected_body(symbol)
  summary = summarize_body(body) if body else {}
  counts = summary.get("instruction_counts", {})
  handoff = summary.get("handoff_inference", {})
  schedule = codegen.get("tensile_schedule", {})
  selected = selection.get("selected", {}).get("rocblas", {})

  gate = {
    "selected_symbol_isolated": found,
    "authority_shape_metadata_present": selected.get("grid") == [512, 96, 1] and selected.get("workgroup") == [128, 1, 1],
    "tensile_resource_metadata_present": selected.get("lds_bytes") == 25088 and selected.get("vgpr") == 256 and selected.get("scratch") == 0,
    "schedule_has_pgr_plr_lrvw": schedule.get("prefetch_global_read_PGR") == "1" and schedule.get("prefetch_local_read_PLR") == "1" and schedule.get("local_read_vec_LRVW") == "16",
    "disasm_has_ds_load_b128": (counts.get("ds_load_b128") or 0) > 0,
    "disasm_has_ds_store_to_lds": (counts.get("ds_store_b128") or 0) > 0 or (counts.get("ds_store_b64") or 0) > 0,
    "disasm_has_wmma": (counts.get("v_wmma") or 0) > 0,
    "disasm_has_waits_and_barriers": (counts.get("s_waitcnt") or 0) > 0 and (counts.get("s_barrier") or 0) > 0,
    "lds_to_wmma_register_handoff_inferred": (handoff.get("wmma_with_recent_ds_load_source_register_overlap") or 0) > 0,
    "global_to_lds_register_handoff_inferred": (handoff.get("ds_store_with_recent_global_load_data_register_overlap") or 0) > 0,
    "tinygrad_authority_zero_lds_baseline_present": capture.get("resource", {}).get("lds_bytes") == 0 and capture.get("resource", {}).get("kernel_descriptor", {}).get("private_segment_fixed_size") == 0,
    "bb5a9_causal_delta_passed": causal.get("gate_pass") is True,
  }
  candidate_ready = all(gate.values())
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.10_tensile_layout_audit",
    "schema": "amd_bb5a10_tensile_layout_audit_v1",
    "verdict": "PASS_TENSILE_LAYOUT_AUDIT_CANDIDATE_SPEC_READY_NOT_BITEXACT" if candidate_ready else "BLOCKED_TENSILE_LAYOUT_AUDIT_INPUTS_INCOMPLETE",
    "gate_pass": candidate_ready,
    "default_behavior_changed": False,
    "performance_claim": False,
    "selected_function": {
      "symbol": symbol,
      "disasm_path": str(DISASM),
      "isolated": found,
      "label": label,
      "line_start": start,
      "line_end": end,
    },
    "oracle_contract": {
      "shape": selection.get("shape"),
      "grid": selected.get("grid"),
      "workgroup": selected.get("workgroup"),
      "vgpr": selected.get("vgpr"),
      "lds_bytes": selected.get("lds_bytes"),
      "scratch": selected.get("scratch"),
      "tflops": selected.get("tflops"),
      "schedule": schedule,
    },
    "disasm_summary": summary,
    "gate": gate,
    "readiness": {
      "enough_to_start_bb5a10_candidate": candidate_ready,
      "enough_to_clone_bitexact_tensile_lds_layout": False,
      "why_not_bitexact": [
        "AMDGPU disassembly gives LDS offsets, registers, waits, and handoff windows but not Tensile's symbolic tensor-layout intent.",
        "The selected code object contains the concrete schedule contract, but no source-level mapping from A/B logical tile coordinates to every LDS byte lane.",
        "A bit-identical clone would need Tensile generator metadata/source reconstruction, not just selected-kernel ISA.",
      ],
      "candidate_scope": [
        "Implement a non-bitexact Tensile-class authority-shape candidate: MT128x128x16, WMMA 16x16x16, PGR1/PLR1-style staged movement, nonzero LDS, selected-kernel-compatible LDS stores, ds_load_b128, waits/barriers, scratch-free resource policy.",
        "Use extracted offsets/register windows as acceptance probes and diagnostics, not as a requirement to duplicate the exact Tensile offset sequence.",
      ],
    },
    "implementation_spec": {
      "B_LDS_layout": [
        "Allocate two logical operand regions A/B in LDS for the authority shape.",
        "Emit global-to-register-to-LDS stores visible in source/disasm; selected rocBLAS authority uses ds_store_b64 while the larger oracle corpus includes ds_store_b128.",
        "Emit wide LDS-to-register reads with ds_load_b128, and ensure the loaded VGPR ranges feed v_wmma source operands.",
        "Gate on nonzero LDS bytes in ELF metadata and no private scratch."
      ],
      "C_K_loop_scheduler": [
        "Generate prologue plus steady-state loop over depthU=16 K tiles.",
        "Separate producer and consumer LDS stages so the next global load/LDS store can be scheduled around current WMMA issue.",
        "Place vmcnt/lgkmcnt waits and barriers based on dependency groups, not textual heuristics.",
      ],
      "D_resource_policy": [
        "Reject if scratch/private segment appears.",
        "Reject if VGPR/SGPR/LDS budget cannot preserve the authority occupancy/resource envelope.",
        "Only time candidates after LDS, selected-kernel-compatible LDS stores, ds_load_b128, WMMA, and wait/barrier structural checks pass.",
      ],
      "candidate_gate": "correctness plus >=60 TFLOPS pure tinygrad authority prefill before q8 transfer reopens",
    },
    "input_artifacts": [
      "bench/qk-tensile-extraction/selection.json",
      "bench/qk-tensile-extraction/codegen_oracle.json",
      "bench/amd-broad-backend-roadmap/bb5a8_authority_kernel_capture_result.json",
      "bench/amd-broad-backend-roadmap/bb5a9_causal_delta_package_result.json",
      str(DISASM),
    ],
  }
  write_json("bb5a10_tensile_layout_audit_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a10_tensile_layout_audit_result.json",
    "verdict": result["verdict"],
    "gate_pass": result["gate_pass"],
    "selected_lines": summary.get("line_count"),
    "counts": {k: counts.get(k) for k in ["ds_load_b128", "ds_store_b128", "ds_store_b64", "v_wmma", "s_waitcnt", "s_barrier"]},
    "bitexact_ready": result["readiness"]["enough_to_clone_bitexact_tensile_lds_layout"],
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
