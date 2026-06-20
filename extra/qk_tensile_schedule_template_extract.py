#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, re, struct
from collections import Counter
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROCBLAS_LIB = pathlib.Path("/opt/rocm-7.2.4/lib/rocblas/library")
DAT = ROCBLAS_LIB / "TensileLibrary_Type_HH_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx1100.dat"
DISASM = pathlib.Path("/tmp/td_all.txt")
OUT = ROOT / "bench/qk-tensile-extraction"

FUNC_RE = re.compile(r"^[0-9a-fA-F]+ <([^>]+)>:")
MNEMONIC_RE = re.compile(r"^\s*([a-zA-Z0-9_]+)")
OFFSET_RE = re.compile(r"offset:([0-9]+)")


def mp_decode(b: memoryview, i: int = 0):
  c = b[i]; i += 1
  if c < 0x80: return c, i
  if c >= 0xe0: return c - 0x100, i
  if 0x80 <= c <= 0x8f: return _mp_map(b, i, c & 0xf)
  if 0x90 <= c <= 0x9f: return _mp_arr(b, i, c & 0xf)
  if 0xa0 <= c <= 0xbf: return _mp_str(b, i, c & 0x1f)
  if c == 0xc0: return None, i
  if c == 0xc2: return False, i
  if c == 0xc3: return True, i
  if c == 0xc4: n = b[i]; return bytes(b[i+1:i+1+n]), i+1+n
  if c == 0xc5: n = struct.unpack_from(">H", b, i)[0]; return bytes(b[i+2:i+2+n]), i+2+n
  if c == 0xc6: n = struct.unpack_from(">I", b, i)[0]; return bytes(b[i+4:i+4+n]), i+4+n
  if c == 0xca: return struct.unpack_from(">f", b, i)[0], i+4
  if c == 0xcb: return struct.unpack_from(">d", b, i)[0], i+8
  if c == 0xcc: return b[i], i+1
  if c == 0xcd: return struct.unpack_from(">H", b, i)[0], i+2
  if c == 0xce: return struct.unpack_from(">I", b, i)[0], i+4
  if c == 0xcf: return struct.unpack_from(">Q", b, i)[0], i+8
  if c == 0xd0: return struct.unpack_from(">b", b, i)[0], i+1
  if c == 0xd1: return struct.unpack_from(">h", b, i)[0], i+2
  if c == 0xd2: return struct.unpack_from(">i", b, i)[0], i+4
  if c == 0xd3: return struct.unpack_from(">q", b, i)[0], i+8
  if c == 0xd9: n = b[i]; return _mp_str(b, i+1, n)
  if c == 0xda: n = struct.unpack_from(">H", b, i)[0]; return _mp_str(b, i+2, n)
  if c == 0xdb: n = struct.unpack_from(">I", b, i)[0]; return _mp_str(b, i+4, n)
  if c == 0xdc: n = struct.unpack_from(">H", b, i)[0]; return _mp_arr(b, i+2, n)
  if c == 0xdd: n = struct.unpack_from(">I", b, i)[0]; return _mp_arr(b, i+4, n)
  if c == 0xde: n = struct.unpack_from(">H", b, i)[0]; return _mp_map(b, i+2, n)
  if c == 0xdf: n = struct.unpack_from(">I", b, i)[0]; return _mp_map(b, i+4, n)
  raise ValueError(f"msgpack byte {hex(c)} @ {i-1} unsupported")


def _mp_str(b: memoryview, i: int, n: int): return bytes(b[i:i+n]).decode("utf-8", "replace"), i+n


def _mp_arr(b: memoryview, i: int, n: int):
  out = []
  for _ in range(n):
    v, i = mp_decode(b, i)
    out.append(v)
  return out, i


def _mp_map(b: memoryview, i: int, n: int):
  out = {}
  for _ in range(n):
    k, i = mp_decode(b, i)
    v, i = mp_decode(b, i)
    out[k] = v
  return out, i


def read_json(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def select_solution(dat: dict[str, Any], symbol: str) -> dict[str, Any]:
  solutions = dat.get("solutions") or []
  for i, sol in enumerate(solutions):
    names = [sol.get("KernelName"), sol.get("Name"), sol.get("SolutionName"), sol.get("name"), sol.get("SolutionIndex")]
    if symbol in {x for x in names if isinstance(x, str)}:
      return {"index": i, "solution": sol, "match_field": "exact"}
    if any(isinstance(x, str) and (x in symbol or symbol in x) for x in names):
      return {"index": i, "solution": sol, "match_field": "substring"}
  raise RuntimeError(f"selected symbol not found in {len(solutions)} .dat solutions")


def compact_solution(sol: dict[str, Any]) -> dict[str, Any]:
  sm = sol.get("sizeMapping") or {}
  keys = [
    "KernelName", "Name", "SolutionName", "SolutionIndex", "name", "index", "ProblemType", "problemType",
    "ISA", "MacroTile0", "MacroTile1",
    "MacroTile", "DepthU", "MatrixInstruction", "WorkGroup", "ThreadTile", "NumThreads", "LoopIters",
    "PrefetchGlobalRead", "PrefetchLocalRead", "ScheduleIterAlg", "ScheduleGlobalRead", "ScheduleLocalWrite",
    "GlobalReadVectorWidth", "GlobalLoadVectorWidthA", "GlobalLoadVectorWidthB", "LocalReadVectorWidth",
    "VectorWidth", "VectorWidthB", "GlobalReadCoalesceGroupA", "GlobalReadCoalesceGroupB",
    "GlobalReadCoalesceVectorA", "GlobalReadCoalesceVectorB", "TransposeLDS", "UnrollMajorLDSA",
    "UnrollMajorLDSB", "LdsPadA", "LdsPadB", "LdsBlockSizePerPadA", "LdsBlockSizePerPadB", "LdsNumElements",
    "LdsOffsetA", "LdsOffsetB", "LdsOffsetA_Blk", "LdsOffsetB_Blk", "1LDSBuffer", "DirectToLdsA",
    "DirectToLdsB", "WorkGroupMapping", "WorkGroupMappingType", "GlobalSplitU", "StreamK", "StaggerU",
    "StaggerUStride", "StaggerUMapping", "NumLoadsCoalescedA", "NumLoadsCoalescedB", "LocalSplitU",
  ]
  ret = {k: sol.get(k) for k in keys if k in sol}
  if sm:
    ret["sizeMapping"] = {k: sm.get(k) for k in [
      "workGroup", "macroTile", "threadTile", "depthU", "staggerU", "globalSplitU", "staggerStrideShift",
      "workGroupMapping", "packSummationDims", "packBatchDims", "magicDivAlg", "streamK", "streamKAtomic",
      "persistentKernel", "persistentKernelAlongBatch", "sourceKernel", "globalAccumulation",
      "workspaceSizePerElemC", "preloadKernargs",
    ] if k in sm}
  return ret


def mnemonic(line: str) -> str:
  m = MNEMONIC_RE.match(line)
  return m.group(1) if m else ""


def selected_body(symbol: str) -> list[tuple[int, str]]:
  if not DISASM.exists(): raise FileNotFoundError(f"{DISASM} missing; run disassembly capture first")
  labels = {symbol, symbol + "_preloaded"}
  active = False
  body: list[tuple[int, str]] = []
  with DISASM.open("r", errors="replace") as f:
    for no, line in enumerate(f, 1):
      fm = FUNC_RE.match(line)
      if fm:
        label = fm.group(1)
        if active and label.startswith("Cijk_"): break
        if label in labels or label.startswith(symbol):
          active = True
      if active: body.append((no, line.rstrip("\n")))
  if not body: raise RuntimeError(f"selected symbol body not found in {DISASM}")
  return body


def summarize_region(rows: list[tuple[int, str]]) -> dict[str, Any]:
  counts: Counter[str] = Counter()
  offsets: dict[str, Counter[int]] = {"ds_store_b64": Counter(), "ds_store_b128": Counter(), "ds_load_b128": Counter()}
  for _, line in rows:
    m = mnemonic(line)
    if not m: continue
    key = "v_wmma" if m.startswith("v_wmma") else m
    if key.startswith(("buffer_load", "global_load")) or key in {"ds_store_b64", "ds_store_b128", "ds_load_b128", "s_waitcnt", "s_barrier", "v_wmma"}:
      counts[key] += 1
    if key in offsets:
      mo = OFFSET_RE.search(line)
      offsets[key][int(mo.group(1)) if mo else 0] += 1
  return {
    "line_range": [rows[0][0], rows[-1][0]] if rows else None,
    "line_count": len(rows),
    "counts": dict(sorted(counts.items())),
    "offsets": {k: {"unique": len(v), "min": min(v) if v else None, "max": max(v) if v else None,
                    "values": sorted(v)[:64]} for k, v in offsets.items()},
  }


def segment_body(body: list[tuple[int, str]]) -> dict[str, Any]:
  wmma_idx = [i for i, (_, line) in enumerate(body) if mnemonic(line).startswith("v_wmma")]
  if not wmma_idx: raise RuntimeError("selected body has no v_wmma")
  first, last = wmma_idx[0], wmma_idx[-1]
  regions = {
    "prologue_before_first_wmma": body[:first],
    "steady_compute_first_to_last_wmma": body[first:last+1],
    "epilogue_after_last_wmma": body[last+1:],
  }
  steady = regions["steady_compute_first_to_last_wmma"]
  wmma_lines = [no for no, line in steady if mnemonic(line).startswith("v_wmma")]
  return {
    "heuristic": "prologue before first v_wmma; steady region first through last v_wmma; epilogue after last v_wmma",
    "function_line_range": [body[0][0], body[-1][0]],
    "function_line_count": len(body),
    "first_wmma_line": body[first][0],
    "last_wmma_line": body[last][0],
    "wmma_count": len(wmma_lines),
    "wmma_line_stride_summary": {
      "min": min((b-a for a,b in zip(wmma_lines, wmma_lines[1:])), default=None),
      "max": max((b-a for a,b in zip(wmma_lines, wmma_lines[1:])), default=None),
      "first_24": wmma_lines[:24],
    },
    "regions": {k: summarize_region(v) for k, v in regions.items()},
  }


def main() -> int:
  selection = read_json("bench/qk-tensile-extraction/selection.json", {})
  contract = read_json("bench/qk-tensile-extraction/ffn_gate_up_contract.json", {})
  symbol = selection["selected"]["rocblas"]["kernel_symbol"]
  dat, end = mp_decode(memoryview(DAT.read_bytes()))
  if end != DAT.stat().st_size: raise RuntimeError(f"partial .dat decode: {end} of {DAT.stat().st_size}")
  picked = select_solution(dat, symbol)
  body = selected_body(symbol)
  result = {
    "schema": "qk_tensile_schedule_template_v1",
    "date": "2026-06-20",
    "role": "ffn_gate/up",
    "shape": {"M": 512, "N": 12288, "K": 4096},
    "selected": {
      "kernel_symbol": symbol,
      "dat_path": str(DAT),
      "dat_solution_index": picked["index"],
      "dat_solution_count": len(dat.get("solutions") or []),
      "match_field": picked["match_field"],
      "launch": selection["selected"]["rocblas"].get("grid"),
      "workgroup": selection["selected"]["rocblas"].get("workgroup"),
      "resources": {
        "sgpr": contract.get("sgpr_count"),
        "vgpr": contract.get("vgpr_count"),
        "lds_bytes": contract.get("group_segment_fixed_size"),
        "private": contract.get("private_segment_fixed_size"),
      },
    },
    "solution_params": compact_solution(picked["solution"]),
    "schedule_segments": segment_body(body),
    "readiness": {
      "dat_solution_row_extracted": True,
      "disasm_segmented": True,
      "logical_lds_tile_map_complete": False,
      "steady_state_segmentation_is_heuristic": True,
      "ready_for_machine_search": False,
      "next_required_artifact": "source-level LDS tile-map sketch from Tensile LraTileAssignment/LocalRead plus selected offsets",
    },
  }
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "ffn_gate_up_schedule_template.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "out": "bench/qk-tensile-extraction/ffn_gate_up_schedule_template.json",
    "solution_index": picked["index"],
    "solution_keys": len(picked["solution"]),
    "regions": {k: v["counts"] for k, v in result["schedule_segments"]["regions"].items()},
    "ready_for_machine_search": result["readiness"]["ready_for_machine_search"],
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
