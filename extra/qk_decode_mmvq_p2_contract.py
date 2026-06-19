#!/usr/bin/env python3
"""P2 contract capture processor for imported llama MMVQ kernels."""
from __future__ import annotations

import json, pathlib, struct
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-mmvq-large-project"
SRC = pathlib.Path("/tmp/qk_decode_mmvq_kernarg.jsonl")

FIELDS = [
  ("vx_ptr", 0, 8, "ptr"),
  ("vy_ptr", 8, 8, "ptr"),
  ("ids_ptr", 16, 8, "ptr"),
  ("fusion", 24, 32, "bytes"),
  ("dst_ptr", 56, 8, "ptr"),
  ("ncols_x", 64, 4, "u32"),
  ("nchannels_y", 68, 12, "u32x3"),
  ("stride_row_x", 80, 4, "u32"),
  ("stride_col_y", 84, 4, "u32"),
  ("stride_col_dst", 88, 4, "u32"),
  ("channel_ratio", 92, 12, "u32x3"),
  ("stride_channel_x", 104, 4, "u32"),
  ("stride_channel_y", 108, 4, "u32"),
  ("stride_channel_dst", 112, 4, "u32"),
  ("sample_ratio", 116, 12, "u32x3"),
  ("stride_sample_x", 128, 4, "u32"),
  ("stride_sample_y", 132, 4, "u32"),
  ("stride_sample_dst", 136, 4, "u32"),
  ("ids_stride", 140, 4, "u32"),
]


def decode_field(raw: bytes, off: int, size: int, kind: str) -> Any:
  if kind == "ptr":
    return struct.unpack_from("<Q", raw, off)[0]
  if kind == "u32":
    return struct.unpack_from("<I", raw, off)[0]
  if kind == "u32x3":
    return list(struct.unpack_from("<III", raw, off))
  if kind == "bytes":
    return list(raw[off:off + size])
  raise ValueError(kind)


def parse_symbol(sym: str) -> dict[str, Any]:
  import re
  m = re.search(r"ggml_type(?P<type>\d+)ELi(?P<ncols>\d+)ELb(?P<b0>[01])ELb(?P<b1>[01])", sym)
  return {
    "type_id": int(m.group("type")) if m else None,
    "ncols_dst": int(m.group("ncols")) if m else None,
    "has_fusion": bool(int(m.group("b0"))) if m else None,
    "small_k": bool(int(m.group("b1"))) if m else None,
  }


def role_guess(row: dict[str, Any], decoded: dict[str, Any]) -> str:
  g = row["num_workgroups"]
  if row["type"] == "Q6_K" and g[0] == 151936:
    return "lm_head"
  if row["type"] == "Q6_K" and g[0] == 4096 and decoded["ncols_x"] == 12288:
    return "ffn_down"
  if row["type"] == "Q4_K" and g[0] == 12288:
    return "ffn_gate_or_up_fused"
  if row["type"] == "Q4_K" and g[0] == 4096:
    return "attn_q_or_o"
  if row["type"] == "Q4_K" and g[0] == 1024:
    return "attn_k_or_v"
  if row["type"] == "Q6_K" and g[0] == 1024:
    return "q6_small_role"
  return "unknown"


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  rows = []
  if not SRC.exists():
    raise FileNotFoundError(f"{SRC} not found; run extra/qk_decode_mmvq_kernarg_capture.cpp under llama-bench first")
  for line in SRC.read_text().splitlines():
    if not line.strip():
      continue
    row = json.loads(line)
    raw = bytes(row["kernarg_bytes"])
    decoded = {name: decode_field(raw, off, size, kind) for name, off, size, kind in FIELDS}
    row["symbol_template"] = parse_symbol(row["kernel_symbol"])
    row["decoded_kernarg"] = decoded
    row["role_guess"] = role_guess(row, decoded)
    row["pointer_offsets"] = [off for name, off, _size, kind in FIELDS if kind == "ptr"]
    row["rebindable_pointer_offsets"] = {"vx": 0, "vy": 8, "ids": 16, "dst": 56}
    rows.append(row)

  by_type = {}
  for r in rows:
    by_type[r["type"]] = by_type.get(r["type"], 0) + 1
  selected = {
    "q4_attn_q_or_o": next((r for r in rows if r["type"] == "Q4_K" and r["role_guess"] == "attn_q_or_o" and not r["symbol_template"]["has_fusion"]), None),
    "q6_ffn_down": next((r for r in rows if r["type"] == "Q6_K" and r["role_guess"] == "ffn_down"), None),
    "q6_lm_head": next((r for r in rows if r["type"] == "Q6_K" and r["role_guess"] == "lm_head"), None),
  }
  result = {
    "schema": "decode_mmvq_large_project_p2_contract_v1",
    "date": "2026-06-19",
    "source_jsonl": str(SRC),
    "capture_count": len(rows),
    "by_type": by_type,
    "fields": [{"name": n, "offset": o, "size": s, "kind": k} for n, o, s, k in FIELDS],
    "selected": selected,
    "rows": rows,
    "verdict": "PASS" if selected["q4_attn_q_or_o"] and selected["q6_ffn_down"] else "KILL",
    "next": "P3 standalone HCQ correctness with pointer rebinding" if selected["q4_attn_q_or_o"] and selected["q6_ffn_down"] else "rerun capture with broader decode workload",
  }
  (OUT / "p2_kernarg_capture.json").write_text(json.dumps(result, indent=2) + "\n")
  (OUT / "p2_kernarg_capture.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
  summary = [
    "# Decode MMVQ large project P2 kernarg capture",
    "",
    f"- verdict: `{result['verdict']}`",
    f"- captures: `{len(rows)}`",
    f"- by type: `{by_type}`",
    "",
    "## Selected P3 Templates",
    "",
  ]
  for key, row in selected.items():
    if row is None:
      summary.append(f"- `{key}`: missing")
    else:
      d = row["decoded_kernarg"]
      summary.append(
        f"- `{key}`: role `{row['role_guess']}`, global `{row['global']}`, local `{row['local']}`, "
        f"num_workgroups `{row['num_workgroups']}`, ncols_x `{d['ncols_x']}`, stride_row_x `{d['stride_row_x']}`, "
        f"stride_col_dst `{d['stride_col_dst']}`, has_fusion `{row['symbol_template']['has_fusion']}`"
      )
  summary.append("")
  (OUT / "p2_kernarg_capture_summary.md").write_text("\n".join(summary))
  print(json.dumps({"verdict": result["verdict"], "capture_count": len(rows), "by_type": by_type}, indent=2))


if __name__ == "__main__":
  main()
