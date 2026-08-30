#!/usr/bin/env python3
"""F1.1: build an installed-vs-vector manifest from finalized graph calls.

This is intentionally a capture consumer.  It never constructs or relaunches
the installed T=512 graph: each entry must carry the graph-owned slice/buffer
binding from the finalized PROGRAM call.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

PROGRAM = "nv_sm120_q16_grid_hd128_loop_attention"
ABI = "nv_sm120_vkv_h4_t64_w4_online128_v1"
IDENTITY = "flash.nv_sm120.vkv_h4_t64_w4_online128.v1.swizzle16.v1"
EXPECTED = 36

def digest(obj):
  return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _calls(capture):
  rows = capture.get("calls", capture.get("entries", []))
  out = []
  for i, row in enumerate(rows):
    name = row.get("program", row.get("name", row.get("identity", "")))
    if name == PROGRAM or row.get("installed_program") == PROGRAM:
      out.append((i, row))
  return out

def _entry(i, row):
  # Require graph-owned binding metadata.  No defaults for shapes/strides are
  # accepted: silently inventing them would turn this into a full-T512 replay.
  binding = row.get("graph_owned_slice", row.get("binding"))
  if not isinstance(binding, dict) or binding.get("source") not in ("graph", "graph-owned"):
    raise ValueError(f"call {i} lacks graph-owned slice binding")
  required = ("layer", "q_head", "kv_head", "causal_length", "shapes", "strides", "dtypes", "buffer_hashes", "output_shape", "buffers")
  missing = [k for k in required if k not in row]
  if missing: raise ValueError(f"call {i} missing {','.join(missing)}")
  if not all(isinstance(buf, dict) and "base" in buf and "offset" in buf for buf in row["buffers"]):
    raise ValueError(f"call {i} has incomplete graph buffer/view metadata")
  if row.get("full_t512_relaunch") or row.get("launch_scope") == "full-T512":
    raise ValueError(f"call {i} is a forbidden full-T512 relaunch")
  return {"ordinal": i, "layer": row["layer"], "q_head": row["q_head"],
    "kv_head": row["kv_head"], "causal_length": row["causal_length"],
    "shapes": row["shapes"], "strides": row["strides"], "dtypes": row["dtypes"],
    "buffer_hashes": row["buffer_hashes"], "output_shape": row["output_shape"], "buffers": row["buffers"],
    "graph_owned_slice": binding, "installed": {"program": PROGRAM, "identity": row.get("installed_identity", PROGRAM)},
    "candidate": {"abi": ABI, "identity": IDENTITY}, "same_logical_call": True}

def build(capture_path, out_path):
  cap = json.loads(Path(capture_path).read_text())
  selected = _calls(cap)
  result = {"schema":"tinygrad.nv_prefill_flash_f1_population.v1", "packet":"F1.1",
    "status":"BLOCKED", "source_capture":str(capture_path), "exact_population":EXPECTED,
    "installed_program":PROGRAM, "candidate":{"abi":ABI,"identity":IDENTITY},
    "graph_owned_slice_semantics":{"required":True,"full_t512_relaunch":False},
    "calls":[], "census":{"predicted":{"installed":EXPECTED,"candidate":EXPECTED},"observed":{"installed":len(selected),"candidate":0}},
    "decision":"BLOCKED: capture does not contain exact 36 finalized installed calls", "next_packet":None}
  try:
    result["calls"] = [_entry(i, row) for i, row in selected]
    if len(result["calls"]) == EXPECTED:
      result["status"] = "PASS"; result["census"]["observed"]["candidate"] = EXPECTED
      result["decision"] = "PASS: exact 36 graph-owned calls captured; comparator manifest ready for F1.2"
    elif len(result["calls"]) > EXPECTED:
      result["decision"] = "STOP: installed population exceeds exact 36-call authority"
  except ValueError as exc:
    result["status"] = "STOP"; result["decision"] = "STOP: " + str(exc)
  result["manifest_sha256"] = digest(result["calls"])
  Path(out_path).parent.mkdir(parents=True, exist_ok=True)
  Path(out_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, sort_keys=True))
  return 0 if result["status"] == "PASS" else 2

if __name__ == "__main__":
  ap=argparse.ArgumentParser(); ap.add_argument("capture"); ap.add_argument("--out", required=True)
  raise SystemExit(build(ap.parse_args().capture, ap.parse_args().out))
