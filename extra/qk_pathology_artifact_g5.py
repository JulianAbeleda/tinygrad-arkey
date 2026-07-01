#!/usr/bin/env python3
"""Capture tinygrad.compiler_pathology.v1 artifact for the G=5 block tile flash kernel.

Patches AMDProgram to capture private_segment_size (scratch/spill), group_segment_size (LDS),
and RSRC1 (encoded VGPR count) from the compiled ELF descriptor. Also statically analyzes
the kernel DSL for barrier count and warp reduce patterns.

Run:
  DEV=AMD JIT=1 QK_MODEL=/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf PYTHONPATH=. python3 extra/qk_pathology_artifact_g5.py
Output:
  bench/g5-block-tile/compiler_pathology_v1.json
"""
import os, sys, json
from pathlib import Path

os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("JIT", "1")

GGUF_PATH = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf")

# --- Patch AMDProgram to intercept compiled kernels ---
captured = {}

def _patch_amd_program():
  from tinygrad.runtime import ops_amd
  _orig_init = ops_amd.AMDProgram.__init__
  def _patched_init(self, dev, name, lib, **kwargs):
    _orig_init(self, dev, name, lib, **kwargs)
    if "flash_block_tiled" in name or "flash_state" in name:
      captured[name] = {
        "group_segment_size": self.group_segment_size,   # LDS bytes
        "private_segment_size": self.private_segment_size,  # scratch/spill bytes
        "rsrc1": self.rsrc1,
        "rsrc2": self.rsrc2,
        "rsrc3": self.rsrc3,
        "wave32": self.wave32,
      }
      print(f"  [captured] {name}: LDS={self.group_segment_size}B scratch={self.private_segment_size}B rsrc1={self.rsrc1:#010x}")
  ops_amd.AMDProgram.__init__ = _patched_init

_patch_amd_program()

# --- Static analysis of G=5 kernel DSL ---
def analyze_kernel_static():
  """Count barriers, REG slots, and structural complexity from the kernel Python DSL."""
  from extra.qk_flash_decode import flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel, _ceildiv
  import inspect, re

  src = inspect.getsource(flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel)

  # Count explicit UOp.barrier calls
  barrier_count_dsl = len(re.findall(r'UOp\.barrier', src))

  # Count warp reduce calls (each is 5 ds_bpermute stages)
  warp_reduce_calls = len(re.findall(r'_warp_reduce_sum_staged|warp_reduce_sum', src))

  # Count REG placeholders
  reg_placeholders = len(re.findall(r'AddrSpace\.REG', src))

  # Count UOp.range (reduce axes)
  reduce_ranges = len(re.findall(r'UOp\.range\(', src))

  # G=5 specific
  Hd, Hq, Hkv = 128, 40, 8
  G = Hq // Hkv  # 5
  LANES = 32; WARPS = G; THREADS = LANES * WARPS  # 160
  TK = 16
  STAGES = _ceildiv(TK * Hd, THREADS)  # ceildiv(2048, 160) = 13
  NB_at_L128 = _ceildiv(128, TK)  # = 8 for L=128
  NB_at_L16 = _ceildiv(16, TK)    # = 1 for L=16 (baseline uses smaller L)
  R = Hd // LANES  # 4
  RP = Hd // 64    # 2

  # Estimated barrier count per workgroup
  # 1 barrier per b iteration (after K+V staging)
  barriers_per_wg_L128 = NB_at_L128  # = 8
  barriers_per_wg_L16 = NB_at_L16    # = 1

  # Warp reduce ds_bpermute count per workgroup
  # _warp_reduce_sum_staged with WARP=32: 5 stages per call
  # Called once per TK token per NB block
  ds_bpermute_per_wg_L128 = NB_at_L128 * TK * warp_reduce_calls * 5  # each call -> 5 stages

  return {
    "G": G, "LANES": LANES, "WARPS": WARPS, "THREADS": THREADS,
    "TK": TK, "STAGES": STAGES, "R": R, "RP": RP,
    "NB_at_L128": NB_at_L128, "NB_at_L16": NB_at_L16,
    "dsl_barrier_calls": barrier_count_dsl,
    "dsl_warp_reduce_calls": warp_reduce_calls,
    "dsl_reg_placeholders": reg_placeholders,
    "dsl_reduce_ranges": reduce_ranges,
    "estimated_barriers_per_wg_L128": barriers_per_wg_L128,
    "estimated_ds_bpermute_per_wg_L128": ds_bpermute_per_wg_L128,
    "lds_ksh_bytes": TK * Hd * 2,   # fp16
    "lds_vsh_bytes": TK * Hd * 2,
    "lds_staging_bytes": TK * Hd * 2 + TK * Hd * 2,
    "lds_reg_per_thread_bytes": (R + 1 + 1 + 1) * 4,   # acc + den + mx + _dotp in f32
    "lds_total_estimate": TK*Hd*2 + TK*Hd*2 + THREADS * (R + 1 + 1 + 1) * 4,
    "stages_div_cleanly": (TK * Hd) % THREADS == 0,  # False for G=5
    "stages_OOB_elements": (STAGES * THREADS) - (TK * Hd),  # 32 OOB for G=5
  }

print("Analyzing G=5 kernel structure statically...")
static = analyze_kernel_static()
print(f"  THREADS={static['THREADS']}, STAGES={static['STAGES']}, NB(L=128)={static['NB_at_L128']}")
print(f"  LDS staging: {static['lds_staging_bytes']}B  REG-in-LDS: {static['lds_reg_per_thread_bytes']}B/thread x {static['THREADS']} = {static['lds_total_estimate']}B total")
print(f"  Barriers per WG (L=128): {static['estimated_barriers_per_wg_L128']}")
print(f"  STAGES clean division: {static['stages_div_cleanly']} (OOB elements: {static['stages_OOB_elements']})")

# --- Trigger G=5 kernel compilation ---
print(f"\nLoading model to trigger G=5 kernel compilation...")
print(f"  Model: {GGUF_PATH}")

try:
  import os
  os.environ["DECODE_FLASH_BLOCK_TILE_G5"] = "1"

  from tinygrad.helpers import getenv, Timing
  from tinygrad.tensor import Tensor
  from tinygrad.dtype import dtypes

  # Minimal trigger: build the kernel without running the full model
  from extra.qk_flash_decode import flash_decode_g5_block_tile, _ceildiv
  Hd, Hq, Hkv, MAXC = 128, 40, 8, 4608

  # Synthetic inputs matching 14B decode at ctx=512
  ctx = 512
  q = Tensor.zeros(Hq, Hd, dtype=dtypes.float)
  cache_kv = Tensor.zeros(2, 1, Hkv, MAXC, Hd, dtype=dtypes.float)

  print("  Compiling G=5 block tile kernels...")
  with Timing("  compilation: "):
    out = flash_decode_g5_block_tile(q, cache_kv, ctx, ctx, Hd, Hq, Hkv, MAXC, L=128)
    out.realize()

  print(f"\nCaptured kernels: {list(captured.keys())}")

except Exception as e:
  print(f"  Model load failed: {e}")
  print("  Using static analysis only.")

# --- Decode RSRC1 for VGPR count ---
def decode_rsrc1_vgpr(rsrc1: int) -> int:
  """VGPR count from COMPUTE_PGM_RSRC1. Bits [5:0] = (vgpr_count / 8) - 1 for wave32."""
  granulated = rsrc1 & 0x3F
  return (granulated + 1) * 8

def decode_rsrc1_sgpr(rsrc1: int) -> int:
  """SGPR count from COMPUTE_PGM_RSRC1. Bits [9:6] = (sgpr_count / 8) - 1."""
  granulated = (rsrc1 >> 6) & 0xF
  return (granulated + 1) * 8

# --- Build the artifact ---
# Find the main block tile partial kernel
block_tile_key = next((k for k in captured if "whole_cache_40_128" in k), None)
baseline_key = None  # we'd need to capture baseline too

print("\nBuilding compiler_pathology_v1.json...")

# Known timing from bench/g5-block-tile/latest.json
baseline_wg_us = 27.0    # flash_partial_40_128 per layer
candidate_wg_us = 2090.0  # G=5 block tile per layer

artifact = {
  "schema": "tinygrad.compiler_pathology.v1",
  "date": "2026-07-01",
  "candidate_id": "decode_flash_block_tile_g5_native_context",
  "kernel_name": "flash_block_tiled_xlane_score_pv_tile_whole_cache_40_128",
  "baseline_kernel_name": "flash_partial_40_128",
  "baseline_workgroup_us": baseline_wg_us,
  "candidate_workgroup_us": candidate_wg_us,
  "slowdown_factor": round(candidate_wg_us / baseline_wg_us, 1),
  "measurement_scope": "per_layer_wall_time",

  # From static analysis
  "static_analysis": static,

  # From runtime capture (if available)
  "runtime_capture": {},

  # Fields for BoltBeam classifier
  "scratch_bytes": None,
  "barrier_count": None,
  "expected_barrier_count": None,
  "static_inst_count": None,
  "math_op_count": None,
  "vgpr_count": None,
  "sgpr_count": None,
  "warp_count": static["WARPS"],
  "expected_warp_count": 4,  # G=4 baseline

  "notes": (
    "scratch_bytes/vgpr/sgpr/inst_count from runtime ELF descriptor capture. "
    "If capture succeeded, filled below. Barrier count estimated from DSL (1 per NB block). "
    "2090us/27us ratio is per-layer wall time (40 layers x kernel_time = total step contribution). "
    "warp_count=5 (WARPS=G=5, THREADS=160=5x32). expected_warp_count=4 (G=4 for 8B)."
  ),
}

if block_tile_key and block_tile_key in captured:
  rd = captured[block_tile_key]
  vgpr = decode_rsrc1_vgpr(rd["rsrc1"])
  sgpr = decode_rsrc1_sgpr(rd["rsrc1"])
  artifact["scratch_bytes"] = rd["private_segment_size"]
  artifact["vgpr_count"] = vgpr
  artifact["sgpr_count"] = sgpr
  artifact["runtime_capture"] = {k: captured[k] for k in captured}
  print(f"  scratch_bytes={rd['private_segment_size']}, vgpr={vgpr}, sgpr={sgpr}")
  print(f"  LDS (group_segment)={rd['group_segment_size']}B")

  # Set barrier_count from static analysis (we know 1 barrier per NB block)
  artifact["barrier_count"] = static["estimated_barriers_per_wg_L128"]
  artifact["expected_barrier_count"] = static["NB_at_L16"]  # baseline uses smaller L

  # REGISTER SPILL CHECK
  if rd["private_segment_size"] > 0:
    print(f"  *** REGISTER SPILL DETECTED: {rd['private_segment_size']} bytes scratch per thread ***")
    artifact["pathology_hypothesis"] = "REGISTER_SPILL"
    artifact["pathology_evidence"] = f"private_segment_size={rd['private_segment_size']} > 0; kernel spills to global scratch"
  else:
    print(f"  No register spill (private_segment_size=0)")
    artifact["pathology_hypothesis"] = "UNKNOWN"
    artifact["pathology_evidence"] = f"No register spill; barrier_count={artifact['barrier_count']} (not flood); need ISA disasm"
else:
  print("  Runtime capture not available, using static analysis only")
  artifact["barrier_count"] = static["estimated_barriers_per_wg_L128"]
  artifact["expected_barrier_count"] = static["NB_at_L16"]
  artifact["pathology_hypothesis"] = "NATIVE_ISA_ORACLE_NEEDED"
  artifact["pathology_evidence"] = "Runtime ELF capture did not fire; need model run to get private_segment_size"

# Save artifact
out_dir = Path("bench/g5-block-tile")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "compiler_pathology_v1.json"
with open(out_path, "w") as f:
  json.dump(artifact, f, indent=2)

print(f"\nWrote: {out_path}")
print(f"Pathology hypothesis: {artifact.get('pathology_hypothesis', 'unknown')}")
print(f"scratch_bytes: {artifact['scratch_bytes']}")
print(f"vgpr_count: {artifact['vgpr_count']}")
print(f"barrier_count: {artifact['barrier_count']}")
