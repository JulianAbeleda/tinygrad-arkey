#!/usr/bin/env python3
"""BubbleBeam -> FutureSight -> BoltBeam campaign: ffn_gate_up Q4_K m=512 n=12288 k=4096 prefill, METAL.

Real Qwen3-8B ffn_gate_up shape (not the wiring-check shape). Reuses
scratchpad/wire_candidate_context_verify.py's candidate/geometry construction,
scaled to a coupled-row search driven by BubbleBeam's propose_legal_dimensions
and FutureSight's classify_coupled_rows/apply_coupled_row, materialized through
BoltBeam's instantiate_candidate_rows. No production code is modified; nothing
is added to PACKED_WMMA_ROUTES.

Two phases:
  1. COMPILE-ONLY pass over every FutureSight-legal coupled row (cheap: no GPU
     execution, no numpy oracle) -- records reached-precontract vs declined vs
     BLOCKED/REJECTED with reason for every row.
  2. CHECK+MEASURE pass (real GPU timing, canonical_packed_reference oracle)
     over the rows that reached the precontract kernel, plus one
     tinygrad_heuristic.v1 control row with no candidate_geometry sidecar.
     Top-2 finalists + control get 2 repeat measurement runs each (mr9-style
     stability/win-fraction decision, thresholds mirrored from
     boltbeam/search/mr9_semantic_search.py::_decision).
"""
import hashlib, json, statistics, sys, time, traceback

sys.path.insert(0, "/Users/julianabeleda/env/BoltBeam")
sys.path.insert(0, ".")

from tinygrad.device import Device
from extra.llm_research.search_provider import MetalAdapter, ProtocolError, _metal_facts
from extra.llm_research.bubblebeam_futuresight import (
  LegalDimensionProposal, propose_legal_dimensions, dimension_mapping, classify_coupled_rows, apply_coupled_row,
)
from boltbeam.search.spec import FullKernelCandidate
from boltbeam.search.full_kernel_candidates import instantiate_candidate_rows

# ---------------------------------------------------------------------------
# Live target facts (M4, queried, not assumed)
# ---------------------------------------------------------------------------
device = Device["METAL"]
live_facts = _metal_facts(device)
arch = live_facts["architecture"]
print("=== live Metal facts ===")
print(json.dumps({k: v for k, v in live_facts.items() if k != "replay"}, indent=2))
assert live_facts["max_threads_per_threadgroup"] == 1024
assert live_facts["max_threadgroup_memory_bytes"] == 32768
assert live_facts["subgroup_width"] == 32

M, N, K = 512, 12288, 4096  # real Qwen3-8B ffn_gate_up prefill shape
TC_DIMS = (8, 8, 8)  # Metal tc.metal descriptor, tinygrad/codegen/opt/tc.py:181

TARGET = {"target_id": "apple_m4_ffn_gate_up_campaign", "backend": "METAL", "arch": arch,
          "subgroup_size": 32, "resolved_target_hash": "0" * 64}
MODEL_SHA = hashlib.sha256(b"qwen3_8b_ffn_gate_up_campaign").hexdigest()

# ---------------------------------------------------------------------------
# Seed v2 candidate (BoltBeam-strict schema; instantiate_candidate_rows requires it)
# ---------------------------------------------------------------------------
SEED_PAYLOAD = {
  "schema_version": "boltbeam.full_kernel_candidate.v2",
  "workload": {
    "profile": "qwen3_8b.ffn_gate_up", "model_sha256": MODEL_SHA, "phase": "prefill", "role": "ffn_gate_up",
    "operation": "matmul", "shape": {"m": M, "n": N, "k": K},
    "operands": {
      "a": {"dtype": "half", "layout": "row_major", "quantization": "none"},
      "b": {"dtype": "uint32", "layout": "packed_q4_k", "quantization": "Q4_K"},
      "c": {"dtype": "float", "layout": "row_major", "quantization": "none"},
    },
    "accumulator_dtype": "float", "target": TARGET,
  },
  "schedule": {
    "plan_kind": "tinygrad_heuristic.v1", "transforms": [],
    "tile": {"m": 256, "n": 64, "k": 32}, "launch": {"threads": 256},
    "mapping": {"lane_policy": "precontract_wmma"},
    "memory": {"a": {"space": "global", "vector_width": 1, "alignment": 1},
               "b": {"space": "global", "vector_width": 1, "alignment": 1},
               "c": {"space": "global", "vector_width": 1, "alignment": 1}},
    "pipeline": {"stage_count": 1},
    "compute": {"family": "wmma"}, "numerical_mode": "default",
  },
  "static_constraints": {"max_local_memory_bytes": None, "max_registers_per_thread": None, "spill_policy": "unknown"},
  "correctness": {"oracle": "canonical_packed_reference", "atol": 1e-3, "rtol": 1e-3},
  "memory_budget": {"status": "unavailable", "bytes": None},
  "provenance": {"generator_id": "bubblebeam_boltbeam_campaign", "generator_revision": "1",
                 "schema_revision": "boltbeam.full_kernel_candidate.v2"},
  "applicability": {"exact_shape": True, "profiles": ["ffn_gate_up"], "roles": ["ffn_gate_up"], "targets": [f"METAL:{arch}"]},
}
seed = FullKernelCandidate(SEED_PAYLOAD)
print("\nseed candidate_hash:", seed.candidate_hash)

# ---------------------------------------------------------------------------
# Step 1: BubbleBeam -- propose legal dimension values from declared target facts
# ---------------------------------------------------------------------------
workload_facts = {"shape": {"m": M, "n": N, "k": K}}
target_facts = {
  "max_threads_per_threadgroup": 1024, "max_threadgroup_memory_bytes": 32768,
  "compiler_transforms": (), "supported_plan_kinds": ("tinygrad_heuristic.v1", "tinygrad_opt_sequence.v1"),
  "generic_control_plan_kind": "tinygrad_heuristic.v1",
}

tm_candidates = (8, 16, 32, 64, 128, 256, 512, 100, 300)  # includes deliberately-illegal probes (non-divisors)
tn_candidates = (8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 100, 5000)
tk_candidates = (8, 16, 24, 32, 40, 64, 128, 256, 4096)
thread_candidates = (32, 64, 128, 256, 512, 1024, 2048)

bb_proposals = propose_legal_dimensions(workload_facts, target_facts, {
  "schedule.tile.m": tm_candidates, "schedule.tile.n": tn_candidates, "schedule.tile.k": tk_candidates,
  "schedule.launch.threads": thread_candidates, "schedule.plan_kind": ("tinygrad_heuristic.v1", "tinygrad_opt_sequence.v1", "bogus_plan"),
})
bb_map = dimension_mapping(bb_proposals)
print("\n=== BubbleBeam propose_legal_dimensions ===")
for row in bb_proposals:
  rejected = None
  if row.path == "schedule.tile.m": rejected = sorted(set(tm_candidates) - set(row.values))
  elif row.path == "schedule.tile.n": rejected = sorted(set(tn_candidates) - set(row.values))
  elif row.path == "schedule.tile.k": rejected = sorted(set(tk_candidates) - set(row.values))
  elif row.path == "schedule.launch.threads": rejected = sorted(set(thread_candidates) - set(row.values))
  print(f"  {row.path}: legal={row.values}" + (f"  rejected(FutureSight-static)={rejected}" if rejected else ""))

legal_tm = set(bb_map["schedule.tile.m"])
legal_tn = set(bb_map["schedule.tile.n"])
legal_tk_static = set(bb_map["schedule.tile.k"])  # only checks positivity here; WMMA-specific tk legality is layered below

# ---------------------------------------------------------------------------
# Step 2: WMMA-precontract-specific geometry legality (NOT expressible in the
# generic candidate schema -- wm/wn/bc ride the candidate_geometry sidecar and
# so cannot be checked by FutureSight's build_static_legality, which only ever
# sees the canonical candidate). Derived directly from
# tinygrad/codegen/opt/kernel_lds.py (derive_precontract_shape_factors /
# derive_precontract_factors) and tinygrad/codegen/opt/kernel_pipeline.py
# (KernelStage1PipelinePlan.buffer_count in {1,2}) plus the task's declared
# bc*(tm+tn)*80 <= 32768 / wm*wn <= 32 formulas.
# ---------------------------------------------------------------------------
STRIDE = 80  # hardcoded in tinygrad/llm/packed_wmma_prefill.py:_candidate_context, independent of tk
MAX_LDS = 32768


def geometry_legal(tm, tn, tk, wm, wn, bc):
  reasons = []
  if wm * wn > 32: reasons.append("wm*wn>32 (threads=wm*wn*32 > max_threads_per_threadgroup)")
  if bc not in (1, 2): reasons.append("buffer_count must be 1 or 2 (KernelStage1PipelinePlan)")
  if tm % (wm * TC_DIMS[1]): reasons.append("tm not divisible by wm*tc.dims[1]")
  if tn % (wn * TC_DIMS[0]): reasons.append("tn not divisible by wn*tc.dims[0]")
  if tk % TC_DIMS[2]: reasons.append("tk not divisible by tc.dims[2]")
  ks = tk // TC_DIMS[2] if TC_DIMS[2] else 0
  if ks < 2: reasons.append("tk//8 < 2 (atomic staging needs >=2 K steps)")
  if STRIDE < tk * 2: reasons.append(f"fixed LDS row stride {STRIDE}B < tk*half_itemsize (tk>40 not representable)")
  threads = wm * wn * 32
  vectors_per_row = tk * 2 // 16
  if vectors_per_row <= 0 or (tk * 2) % 16: reasons.append("K row is not whole b128 vectors")
  elif (tm * vectors_per_row) % threads or (tn * vectors_per_row) % threads:
    reasons.append("operand vectors do not divide evenly across cooperative threads")
  if bc * (tm + tn) * STRIDE > MAX_LDS: reasons.append(f"bc*(tm+tn)*80={bc*(tm+tn)*STRIDE} > {MAX_LDS}")
  if M % tm: reasons.append("tm does not divide M")
  if N % tn: reasons.append("tn does not divide N")
  if K % tk: reasons.append("tk does not divide K")
  return reasons


wave_pairs = []
for wm in (1, 2, 4, 8, 16, 32):
  for wn in (1, 2, 4, 8, 16, 32):
    if wm * wn <= 32: wave_pairs.append((wm, wn))

geometry_rows, geometry_rejections = [], []
for wm, wn in wave_pairs:
  for bc in (1, 2, 3):  # bc=3 deliberately included to record the buffer_count rejection
    for tm_mult in (1, 2, 3, 4):
      tm = wm * TC_DIMS[1] * tm_mult
      if tm not in legal_tm: continue
      for tn_mult in (1, 2, 3, 4):
        tn = wn * TC_DIMS[0] * tn_mult
        if tn not in legal_tn: continue
        for tk in (16, 32):
          reasons = geometry_legal(tm, tn, tk, wm, wn, bc)
          row = {"tm": tm, "tn": tn, "tk": tk, "wm": wm, "wn": wn, "bc": bc, "threads": wm * wn * 32}
          if reasons: geometry_rejections.append({**row, "reasons": reasons})
          else: geometry_rows.append(row)

print(f"\n=== WMMA sidecar geometry legality ===")
print(f"legal geometries: {len(geometry_rows)}  rejected geometries: {len(geometry_rejections)}")
bc3 = [r for r in geometry_rejections if r["bc"] == 3]
print(f"  (of which {len(bc3)} rejected solely for bc=3 buffer_count constraint)")
from collections import Counter
reason_hist = Counter(r for row in geometry_rejections if row["bc"] != 3 for r in row["reasons"])
print("  rejection reason histogram (bc in {1,2} only):")
for reason, count in reason_hist.most_common(): print(f"    {count:4d}  {reason}")

# ---------------------------------------------------------------------------
# Step 3: FutureSight -- classify coupled rows (generic candidate legality:
# tile divides shape, threads <= live limit) via classify_coupled_rows/apply_coupled_row
# ---------------------------------------------------------------------------
def _coupled_row(g):
  return {"schedule.tile.m": g["tm"], "schedule.tile.n": g["tn"], "schedule.tile.k": g["tk"],
          "schedule.launch.threads": g["threads"]}

# Multiple distinct (wm, wn, bc) geometries can legally share one (tm, tn, tk, threads)
# tuple (e.g. wm=1,wn=8 and wm=2,wn=4 both tile 32x64 at 256 threads) -- and since
# candidate_hash is a function of the schema-visible tile/threads only, they also share
# one candidate_hash. Classify one row at a time (never bulk-classify-then-reindex by
# position) so each ORIGINAL geometry stays paired with its own legality verdict, with
# no risk of a dict-keyed lookup silently collapsing distinct (wm, wn, bc) sidecars that
# happen to render the same tile/thread key.
fs_legal_geoms, fs_rejected = [], []
for g in geometry_rows:
  row = _coupled_row(g)
  legal, rejected = classify_coupled_rows(SEED_PAYLOAD, [row], workload_facts, target_facts)
  if legal: fs_legal_geoms.append(g)
  else: fs_rejected.append({"geometry": g, **rejected[0]})
print(f"\n=== FutureSight classify_coupled_rows (per-geometry-row) ===")
print(f"FutureSight-legal rows: {len(fs_legal_geoms)}  FutureSight-rejected rows: {len(fs_rejected)}")
for r in fs_rejected[:10]: print("  REJECTED:", r)
fs_legal = [_coupled_row(g) for g in fs_legal_geoms]

# ---------------------------------------------------------------------------
# Step 4: BoltBeam -- deterministic finite expansion, capped at max_candidates=256
# ---------------------------------------------------------------------------
MAX_CANDIDATES = 256
if len(fs_legal) > MAX_CANDIDATES:
  raise SystemExit(f"FutureSight-legal row count {len(fs_legal)} exceeds max_candidates={MAX_CANDIDATES}; trim before instantiating")
candidates = instantiate_candidate_rows(seed, fs_legal, max_candidates=MAX_CANDIDATES)
print(f"\n=== BoltBeam instantiate_candidate_rows ===")
print(f"instantiated candidates: {len(candidates)} (max_candidates={MAX_CANDIDATES}) "
      f"-- deduped from {len(fs_legal)} rows because candidate_hash is geometry-blind "
      f"(wm/wn/bc live only in the sidecar, not the hashed schema)")

# instantiate_candidate_rows dedups on candidate_hash, which is geometry-blind: several
# distinct (wm, wn, bc) sidecars can legally share one (tm, tn, tk, threads) candidate
# JSON (see the classify loop above -- fs_legal/fs_legal_geoms stay index-aligned 1:1,
# by construction, since each was classified and appended together). Recompute each
# row's candidate_hash directly and look it up in the deduped `candidates` set so every
# surviving physical (candidate, geometry) pair is measured, none silently dropped.
hash_to_candidate = {c.candidate_hash: c for c in candidates}
candidate_geometry_pairs = []
seen_hash_geometry = set()
assert len(fs_legal) == len(fs_legal_geoms)
for row, g in zip(fs_legal, fs_legal_geoms):
  cand_payload = apply_coupled_row(SEED_PAYLOAD, row)
  cand_hash = FullKernelCandidate(cand_payload).candidate_hash
  dedup_key = (cand_hash, g["wm"], g["wn"], g["bc"])
  if dedup_key in seen_hash_geometry: continue  # true exact duplicate row, not a hash collision across geometries
  seen_hash_geometry.add(dedup_key)
  candidate_geometry_pairs.append((hash_to_candidate[cand_hash], {"wm": g["wm"], "wn": g["wn"], "bc": g["bc"]}))
print(f"(candidate, geometry) pairs to run: {len(candidate_geometry_pairs)}")
dup_hashes = len(fs_legal) - len({p[0].candidate_hash for p in candidate_geometry_pairs})
print(f"distinct candidate_hash values among pairs: {len({p[0].candidate_hash for p in candidate_geometry_pairs})} "
      f"({dup_hashes} geometry rows share a hash with another distinct (wm,wn,bc) geometry)")

with open("/Users/julianabeleda/env/tinygrad-arkey-exp/scratchpad/_campaign_state.json", "w") as f:
  json.dump({
    "geometry_rows": geometry_rows, "geometry_rejections": geometry_rejections,
    "fs_rejected": fs_rejected,
    "bb_proposals": {row.path: list(row.values) for row in bb_proposals},
    "candidates": [{"candidate": c.to_dict(), "candidate_hash": c.candidate_hash, "geometry": g}
                    for c, g in candidate_geometry_pairs],
    "seed_hash": seed.candidate_hash, "target": TARGET, "model_sha256": MODEL_SHA,
  }, f, indent=2)
print("wrote scratchpad/_campaign_state.json")
print(f"\nTOTAL population for phase 2 (compile-only sweep): {len(candidate_geometry_pairs)}")
