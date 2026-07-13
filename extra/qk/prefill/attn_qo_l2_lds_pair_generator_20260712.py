"""CPU-only paired candidate generator for the attn_qo prefill decision.

The semantic tile is immutable.  Only the transport (register-resident or
LDS) is selected when strict boltbeam payloads are materialized.  Nothing in
this module imports a device, renderer, compiler, or runtime dispatcher.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json
from types import MappingProxyType
from typing import Any

from extra.qk.runtime_specs import (FULL_KERNEL_CANDIDATE_SCHEMA, admit_full_kernel_candidate_set,
                                     full_kernel_candidate_set_from_legacy)

ROLE = "attn_qo"
SHAPE = (512, 4096, 4096)
PROFILE = "qwen3_8b_q4k_m_gfx1100"
TARGET = {"backend": "AMD", "arch": "gfx1100", "wave_size": 32}

def _freeze(x: Any) -> Any:
  if isinstance(x, dict): return MappingProxyType({k: _freeze(v) for k, v in x.items()})
  if isinstance(x, list): return tuple(_freeze(v) for v in x)
  return x

@dataclass(frozen=True)
class SemanticSchedule:
  tile: tuple[int, int, int] = (128, 128, 32)
  waves: tuple[int, int] = (4, 2)
  threads: int = 256
  lane_ownership: str = "rdna3_wmma_f32_16x16x16_f16_lds2_static"
  numerical_mode: str = "ieee_fp16_acc_fp32"

SEMANTIC_SCHEDULE = SemanticSchedule()

def _digest(schedule: SemanticSchedule) -> str:
  row = {"tile": list(schedule.tile), "waves": list(schedule.waves), "threads": schedule.threads,
         "lane_ownership": schedule.lane_ownership, "numerical_mode": schedule.numerical_mode}
  return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _payload(storage: str, schedule: SemanticSchedule = SEMANTIC_SCHEDULE) -> dict[str, Any]:
  register = storage == "direct_l2"
  return {"schema_version": FULL_KERNEL_CANDIDATE_SCHEMA,
    "workload": {"profile": PROFILE, "role": ROLE, "shape": dict(zip(("m","n","k"), SHAPE)),
      "dtypes": {"a":"fp16","b":"fp16","c":"fp16","accumulator":"fp32"},
      "layout": {"a":"row_major","b":"transposed_row_major","c":"row_major"}, "target": dict(TARGET)},
    "schedule": {"tile": dict(zip(("m","n","k"), schedule.tile)), "waves": dict(zip(("m","n"), schedule.waves)),
      "threads": schedule.threads, "lane_ownership": schedule.lane_ownership,
      "cooperative_load": {r:{"lane_mapping":"cooperative_row_stride_64_b128","vector_width":8,"alignment":16} for r in ("a","b")},
      "lds": {"windows":{"a":[0,10240],"b":[10240,20480]},"strides":{"a":80,"b":80},"padding":16,"banks":32,"store_vector_width":8,"load_vector_width":8},
      "pipeline": {"buffer_count":1 if register else 2,"stage_count":2 if register else 1,"epoch_graph":[{"epoch":"body","slot":0,"produce":["a","b"],"wait":["global"] if register else ["global","lds"],"barrier":"before_fragment_load","consume":["a","b"]}]},
      "wmma": {"instruction_family":"wmma_f32_16x16x16_f16","fragment_layout":"rdna3_wmma_f32_16x16x16_f16_register_static" if register else "rdna3_wmma_f32_16x16x16_f16_lds2_static","accumulator_ownership":"wmma_accum_wm_x_wn_8_vgprs"},
      "dependency_policy":{"waitcnt":{"vm":0,"lgkm":0},"barriers":["before_fragment_load","after_wmma_before_slot_reuse"]},
      "residency":{"preload":["a","b"],"resident":["accumulator","stage_ab_register"] if register else ["accumulator"],"reuse":{"a":4,"b":2}},
      "epilogue":{"lane_mapping":"wmma_accumulator_scalar_b16","vector_width":1},"numerical_mode":schedule.numerical_mode},
    "static_constraints":{"max_lds_bytes":65536,"max_vgpr_per_thread":256,"allow_spill":False},
    "applicability":{"exact_shape":True,"profiles":[PROFILE],"roles":[ROLE],"targets":["AMD:gfx1100:wave32"]}}

def generate_pair() -> dict[str, Any]:
  digest = _digest(SEMANTIC_SCHEDULE)
  payloads = {name: _payload(name) for name in ("direct_l2", "lds")}
  # Route each candidate through the candidate-SET admission: the two-buffer LDS
  # candidate admits GFX1100_TWO_BUFFER_STAGE1_CAPABILITY (only reachable via the
  # set path), while direct_l2 still resolves its register capability.
  admitted = tuple(admit_full_kernel_candidate_set(
    full_kernel_candidate_set_from_legacy(p, _identity(p))).admissions[0] for p in payloads.values())
  rows = {name: {"payload": payloads[name], "canonical_identity": _identity(payloads[name]),
                  "storage": name, "active_lds_bytes": admitted[i].active_lds_bytes}
          for i, name in enumerate(("direct_l2", "lds"))}
  return {"schema":"attn_qo_l2_lds_pair.v1", "pair_key":f"{ROLE}:{SHAPE}:{digest}",
          "schedule_digest":digest, "semantic_schedule":_freeze(SEMANTIC_SCHEDULE.__dict__), "candidates":rows}

def _identity(payload: dict[str, Any]) -> str:
  return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
