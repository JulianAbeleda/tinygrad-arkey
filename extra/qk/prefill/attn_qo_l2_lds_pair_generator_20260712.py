"""CPU-only paired candidate generator for the attn_qo prefill decision.

The mathematical workload is shared; work decomposition is storage-aware.
LDS receives a cooperative multi-wave tile, while register residency receives
a wave-owned tile.  Both are materialized from the same typed compiler policy
composition rather than redefining geometry in transport adapters.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json
from types import MappingProxyType
from typing import Any

from extra.qk.runtime_specs import (FULL_KERNEL_CANDIDATE_SCHEMA, admit_full_kernel_candidate_set,
                                     full_kernel_candidate_set_from_legacy)
from tinygrad.codegen.opt.compiler_policies import GEMMSchedulePolicy

ROLE = "attn_qo"
SHAPE = (512, 4096, 4096)
PROFILE = "qwen3_8b_q4k_m_gfx1100"
TARGET = {"backend": "AMD", "arch": "gfx1100", "wave_size": 32}

def _freeze(x: Any) -> Any:
  if isinstance(x, dict): return MappingProxyType({k: _freeze(v) for k, v in x.items()})
  if isinstance(x, list): return tuple(_freeze(v) for v in x)
  return x

@dataclass(frozen=True)
class SemanticProblem:
  operation: str = "a@b_transpose"
  input_dtype: str = "fp16"
  accumulator_dtype: str = "fp32"
  output_dtype: str = "fp16"
  numerical_mode: str = "ieee_fp16_acc_fp32"

SEMANTIC_PROBLEM = SemanticProblem()
SCHEDULE_POLICIES = {
  "direct_l2": GEMMSchedulePolicy.register_native(pipe_tm=2, pipe_tn=2, k_steps=2, wave_size=32),
  "lds": GEMMSchedulePolicy.lds_cooperative(tile=(128, 128, 32), waves=(4, 2), slot_bytes=20480,
                                             buffer_count=2, wave_size=32, reuse=(4, 2)),
}

def _digest(problem: SemanticProblem) -> str:
  row = dict(problem.__dict__)
  return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _payload(storage: str) -> dict[str, Any]:
  if storage not in SCHEDULE_POLICIES: raise ValueError(f"unknown attn_qo storage {storage!r}")
  policy = SCHEDULE_POLICIES[storage]
  workgroup, pipeline = policy.workgroup, policy.pipeline
  register = storage == "direct_l2"
  return {"schema_version": FULL_KERNEL_CANDIDATE_SCHEMA,
    "workload": {"profile": PROFILE, "role": ROLE, "shape": dict(zip(("m","n","k"), SHAPE)),
      "dtypes": {"a":"fp16","b":"fp16","c":"fp16","accumulator":"fp32"},
      "layout": {"a":"row_major","b":"transposed_row_major","c":"row_major"}, "target": dict(TARGET)},
    "schedule": {"tile": dict(zip(("m","n","k"), workgroup.tile)), "waves": dict(zip(("m","n"), workgroup.waves)),
      "threads": workgroup.threads,
      "lane_ownership":"rdna3_wmma_wave_private_2x2" if register else "rdna3_wmma_f32_16x16x16_f16_lds2_static",
      "cooperative_load": {r:{"lane_mapping":"wave_contiguous_b128" if register else "cooperative_row_stride_64_b128",
                               "vector_width":8,"alignment":16} for r in ("a","b")},
      "lds": {"windows":{"a":[0,10240],"b":[10240,20480]},"strides":{"a":80,"b":80},"padding":16,"banks":32,"store_vector_width":8,"load_vector_width":8},
      "pipeline": {"buffer_count":pipeline.storage.buffer_count,"stage_count":pipeline.logical_stage_count,
        "epoch_graph":[{"epoch":"body","slot":0,"produce":["a","b"],"wait":["global"] if register else ["global","lds"],
                        "barrier":"wave_dependency" if register else "before_fragment_load","consume":["a","b"]}]},
      "wmma": {"instruction_family":"wmma_f32_16x16x16_f16","fragment_layout":"rdna3_wmma_f32_16x16x16_f16_register_static" if register else "rdna3_wmma_f32_16x16x16_f16_lds2_static","accumulator_ownership":"wmma_accum_wm_x_wn_8_vgprs"},
      "dependency_policy":{"waitcnt":{"vm":0,"lgkm":0},"barriers":[] if register else ["before_fragment_load","after_wmma_before_slot_reuse"]},
      "residency":{"preload":["a","b"],"resident":["accumulator","stage_ab_register"] if register else ["accumulator"],
                   "reuse":{"a":workgroup.reuse[0],"b":workgroup.reuse[1]}},
      "epilogue":{"lane_mapping":"wmma_accumulator_scalar_b16","vector_width":1},"numerical_mode":SEMANTIC_PROBLEM.numerical_mode},
    "static_constraints":{"max_lds_bytes":65536,"max_vgpr_per_thread":256,"allow_spill":False},
    "applicability":{"exact_shape":True,"profiles":[PROFILE],"roles":[ROLE],"targets":["AMD:gfx1100:wave32"]}}

def generate_pair() -> dict[str, Any]:
  digest = _digest(SEMANTIC_PROBLEM)
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
          "schedule_digest":digest, "semantic_schedule":_freeze(SEMANTIC_PROBLEM.__dict__),
          "schedule_policies":_freeze({name:{"tile":list(policy.workgroup.tile),"waves":list(policy.workgroup.waves),
            "threads":policy.workgroup.threads,"ownership":policy.workgroup.ownership,
            "storage":policy.pipeline.storage_kind,"stages":policy.pipeline.logical_stage_count}
            for name,policy in SCHEDULE_POLICIES.items()}), "candidates":rows}

def _identity(payload: dict[str, Any]) -> str:
  return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
