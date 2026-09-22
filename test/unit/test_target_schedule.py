"""T2: one derive authority for the typed prefill schedule."""
import dataclasses
import json
from pathlib import Path

import pytest

from extra.llm_research.runtime_specs import (
  GFX1100_Q4K_Q8_FIVE_BUFFER_CAPABILITY, GFX1100_REGISTER_RESIDENT_CAPABILITY,
  GFX1100_TWO_BUFFER_STAGE1_CAPABILITY, METAL_M4_10C_TWO_BUFFER_STAGE1_CAPABILITY,
  NV_SM120_TWO_BUFFER_STAGE1_CAPABILITY, _CAPABILITY_ROWS,
)
from extra.llm_research.target_contracts import TARGET_SCHEDULE_CONTRACTS
from extra.llm_research.target_schedule import derive_target_schedule

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMOTED_SET = _REPO_ROOT / "tinygrad" / "llm" / "generated" / "prefill_wmma_lds_dbuf_candidate_set.json"


def _seed_geometry() -> dict:
  return {"tile": {"m": 128, "n": 128, "k": 32}, "waves": {"m": 4, "n": 2},
          "buffer_count": 2, "stage_count": 1}


def _seed_shape() -> dict:
  return {"m": 512, "n": 12288, "k": 4096,
          "dtypes": {"a": "fp16", "b": "fp16", "c": "fp16", "accumulator": "fp32"}}


def _canonical_json(value) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def test_amd_derive_is_byte_identical_to_promoted_template():
  template = json.loads(_PROMOTED_SET.read_text())["template"]
  out = derive_target_schedule(GFX1100_TWO_BUFFER_STAGE1_CAPABILITY, _seed_geometry(), _seed_shape())
  assert _canonical_json(out["schedule"]) == _canonical_json(template["schedule"])
  assert _canonical_json(out["static_constraints"]) == _canonical_json(template["static_constraints"])


def test_amd_derive_at_single_buffer_geometry_matches_the_lane_dispatch():
  geometry = dict(_seed_geometry(), buffer_count=1)
  out = derive_target_schedule(GFX1100_TWO_BUFFER_STAGE1_CAPABILITY, geometry, _seed_shape())
  assert out["schedule"]["threads"] == 256
  assert out["schedule"]["lds"]["windows"] == {"a": [0, 10240], "b": [10240, 20480]}
  assert out["schedule"]["lds"]["strides"] == {"a": 80, "b": 80}
  assert out["schedule"]["pipeline"]["buffer_count"] == 1
  assert out["static_constraints"]["max_lds_bytes"] == 65536


def test_nv_derive_emits_typed_schedule_without_amd_vocabulary():
  out = derive_target_schedule(NV_SM120_TWO_BUFFER_STAGE1_CAPABILITY, _seed_geometry(), _seed_shape())
  wmma = out["schedule"]["wmma"]
  assert wmma["instruction_family"] == NV_SM120_TWO_BUFFER_STAGE1_CAPABILITY.instruction_family
  assert wmma["fragment_layout"] == "cuda_mma_f32_8x16x16_f16_lds2_static"
  assert out["schedule"]["lane_ownership"] == "cuda_mma_f32_8x16x16_f16_lds2_static"
  assert out["static_constraints"]["max_lds_bytes"] == 49152
  assert out["schedule"]["dependency_policy"]["waitcnt"] == {"vm": None, "lgkm": None}
  text = _canonical_json(out)
  assert "rdna3" not in text and "gfx1100" not in text and "65536" not in text


def test_metal_derive_emits_typed_schedule_without_amd_vocabulary():
  out = derive_target_schedule(METAL_M4_10C_TWO_BUFFER_STAGE1_CAPABILITY, _seed_geometry(), _seed_shape())
  wmma = out["schedule"]["wmma"]
  assert wmma["instruction_family"] == METAL_M4_10C_TWO_BUFFER_STAGE1_CAPABILITY.instruction_family
  assert wmma["fragment_layout"] == "metal_simdgroup_matrix_f32_8x8x8_f16_lds2_static"
  assert out["static_constraints"]["max_lds_bytes"] == 32768
  text = _canonical_json(out)
  assert "rdna3" not in text
  assert out["schedule"]["dependency_policy"]["waitcnt"]["lgkm"] is None


def test_derive_never_fabricates_a_wmma_prefix():
  # The family string is vocabulary read from the row, never assembled from the tc descriptor.
  non_wmma = dataclasses.replace(NV_SM120_TWO_BUFFER_STAGE1_CAPABILITY,
                                 instruction_family="simdgroup_matrix_f32_8x8x8_f16")
  out = derive_target_schedule(non_wmma, _seed_geometry(), _seed_shape())
  assert out["schedule"]["wmma"]["instruction_family"] == "simdgroup_matrix_f32_8x8x8_f16"
  assert out["schedule"]["wmma"]["instruction_family"].startswith("simdgroup_matrix")


def test_derive_rejects_non_lds_transport_rows():
  for row in (GFX1100_REGISTER_RESIDENT_CAPABILITY, GFX1100_Q4K_Q8_FIVE_BUFFER_CAPABILITY):
    with pytest.raises(ValueError, match="LDS buffer family"):
      derive_target_schedule(row, _seed_geometry(), _seed_shape())


@pytest.mark.parametrize("mutation", (
  {"geometry": {"tile": {"m": 128, "n": 128}, "waves": {"m": 4, "n": 2}, "buffer_count": 2, "stage_count": 1}},
  {"geometry": {"tile": {"m": 128, "n": 128, "k": 0}, "waves": {"m": 4, "n": 2}, "buffer_count": 2, "stage_count": 1}},
  {"geometry": {"tile": {"m": 128, "n": 128, "k": 32}, "waves": {"m": 4, "n": 2}, "buffer_count": 2}},
  {"shape": {"m": 512, "n": 12288, "k": 4096}},
  {"shape": {"m": 512, "n": 12288, "k": 4096, "dtypes": {"a": "fp32", "b": "fp16", "c": "fp16", "accumulator": "fp32"}}},
))
def test_derive_validates_geometry_and_shape(mutation):
  geometry, shape = _seed_geometry(), _seed_shape()
  if "geometry" in mutation: geometry = mutation["geometry"]
  else: shape = mutation["shape"]
  with pytest.raises(ValueError):
    derive_target_schedule(GFX1100_TWO_BUFFER_STAGE1_CAPABILITY, geometry, shape)


_ROW_ATTR = {
  "lane_ownership": "lane_ownership",
  "cooperative_load.lane_mapping": "cooperative_lane_mapping",
  "lds.banks": "lds_banks",
  "lds.padding": "lds_padding",
  "pipeline.epoch_graph": "epoch_graph",
  "wmma.fragment_layout": "fragment_layout",
  "wmma.accumulator_ownership": "accumulator_ownership",
  "dependency_policy.waitcnt": "waitcnt",
  "dependency_policy.barriers": "dependency_barriers",
  "epilogue.lane_mapping": "epilogue_lane_mapping",
  "epilogue.vector_width": "epilogue_vector_width",
  "residency.preload": "residency_preload",
  "residency.resident": "residency_resident",
  "residency.reuse": "residency_reuse",
  "numerical_mode": "numerical_mode",
  "static_constraints.max_vgpr_per_thread": "max_vgpr_per_thread",
  "static_constraints.allow_spill": "allow_spill",
}


def _row_value_canonical(row, path: str):
  value = getattr(row, _ROW_ATTR[path])
  if path == "pipeline.epoch_graph":
    return [dict((k, list(v) if isinstance(v, tuple) else v) for k, v in value)]
  if path in ("dependency_policy.waitcnt", "residency.reuse"):
    return dict(value)
  return list(value) if isinstance(value, tuple) else value


def test_lds_capability_rows_agree_with_the_contract_table():
  for (backend, arch), contracts in TARGET_SCHEDULE_CONTRACTS.items():
    rows = _CAPABILITY_ROWS.get((backend, arch), {})
    assert rows, f"contract table target {backend}:{arch} has no capability rows"
    for shape, row in rows.items():
      if row.transport != "lds":
        continue  # register/direct-global vocabulary is its own emitter authority
      for path, contract in contracts.items():
        assert _row_value_canonical(row, path) == contract.value, (
          f"{backend}:{arch} {shape} row {path} drifted from the contract table")


def test_contract_table_pins_are_the_rows_the_derive_reads():
  # The derive output for every declared target must match the table's declared values
  # field for field -- the table is the census, the row is the runtime authority.
  for (backend, arch), contracts in TARGET_SCHEDULE_CONTRACTS.items():
    row = _CAPABILITY_ROWS[(backend, arch)]["single_buffer"]
    out = derive_target_schedule(row, _seed_geometry(), _seed_shape())
    for path, contract in contracts.items():
      if contract.status != "declared":
        continue
      if path == "cooperative_load.lane_mapping":
        assert all(out["schedule"]["cooperative_load"][operand]["lane_mapping"] == contract.value
                   for operand in ("a", "b"))
        continue
      parts = path.split(".")
      root = "static_constraints" if parts[0] == "static_constraints" else "schedule"
      value = out[root]
      for part in (parts[1:] if root == "static_constraints" else parts):
        value = value[part]
      assert value == contract.value, f"{backend}:{arch} derived {path} != declared {contract.value!r}"
