"""Census pins for T1: the promoted AMD template is declared facts with citations."""
import json
from pathlib import Path

import pytest

from extra.llm_research.target_contracts import (
  CONTRACT_FIELD_KEYS, TARGET_SCHEDULE_CONTRACTS, contract_row,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMOTED_SET = _REPO_ROOT / "tinygrad" / "llm" / "generated" / "prefill_wmma_lds_dbuf_candidate_set.json"

_EMITTER_REFERENCES = ("cstyle.py", "cuda.py", "kernel_lds.py", "KernelStage1PipelinePlan", "s_waitcnt",
                       "M1b/M1c/M1d", "five-buffer row", "tc descriptor", "ISA", "promoted template",
                       "generator")


def _promoted_template() -> dict:
  candidate_set = json.loads(_PROMOTED_SET.read_text())
  return candidate_set["template"]


def _lookup_dotted(container: dict, path: str):
  value = container["static_constraints"] if path.startswith("static_constraints.") else container["schedule"]
  path = path.removeprefix("static_constraints.")
  if path == "cooperative_load.lane_mapping":
    return [value["cooperative_load"][operand]["lane_mapping"] for operand in ("a", "b")]
  for part in path.split("."):
    if isinstance(value, dict) and part in value:
      value = value[part]
    else:
      return None
  return value


def test_amd_contract_table_matches_promoted_template_field_for_field():
  template = _promoted_template()
  for path, contract in TARGET_SCHEDULE_CONTRACTS[("AMD", "gfx1100")].items():
    value = _lookup_dotted(template, path)
    if path == "cooperative_load.lane_mapping":
      assert value == [contract.value, contract.value], (
        f"AMD contract {path} drifted from the promoted template")
    else:
      assert value == contract.value, f"AMD contract {path} drifted from the promoted template"


def test_amd_contracts_are_fully_declared_with_citations():
  for path, contract in TARGET_SCHEDULE_CONTRACTS[("AMD", "gfx1100")].items():
    assert contract.status == "declared", f"AMD field {path} must be declared, not pending"
    assert any(ref in contract.citation for ref in _EMITTER_REFERENCES), (
      f"AMD field {path} citation must name an emitter/lowering, got {contract.citation!r}")


def test_every_target_row_covers_the_same_field_set():
  for (backend, arch), row in TARGET_SCHEDULE_CONTRACTS.items():
    assert set(row) == set(CONTRACT_FIELD_KEYS), (
      f"target {backend}:{arch} contract row must cover every C-class field")


def test_nv_and_metal_declare_only_measured_fields():
  for key in (("CUDA", "sm120"), ("Metal", "m4_10c")):
    row = TARGET_SCHEDULE_CONTRACTS[key]
    declared = {path for path, c in row.items() if c.status == "declared"}
    # Fragment layout + its LDS-family lane_ownership coincidence are renderer-proven;
    # Metal additionally proves the cooperative b128 lane mapping (M1b/M1c/M1d ran with it).
    assert declared == {"wmma.fragment_layout", "lane_ownership"} or \
      declared == {"wmma.fragment_layout", "lane_ownership", "cooperative_load.lane_mapping"}, (
      f"{key} declared set is {declared}")
    assert "wmma.fragment_layout" in declared and "lane_ownership" in declared
    # The AMD-only waitcnt vocabulary must never be carried as a value for NV/Metal.
    assert row["dependency_policy.waitcnt"].value == {"vm": None, "lgkm": None}
    assert row["dependency_policy.waitcnt"].status == "pending"
    # No rdna3_* vocabulary may be carried for NV/Metal.
    for path, contract in row.items():
      text = str(contract.value)
      assert "rdna3" not in text, f"{key} {path} carries AMD vocabulary {text!r}"


def test_contract_row_fails_closed_for_undeclared_target():
  with pytest.raises(ValueError, match="no declared schedule contracts"):
    contract_row("CUDA", "sm999")
  with pytest.raises(ValueError, match="no declared schedule contracts"):
    contract_row("AMD", "gfx1200")


def test_promoted_template_has_no_unaccounted_c_class_field():
  expected = {
    "lane_ownership", "cooperative_load.lane_mapping",
    "lds.banks", "lds.padding", "pipeline.epoch_graph",
    "wmma.fragment_layout", "wmma.accumulator_ownership",
    "dependency_policy.waitcnt", "dependency_policy.barriers",
    "epilogue.lane_mapping", "epilogue.vector_width",
    "residency.preload", "residency.resident", "residency.reuse",
    "numerical_mode", "static_constraints.max_vgpr_per_thread", "static_constraints.allow_spill",
  }
  assert set(CONTRACT_FIELD_KEYS) == expected
