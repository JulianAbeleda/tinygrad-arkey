"""CPU-only contracts for the fork side of BubbleBeam/FutureSight.

The policy tests, including the flash legality and priority tests, moved to
BoltBeam with the policy (BoltBeam tests/test_bubblebeam_futuresight.py). This
file checks what the fork still owns: the re-export shim and its BoltBeam
loader. The Q4_K lane map moved to test_q4k_lane_map.py with its module.
"""
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import extra.llm_research.boltbeam_checkout as checkout
import extra.llm_research.bubblebeam_futuresight as bf

ROOT = Path(__file__).resolve().parents[2]
BOLTBEAM_POLICY = {
  "boltbeam.search.bubblebeam": ("JSONValue", "LegalDimensionProposal", "ScheduleVocabulary", "dimension_mapping",
                                 "propose_legal_dimensions", "target_schedule_vocabulary"),
  "boltbeam.search.futuresight": ("CanonicalCandidate", "Legality", "Priority", "StaticAssessment", "StaticRejection",
                                  "apply_coupled_row", "build_flash_legality", "build_flash_static_priority",
                                  "build_static_legality", "build_static_priority", "candidate_report", "classify_candidates",
                                  "classify_coupled_rows", "rank_static_candidates"),
}
# A shim defines nothing. Every name it exports is BoltBeam's object, and the Q4_K lane map
# that used to be the exception now lives in extra/llm_research/q4k_lane_map.py.
FORK_OWNED: set[str] = set()


def test_policy_names_are_the_boltbeam_objects():
  for module, names in BOLTBEAM_POLICY.items():
    source = importlib.import_module(module)
    for name in names: assert getattr(bf, name) is getattr(source, name), f"bubblebeam_futuresight.{name} is not {module}.{name}"
  own = {name for name, value in vars(bf).items() if not name.startswith("_") and getattr(value, "__module__", None) == bf.__name__}
  assert own == FORK_OWNED, (
    f"the shim defines its own {sorted(own)}; a re-export module should define nothing")


def test_missing_boltbeam_names_every_place_tried(tmp_path, monkeypatch):
  monkeypatch.delenv("BOLTBEAM_ROOT", raising=False)
  monkeypatch.setitem(sys.modules, "boltbeam", None)  # no importable boltbeam
  monkeypatch.setattr(checkout, "SIBLING_ROOT", tmp_path / "BoltBeam")  # no sibling checkout
  with pytest.raises(ImportError) as info: checkout.require_boltbeam("boltbeam.search.bubblebeam")
  message = str(info.value)
  assert "boltbeam.search.bubblebeam unavailable: no place has a boltbeam package" in message
  assert "production tinygrad/ does not" in message
  for place in ("BOLTBEAM_ROOT (unset)", "importable boltbeam (none)", f"sibling checkout {tmp_path / 'BoltBeam'}"): assert place in message


def test_a_wrong_boltbeam_root_fails_the_research_import_loudly(tmp_path):
  env = {**os.environ, "PYTHONPATH": str(ROOT), "BOLTBEAM_ROOT": str(tmp_path)}
  out = subprocess.run([sys.executable, "-c", "import extra.llm_research.bubblebeam_futuresight"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
  assert out.returncode != 0
  assert "ImportError: BoltBeam module(s) boltbeam.search.bubblebeam, boltbeam.search.futuresight unavailable" in out.stderr
  assert f"BOLTBEAM_ROOT={tmp_path}" in out.stderr


def test_production_tinygrad_never_imports_the_shim_or_the_loader():
  names = ("bubblebeam_futuresight", "boltbeam_checkout", "flash_candidate_schema")
  offenders = [f"{path.relative_to(ROOT)} mentions {name}" for path in sorted((ROOT / "tinygrad").rglob("*.py"))
               for name in names if name in path.read_text(errors="ignore")]
  assert offenders == []


def test_module_does_not_own_candidate_schema_hash_or_population_expansion():
  assert not hasattr(bf, "StaticCandidate")
  assert not hasattr(bf, "CandidateDimension")
  assert not hasattr(bf, "enumerate_candidates")
  assert not hasattr(bf, "sha256")
  assert "boltbeam.full_kernel_candidate" not in bf.__doc__
