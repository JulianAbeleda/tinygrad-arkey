"""CPU-only contracts for the fork's BoltBeam loader.

BubbleBeam and FutureSight live in BoltBeam (boltbeam/search/bubblebeam.py,
boltbeam/search/futuresight.py; their tests are BoltBeam's). The fork keeps
no copy and no re-export module: research code imports BoltBeam's modules
after extra/llm_research/boltbeam_checkout.py finds the checkout. This file
checks that loader. The Q4_K lane map is covered by test_q4k_lane_map.py.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

import extra.llm_research.boltbeam_checkout as checkout

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_IMPORT = "from extra.llm_research.boltbeam_checkout import require_boltbeam; require_boltbeam('boltbeam.search.futuresight')"


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
  out = subprocess.run([sys.executable, "-c", RESEARCH_IMPORT], cwd=ROOT, env=env, capture_output=True, text=True)
  assert out.returncode != 0
  assert "ImportError: BoltBeam module(s) boltbeam.search.futuresight unavailable" in out.stderr
  assert f"BOLTBEAM_ROOT={tmp_path}" in out.stderr


def test_production_tinygrad_never_imports_the_loader():
  names = ("boltbeam_checkout", "flash_candidate_schema")
  offenders = [f"{path.relative_to(ROOT)} mentions {name}" for path in sorted((ROOT / "tinygrad").rglob("*.py"))
               for name in names if name in path.read_text(errors="ignore")]
  assert offenders == []
