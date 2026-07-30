import hashlib, json
from pathlib import Path
import pytest

from extra.llm_research import route_manifest

EXPORT = Path(route_manifest.__file__).parent / "generated/boltbeam_route_manifest.v1.json"
EXP_ROUTE_SNAPSHOT = frozenset((
  "decode_flash_block_tile_g5_konly", "decode_flash_live_split_g4_kvboth", "decode_flash_live_split_g5_kvboth",
  "decode_q4k_g3_generated", "decode_q6k_coop_generated", "packed_wmma_prefill_generated",
  "prefill_flash_attention_generated", "prefill_q4k_direct_tile4x4_default", "prefill_q4k_int8_wmma_generated_research",
  "prefill_q4k_int8_wmma_tiled_research", "prefill_q4k_packed_ds4_research", "prefill_q4k_packed_fused_research",
  "prefill_q4k_packed_row_major_research", "prefill_q4k_q8_1_hybrid_mmq_atom", "prefill_q4k_reduce_out_research",
  "prefill_q6k_direct_generated", "prefill_v2_scheduler_matmul_default", "prefill_wmma_lds_dbuf_generated",
))
BOLT_HISTORY_SNAPSHOT = frozenset((
  "decode_attention_native_correct_not_fast", "decode_attention_owned_two_kernel", "decode_flash_block_tile_g5_8b_refuted",
  "decode_flash_live_split_g4_8b_kvboth", "decode_q4k_owned_warp", "decode_q6k_coop_shipped", "decode_q6k_direct_refuted",
  "prefill_pipe_global_rollback", "prefill_pipe_role_selective_default", "prefill_pipe_role_selective_generated",
))

def _semantic_snapshot(routes):
  # Guards are intentionally an EXP execution-fact overlay, so selection-policy
  # parity excludes only the dynamically derived guard field.
  return {rid:{k:v for k,v in row.items() if k != "shape_guards"} for rid,row in routes.items()}

def test_export_has_all_exp_routes_and_bolt_history():
  source = json.loads(EXPORT.read_text())
  assert EXP_ROUTE_SNAPSHOT <= set(source["routes"])
  assert BOLT_HISTORY_SNAPSHOT <= set(source["routes"])
  assert len(EXP_ROUTE_SNAPSHOT) == 18 and len(BOLT_HISTORY_SNAPSHOT) == 10
  assert set(source["routes"]) <= set(route_manifest.ROUTES)
  assert {"decode_q4k_owned_warp", "decode_q6k_direct_refuted", "prefill_pipe_global_rollback"} <= set(source["routes"])
  assert {"decode_flash_live_split_g4_kvboth", "packed_wmma_prefill_generated"} <= set(source["routes"])
  assert _semantic_snapshot(route_manifest.ROUTES) == _semantic_snapshot(source["routes"])

def test_export_hash_failure_is_fail_closed(tmp_path, monkeypatch):
  bad = tmp_path / "route.json"; bad.write_text(EXPORT.read_text() + " ")
  bad.with_suffix(".json.sha256").write_text(EXPORT.with_suffix(".json.sha256").read_text())
  monkeypatch.setattr(route_manifest, "_EXPORT", bad)
  with pytest.raises(RuntimeError, match="hash mismatch"):
    route_manifest._load_export()

def test_missing_export_is_fail_closed(tmp_path, monkeypatch):
  monkeypatch.setattr(route_manifest, "_EXPORT", tmp_path / "missing.json")
  with pytest.raises(RuntimeError, match="unavailable"):
    route_manifest._load_export()

def test_stale_version_is_fail_closed(tmp_path, monkeypatch):
  stale = json.loads(EXPORT.read_text()); stale["version"] = 0
  raw = (json.dumps(stale, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
  path = tmp_path / "route.json"; path.write_bytes(raw)
  path.with_suffix(".json.sha256").write_text(hashlib.sha256(raw).hexdigest())
  monkeypatch.setattr(route_manifest, "_EXPORT", path)
  with pytest.raises(RuntimeError, match="schema mismatch"):
    route_manifest._load_export()

def test_no_production_cross_repo_imports():
  production = Path(route_manifest.__file__).read_text()
  assert "import boltbeam" not in production and "from boltbeam" not in production
  assert not (Path(route_manifest.__file__).parent / "route_manifest.json").exists()
  source_root = Path(route_manifest.__file__).resolve().parents[2] / "tinygrad"
  assert all("extra.llm_research" not in path.read_text() for path in source_root.rglob("*.py"))

def test_export_sidecar_matches_content():
  assert hashlib.sha256(EXPORT.read_bytes()).hexdigest() == EXPORT.with_suffix(".json.sha256").read_text().split()[0]

def test_generated_export_has_no_drift_from_boltbeam_authority():
  # Test-only byte comparison; production deliberately has no sibling-repo import.
  source = Path(route_manifest.__file__).resolve().parents[3] / "BoltBeam/boltbeam/policy/assets/route_manifest.v1.json"
  assert EXPORT.read_bytes() == source.read_bytes()
