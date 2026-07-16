from extra.qk.wave32_geometry_compile_sweep import TILES, WAVES, _one

def test_wave32_sweep_matrix_and_resource_contract():
  assert TILES == (16, 32, 64, 128) and WAVES == ((1, 1), (2, 2), (4, 2))
  row = _one(16, (1, 1))
  assert row["status"] == "FAIL" and row["lds_bytes"] == 2048
  assert row["failure_stage"] == "rewrite_or_lowering"
  assert row["register_evidence"]["status"] == "unavailable"
