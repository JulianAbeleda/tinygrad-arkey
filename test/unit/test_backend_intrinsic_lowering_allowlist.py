import json

from extra.qk import backend_intrinsic_lowering_allowlist as allowlist


def test_build_output_has_backend_owned_and_route_local_rows():
  out = allowlist.build()
  assert out["schema"] == "backend_intrinsic_lowering_allowlist.v1"
  assert out["summary"]["backend_owned_count"] == 5
  assert out["summary"]["route_local_count"] == 1
  assert out["summary"]["row_count"] == len(out["rows"])
  assert out["summary"]["marker_conflict_row_count"] == 0
  assert out["marker_conflicts"] == {}
  assert json.dumps(out)  # sanity-check JSON-serializable content


def test_backend_owned_l5_entries_exist():
  rows = {r["name"]: r for r in allowlist.rows()}
  expected = {"wmma_mfma", "dot4", "v_dot2_fdot2", "cross_lane_reduction", "exp2_fast_math"}
  assert expected.issubset(rows.keys())

  assert rows["wmma_mfma"]["scope"] == "backend_owned"
  assert "Ops.WMMA" in rows["wmma_mfma"]["allow_markers"]
  assert rows["dot4"]["scope"] == "backend_owned"
  assert "__builtin_amdgcn_udot4" in rows["dot4"]["allow_markers"]
  assert rows["v_dot2_fdot2"]["scope"] == "backend_owned"
  assert "__builtin_amdgcn_fdot2" in rows["v_dot2_fdot2"]["allow_markers"]
  assert rows["cross_lane_reduction"]["scope"] == "backend_owned"
  assert "__builtin_amdgcn_ds_bpermute" in rows["cross_lane_reduction"]["allow_markers"]
  assert rows["exp2_fast_math"]["scope"] == "backend_owned"
  assert "__builtin_amdgcn_exp2f" in rows["exp2_fast_math"]["allow_markers"]


def test_route_local_markers_are_banned():
  raw = allowlist.row("route_local_raw_markers")
  for marker in ("asm volatile", "Ops.INS", "Ops.BINARY", "Tensor.custom_kernel", ".custom_kernel("):
    assert marker in raw["banned_markers"]
  assert raw["scope"] == "route_local"


def test_allow_and_banned_markers_are_disjoint_per_row():
  assert allowlist.marker_conflicts() == {}
  for row in allowlist.rows():
    overlap = set(row["allow_markers"]) & set(row["banned_markers"])
    assert overlap == set(), f"{row['name']} has contradictory markers: {sorted(overlap)}"
