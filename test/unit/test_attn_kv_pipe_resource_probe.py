import json

from extra.qk.prefill.attn_kv_pipe_resource_probe import build_report


def test_attn_kv_pipe_resource_probe_reproduces_overflow_and_ranks_candidates():
  report = build_report()

  assert report["schema"] == "attn-kv-pipe-resource-probe.v1"
  assert report["role"] == "attn_kv"
  assert report["shape"] == {"m": 512, "n": 1024, "k": 4096}
  assert report["active_route_changed"] is False
  assert report["failure_source_analysis"]["inferred_prefill_role"] == "attn_kv"
  assert report["failure_source_analysis"]["shared_bytes"] == 69632
  assert report["failure_source_analysis"]["shared_over_limit"] is True

  baseline = report["baseline"]
  assert baseline["shared_bytes"] == 69632
  assert baseline["fits_lds"] is False

  rows = {row["name"]: row for row in report["candidates"]}
  assert rows["disable_attn_kv_local_staging"]["fits_lds"] is True
  assert rows["disable_attn_kv_local_staging"]["preserves_pipe_route"] is True
  assert rows["retile_n_1024_to_512"]["fits_lds"] is True
  assert rows["retile_n_1024_to_512"]["route_change"] == "tile_shape"
  assert rows["byte_budgeted_local_staging"]["fits_lds"] is True
  assert rows["byte_budgeted_local_staging"]["preserves_pipe_route"] is True
  assert report["next_primitive_fix"] == "disable_attn_kv_local_staging"
  assert "byte_budgeted_local_staging" in report["legal_candidate_names"]
  json.dumps(report, allow_nan=False)
