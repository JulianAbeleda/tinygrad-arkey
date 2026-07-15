import pytest

from extra.qk.q6k_coop_tile_probe import Q6KCoopTileProbe, run_q6k_coop_tile_probe


def test_q6_coop_probe_is_default_off_and_has_resource_evidence():
  report = run_q6k_coop_tile_probe()
  assert report["default_off"] is True
  assert report["production_dispatch_changed"] is False
  assert report["resources"]["status"] == "MODEL_ONLY"
  assert report["resources"]["scratch_bytes"] == "UNKNOWN"
  assert report["blocker"]


def test_q6_coop_bounded_staged_reference_is_finite():
  report = run_q6k_coop_tile_probe(run_correctness=True)
  assert report["correctness"]["status"] == "PASS"
  assert report["correctness"]["finite"] is True
  assert report["correctness"]["full_output"] is True


def test_q6_coop_enabled_mode_fails_closed():
  with pytest.raises(ValueError, match="no lowered emitter"):
    Q6KCoopTileProbe(enabled=True).evidence()


def test_q6_coop_rejects_unbounded_geometry():
  with pytest.raises(ValueError, match="bounded"):
    Q6KCoopTileProbe(tile_m=32).validate()
