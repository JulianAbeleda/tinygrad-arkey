from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_exact_role_spec import ExactRoleSpec
from extra.qk import mmq_frozen_epoch_static_certificate as static


ROLE = ExactRoleSpec("unit_static", 128, 128, 256, "a" * 64)


def _resource(**changes):
  row = {
    "schema": "tinygrad.amd.native_program_resources.v1",
    "target": "gfx1100",
    "vgpr": 256,
    "sgpr": 32,
    "allocated_vgpr": 256,
    "allocated_sgpr": None,
    "used_vgpr": 255,
    "lds_bytes": 57856,
    "scratch_bytes": 0,
    "vgpr_spills": 0,
    "sgpr_spills": 0,
    "wavefront_size": 32,
    "workgroup_threads": 256,
    "max_workgroup_threads": 256,
    "global_size": [1, 1, 1],
    "local_size": [256, 1, 1],
  }
  return row | changes


@pytest.fixture
def authorities(monkeypatch):
  sink = UOp(Ops.SINK)
  program = UOp(Ops.PROGRAM, src=(sink,))
  provenance = {"schema": static.PROVENANCE_SCHEMA, "source_revision": "b" * 40}
  manifest = {
    "schema": static.FROZEN_SCHEMA,
    "c1_certification": {"certified": True},
    "generation_provenance": provenance,
    "content_address": "sha256:" + "c" * 64,
  }
  artifact = SimpleNamespace(manifest=manifest, programs=(program,), sinks=(sink,))
  binding = SimpleNamespace(
    artifact=artifact, role_spec=ROLE, program_keys=(program.key.hex(),), family_identity="c" * 64)
  seen = {"require_c1": None, "resources": 0, "c3": 0}

  def load(role_spec, bundle, **kwargs):
    assert role_spec == ROLE and bundle == "/bundle"
    seen["require_c1"] = kwargs.get("require_c1")
    return binding

  def resources(loaded, *, target):
    assert loaded is program and target == static.AMD_ISA_TARGET
    seen["resources"] += 1
    return _resource()

  def c3(role_spec, sinks, programs):
    assert role_spec == ROLE and sinks == (sink,) and programs == (program,)
    seen["c3"] += 1
    return {"schema": "c3", "state": "PASS", "cpu_only": True, "certificate_sha256": "d" * 64}

  monkeypatch.setattr(static, "load_frozen_epoch_program_set_binding", load)
  monkeypatch.setattr(static, "amd_native_program_resources", resources)
  monkeypatch.setattr(static, "certify_frozen_epoch_program_family", c3)
  return seen


def test_composes_strict_c1_all_program_c2_and_c3_into_deterministic_report(authorities):
  first = static.certify_frozen_epoch_static(ROLE, "/bundle")
  second = static.certify_frozen_epoch_static(ROLE, "/bundle")
  assert authorities == {"require_c1": True, "resources": 2, "c3": 2}
  assert first == second and first["state"] == "PASS" and first["cpu_only"] is True
  assert first["gates"] == {"C1": "PASS", "C2": "PASS", "C3": "PASS"}
  assert first["c2"]["constraints"] == {
    "target": "AMD:ISA:gfx1100",
    "max_vgpr_per_thread": 256,
    "max_lds_bytes": 65536,
    "expected_lds_bytes": 57856,
    "allow_scratch": False,
    "allow_spills": False,
    "wavefront_size": 32,
    "global_size": [1, 1, 1],
    "local_size": [256, 1, 1],
  }
  body = {key: value for key, value in first.items() if key != "certificate_sha256"}
  expected = hashlib.sha256(
    (json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()).hexdigest()
  assert first["certificate_sha256"] == expected
  assert static.static_certificate_json(first) == static.static_certificate_json(second)


@pytest.mark.parametrize(("change", "match"), (
  ({"allocated_vgpr": 257}, "VGPR"),
  ({"scratch_bytes": 4}, "scratch_bytes"),
  ({"vgpr_spills": 1}, "vgpr_spills"),
  ({"lds_bytes": 65536}, "LDS"),
  ({"global_size": [2, 1, 1]}, "launch geometry"),
  ({"wavefront_size": 64}, "wavefront"),
))
def test_c2_fails_closed_on_resource_or_launch_limit_drift(monkeypatch, authorities, change, match):
  monkeypatch.setattr(static, "amd_native_program_resources", lambda *_args, **_kwargs: _resource(**change))
  with pytest.raises(ValueError, match=match):
    static.certify_frozen_epoch_static(ROLE, "/bundle")


def test_rejects_noncanonical_target_before_loading(authorities):
  with pytest.raises(ValueError, match="target drift"):
    static.certify_frozen_epoch_static(ROLE, "/bundle", target="AMD:ISA:gfx1200")
  assert authorities == {"require_c1": None, "resources": 0, "c3": 0}
