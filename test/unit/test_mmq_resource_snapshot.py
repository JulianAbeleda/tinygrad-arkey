import json

import pytest

from extra.qk.mmq_resource_snapshot import (
  SCHEMA, build_kernel_resource_trace_bundle, validate_kernel_resource_trace_bundle,
)


def test_build_kernel_resource_trace_bundle_packages_known_candidate_resources():
  source_sha = "1" * 64
  binary_sha = "a" * 64

  bundle = build_kernel_resource_trace_bundle(
    candidate_id="amd_ds4_lds_skeleton",
    kernel_name="q4k_q8_1_mmq_ds4_lds_skeleton_atom",
    source_sha256=source_sha,
    binary_sha256=binary_sha,
    vgpr=72,
    sgpr=48,
    lds_bytes=256,
    scratch_bytes=0,
    workgroup=(256, 1, 1),
    grid=[4, 2, 1],
    occupancy=0.5,
  )

  assert bundle == {
    "schema": SCHEMA,
    "candidate_id": "amd_ds4_lds_skeleton",
    "kernel_name": "q4k_q8_1_mmq_ds4_lds_skeleton_atom",
    "source_sha256": source_sha,
    "binary_sha256": binary_sha,
    "resources": {
      "vgpr": 72,
      "sgpr": 48,
      "lds_bytes": 256,
      "scratch_bytes": 0,
      "workgroup": [256, 1, 1],
      "grid": [4, 2, 1],
      "occupancy": 0.5,
    },
  }
  assert json.loads(json.dumps(bundle))["resources"]["workgroup"] == [256, 1, 1]
  assert validate_kernel_resource_trace_bundle(bundle)["resources"]["scratch_bytes"] == 0


def test_build_kernel_resource_trace_bundle_omits_unknown_values_instead_of_zeroing():
  bundle = build_kernel_resource_trace_bundle(
    candidate_id="llama_mmq_r4_store_only_owner_map_probe",
    kernel_name="research_only_store_owner_map_static",
    vgpr=None,
    sgpr=None,
    lds_bytes=0,
    scratch_bytes=None,
  )

  assert bundle["resources"] == {"lds_bytes": 0}
  assert "vgpr" not in bundle["resources"]
  assert "sgpr" not in bundle["resources"]
  assert "scratch_bytes" not in bundle["resources"]
  assert "source_sha256" not in bundle
  assert "binary_sha256" not in bundle


def test_build_kernel_resource_trace_bundle_accepts_exact_amdgpu_metadata():
  bundle = build_kernel_resource_trace_bundle(candidate_id="c", kernel_name="k", vgpr=27, sgpr=29, lds_bytes=256,
    scratch_bytes=0, vgpr_spills=0, sgpr_spills=0, workgroup_threads=32, max_workgroup_threads=32,
    wavefront_size=32, dynamic_stack=False)
  assert validate_kernel_resource_trace_bundle(bundle)["resources"]["dynamic_stack"] is False


def test_build_kernel_resource_trace_bundle_omits_empty_resources_mapping():
  bundle = build_kernel_resource_trace_bundle(candidate_id="c0", kernel_name="kernel")

  assert bundle == {"schema": SCHEMA, "candidate_id": "c0", "kernel_name": "kernel"}


@pytest.mark.parametrize(
  ("kwargs", "message"),
  [
    ({"candidate_id": ""}, "candidate_id must be a non-empty string"),
    ({"kernel_name": ""}, "kernel_name must be a non-empty string"),
    ({"source_sha256": "A" * 64}, "source_sha256 must be a lowercase hex sha256 string"),
    ({"vgpr": -1}, "resources.vgpr must be a non-negative integer"),
    ({"sgpr": True}, "resources.sgpr must be a non-negative integer"),
    ({"workgroup": []}, "resources.workgroup must not be empty"),
    ({"grid": [1, 0, 1]}, r"resources.grid\[1\] must be a positive integer"),
    ({"occupancy": -0.1}, "resources.occupancy must be a non-negative number"),
  ],
)
def test_build_kernel_resource_trace_bundle_rejects_invalid_synthetic_data(kwargs, message):
  args = {"candidate_id": "c0", "kernel_name": "kernel", **kwargs}

  with pytest.raises(ValueError, match=message):
    build_kernel_resource_trace_bundle(**args)


def test_validate_kernel_resource_trace_bundle_rejects_wrong_schema_and_unknown_resource_fields():
  with pytest.raises(ValueError, match="schema must be"):
    validate_kernel_resource_trace_bundle({"schema": "other", "candidate_id": "c0", "kernel_name": "kernel"})

  with pytest.raises(ValueError, match="unknown fields"):
    validate_kernel_resource_trace_bundle({
      "schema": SCHEMA,
      "candidate_id": "c0",
      "kernel_name": "kernel",
      "resources": {"vgpr": 1, "waves": 2},
    })
