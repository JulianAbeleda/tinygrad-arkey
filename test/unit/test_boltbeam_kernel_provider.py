import copy

import pytest

from extra.llm_research.boltbeam_kernel_provider import candidate_hash, generate_kernel, generate_route


def _candidate(family, kernel_id, shape, parameters, launch):
  return {"schema_version": "boltbeam.kernel_candidate.v1", "kernel_id": kernel_id, "family": family,
    "phase": "prefill", "role": "ffn_down", "operation": "fixture", "shape": shape,
    "abi": [{"name": "buffer", "access": "read", "dtype": "uint8", "layout": "fixture"}],
    "parameters": parameters, "launch": launch, "resources": {"workspace_bytes": 0, "static_shared_memory_bytes": 0},
    "target": {"target_id": "nvidia_sm120", "backend": "CUDA", "arch": "sm_120", "subgroup_size": 32,
               "resolved_target_hash": "a"*64},
    "correctness": {"oracle": "fixture", "atol": 0.0, "rtol": 0.0, "bit_exact": True},
    "provenance": {"generator_id": "fixture", "generator_revision": "fixture", "schema_revision": "boltbeam.kernel_candidate.v1"}}


def _route():
  main = _candidate("packed_q6_q8_streamk_main.v1", "main", {"m": 512, "n": 4096, "k": 12288},
    {"tile_m": 128, "tile_n": 128, "tile_k": 256, "streamk_owners": 170, "streamk_segment": 0,
     "streamk_segments_in_cta": True, "segment_order": "ascending", "prefetch_second_panel": True,
     "combined_initial_publish": True, "factor_dA": False, "oracle_publisher": True,
     "weight_scale_contract": "trusted_fp16_packed", "partial_output_layout": "destination_major"},
    {"grid": [170, 1, 1], "block": [256, 1, 1], "dynamic_shared_memory_bytes": 58368})
  fixup = _candidate("q6_destination_major_fixup.v1", "fixup", {"m": 512, "n": 4096, "contributors": 3},
    {"rows": 128, "cols": 128, "tiles_m": 4, "tiles_n": 32, "slices": 4, "threads": 128,
     "outputs_per_thread": 32, "fold_order": "slot0_slot1_slot2", "fadd": "rn"},
    {"grid": [128, 4, 1], "block": [128, 1, 1], "dynamic_shared_memory_bytes": 0})
  return {"schema_version": "boltbeam.kernel_route.v1", "route_id": "q6", "phase": "prefill", "role": "ffn_down",
    "components": [{"name": "main", "candidate_hash": candidate_hash(main), "candidate": main, "depends_on": []},
                   {"name": "fixup", "candidate_hash": candidate_hash(fixup), "candidate": fixup, "depends_on": ["main"]}],
    "outputs": ["fixup"], "provenance": {"generator_id": "fixture", "generator_revision": "fixture",
    "schema_revision": "boltbeam.kernel_route.v1"}}


def test_q6_route_generates_real_main_uops_and_real_fixup_source():
  generated = generate_route(_route())
  assert [x.kind for x in generated] == ["uop", "source"]
  assert generated[0].artifact.op.name == "SINK"
  assert "nv_q6_destination_major_fixup" in generated[1].artifact


def test_candidate_identity_and_family_contract_fail_closed():
  route = _route(); main = route["components"][0]
  with pytest.raises(ValueError, match="hash mismatch"): generate_kernel(main["candidate"], "0"*64)
  changed = copy.deepcopy(main["candidate"]); changed["parameters"]["streamk_owners"] = 169
  with pytest.raises(ValueError, match="admitted contract"): generate_kernel(changed)
  changed = copy.deepcopy(main["candidate"]); changed["family"] = "unregistered.v1"
  with pytest.raises(ValueError, match="no tinygrad emitter"): generate_kernel(changed)


def test_route_rejects_a_hidden_or_reordered_dependency():
  route = _route(); route["components"][0]["depends_on"] = ["fixup"]
  with pytest.raises(ValueError, match="missing or forward"): generate_route(route)
