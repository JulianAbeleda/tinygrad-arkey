import numpy as np
import inspect

import extra.qk.mmq_coop_tile_harness as harness
from extra.qk.mmq_coop_tile_harness import validate_bounded_coop_tile
from extra.qk.mmq_q4k_q8_atom import q8_1_mmq_ds4_from_row_major
from extra.qk.mmq_q4k_q8_reference import Q4KQ81MMQTileSpec
from extra.qk.mmq_owner_coverage import structural_static_store_only_owner_map


def test_bounded_coop_harness_owner_geometry_is_complete():
  spec = Q4KQ81MMQTileSpec("ffn_gate_up", 16, 16, 256, 0, 0, 16, 16)
  stores = structural_static_store_only_owner_map(spec)
  assert len(stores) == 256 and len({(s.m, s.n) for s in stores}) == 256


def test_bounded_coop_harness_is_fail_closed_without_amd():
  raw = np.zeros((16, 1, 144), dtype=np.uint8)
  ds4 = q8_1_mmq_ds4_from_row_major(np.zeros((16, 256), np.int8), np.ones((16, 8), np.float32))
  evidence = validate_bounded_coop_tile(raw, ds4, timeout_seconds=5, compare_direct=False)
  assert evidence["passed"] is False or evidence["dispatch_state"] == "completed"
  assert evidence["owner_coverage"]["complete"] is True
  assert evidence["reference_authority"] == {"kind": "canonical_cpu", "compared": False}
  assert evidence["gpu_kernel_authority"]["claimable"] is False
  assert evidence["gpu_kernel_authority"]["dispatch_performed"] is False


def test_harness_builds_emitted_program_in_child_not_cpu_atom():
  source = inspect.getsource(harness._build_emitted_amd_bundle)
  assert "compile_mmq_program" in source
  assert "build_tinygrad_bundle" in source
  assert "run_q4k_q8_1_mmq_bounded_amd_ds4_coop_tile" not in source
