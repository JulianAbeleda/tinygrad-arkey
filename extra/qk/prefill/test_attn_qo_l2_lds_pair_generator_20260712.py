from extra.qk.prefill.attn_qo_l2_lds_pair_generator_20260712 import generate_pair
from extra.qk.runtime_specs import (GFX1100_REGISTER_RESIDENT_CAPABILITY, GFX1100_TWO_BUFFER_STAGE1_CAPABILITY,
                                    admit_full_kernel_candidate_set, full_kernel_candidate_set_from_legacy)

def test_attn_qo_pair_is_exact_and_transport_distinct_cpu_only():
  pair = generate_pair()
  assert pair["schedule_digest"] in pair["pair_key"]
  assert pair["candidates"]["direct_l2"]["active_lds_bytes"] == 0
  assert pair["candidates"]["lds"]["active_lds_bytes"] == 40960  # two-buffer WMMA-LDS
  rows = pair["candidates"]
  # The two-buffer LDS candidate admits GFX1100_TWO_BUFFER_STAGE1_CAPABILITY only
  # through the candidate-set path; direct_l2 resolves its register capability.
  admitted = tuple(admit_full_kernel_candidate_set(
    full_kernel_candidate_set_from_legacy(rows[n]["payload"], rows[n]["canonical_identity"])).admissions[0]
    for n in ("direct_l2", "lds"))
  assert len(admitted) == 2
  assert admitted[0].capability.capability_id == GFX1100_REGISTER_RESIDENT_CAPABILITY.capability_id
  assert admitted[1].capability.capability_id == GFX1100_TWO_BUFFER_STAGE1_CAPABILITY.capability_id
  assert rows["direct_l2"]["canonical_identity"] != rows["lds"]["canonical_identity"]
  assert rows["direct_l2"]["payload"]["schedule"]["residency"]["resident"][-1] == "stage_ab_register"
  assert "stage_ab_register" not in rows["lds"]["payload"]["schedule"]["residency"]["resident"]
