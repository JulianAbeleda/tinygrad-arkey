from extra.qk.prefill.attn_qo_l2_lds_pair_generator_20260712 import generate_pair
from extra.qk.runtime_specs import admit_full_kernel_candidate

def test_attn_qo_pair_is_exact_and_transport_distinct_cpu_only():
  pair = generate_pair()
  assert pair["schedule_digest"] in pair["pair_key"]
  assert pair["candidates"]["direct_l2"]["active_lds_bytes"] == 0
  assert pair["candidates"]["lds"]["active_lds_bytes"] > 0
  rows = pair["candidates"]
  admitted = tuple(admit_full_kernel_candidate(rows[n]["payload"], rows[n]["canonical_identity"],
    profile="qwen3_8b_q4k_m_gfx1100", role="attn_qo", shape=(512, 4096, 4096),
    target={"backend":"AMD", "arch":"gfx1100", "wave_size":32}) for n in ("direct_l2", "lds"))
  assert len(admitted) == 2
  assert rows["direct_l2"]["canonical_identity"] != rows["lds"]["canonical_identity"]
  assert rows["direct_l2"]["payload"]["schedule"]["residency"]["resident"][-1] == "stage_ab_register"
  assert "stage_ab_register" not in rows["lds"]["payload"]["schedule"]["residency"]["resident"]
