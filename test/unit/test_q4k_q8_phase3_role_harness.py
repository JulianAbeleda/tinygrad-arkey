import json

from extra.qk.prefill.q4k_q8_phase3_role_harness import DEFAULT_INVENTORY, _request, _select


def test_phase3_requests_use_each_admitted_candidate_transport(tmp_path):
  inventory = json.loads(open(DEFAULT_INVENTORY).read())
  physical, _, direct = _select(inventory, "attn_kv", (512, 1024, 5120))
  common = dict(comparator_id="other", input_npz=tmp_path / "input.npz", workload_digest="w",
                session_id="s", timeout_ms=1, warmups=0, rounds=1)
  candidate = _request(candidate_id="candidate", adapter_id="five", entry=physical,
                       input_format="fp32_activation", **common)
  comparator = _request(candidate_id="comparator", adapter_id="direct", entry=direct, **common)
  assert candidate.transport_plan.transport == "direct_global"
  assert comparator.transport_plan.transport == "lds"
