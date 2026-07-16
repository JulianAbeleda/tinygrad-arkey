import copy, json
from pathlib import Path

import pytest

from extra.qk.prefill.q4k_q8_five_buffer_role_gate import admitted_q4k_non_fitting_roles, build_role_gate


def _inventory():
  return json.loads(Path("bench/prefill-pure-full-kernel/qwen3-14b-mixed-quant-candidate-inventory-v1.json").read_text())


def _fake_compiler(payload, identity):
  from extra.qk.prefill.q4k_q8_five_buffer_compile_adapter import admit_q4k_q8_five_buffer_compile
  admission = admit_q4k_q8_five_buffer_compile(payload, identity)
  class Arg: pass
  class Kernel: pass
  class Program: pass
  arg, kernel, program = Arg(), Kernel(), Program()
  arg.candidate_context = admission.context; kernel.arg = arg; program.src = [kernel]
  evidence = {"passed": True, "canonical_identity": identity, "abi_digest": "a" * 64,
    "abi": {"argument_order": ["output", "q4_packed_words", "q8_ds4_values", "q8_scales", "q8_weighted_sums"]},
    "resource_summary": {"lds_bytes": 0, "scratch_bytes": 0, "vgpr_spills": 0, "sgpr_spills": 0,
      "vgpr": 32, "sgpr": 16, "workgroup": [32, 1, 1], "workgroup_threads": 32,
      "grid": [2, 3, 1], "wavefront_size": 32}}
  return program, evidence


def test_enumerates_all_admitted_real_q4k_roles_deterministically():
  obligations = admitted_q4k_non_fitting_roles(_inventory())
  got = [(x.payload["workload"]["role"], tuple(x.payload["workload"]["shape"][k] for k in ("m", "n", "k")))
         for x, _ in obligations]
  assert got == [("attn_kv", (512, 1024, 5120)), ("attn_qo", (512, 5120, 5120)),
                 ("ffn_down", (512, 5120, 17408)), ("ffn_gate_up", (512, 17408, 5120))]


def test_gate_exposes_identity_launch_and_resource_evidence_without_allocating():
  report = build_role_gate(_inventory(), compiler=_fake_compiler)
  assert report["passed"] and report["status"] == "pass" and report["role_count"] == 4
  assert [row["role"] for row in report["rows"]] == ["attn_kv", "attn_qo", "ffn_down", "ffn_gate_up"]
  for row in report["rows"]:
    assert row["compile_status"] == "pass" and row["canonical_identity"] == row["context_identity"]
    assert len(row["abi_identity"]) == 64 and row["workgroup"] == [32, 1, 1] and row["grid"] == [2, 3, 1]
    assert {x: row["resources"][x] for x in ("lds_bytes", "scratch_bytes", "vgpr_spills", "sgpr_spills")} == \
           {"lds_bytes": 0, "scratch_bytes": 0, "vgpr_spills": 0, "sgpr_spills": 0}


def test_compiler_blocker_is_distinct_from_gate_contract_failure():
  calls = 0
  def compiler(payload, identity):
    nonlocal calls
    calls += 1
    if calls == 1: raise RuntimeError("static lowering is not implemented")
    program, evidence = _fake_compiler(payload, identity)
    if calls == 2: evidence["resource_summary"]["scratch_bytes"] = 4
    return program, evidence
  report = build_role_gate(_inventory(), compiler=compiler)
  assert not report["passed"] and [x["blocker_kind"] for x in report["blockers"]] == ["compiler", "gate_contract"]
  assert report["blockers"][0]["compile_status"] == "blocked"
  assert report["blockers"][1]["compile_status"] == "fail"
  assert len(report["rows"]) == 4  # one blocker never truncates remaining role evidence


@pytest.mark.parametrize("mutation", ["missing_binding", "identity_drift"])
def test_inventory_mismatch_fails_closed_before_compilation(mutation):
  artifact = copy.deepcopy(_inventory())
  if mutation == "missing_binding": artifact["bindings"] = artifact["bindings"][1:]
  else: artifact["bindings"][0]["inventory_key"]["role"] = "made_up_role"
  with pytest.raises(ValueError, match="five-buffer role gate"):
    build_role_gate(artifact, compiler=lambda *_: pytest.fail("compiler must not run"))
