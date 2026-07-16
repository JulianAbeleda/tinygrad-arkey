import copy
import json

import pytest

from extra.qk.runtime_specs import (
  FULL_KERNEL_CANDIDATE_SCHEMA, GFX1100_Q4K_Q8_COOPERATIVE_CAPABILITY,
  GFX1100_Q4K_Q8_FIVE_BUFFER_CAPABILITY, Q4KQ8CooperativePlan,
  Q4K_Q8_1_COOPERATIVE_SCHEDULE_FAMILY, Q4K_Q8_1_DIRECT_SCHEDULE_FAMILY,
  FullKernelAdmissionError, admit_full_kernel_candidate, derive_q4k_q8_1_cooperative_five_buffer_candidate,
  derive_q4k_q8_1_five_buffer_candidate, full_kernel_candidate_capability, q4k_q8_1_five_buffer_abi_plan,
)


def _payload(shape=(129, 257, 512), role="arbitrary_role", profile="arbitrary_profile"):
  m, n, k = shape
  return {"schema_version":FULL_KERNEL_CANDIDATE_SCHEMA,
    "workload":{"profile":profile,"role":role,"shape":{"m":m,"n":n,"k":k},
      "dtypes":{"a":"fp16","b":"fp16","c":"fp16","accumulator":"fp32"},
      "layout":{"a":"row_major","b":"transposed_row_major","c":"row_major"},
      "target":{"backend":"AMD","arch":"gfx1100","wave_size":32}},
    "schedule":{}, "static_constraints":{},
    "applicability":{"exact_shape":True,"profiles":[profile],"roles":[role],"targets":["AMD:gfx1100:wave32"]}}


def _admit(entry):
  workload = entry.payload["workload"]
  shape = tuple(workload["shape"][x] for x in ("m", "n", "k"))
  return admit_full_kernel_candidate(entry.payload, entry.canonical_identity, profile=workload["profile"],
    role=workload["role"], shape=shape, target=dict(workload["target"]))


def test_cooperative_descriptor_admits_from_schedule_family_and_derives_ceil_grid():
  entry = derive_q4k_q8_1_cooperative_five_buffer_candidate(_payload())
  schedule = entry.payload["schedule"]
  assert schedule["family"] == Q4K_Q8_1_COOPERATIVE_SCHEDULE_FAMILY
  assert (schedule["tile"], schedule["waves"], schedule["wave_size"], schedule["threads"]) == \
         ({"m":128,"n":128,"k":32}, {"m":4,"n":2}, 32, 256)
  assert schedule["lds"] == {"active_bytes":12288,"max_bytes":65536,"slots":1,"pipeline":"sequential"}
  assert schedule["wmma"]["signed"] is True and schedule["wmma"]["input_dtype"] == "int8"
  assert full_kernel_candidate_capability(entry.payload) is GFX1100_Q4K_Q8_COOPERATIVE_CAPABILITY
  admission = _admit(entry)
  assert admission.plan == admission.pipeline_plan == admission.context.pipeline == \
         Q4KQ8CooperativePlan(outer_grid=(3, 2, 1))
  assert admission.active_lds_bytes == 12288
  assert 0 < admission.active_lds_bytes <= 65536


def test_cooperative_preserves_exact_five_buffer_abi_but_has_distinct_canonical_identity():
  source = _payload((256, 256, 512))
  cooperative = derive_q4k_q8_1_cooperative_five_buffer_candidate(source)
  direct = derive_q4k_q8_1_five_buffer_candidate(source)
  assert json.loads(json.dumps(cooperative.payload["kernel_abi"])) == \
         json.loads(json.dumps(direct.payload["kernel_abi"])) == q4k_q8_1_five_buffer_abi_plan()
  assert cooperative.canonical_identity != direct.canonical_identity
  assert direct.payload["schedule"]["family"] == Q4K_Q8_1_DIRECT_SCHEDULE_FAMILY
  assert full_kernel_candidate_capability(direct.payload) is GFX1100_Q4K_Q8_FIVE_BUFFER_CAPABILITY
  assert _admit(direct).active_lds_bytes == 0


@pytest.mark.parametrize("path,value", (
  (("schedule","tile","m"),64), (("schedule","waves","n"),4),
  (("schedule","wave_size"),64), (("schedule","threads"),128),
  (("schedule","lds","slots"),2), (("schedule","lds","active_bytes"),0),
  (("schedule","lds","active_bytes"),65537), (("schedule","wmma","signed"),False),
  (("schedule","wmma","input_dtype"),"uint8"),
))
def test_cooperative_descriptor_rejects_schedule_drift(path, value):
  entry = derive_q4k_q8_1_cooperative_five_buffer_candidate(_payload())
  payload = entry.to_json()["payload"]
  cursor = payload
  for key in path[:-1]: cursor = cursor[key]
  cursor[path[-1]] = value
  with pytest.raises(FullKernelAdmissionError, match="payload_schema"):
    admit_full_kernel_candidate(payload, entry.canonical_identity, profile="arbitrary_profile", role="arbitrary_role",
      shape=(129,257,512), target={"backend":"AMD","arch":"gfx1100","wave_size":32})


def test_cooperative_admission_is_not_inferred_from_role_profile_or_shape():
  a = derive_q4k_q8_1_cooperative_five_buffer_candidate(_payload(role="lm_head", profile="model_a"))
  b = derive_q4k_q8_1_cooperative_five_buffer_candidate(_payload(role="unknown", profile="model_b"))
  assert _admit(a).capability is _admit(b).capability is GFX1100_Q4K_Q8_COOPERATIVE_CAPABILITY
  wrong_capability = copy.deepcopy(a.to_json()["payload"])
  with pytest.raises(FullKernelAdmissionError, match="typed LDS capability"):
    workload = wrong_capability["workload"]
    admit_full_kernel_candidate(wrong_capability, a.canonical_identity, profile=workload["profile"], role=workload["role"],
      shape=(129,257,512), target=workload["target"], capability=GFX1100_Q4K_Q8_FIVE_BUFFER_CAPABILITY)
