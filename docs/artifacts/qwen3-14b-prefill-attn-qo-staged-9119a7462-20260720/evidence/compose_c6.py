import json, sys
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")
from extra.qk.mmq_attn_qo_c6_binding import compose_attn_qo_c6_binding, read_json
from extra.qk.mmq_frozen_staged_family import load_frozen_staged_family_manifest
from extra.qk.mmq_exact_role_spec import exact_role_spec

OUT = "/tmp/qk-attn-qo-9119a7462-20260720"
BUNDLE = "/home/ubuntu/tinygrad-arkey/docs/artifacts/qwen3-14b-prefill-attn-qo-staged-951d3615c-20260719/bundle"
FAM = "/home/ubuntu/tinygrad-arkey/docs/artifacts/qwen3-14b-prefill-attn-qo-staged-951d3615c-20260719/evidence/qk-attn-qo-staged-951d3615c-final-r1-20260719-family.json"

role = exact_role_spec("attn_qo")
family = load_frozen_staged_family_manifest(FAM, role_spec=role, frozen_bundle=BUNDLE)

raw_c6_by_queue = {
  "PM4": read_json(f"{OUT}/c6-pm4-full20.json", "PM4 C6"),
  "AQL": read_json(f"{OUT}/c6-aql-full20.json", "AQL C6"),
}
c7_memory_ledger = read_json(f"{OUT}/c7-ledger.json", "C7 ledger")
c7_authority_snapshot = read_json(f"{OUT}/c7-authority-snapshot.json", "C7 authority")
c7_captures_by_queue = {
  "PM4": read_json(f"{OUT}/c7-capture-pm4.json", "PM4 C7 capture"),
  "AQL": read_json(f"{OUT}/c7-capture-aql.json", "AQL C7 capture"),
}

composition = compose_attn_qo_c6_binding(
  family=family, raw_c6_by_queue=raw_c6_by_queue,
  c7_memory_ledger=c7_memory_ledger,
  c7_authority_snapshot=c7_authority_snapshot,
  c7_captures_by_queue=c7_captures_by_queue,
)

with open(f"{OUT}/c6-composition.json", "w") as f:
  json.dump(composition, f, indent=2, sort_keys=True, allow_nan=False)
  f.write("\n")

print("status:", composition["status"])
print("evidence_identity:", composition["evidence_identity"])
print("family_identity:", composition["family_identity"])
