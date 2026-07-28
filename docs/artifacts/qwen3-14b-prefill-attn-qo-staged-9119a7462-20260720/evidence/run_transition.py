import json, sys, argparse
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")
from extra.qk.mmq_frozen_staged_c8_sessions import run_guarded_persistent_c8_route_sequence
from extra.qk.mmq_attn_qo_c8_runtime import attn_qo_c8_runner_factory
from extra.qk.mmq_frozen_staged_family import load_frozen_staged_family_manifest
from extra.qk.mmq_exact_role_spec import exact_role_spec

OUT = "/tmp/qk-attn-qo-9119a7462-20260720"
BUNDLE = "/home/ubuntu/tinygrad-arkey/docs/artifacts/qwen3-14b-prefill-attn-qo-staged-951d3615c-20260719/bundle"
FAM = "/home/ubuntu/tinygrad-arkey/docs/artifacts/qwen3-14b-prefill-attn-qo-staged-951d3615c-20260719/evidence/qk-attn-qo-staged-951d3615c-final-r1-20260719-family.json"

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--queue-mode", required=True, choices=["PM4", "AQL"])
  ap.add_argument("--sequence", required=True, help="comma-separated: staged_candidate,direct_packed")
  ap.add_argument("--label", required=True)
  args = ap.parse_args()
  sequence = args.sequence.split(",")

  role = exact_role_spec("attn_qo")
  family = load_frozen_staged_family_manifest(FAM, role_spec=role, frozen_bundle=BUNDLE)

  with open(f"{OUT}/c6-composition.json") as f:
    composition = json.load(f)
  c6_correctness_evidence = composition["c6_correctness_evidence"]

  runner_config = json.load(open(f"{OUT}/runner-config.json"))

  result = run_guarded_persistent_c8_route_sequence(
    family=family,
    c6_correctness_evidence=c6_correctness_evidence,
    queue_mode=args.queue_mode,
    sequence=sequence,
    runner_factory=attn_qo_c8_runner_factory,
    runner_config=runner_config,
    timeout_seconds=300.0,
  )
  out_path = f"{OUT}/transition-{args.label}-{args.queue_mode.lower()}.json"
  with open(out_path, "w") as f:
    json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
    f.write("\n")
  print("status:", result["status"])
  print("exact_blocker:", result.get("exact_blocker"))
  print("health_before/after:", result.get("health_before"), result.get("health_after"))
  print("kernel_faults:", result.get("kernel_faults"))
  print("written:", out_path)

if __name__ == "__main__":
  main()
