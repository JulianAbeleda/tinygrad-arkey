"""Validate the current252 vocabulary-only bracket, including route and logits."""
import argparse, hashlib, json, pathlib, statistics
import numpy as np


def compare(root: pathlib.Path) -> dict:
  names = ("control-a", "candidate-b", "control-c")
  arms = {name: json.loads((root/f"{name}.json").read_text()) for name in names}
  logits = {name: np.load(root/f"{name}.npz") for name in names}
  # Renderer-emitted symbol, rather than the KernelProgram display name.
  vocab = "q6k_v_four_warp_fp16_direct_151936_4096"
  contracts = {}
  for name, arm in arms.items():
    programs = arm["program_names"]
    selected = programs.get(vocab, 0)
    generic = sum(count for symbol, count in programs.items() if symbol.startswith("r_1187_32_4_16_2_2_2_4_8_"))
    census = arm["census"]
    contracts[name] = {
      "arm_pass": arm["status"] == "PASS",
      "projection_mains": census["compiler_main_total"],
      "projection_producers": census["q8_producer_total"],
      "projection_weights": census["candidate_weight_args"],
      "canonical_weights": census["all_weights_canonical"],
      "remaining_overlays": census["remaining_v_down_fp16_overlays"],
      "vocabulary_calls": selected, "generic_vocabulary_calls": generic,
      "route_pass": (selected, generic) == ((1, 0) if name == "candidate-b" else (0, 1)),
    }
  quality = {}
  candidate = logits["candidate-b"]["logits"].astype(np.float32)
  for name in ("control-a", "control-c"):
    control = logits[name]["logits"].astype(np.float32)
    if control.shape != candidate.shape: raise ValueError("full logits shape changed")
    diff = np.abs(control-candidate)
    quality[name] = {"shape": list(candidate.shape), "finite": bool(np.isfinite(candidate).all() and np.isfinite(control).all()),
      "same_token": int(logits[name]["token"]) == int(logits["candidate-b"]["token"]),
      "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
      "allclose": bool(np.allclose(candidate, control, rtol=.02, atol=.5))}
  medians = {name: statistics.median(arm["wall"]["samples_ms"]) for name, arm in arms.items()}
  control_mean = statistics.mean(medians[name] for name in ("control-a", "control-c"))
  structural = all(row["arm_pass"] and row["route_pass"] and row["canonical_weights"] and
    row["projection_mains"] == row["projection_producers"] == row["projection_weights"] == 252 and row["remaining_overlays"] == 0
    for row in contracts.values())
  correct = all(row["finite"] and row["same_token"] and row["allclose"] for row in quality.values())
  result = {"schema": "tinygrad.current252.vocabulary_lifecycle_audit.v1", "contracts": contracts, "quality": quality,
    "structural_pass": structural, "correctness_pass": correct, "medians_ms": medians,
    "control_mean_ms": control_mean, "recovery_ms": control_mean-medians["candidate-b"],
    "recovery_percent": 100*(control_mean-medians["candidate-b"])/control_mean,
    "control_drift_ms": abs(medians["control-c"]-medians["control-a"]),
    "beats_both_controls_by_0p5ms": all(medians[name]-medians["candidate-b"] >= .5 for name in ("control-a", "control-c")),
    "input_sha256": {f"{name}.json": hashlib.sha256((root/f"{name}.json").read_bytes()).hexdigest() for name in names},
    "scope": "Same-harness whole-prefill replay; GPU states are process-boundary observations, not locked clocks or CUPTI active-body times."}
  result["projection_identities_unchanged"] = all(arm["route"]["identities"] == arms["control-a"]["route"]["identities"] for arm in arms.values())
  if (root/"candidate-default.json").exists():
    default = json.loads((root/"candidate-default.json").read_text())
    default_logits = np.load(root/"candidate-default.npz")
    result["automatic_selection"] = {"status": default["status"], "median_ms": default["wall"]["median_ms"],
      "generated_vocab_calls": default["program_names"].get(vocab, 0),
      "same_logits_as_explicit_candidate": bool(np.array_equal(default_logits["logits"], logits["candidate-b"]["logits"])),
      "same_token": int(default_logits["token"]) == int(logits["candidate-b"]["token"])}
  if (root/"llama-fresh.json").exists():
    llama = json.loads((root/"llama-fresh.json").read_text())[0]
    settled = statistics.median(llama["samples_ns"][1:])/1e6
    result["llama_reference"] = {"build_commit": llama["build_commit"], "settled_median_ms": settled,
      "excluded_sample_indices": [0], "candidate_minus_llama_ms": medians["candidate-b"]-settled,
      "scope": "Fresh sequential cross-harness reference; different prompt/token protocol, not a bracketing parity qualification."}
  return result


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("directory", type=pathlib.Path); args = ap.parse_args()
  result = compare(args.directory)
  (args.directory/"comparison.json").write_text(json.dumps(result, indent=2)+"\n")
  print(json.dumps(result, indent=2))
  if not result["structural_pass"] or not result["correctness_pass"] or not result["projection_identities_unchanged"]: raise SystemExit(1)
  if (auto := result.get("automatic_selection")) is not None:
    if auto["status"] != "PASS" or auto["generated_vocab_calls"] != 1 or not auto["same_logits_as_explicit_candidate"] or not auto["same_token"]:
      raise SystemExit(1)
