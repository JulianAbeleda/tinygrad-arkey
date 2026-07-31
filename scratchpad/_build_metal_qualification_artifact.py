#!/usr/bin/env python3
"""Assemble the M1b Metal qualification artifact JSON from the real child-result data captured by
`m1b_metal_qualification_run.py` (/tmp/m1b_metal_child_result.json). Writes only from measured
numbers; nothing here is invented. Mirrors the shape of
docs/qwen3-14b-prefill-q6-ffn-down-qualification-20260718.json (candidate_identity/fixture/health/
measurement_definition/qualification_identity/result.phases/route_id/schema/status/workload), with
fields this run cannot honestly produce marked explicitly absent rather than fabricated or copied.
"""
from __future__ import annotations
import json, re, hashlib

RAW_PATH = "/tmp/m1b_metal_child_result.json"
OUT_PATH = "/Users/julianabeleda/env/tinygrad-arkey-exp/docs/qwen3-8b-prefill-q4k-ffn-gate-up-qualification-metal-20260730.json"


def _strip_ansi(s: str) -> str:
  return re.sub(r"\x1b\[[0-9;]*m", "", s)


def main() -> None:
  d = json.load(open(RAW_PATH))
  r = d["result"]
  compile_rec = dict(r["compile"])
  compile_rec["kernel_name"] = _strip_ansi(compile_rec["kernel_name"])
  rounds = r["rounds"]
  errs = [rr["max_abs_error"] for rr in rounds]
  elapsed = [rr["elapsed_seconds"] for rr in rounds]
  median_elapsed = sorted(elapsed)[len(elapsed) // 2]
  canonical_identity = d["canonical_identity"]
  fixture = d["fixture"]

  request_digest_input = {"experiment_id": "m1b-metal-packed-wmma-ffn-gate-up-qualification",
    "candidate_id": canonical_identity, "role": "ffn_gate_up", "shape": fixture["shape"], "quant_format": "Q4_K"}
  request_digest = hashlib.sha256(json.dumps(request_digest_input, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

  correctness_status = "passed" if all(e <= 0.02 for e in errs) else "failed"
  overall_status = "qualified" if correctness_status == "passed" else "not_qualified"

  compile_evidence = {
    "passed": True,
    "source_sha256": compile_rec["source_sha256"],
    "binary_sha256": compile_rec["binary_sha256"],
    "compiled_device": compile_rec["compiled_device"],
    "compile_target": compile_rec["compile_target"],
    "kernel_name": compile_rec["kernel_name"],
    "local_size": compile_rec["local_size"],
    "global_size": compile_rec["global_size"],
    "resource_summary": {
      "schema": "tinygrad.metal.compile_resource_summary.v1",
      "authority": "compiled_program_launch_geometry_and_admitted_schedule",
      "active_lds_bytes": compile_rec["active_lds_bytes"],
      "threads": compile_rec["schedule"]["threads"],
      "tile": compile_rec["schedule"]["tile"],
      "pipeline": compile_rec["schedule"]["pipeline"],
      "local_size": compile_rec["local_size"],
      "global_size": compile_rec["global_size"],
    },
    "final_isa_manifest": {
      "status": "unavailable",
      "reason": "current_prefill_execution_adapter.prepare_current_prefill_compile's ISA-disassembly "
                "evidence (disassemble_amdgpu/parse_amdgpu_metadata/kernel_descriptor_from_elf) requires "
                "a ROCm llvm-objdump/llvm-readelf toolchain. This machine has none (confirmed: "
                "FileNotFoundError from _run_binary_tool for all of "
                "('/opt/rocm/llvm/bin/llvm-objdump','llvm-objdump-21','llvm-objdump-20','llvm-objdump')). "
                "That step is unconditional -- run regardless of device -- so it also blocks AMD's own "
                "compile path on this exact box (matches the M1a/PG0/PG1 finding: 'this box has no ROCm "
                "compiler'). Explicitly recorded absent, never fabricated or copied from AMD's artifact.",
    },
    "isa_structure": {"status": "unavailable", "reason": "same as final_isa_manifest"},
  }

  execution_evidence = dict(rounds[0])
  execution_evidence["scope"] = "guarded_round_trip"  # allocate+upload+dispatch+readback+guard-check, not kernel-only
  execution_status = "passed" if rounds[0]["dispatch_performed"] and not rounds[0]["device_fault"] else "failed"

  correctness_evidence = {
    "element_count": fixture["reference_elements"],
    "output_shape": rounds[0]["output_shape"],
    "finite_output": rounds[0]["finite_output"],
    "tolerance_abs": 0.02,
    "tolerance_rel": 0.002,
    "max_abs_error_by_round": errs,
    "max_abs_error": errs[0],
    "numerical_passed": correctness_status == "passed",
    "note": "max_abs_error is consistently ~29,000 across all 3 rounds (not a small rounding error and "
            "not exactly 0.0) with finite_min=-0.09375 remaining plausible -- consistent with a real, "
            "structural mismatch (most likely the packed-dequant precontract fusion's fragment/lane "
            "layout, declared 'rdna3_wmma_f32_16x16x16_f16_lds2_static' in the candidate schedule, "
            "being wrong for Metal's simdgroup_multiply_accumulate 8x8x8 fragment/lane convention) "
            "rather than a numerically-close AMD-vs-Metal accumulation difference.",
  }

  timing_evidence = {
    "inclusion": "guarded_round_trip",
    "scope": "guarded_round_trip",
    "sync": "explicit_device_synchronize_before_and_after_each_round",
    "statistic": "median",
    "warmups": 1,
    "repetitions": len(elapsed),
    "samples": elapsed,
    "units": "s",
    "value": median_elapsed,
    "noise_threshold": 0.0,
    "performance_claim_valid": False,
    "note": "Correctness failed this round (see the correctness phase), so this timing number does not "
            "characterize the speed of a correct computation -- recorded for completeness only, per "
            "measurement_definition.performance_claim=false, not as a valid throughput result.",
  }

  artifact = {
    "candidate_identity": canonical_identity,
    "canonical_identity": canonical_identity,
    "fixture": {
      "quant_format": fixture["quant_format"],
      "shape": fixture["shape"],
      "packed_bytes": fixture["packed_bytes"],
      "reference_elements": fixture["reference_elements"],
      "generator": "extra.llm_research.prefill.packed_wmma_correctness_canary.build_artifact "
                   "(deterministic single-nonzero-K-element-per-row activation; independent pure-numpy "
                   "Q4_K reference decode)",
    },
    "health": {
      "before": {"schema": "host-safety-canary.v1.tiny_add_probe", "device": "METAL", "alive": d["health_before"]},
      "after": {"schema": "host-safety-canary.v1.tiny_add_probe", "device": "METAL", "alive": r["health_after"]},
      "system_snapshot": {
        "status": "partial",
        "identification": {
          "command": ["system_profiler", "SPDisplaysDataType"],
          "chipset_model": "Apple M4", "cores": 10, "metal_support": "Metal 4",
        },
        "live_telemetry": {
          "status": "absent",
          "reason": "AMD's reference artifacts recorded a live rocm-smi utilization/temperature/VRAM "
                     "snapshot. No equivalent verified command exists in this campaign for Metal/macOS; "
                     "recorded absent rather than fabricated or copied from AMD's rocm-smi output.",
        },
      },
    },
    "measurement_definition": {
      "performance_claim": False, "rounds": len(elapsed), "scope": "full role output",
      "statistic": "median", "warmups": 1,
    },
    "qualification_identity": f"packed_wmma_metal_precontract:sha256:{request_digest}",
    "result": {
      "candidate_id": canonical_identity,
      "experiment_id": "m1b-metal-packed-wmma-ffn-gate-up-qualification",
      "extensions": {
        "identity": {"session_id": "m1b-metal-ffn_gate_up-20260730", "system_snapshot_id": "apple-m4-macos",
                     "target_id": "METAL:Apple9:wave32"},
        "workload": {"role": "ffn_gate_up", "shape": fixture["shape"]},
      },
      "phases": [
        {"phase": "compile", "status": "passed", "identity": {"canonical_identity": canonical_identity,
          "source_sha256": compile_rec["source_sha256"], "binary_sha256": compile_rec["binary_sha256"],
          "target_id": "METAL:Apple9:wave32"}, "evidence": compile_evidence, "unsupported": [], "error": None,
         "schema": "execution_bridge.phase_result.v1"},
        {"phase": "execution", "status": execution_status, "identity": {"canonical_identity": canonical_identity,
          "target_id": "METAL:Apple9:wave32"}, "evidence": {"dispatch_state": "completed", "guarded": execution_evidence,
          "health": {"preflight": d["health_before"], "postflight": r["health_after"]}}, "unsupported": [], "error": None,
         "schema": "execution_bridge.phase_result.v1"},
        {"phase": "correctness", "status": correctness_status, "identity": {"candidate_id": canonical_identity},
         "evidence": correctness_evidence, "unsupported": [], "error": None, "schema": "execution_bridge.phase_result.v1"},
        {"phase": "timing", "status": "passed", "identity": {}, "evidence": timing_evidence, "unsupported": [],
         "error": None, "schema": "execution_bridge.phase_result.v1"},
      ],
      "request_digest": request_digest,
      "schema": "execution_bridge.result.v1",
    },
    "route_id": "direct_packed",
    "schema": "tinygrad.q4k_direct_packed_qualification.v1",
    "status": overall_status,
    "workload": {
      "phase": "prefill", "quant_format": "Q4_K", "role": "ffn_gate_up",
      "shape": {"m": fixture["shape"][0], "n": fixture["shape"][1], "k": fixture["shape"][2]},
      "geometry": {"tm": 256, "tn": 64, "tk": 32, "wm": 8, "wn": 1, "bc": 1},
      "target": {"backend": "METAL", "arch": "Apple9", "wave_size": 32},
      "note": "The admitted candidate payload's own workload.target metadata field is retained as "
              "AMD:gfx1100:wave32 (extra/llm_research/runtime_specs.py's capability lattice -- "
              "FullKernelCapability -- is backend/arch-fixed to AMD/gfx1100 for every existing capability "
              "constant; this was left untouched, target-additive scope). The `target` recorded here "
              "instead describes where compilation and dispatch actually happened: real Context(DEV="
              "'METAL') compile, real Metal PROGRAM binary, real dispatch on this Apple M4.",
    },
  }

  with open(OUT_PATH, "w") as f:
    json.dump(artifact, f, indent=2, sort_keys=True)
    f.write("\n")
  print("wrote", OUT_PATH)
  print("status:", overall_status, "correctness:", correctness_status, "max_abs_error (round1):", errs[0])


if __name__ == "__main__":
  main()
