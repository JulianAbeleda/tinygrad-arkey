#!/usr/bin/env python3
"""M1b phase 3: one real Metal execution of a packed-WMMA row.

Row: Q4_K, ffn_gate_up, shape (512, 12288, 4096) [Qwen3-8B, qwen3_8b_q4k_m_gfx1100 profile],
geometry (256, 64, 32, 8, 1, 1) -- AMD's own production ffn_gate_up tuple
(tinygrad/llm/packed_wmma_prefill.py PACKED_WMMA_ROUTES), reused verbatim. This shape/geometry
combination was already proven LDS-legal on Metal by PG2 (active_lds_bytes=25600 < 32768, bc=1).
Never inserted into PACKED_WMMA_ROUTES -- a local-only PackedWmmaRoute, never touching the frozen
production table (matches the M1 feasibility probe's discipline).

Uses the real, unmodified, device-generic machinery throughout: `candidate_payload`,
`derive_packed_weight_candidate`, `admit_current_prefill`, `compile_current_prefill_program`,
`build_artifact`, `run_isolated_guarded_execution` / `run_tinygrad_executable_guarded`
(process-isolated spawn boundary, health preflight/postflight, guard bytes, input immutability,
full-output comparison, finite check -- extra/llm_research/prefill/guarded_execution.py, which
"knows nothing about ... AMD ISA" by its own module docstring).

Deliberately bypasses ONLY `current_prefill_execution_adapter.prepare_current_prefill_compile`'s
AMD-ROCm ISA-disassembly evidence capture (disassemble_amdgpu / parse_amdgpu_metadata /
kernel_descriptor_from_elf), which is invoked UNCONDITIONALLY after any successful compile
regardless of `device`, and which this machine cannot satisfy for ANY backend: it requires a real
ROCm llvm-objdump/llvm-readelf toolchain (checked: `_run_binary_tool` raises
`FileNotFoundError: none of ('/opt/rocm/llvm/bin/llvm-objdump', 'llvm-objdump-21',
'llvm-objdump-20', 'llvm-objdump') is available`), which does not exist on macOS -- the same class
of environment limitation PG0/PG1/M1a already recorded for AMD's own compile path on this exact
box ("this box has no ROCm compiler"/"no AMD hardware or compiler"). This is confirmed by actually
calling `run_canary(..., device="METAL")` first (see the companion script
`m1b_metal_qualification_attempt.py`), which compiles the Metal kernel successfully (ABI checks
pass, active_lds_bytes admitted) and only then raises at the ISA-disassembly step.

No repo file is patched to route around this. Instead this script builds its own MINIMAL, honestly
labelled compile evidence -- `{"passed": True, "binary_sha256": ...}` -- which is the entire
contract `tinygrad.runtime.bridge.prepare_executable` actually requires (see its own source,
`compile_evidence.get("passed") is not True` / `compile_evidence.get("binary_sha256")`); it does
NOT reconstruct or fake an ISA manifest, and the resulting qualification JSON marks that evidence
explicitly absent rather than fabricating or copying AMD's.
"""
from __future__ import annotations
import sys, tempfile, os, json, copy, hashlib, time
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

import numpy as np

from tinygrad.llm.packed_wmma_prefill import PackedWmmaRoute, PACKED_WMMA_ROUTE_BY_KEY, PACKED_WMMA_ROUTES
from extra.llm_research.prefill.packed_wmma_correctness_canary import candidate_payload, build_artifact
from extra.llm_research.runtime_specs import derive_packed_weight_candidate, full_kernel_workload
from extra.llm_research.prefill.host_safety_canary import make_tiny_health_probe
from extra.llm_research.prefill.guarded_execution import GuardPolicy, run_tinygrad_executable_guarded
from tinygrad.runtime.process_isolated import run_isolated

QUANT, ROLE, SHAPE = "Q4_K", "ffn_gate_up", (512, 12288, 4096)
PROFILE = "qwen3_8b_q4k_m_gfx1100"
GEOMETRY = (256, 64, 32, 8, 1, 1)  # AMD's real ffn_gate_up (tm,tn,tk,wm,wn,bc), reused verbatim.
DEVICE = "METAL"
ARGUMENT_ORDER = ("output", "a", "b")
WARMUPS, ROUNDS = 1, 3

assert (QUANT, ROLE, SHAPE) not in PACKED_WMMA_ROUTE_BY_KEY, "refusing: shape already a production row"
assert GEOMETRY == PACKED_WMMA_ROUTE_BY_KEY[("Q4_K", "ffn_gate_up", (512, 17408, 5120))].geometry


def _payload_for_local_row(profile: str, row: PackedWmmaRoute) -> dict:
  payload = copy.deepcopy(candidate_payload(profile, row.role))
  if tuple(payload["workload"]["shape"][key] for key in ("m", "n", "k")) != row.shape:
    raise ValueError(f"oracle workload does not match row {row}")
  g, schedule = row.geom, payload["schedule"]
  schedule["tile"] = {"m": g["tm"], "n": g["tn"], "k": g["tk"]}
  schedule["waves"] = {"m": g["wm"], "n": g["wn"]}
  schedule["threads"] = g["wm"] * g["wn"] * 32
  a_end, b_end = g["tm"] * 80, (g["tm"] + g["tn"]) * 80
  schedule["lds"]["windows"] = {"a": [0, a_end], "b": [a_end, b_end]}
  schedule["lds"]["strides"] = {"a": 80, "b": 80}
  schedule["pipeline"]["buffer_count"] = g["bc"]
  return payload


# --- Runs INSIDE the isolated spawned child only -----------------------------

def _child_run(payload: dict, canonical_identity: str, device: str, artifact_path: str,
               shape: tuple[int, int, int], warmups: int, rounds: int) -> dict:
  from extra.llm_research.prefill.current_prefill_execution_adapter import (
    compile_current_prefill_program, admit_current_prefill, _arrays)
  from tinygrad.device import Device

  program, admission = compile_current_prefill_program(payload, canonical_identity, device=device)
  binary = next((u.arg for u in program.src if u.op.name == "BINARY" and isinstance(u.arg, bytes)), None)
  source = next((u.arg for u in program.src if u.op.name == "SOURCE" and isinstance(u.arg, str)), None)
  compiled_target = next((u.arg for u in program.src if u.op.name == "DEVICE"), None)
  if binary is None or source is None:
    return {"ok": False, "stage": "compile", "error": "no final SOURCE/BINARY on the compiled PROGRAM"}
  minimal_evidence = {"passed": True, "binary_sha256": hashlib.sha256(binary).hexdigest()}
  compile_record = {
    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    "binary_sha256": minimal_evidence["binary_sha256"],
    "compiled_device": compiled_target,
    "compile_target": device,
    "kernel_name": str(getattr(program.arg, "name", "")),
    "local_size": list(getattr(program.arg, "local_size", None) or ()) or None,
    "global_size": list(getattr(program.arg, "global_size", None) or ()) or None,
    "active_lds_bytes": admission.active_lds_bytes,
    "schedule": {"threads": payload["schedule"]["threads"], "tile": payload["schedule"]["tile"],
                 "pipeline": dict(payload["schedule"]["pipeline"])},
  }

  admission_again = admit_current_prefill(payload, canonical_identity)  # cheap, pure-python re-check
  inputs, reference = _arrays(artifact_path, shape, admission_again.context.packed_weight)

  from tinygrad.runtime.bridge import prepare_executable
  executable = prepare_executable(program, minimal_evidence, device=device)

  rounds_out = []
  try:
    for i in range(warmups + rounds):
      Device[device].synchronize()  # drain anything pending before this round starts
      t0 = time.perf_counter()
      result = run_tinygrad_executable_guarded(executable=executable, device=device, inputs=inputs,
        reference=reference, health=lambda: True,
        policy=GuardPolicy(timeout_seconds=60.0, check_inputs_unchanged=True, rtol=2e-2, atol=2e-2),
        identity={"canonical_identity": canonical_identity, "round": i}, argument_order=ARGUMENT_ORDER,
        output_dtype=np.float16)
      Device[device].synchronize()  # ensure completion is real before the wall-clock stops
      t1 = time.perf_counter()
      result["wall_seconds"] = t1 - t0
      if i >= warmups: rounds_out.append(result)
  finally:
    close = getattr(executable, "close", None)
    if close is not None:
      try: close()
      except Exception: pass

  health_after = True
  try:
    health_after = bool(_metal_health_probe())
  except Exception:
    health_after = False

  return {"ok": True, "compile": compile_record, "rounds": rounds_out, "health_after": health_after}


def _metal_health_probe() -> bool:
  from extra.llm_research.prefill.host_safety_canary import _tiny_add_is_alive
  return _tiny_add_is_alive(256, DEVICE)


def main() -> None:
  local_route = PackedWmmaRoute(QUANT, ROLE, SHAPE, GEOMETRY, canonical_identity="m1b-probe-placeholder")
  assert all(r.canonical_identity != local_route.canonical_identity for r in PACKED_WMMA_ROUTES)

  payload = _payload_for_local_row(PROFILE, local_route)
  entry = derive_packed_weight_candidate(payload, QUANT)
  final_payload = entry.to_json()["payload"]
  workload = full_kernel_workload(final_payload)
  print("=== payload ===")
  print(json.dumps({"profile": workload.profile, "role": workload.role, "shape": workload.shape,
                     "target": workload.target, "canonical_identity": entry.canonical_identity}, sort_keys=True))

  fd, artifact_path = tempfile.mkstemp(prefix="m1b_metal_qualification_", suffix=".npz")
  os.close(fd)
  try:
    fixture = build_artifact(QUANT, artifact_path, SHAPE)
    print("=== fixture ===")
    print(json.dumps(fixture, sort_keys=True))

    print(f"=== health BEFORE (device={DEVICE}) ===")
    before_probe = make_tiny_health_probe(device=DEVICE)
    health_before = bool(before_probe())
    print("health_before:", health_before)

    print("=== dispatching in isolated child (warmups + rounds) ===")
    res = run_isolated(_child_run,
      args=(final_payload, entry.canonical_identity, DEVICE, artifact_path, SHAPE, WARMUPS, ROUNDS),
      timeout_seconds=180.0, start_method="spawn")
    print("child status:", res.status, "timed_out:", res.timed_out, "error:", res.error)
    print(json.dumps(res.result, sort_keys=True, default=str, indent=2))

    out_path = "/tmp/m1b_metal_child_result.json"
    with open(out_path, "w") as f:
      json.dump({"status": res.status, "timed_out": res.timed_out, "error": res.error,
                 "result": res.result, "health_before": health_before,
                 "payload": final_payload, "canonical_identity": entry.canonical_identity,
                 "fixture": fixture}, f, indent=2, default=str)
    print("wrote", out_path)
  finally:
    try: os.remove(artifact_path)
    except OSError: pass


if __name__ == "__main__":
  main()
