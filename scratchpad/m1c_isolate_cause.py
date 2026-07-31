#!/usr/bin/env python3
"""M1c: capture full reference (R) and actual Metal output (A) arrays from the exact same
dispatch M1b already qualified as a real, reproducible Metal-execution failure, so the
multiset/permutation-vs-wrong-values question can be answered from real numbers.

This is NOT a new dispatch path. It is `scratchpad/m1b_metal_qualification_run.py` verbatim
(same payload construction, same PackedWmmaRoute geometry never inserted into
PACKED_WMMA_ROUTES, same compile/admit/guarded-execution machinery), with exactly one addition:
the guarded-execution hooks' `readback` callback is wrapped (via dataclasses.replace, not by
editing extra/llm_research/prefill/guarded_execution.py) to stash a copy of the output array
for every round instead of only summary statistics. `run_guarded_execution` (the real lifecycle
function -- health preflight, guard bytes, allclose, finite checks) is called directly instead
of via the `run_tinygrad_executable_guarded` convenience wrapper, only so the wrapped hooks can
be passed through; the lifecycle itself is untouched and reused as-is.

Writes full R (reference) and A (actual output, one array per post-warmup round) to an npz file
on disk from inside the isolated child (same machine/filesystem, so no need to pipe ~12MB arrays
through the multiprocessing queue) and returns only the path + cheap summary stats to the parent.
"""
from __future__ import annotations
import sys, tempfile, os, json, copy, hashlib, time, dataclasses
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

import numpy as np

from tinygrad.llm.packed_wmma_prefill import PackedWmmaRoute, PACKED_WMMA_ROUTE_BY_KEY, PACKED_WMMA_ROUTES
from extra.llm_research.prefill.packed_wmma_correctness_canary import candidate_payload, build_artifact
from extra.llm_research.runtime_specs import derive_packed_weight_candidate, full_kernel_workload
from extra.llm_research.prefill.host_safety_canary import make_tiny_health_probe
from extra.llm_research.prefill.guarded_execution import GuardPolicy, run_guarded_execution, make_tinygrad_executable_hooks
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
               shape: tuple[int, int, int], warmups: int, rounds: int, dump_npz_path: str) -> dict:
  from extra.llm_research.prefill.current_prefill_execution_adapter import (
    compile_current_prefill_program, admit_current_prefill, _arrays)
  from tinygrad.device import Device

  program, admission = compile_current_prefill_program(payload, canonical_identity, device=device)
  binary = next((u.arg for u in program.src if u.op.name == "BINARY" and isinstance(u.arg, bytes)), None)
  source = next((u.arg for u in program.src if u.op.name == "SOURCE" and isinstance(u.arg, str)), None)
  if binary is None or source is None:
    return {"ok": False, "stage": "compile", "error": "no final SOURCE/BINARY on the compiled PROGRAM"}
  minimal_evidence = {"passed": True, "binary_sha256": hashlib.sha256(binary).hexdigest()}

  admission_again = admit_current_prefill(payload, canonical_identity)  # cheap, pure-python re-check
  inputs, reference = _arrays(artifact_path, shape, admission_again.context.packed_weight)

  from tinygrad.runtime.bridge import prepare_executable
  executable = prepare_executable(program, minimal_evidence, device=device)

  # Wrap only `readback` so every round's raw output array is captured, in addition to the
  # summary dict run_guarded_execution already returns. The lifecycle function itself
  # (health/guard/allclose checks) is called unmodified.
  captured: dict[str, np.ndarray] = {}
  base_hooks = make_tinygrad_executable_hooks(device, lambda: True, ARGUMENT_ORDER)

  def _readback_and_capture(buffer):
    arr = base_hooks.readback(buffer)
    if buffer.name == "output":
      captured["last"] = np.array(arr, copy=True)
    return arr

  wrapped_hooks = dataclasses.replace(base_hooks, readback=_readback_and_capture)

  rounds_out = []
  captured_by_round = []
  try:
    for i in range(warmups + rounds):
      Device[device].synchronize()  # drain anything pending before this round starts
      t0 = time.perf_counter()
      result = run_guarded_execution(executable=executable, inputs=inputs, reference=reference,
        hooks=wrapped_hooks,
        policy=GuardPolicy(timeout_seconds=60.0, check_inputs_unchanged=True, rtol=2e-2, atol=2e-2),
        identity={"canonical_identity": canonical_identity, "round": i}, output_dtype=np.float16)
      Device[device].synchronize()  # ensure completion is real before the wall-clock stops / readback is trusted
      t1 = time.perf_counter()
      result["wall_seconds"] = t1 - t0
      if i >= warmups:
        rounds_out.append(result)
        captured_by_round.append(captured.get("last"))
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

  # Persist full arrays to disk (same machine/filesystem as the parent) rather than piping
  # ~12MB x N arrays back through the multiprocessing result queue.
  save_kwargs = {"reference": reference}
  for idx, arr in enumerate(captured_by_round):
    if arr is not None:
      save_kwargs[f"output_round{idx}"] = arr
  np.savez(dump_npz_path, **save_kwargs)

  return {"ok": True, "rounds": rounds_out, "health_after": health_after,
          "dump_npz_path": dump_npz_path, "rounds_captured": len(captured_by_round)}


def _metal_health_probe() -> bool:
  from extra.llm_research.prefill.host_safety_canary import _tiny_add_is_alive
  return _tiny_add_is_alive(256, DEVICE)


def main() -> None:
  local_route = PackedWmmaRoute(QUANT, ROLE, SHAPE, GEOMETRY, canonical_identity="m1c-probe-placeholder")
  assert all(r.canonical_identity != local_route.canonical_identity for r in PACKED_WMMA_ROUTES)

  payload = _payload_for_local_row(PROFILE, local_route)
  entry = derive_packed_weight_candidate(payload, QUANT)
  final_payload = entry.to_json()["payload"]
  workload = full_kernel_workload(final_payload)
  print("=== payload ===")
  print(json.dumps({"profile": workload.profile, "role": workload.role, "shape": workload.shape,
                     "target": workload.target, "canonical_identity": entry.canonical_identity}, sort_keys=True))

  fd, artifact_path = tempfile.mkstemp(prefix="m1c_metal_isolate_", suffix=".npz")
  os.close(fd)
  dump_npz_path = "/tmp/m1c_metal_RA_arrays.npz"
  try:
    fixture = build_artifact(QUANT, artifact_path, SHAPE)
    print("=== fixture ===")
    print(json.dumps(fixture, sort_keys=True))

    print(f"=== health BEFORE (device={DEVICE}) ===")
    before_probe = make_tiny_health_probe(device=DEVICE)
    health_before = bool(before_probe())
    print("health_before:", health_before)

    print("=== dispatching in isolated child (warmups + rounds), capturing full R/A ===")
    res = run_isolated(_child_run,
      args=(final_payload, entry.canonical_identity, DEVICE, artifact_path, SHAPE, WARMUPS, ROUNDS, dump_npz_path),
      timeout_seconds=180.0, start_method="spawn")
    print("child status:", res.status, "timed_out:", res.timed_out, "error:", res.error)
    if res.stdout: print("--- child stdout ---\n" + res.stdout)
    if res.stderr: print("--- child stderr ---\n" + res.stderr)
    summary = copy.deepcopy(res.result) if isinstance(res.result, dict) else res.result
    print(json.dumps(summary, sort_keys=True, default=str, indent=2))

    out_path = "/tmp/m1c_metal_child_result.json"
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
