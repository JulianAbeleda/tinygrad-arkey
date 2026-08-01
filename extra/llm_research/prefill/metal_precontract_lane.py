"""M1e -- one reusable Metal precontract test lane.

M1b, M1c, and M1d each rebuilt the same payload/compile/admit/guarded-execute setup as a
throwaway scratchpad driver (`scratchpad/m1b_metal_qualification_run.py`,
`scratchpad/m1c_isolate_cause.py`, `scratchpad/m1d_confirm_c_fragment.py`). This module
consolidates that machinery into one committed, importable entry point,
:func:`run_precontract_probe`, so a configuration sweep (Part 2 of M1e) does not need a fourth
bespoke driver -- it just calls this function once per `(quant, role, shape, geometry, device,
rounds)`.

Everything below is a generalization of `scratchpad/m1c_isolate_cause.py`'s proven payload
construction / compile / admit / guarded-execution machinery (itself a line-for-line reuse of
`scratchpad/m1b_metal_qualification_run.py`), not a reimplementation:

  * Payload construction (`_payload_for_config`) is the same schedule-field injection as M1c's
    `_payload_for_local_row`, generalized off of a fixed `PackedWmmaRoute` so any
    `(shape, geometry)` pair can be probed, not only AMD's own frozen production tuples.
  * Compile/execute (`_child_run`) is the same compile -> minimal-evidence -> `prepare_executable`
    -> wrapped-readback-capture -> `run_guarded_execution` sequence M1c used, run inside a
    `run_isolated(..., start_method="spawn")` child exactly as M1b/M1c required.
  * `run_canary` (`packed_wmma_correctness_canary.py`) is already `device`-parameterized as of
    M1b (`c9e3b9bd1`); this module builds on that rather than forking it, and never adds a row to
    `tinygrad.llm.packed_wmma_prefill.PACKED_WMMA_ROUTES`.

Same production-avoidance discipline M1b/M1c/M1d established, restated here because it is
load-bearing for every call this module makes:

  * `admit_current_prefill`/`derive_packed_weight_candidate`/`rebind_full_kernel_workload` are
    pure Python (no `Device[...]`, no GPU, no ROCm) -- admission is checked in the parent process
    before any child is ever spawned, so illegal configs are skipped for free.
  * GPU compile+dispatch calls the one device-parameterized producer
    `current_prefill_execution_adapter.prepare_current_prefill_compile`: on AMD it produces the
    full ISA manifest, on every other declared target it produces the minimal evidence this lane
    used to hand-build (`{passed, binary_sha256}` plus source sha and target facts) with the
    AMD-only fields present and explicitly not applicable. The lane's old `_child_run` workaround
    (calling `compile_current_prefill_program` directly) is deleted: one producer for every
    target, and the ROCm toolchain is only reached on the AMD branch.
  * `Device[device].synchronize()` is called before and after every round, exactly as M1b/M1c did.
"""
from __future__ import annotations

import copy, dataclasses, os, re, tempfile, time
from dataclasses import dataclass
from typing import Any

import numpy as np

from extra.llm_research.prefill.candidate_payloads import find_role_template, load_candidate_payloads
from extra.llm_research.prefill.packed_wmma_correctness_canary import DEFAULT_PROFILE, build_artifact
from extra.llm_research.prefill.guarded_execution import GuardPolicy, run_guarded_execution, make_tinygrad_executable_hooks
from extra.llm_research.runtime_specs import (FullKernelAdmissionError, derive_packed_weight_candidate,
  full_kernel_workload, rebind_full_kernel_workload)
from tinygrad.runtime.process_isolated import run_isolated

SCHEMA = "metal-precontract-probe.v1"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ARGUMENT_ORDER = ("output", "a", "b")
GEOMETRY_FIELDS = ("tm", "tn", "tk", "wm", "wn", "bc")
_ROUND_GUARD_TIMEOUT_SECONDS = 60.0  # unchanged from M1b/M1c's own GuardPolicy timeout for this dispatch


@dataclass(frozen=True)
class ProbeConfig:
  """One lane input: `(quant, role, shape, geometry, device, rounds)` plus minor knobs."""
  quant: str
  role: str
  shape: tuple[int, int, int]
  geometry: tuple[int, int, int, int, int, int]  # (tm, tn, tk, wm, wn, bc)
  device: str = "METAL"
  rounds: int = 3
  warmups: int = 1
  profile: str = DEFAULT_PROFILE
  timeout_seconds: float = 180.0

  def __post_init__(self):
    if self.quant not in ("Q4_K", "Q6_K"): raise ValueError("quant must be Q4_K or Q6_K")
    if not isinstance(self.role, str) or not self.role: raise ValueError("role is required")
    if len(self.shape) != 3 or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in self.shape):
      raise ValueError("shape must be exactly 3 positive ints (m, n, k)")
    if len(self.geometry) != 6 or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in self.geometry):
      raise ValueError("geometry must be exactly 6 positive ints (tm, tn, tk, wm, wn, bc)")
    if not isinstance(self.device, str) or not self.device: raise ValueError("device is required")
    if self.rounds < 1: raise ValueError("rounds must be >= 1")
    if self.warmups < 0: raise ValueError("warmups must be >= 0")
    if self.timeout_seconds <= 0: raise ValueError("timeout_seconds must be positive")

  @property
  def geom(self) -> dict[str, int]:
    return dict(zip(GEOMETRY_FIELDS, self.geometry))

  def to_json(self) -> dict[str, Any]:
    return {"quant": self.quant, "role": self.role, "shape": list(self.shape), "geometry": list(self.geometry),
            "device": self.device, "rounds": self.rounds, "warmups": self.warmups, "profile": self.profile}


@dataclass(frozen=True)
class ProbeResult:
  """Structured lane output. `status` is one of "measured", "skipped", "error"."""
  schema: str
  config: dict[str, Any]
  status: str
  skip_reason: str | None
  canonical_identity: str | None
  compile: dict[str, Any] | None
  max_abs_error: float | None
  coverage: dict[str, Any] | None
  determinism: dict[str, Any] | None
  rounds: tuple[dict[str, Any], ...]
  passed: bool

  def to_json(self) -> dict[str, Any]:
    return {"schema": self.schema, "config": self.config, "status": self.status, "skip_reason": self.skip_reason,
            "canonical_identity": self.canonical_identity, "compile": self.compile,
            "max_abs_error": self.max_abs_error, "coverage": self.coverage, "determinism": self.determinism,
            "rounds": list(self.rounds), "passed": self.passed}


def _skipped(config: ProbeConfig, reason: str) -> ProbeResult:
  return ProbeResult(SCHEMA, config.to_json(), "skipped", reason, None, None, None, None, None, (), False)


def _errored(config: ProbeConfig, reason: str) -> ProbeResult:
  return ProbeResult(SCHEMA, config.to_json(), "error", reason, None, None, None, None, None, (), False)


def _base_payload_for_shape(profile: str, role: str, shape: tuple[int, int, int]) -> dict:
  """The same construction `packed_wmma_correctness_canary.candidate_payload` uses, except
  `shape` is caller-supplied instead of derived from the profile's own natural role shape --
  needed so the lane can probe shapes the model profile does not itself have (Part 2's scale
  sweep). For a profile/role/shape triple that IS the profile's own natural shape this returns
  the identical payload `candidate_payload` would.
  """
  payloads = load_candidate_payloads()
  exact = next((p for p in payloads if p["workload"]["profile"] == profile and p["workload"]["role"] == role), None)
  if exact is not None and tuple(exact["workload"]["shape"][k] for k in ("m", "n", "k")) == shape:
    return exact
  template = find_role_template(payloads, role)
  return rebind_full_kernel_workload(template, profile=profile, role=role, shape=shape).to_json()["payload"]


def _payload_for_config(config: ProbeConfig) -> dict:
  """Build the exact candidate payload for one probe config.

  Schedule-field injection is line-for-line `scratchpad/m1c_isolate_cause.py`'s
  `_payload_for_local_row`, generalized off of `PackedWmmaRoute` so any `(shape, geometry)` pair
  can be probed, not only a route already recorded in a frozen production table.
  """
  payload = copy.deepcopy(_base_payload_for_shape(config.profile, config.role, config.shape))
  if tuple(payload["workload"]["shape"][k] for k in ("m", "n", "k")) != config.shape:
    raise ValueError("rebound payload shape does not match the requested probe shape")
  g, schedule = config.geom, payload["schedule"]
  schedule["tile"] = {"m": g["tm"], "n": g["tn"], "k": g["tk"]}
  schedule["waves"] = {"m": g["wm"], "n": g["wn"]}
  schedule["threads"] = g["wm"] * g["wn"] * 32
  a_end, b_end = g["tm"] * 80, (g["tm"] + g["tn"]) * 80
  schedule["lds"]["windows"] = {"a": [0, a_end], "b": [a_end, b_end]}
  schedule["lds"]["strides"] = {"a": 80, "b": 80}
  schedule["pipeline"]["buffer_count"] = g["bc"]
  return payload


def admit_probe_config(config: ProbeConfig):
  """Pure-Python admission (no `Device[...]`, no GPU, no ROCm): returns `(entry, admission)` or
  raises `FullKernelAdmissionError`/`ValueError`. Exposed standalone so a caller can check
  legality without ever spawning a GPU child."""
  from extra.llm_research.prefill.current_prefill_execution_adapter import admit_current_prefill
  payload = _payload_for_config(config)
  entry = derive_packed_weight_candidate(payload, config.quant)
  final_payload = entry.to_json()["payload"]
  admission = admit_current_prefill(final_payload, entry.canonical_identity, device=config.device)
  return entry, admission


# --- Runs INSIDE the isolated spawned child only -----------------------------
# Module-level and picklable, as `run_isolated(..., start_method="spawn")` requires.

def _child_run(payload: dict, canonical_identity: str, device: str, artifact_path: str,
               shape: tuple[int, int, int], warmups: int, rounds: int, dump_npz_path: str) -> dict:
  from extra.llm_research.prefill.current_prefill_execution_adapter import (
    prepare_current_prefill_compile, admit_current_prefill, _arrays)
  from tinygrad.device import Device
  from tinygrad.runtime.bridge import prepare_executable

  program, compile_evidence = prepare_current_prefill_compile(payload, canonical_identity, device=device)
  admission = admit_current_prefill(payload, canonical_identity, device=device)
  compile_record = {
    "kernel_name": str(getattr(program.arg, "name", "")),
    "local_size": list(getattr(program.arg, "local_size", None) or ()) or None,
    "global_size": list(getattr(program.arg, "global_size", None) or ()) or None,
    "active_lds_bytes": admission.active_lds_bytes,
    "threads": payload["schedule"]["threads"],
    "tile": dict(payload["schedule"]["tile"]),
    "pipeline": dict(payload["schedule"]["pipeline"]),
    "admitted": True,
  }

  inputs, reference = _arrays(artifact_path, shape, admission.context.packed_weight)

  executable = prepare_executable(program, compile_evidence, device=device)

  # Wrap only `readback` so every round's raw output array is captured, in addition to the
  # summary dict run_guarded_execution already returns -- exactly M1c's technique.
  captured: dict[str, np.ndarray] = {}
  base_hooks = make_tinygrad_executable_hooks(device, lambda: True, ARGUMENT_ORDER)

  def _readback_and_capture(buffer):
    arr = base_hooks.readback(buffer)
    if buffer.name == "output":
      captured["last"] = np.array(arr, copy=True)
    return arr

  wrapped_hooks = dataclasses.replace(base_hooks, readback=_readback_and_capture)

  rounds_out, captured_by_round = [], []
  try:
    for i in range(warmups + rounds):
      Device[device].synchronize()  # drain anything pending before this round starts
      t0 = time.perf_counter()
      result = run_guarded_execution(executable=executable, inputs=inputs, reference=reference,
        hooks=wrapped_hooks,
        policy=GuardPolicy(timeout_seconds=_ROUND_GUARD_TIMEOUT_SECONDS, check_inputs_unchanged=True, rtol=2e-2, atol=2e-2),
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

  save_kwargs = {"reference": reference}
  for idx, arr in enumerate(captured_by_round):
    if arr is not None: save_kwargs[f"output_round{idx}"] = arr
  np.savez(dump_npz_path, **save_kwargs)

  return {"ok": True, "compile": compile_record, "rounds": rounds_out, "dump_npz_path": dump_npz_path,
          "rounds_captured": len(captured_by_round)}


# --- Parent-side entry point --------------------------------------------------

def _summarize(config: ProbeConfig, canonical_identity: str, admission, child: dict, dump_npz_path: str) -> ProbeResult:
  rounds = child["rounds"]
  round_errors = [r.get("max_abs_error") for r in rounds]
  finite_errors = [e for e in round_errors if e is not None]
  max_abs_error = max(finite_errors) if finite_errors else None

  with np.load(dump_npz_path) as npz:
    arrays = [np.array(npz[f"output_round{i}"]) for i in range(len(rounds)) if f"output_round{i}" in npz.files]

  coverage = None
  if arrays:
    a0 = arrays[0]
    total = int(a0.size)
    written = int(np.count_nonzero(a0))
    nonzero = a0[a0 != 0]
    coverage = {
      "total": total, "written_count": written, "never_written_count": total - written,
      "written_fraction": written / total, "never_written_fraction": (total - written) / total,
      "value_range_written": [float(np.min(nonzero)), float(np.max(nonzero))] if nonzero.size else None,
      "basis": "round 0 of the captured output arrays; the output buffer is zero-initialized by "
               "guarded_execution.allocate before dispatch, so a zero after dispatch means the kernel "
               "never wrote that position -- 'nonzero == written' is therefore a lower bound on true "
               "coverage (a position that legitimately computes to exactly 0.0 would be undercounted), "
               "matching M1c's documented methodology"}

  determinism = None
  if len(arrays) >= 2:
    pairs = [(i, j) for i in range(len(arrays)) for j in range(i + 1, len(arrays))]
    diffs = [float(np.max(np.abs(arrays[i].astype(np.float64) - arrays[j].astype(np.float64)))) for i, j in pairs]
    bit_identical = all(np.array_equal(arrays[i], arrays[j]) for i, j in pairs)
    determinism = {"rounds_compared": len(arrays), "bit_identical": bit_identical,
                    "max_inter_round_diff": max(diffs) if diffs else 0.0}
  elif len(arrays) == 1:
    determinism = {"rounds_compared": 1, "bit_identical": None, "max_inter_round_diff": None,
                    "note": "only one round's output array was captured; determinism cannot be assessed"}

  kernel_name = child["compile"].get("kernel_name")
  if isinstance(kernel_name, str): kernel_name = _ANSI_RE.sub("", kernel_name)  # strip tinygrad's colorized debug codes
  compile_info = {"active_lds_bytes": admission.active_lds_bytes, "threads": config.geom["wm"] * config.geom["wn"] * 32,
                   "admitted": True, "kernel_name": kernel_name,
                   "local_size": child["compile"].get("local_size"), "global_size": child["compile"].get("global_size")}

  determ_ok = determinism is not None and determinism.get("bit_identical") is True
  cov_full = coverage is not None and coverage["written_fraction"] == 1.0
  overall_passed = bool(max_abs_error is not None and max_abs_error <= 0.02 and determ_ok and cov_full)

  round_summaries = tuple({"round": i, "max_abs_error": r.get("max_abs_error"), "passed": r.get("passed"),
                            "numerics_passed": r.get("numerics_passed"), "guards_intact": r.get("guards_intact"),
                            "device_healthy_after": r.get("device_healthy_after"), "wall_seconds": r.get("wall_seconds")}
                           for i, r in enumerate(rounds))

  return ProbeResult(SCHEMA, config.to_json(), "measured", None, canonical_identity, compile_info,
                      max_abs_error, coverage, determinism, round_summaries, overall_passed)


def run_precontract_probe(config: ProbeConfig, *, keep_npz: bool = False) -> ProbeResult:
  """The lane's one entry point.

  Pure-Python admission runs first, in the caller's own process, with no `Device[...]` touched --
  a config the admission machinery rejects is returned as `status="skipped"` with the reason and
  no GPU is ever spawned for it. A config that admits is compiled and dispatched exactly the way
  M1b/M1c proved out: a `run_isolated(..., start_method="spawn")` child (fresh device init, hard
  timeout, process-group cleanup), `Device[config.device].synchronize()` before and after every
  round, full-array readback capture for every measured round.
  """
  try:
    entry, admission = admit_probe_config(config)
  except (FullKernelAdmissionError, ValueError) as exc:
    return _skipped(config, f"admission rejected: {exc}")

  final_payload = entry.to_json()["payload"]
  full_kernel_workload(final_payload)  # cheap shape/target sanity re-check, mirrors M1c

  fd, artifact_path = tempfile.mkstemp(prefix="m1e_precontract_probe_", suffix=".npz")
  os.close(fd)
  dump_fd, dump_npz_path = tempfile.mkstemp(prefix="m1e_precontract_RA_", suffix=".npz")
  os.close(dump_fd)
  try:
    build_artifact(config.quant, artifact_path, config.shape)
    res = run_isolated(_child_run,
      args=(final_payload, entry.canonical_identity, config.device, artifact_path, config.shape,
            config.warmups, config.rounds, dump_npz_path),
      timeout_seconds=config.timeout_seconds, start_method="spawn")
    if res.status != "passed" or not isinstance(res.result, dict) or not res.result.get("ok"):
      reason = (f"child status={res.status!r} timed_out={res.timed_out} error={res.error!r} "
                f"result_error={res.result.get('error') if isinstance(res.result, dict) else None}")
      return _errored(config, reason)
    return _summarize(config, entry.canonical_identity, admission, res.result, dump_npz_path)
  finally:
    try: os.remove(artifact_path)
    except OSError: pass
    if not keep_npz:
      try: os.remove(dump_npz_path)
      except OSError: pass


__all__ = ["ProbeConfig", "ProbeResult", "admit_probe_config", "run_precontract_probe"]
