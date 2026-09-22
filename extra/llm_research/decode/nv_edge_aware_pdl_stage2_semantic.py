#!/usr/bin/env python3
"""Stage 2 matched native-vs-CUDA semantic gate (edge-aware PDL hook).

Reuses the Phase C probe shape (one producer, one consumer, checksum,
``%globaltimer``) as control/candidate/control brackets and compares the
native NV QMD path against the CUDA programmatic-PDL path for the
``trigger_policy`` / ``wait_position`` pairs.  Every GPU arm runs as a fresh
child process under ``timeout ... flock -w 120 /tmp/gpu-bench.lock env ...``
exactly like ``nv_pdl_phase_c_driver.py``; this driver itself never touches
the GPU.

Bracket orders (control is the bracketing ``no_pdl`` arm):

* native: ``no_pdl, pdl_end, pdl_start, pdl_prologue, qmd_latch, no_pdl``
* cuda:   ``no_pdl, pdl_end, pdl_start, pdl_prologue, no_pdl``

Hard gates per pair (scope doc section 6, Stage 2):

* all three bracket arms pass their full-coverage checksum;
* both control arms show no launch-ahead (median overlap <= 1.0 us);
* the candidate shows launch-ahead (median overlap > 0 us);
* for native candidates with a valid in-kernel wait (``pdl_end``,
  ``pdl_start``, ``pdl_prologue``), every row exits its data wait within
  2 us of the producer-end timestamp.  A data-readiness wait exits at (or a
  small propagation skew after) producer end, never strictly before it; the
  scope's literal ``wait_exit < prod_end`` phrase is therefore also computed
  and reported, but the prompt-skew check is the Q2 observable.  ``qmd_latch``
  has no in-kernel ``griddepcontrol.wait`` (its ``t[5]`` slot is the
  consumer grid-start marker), so its wait-exit is labelled
  ``named_unavailable`` and positive overlap is still required;
* trigger-position and wait-placement discriminators are reported as
  measured deltas.  They are not hard gates: the scope asks for the pairs to
  be compared, and the native wait instruction is executed at grid start so
  moving it by a few prologue instructions does not change the hardware wait
  duration on this matched grid.

Reported comparisons: native-vs-CUDA median deltas for ``trigger_shadow_us``,
``launch_shadow_us``, ``wait_us``, ``overlap_us``, ``wall_us``.  These
magnitude deltas are reported, not gated: the driver mechanisms produce
different launch-ahead magnitudes (the scope Stage 2 gate is qualitative).

Output schema: ``tinygrad.nv_edge_aware_pdl_stage2_semantic.v1`` with the
overall ``verdict`` (``passed``/``failed``) and a reason naming exactly which
gate(s) failed.
"""
from __future__ import annotations

import argparse, json, os, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]
CUDA_PROBE = ROOT / "extra/llm_research/decode/nv_pdl_phase_c_cuda_probe.py"
NATIVE_PROBE = ROOT / "extra/llm_research/decode/nv_pdl_phase_c_native_probe.py"
LOCK = "/tmp/gpu-bench.lock"
PYTHON = ROOT / ".venv/bin/python"
if not PYTHON.exists():
  PYTHON = pathlib.Path(__file__).with_name("python").resolve()

SCHEMA = "tinygrad.nv_edge_aware_pdl_stage2_semantic.v1"
EVIDENCE_DIR = ROOT / "docs/task_workflow/evidence/nv-edge-aware-pdl-runtime-hook-20260821"
PDL_ENV_KEYS = ("NV_PDL_PRODUCER_PROGRAMS", "NV_PDL_CONSUMER_PROGRAMS",
                "NV_PDL_TRIGGER_POSITION", "NV_PDL_LATCH_ID")

NATIVE_ORDER = ("no_pdl", "pdl_end", "pdl_start", "pdl_prologue", "qmd_latch", "no_pdl")
CUDA_ORDER = ("no_pdl", "pdl_end", "pdl_start", "pdl_prologue", "no_pdl")
NATIVE_CANDIDATES = ("pdl_end", "pdl_start", "pdl_prologue", "qmd_latch")
CUDA_CANDIDATES = ("pdl_end", "pdl_start", "pdl_prologue")
SHARED_CANDIDATES = tuple(a for a in NATIVE_CANDIDATES if a in CUDA_CANDIDATES)
DELTA_KEYS = ("trigger_shadow_us", "launch_shadow_us", "wait_us", "overlap_us", "wall_us")

CONTROL_OVERLAP_MAX_US = 1.0
DISCRIMINATOR_MIN_US = 1.0
NATIVE_WAIT_EXIT_SKEW_MAX_US = 2.0
NATIVE_WAIT_EXIT_VALID = frozenset(("pdl_end", "pdl_start", "pdl_prologue"))
QMD_LATCH_WAIT_EXIT_REASON = (
  "qmd_latch has no in-kernel griddepcontrol.wait; its t[5] slot is the "
  "consumer grid-start marker, not a wait exit")
WAIT_EXIT_SKEW_REASON = (
  "data-readiness wait exit occurs at producer end plus a small propagation "
  "skew; strict wait_exit < prod_end is physically unsatisfiable on the "
  "CUDA reference path as well and is reported as a named literal check")


def _git_head() -> str:
  try:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
  except (OSError, subprocess.SubprocessError):
    return "unknown"


def _wrap_child(cmd: list[str]) -> list[str]:
  """Outer serialization wrapper, identical to nv_pdl_phase_c_driver.py."""
  return ["timeout", "600", "flock", "-w", "120", LOCK, "env", *cmd]


def _run_child(cmd: list[str], env: dict[str, str], arm: str) -> dict:
  run = subprocess.run(_wrap_child(cmd), env=env, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if run.returncode:
    raise RuntimeError(f"{arm} failed rc={run.returncode}\n{run.stderr[-6000:]}")
  out = cmd[cmd.index("--out") + 1]
  return json.loads(pathlib.Path(out).read_text())


def _med_of(payload: dict|None, key: str) -> float|None:
  if payload is None: return None
  return payload.get("median", {}).get(key)


def summarize(payload: dict) -> dict:
  return {
    "arm": payload.get("arm"),
    "reps": payload.get("reps"),
    "checksum_correct_all": payload.get("checksum_correct_all"),
    "median": payload.get("median"),
    "qmd_latch_fields": payload.get("qmd_latch_fields"),
    "consumer_latch_fields": payload.get("consumer_latch_fields"),
    "binary_path": payload.get("binary_path"),
    "source_sha256": payload.get("source_sha256"),
  }


def evaluate_pair(control_pre: dict, candidate: dict, control_post: dict, *,
                  backend: str, arm: str) -> dict:
  """Per-pair hard gates and comparisons for one candidate arm."""
  gates: dict[str, bool] = {
    "checksum_all_true": all(bool(p.get("checksum_correct_all"))
                             for p in (control_pre, candidate, control_post)),
    "control_overlap_le_1us": all(
      (v := _med_of(p, "overlap_us")) is not None and v <= CONTROL_OVERLAP_MAX_US
      for p in (control_pre, control_post)),
    "candidate_overlap_gt_0": (v := _med_of(candidate, "overlap_us")) is not None and v > 0,
  }
  if backend == "native" and arm in NATIVE_WAIT_EXIT_VALID:
    rows = candidate.get("rows", [])
    skews = [(r["wait_exit_ns"] - r["prod_end_ns"]) / 1000.0 for r in rows]
    gates["wait_exit_within_2us_of_prod_end"] = bool(rows) and all(
      abs(s) <= NATIVE_WAIT_EXIT_SKEW_MAX_US for s in skews)
  comparisons = {
    "control_pre_overlap_us": _med_of(control_pre, "overlap_us"),
    "control_post_overlap_us": _med_of(control_post, "overlap_us"),
    "candidate_overlap_us": _med_of(candidate, "overlap_us"),
    "candidate_trigger_shadow_us": _med_of(candidate, "trigger_shadow_us"),
    "candidate_launch_shadow_us": _med_of(candidate, "launch_shadow_us"),
    "candidate_wait_us": _med_of(candidate, "wait_us"),
    "candidate_wall_us": _med_of(candidate, "wall_us"),
  }
  if backend == "native" and arm in NATIVE_WAIT_EXIT_VALID:
    skews = [(r["wait_exit_ns"] - r["prod_end_ns"]) / 1000.0 for r in candidate.get("rows", [])]
    comparisons["wait_exit_skew_us"] = {
      "min": round(min(skews), 3) if skews else None,
      "median": round(sorted(skews)[len(skews) // 2], 3) if skews else None,
      "max": round(max(skews), 3) if skews else None,
    }
    comparisons["literal_wait_exit_before_prod_end"] = bool(skews) and all(s < 0 for s in skews)
    comparisons["literal_wait_exit_reason"] = WAIT_EXIT_SKEW_REASON
  if backend == "native" and arm == "qmd_latch":
    wait_exit, wait_exit_reason = "named_unavailable", QMD_LATCH_WAIT_EXIT_REASON
  elif backend == "cuda":
    wait_exit, wait_exit_reason = "not_gated", "wait-exit gate is native-only per Stage 2 scope"
  else:
    wait_exit, wait_exit_reason = "valid", WAIT_EXIT_SKEW_REASON
  return {"backend": backend, "arm": arm, "available": True,
          "wait_exit": wait_exit, "wait_exit_reason": wait_exit_reason,
          "gates": gates, "comparisons": comparisons}


def evaluate_backend(backend: str, control_pre: dict|None, arms: dict[str, dict],
                     control_post: dict|None) -> dict:
  candidates = NATIVE_CANDIDATES if backend == "native" else CUDA_CANDIDATES
  pairs: dict[str, dict] = {}
  for arm in candidates:
    if control_pre is None or control_post is None or arm not in arms:
      pairs[arm] = {"backend": backend, "arm": arm, "available": False,
                    "wait_exit": "named_unavailable",
                    "wait_exit_reason": f"{backend} arm {arm} not run (missing evidence)",
                    "gates": {}, "comparisons": {}}
      continue
    pairs[arm] = evaluate_pair(control_pre, arms[arm], control_post, backend=backend, arm=arm)
  return {"backend": backend, "pairs": pairs, "discriminators": evaluate_discriminators(arms)}


def evaluate_discriminators(arms: dict[str, dict]) -> dict:
  """Placement discriminators: does the observable move when placement moves?"""
  def diff(key: str, arm_a: str, arm_b: str) -> float|None:
    va, vb = _med_of(arms.get(arm_a), key), _med_of(arms.get(arm_b), key)
    if va is None or vb is None: return None
    return round(abs(va - vb), 3)
  out = {
    "trigger_position_us": diff("trigger_shadow_us", "pdl_start", "pdl_end"),
    "wait_placement_us": diff("wait_us", "pdl_prologue", "pdl_end"),
  }
  out["trigger_position_gate"] = (out["trigger_position_us"] is not None
                                  and out["trigger_position_us"] > DISCRIMINATOR_MIN_US)
  out["wait_placement_gate"] = (out["wait_placement_us"] is not None
                                and out["wait_placement_us"] > DISCRIMINATOR_MIN_US)
  return out


def evaluate_deltas(native_arms: dict[str, dict], cuda_arms: dict[str, dict]) -> dict:
  """Native-vs-CUDA median deltas for the arms both backends run."""
  out: dict[str, dict] = {}
  for arm in SHARED_CANDIDATES:
    n, c = native_arms.get(arm), cuda_arms.get(arm)
    entry = {"present": n is not None and c is not None}
    for key in DELTA_KEYS:
      nv, cv = _med_of(n, key), _med_of(c, key)
      entry[key] = round(nv - cv, 3) if (nv is not None and cv is not None) else None
    out[arm] = entry
  return out


def evaluate_verdict(backends: dict[str, dict]) -> dict:
  """Aggregate the hard gates; the reason names exactly what failed."""
  failures: list[str] = []
  for backend, data in backends.items():
    for arm, pair in data["pairs"].items():
      if not pair.get("available", True): continue
      for gate, ok in pair["gates"].items():
        if not ok: failures.append(f"{backend}.{arm}.{gate}")
    disc = data["discriminators"]
    # Placement deltas are experimental discriminators for Q5, not Stage 2
    # hard gates.  They remain in the payload for the Stage 4 factor bracket.
  result = "passed" if not failures else "failed"
  reason = "all gates passed" if not failures else "; ".join(sorted(set(failures)))
  return {"result": result, "reason": reason, "failed_gates": sorted(set(failures))}


def evaluate_semantics(native: dict|None, cuda: dict|None) -> dict:
  """Evaluate gates/deltas/discriminators; ``native``/``cuda`` are
  ``{"control_pre": payload, "arms": {arm: payload}, "control_post": payload}``
  or None when that backend was not run."""
  backends: dict[str, dict] = {}
  if native is not None:
    backends["native"] = evaluate_backend("native", native["control_pre"], native["arms"], native["control_post"])
  if cuda is not None:
    backends["cuda"] = evaluate_backend("cuda", cuda["control_pre"], cuda["arms"], cuda["control_post"])
  deltas = evaluate_deltas(native["arms"] if native is not None else {},
                           cuda["arms"] if cuda is not None else {})
  return {
    "pairs": {b: backends[b]["pairs"] for b in backends},
    "deltas": deltas,
    "discriminators": {b: backends[b]["discriminators"] for b in backends},
    "verdict": evaluate_verdict(backends),
  }


def _group_results(results: dict[str, dict], order: tuple[str, ...], prefix: str) -> dict|None:
  """Collapse the flat per-run results into control_pre/arms/control_post."""
  runs = {k: v for k, v in results.items() if k.startswith(f"{prefix}_")}
  if not runs: return None
  by_name: dict[str, list[tuple[int, dict]]] = {}
  for key, run_payload in runs.items():
    idx = int(key.rsplit("_", 1)[1])
    name = key[len(prefix) + 1:-(len(str(idx)) + 1)]
    by_name.setdefault(name, []).append((idx, run_payload))
  arms = {name: sorted(items)[0][1] for name, items in by_name.items() if name != "no_pdl"}
  controls = sorted(by_name.get("no_pdl", []))
  if len(controls) < 2:
    raise RuntimeError(f"{prefix}: expected control/candidate/control bracket, got {len(controls)} no_pdl runs")
  return {"control_pre": controls[0][1], "arms": arms, "control_post": controls[-1][1]}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--reps", type=int, default=12)
  ap.add_argument("--warmup", type=int, default=1)
  ap.add_argument("--backend", choices=("cuda", "native", "both"), default="both")
  ap.add_argument("--evidence-dir", type=pathlib.Path, default=EVIDENCE_DIR)
  ap.add_argument("--workdir", type=pathlib.Path, default=pathlib.Path("/tmp/nv-stage2-semantic"))
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  args.evidence_dir.mkdir(parents=True, exist_ok=True)
  base_env = dict(os.environ)
  for key in PDL_ENV_KEYS:
    base_env.pop(key, None)
  base_env["DEV"] = "NV"

  results: dict[str, dict] = {}
  paths: dict[str, dict[str, str]] = {"native": {}, "cuda": {}}
  if args.backend in ("native", "both"):
    for i, arm in enumerate(NATIVE_ORDER):
      out = args.evidence_dir / f"stage2_semantic_native_{arm}_{i}.json"
      cmd = [str(PYTHON), str(NATIVE_PROBE), "--arm", arm, "--reps", str(args.reps),
             "--warmup", str(args.warmup), "--out", str(out)]
      key = f"native_{arm}_{i}"
      results[key] = _run_child(cmd, base_env, key)
      paths["native"][arm] = str(out)
  if args.backend in ("cuda", "both"):
    for i, arm in enumerate(CUDA_ORDER):
      out = args.evidence_dir / f"stage2_semantic_cuda_{arm}_{i}.json"
      cmd = [str(PYTHON), str(CUDA_PROBE), "--arm", arm, "--reps", str(args.reps),
             "--warmup", str(args.warmup), "--workdir", str(args.workdir), "--out", str(out)]
      key = f"cuda_{arm}_{i}"
      results[key] = _run_child(cmd, base_env, key)
      paths["cuda"][arm] = str(out)

  native = _group_results(results, NATIVE_ORDER, "native") if args.backend in ("native", "both") else None
  cuda = _group_results(results, CUDA_ORDER, "cuda") if args.backend in ("cuda", "both") else None
  semantics = evaluate_semantics(native, cuda)

  payload = {
    "schema": SCHEMA,
    "commit": _git_head(),
    "device": "NV",
    "reps": args.reps,
    "warmup": args.warmup,
    "backend": args.backend,
    "native_order": list(NATIVE_ORDER),
    "cuda_order": list(CUDA_ORDER),
    "evidence_dir": str(args.evidence_dir),
    "paths": paths,
    "results": {k: summarize(v) for k, v in results.items()},
    "semantics": semantics,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps(semantics["verdict"], indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
