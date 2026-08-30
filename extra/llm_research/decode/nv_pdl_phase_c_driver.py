#!/usr/bin/env python3
"""Phase C driver: same-grid CUDA-PDL vs native-QMD semantic discriminator.

Every arm runs as a fresh process under ``flock /tmp/gpu-bench.lock``.  The
driver itself never touches the GPU; it only serializes the child probes and
merges their in-kernel timestamp evidence.
"""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]
CUDA_PROBE = ROOT / "extra/llm_research/decode/nv_pdl_phase_c_cuda_probe.py"
NATIVE_PROBE = ROOT / "extra/llm_research/decode/nv_pdl_phase_c_native_probe.py"
LOCK = "/tmp/gpu-bench.lock"
PYTHON = ROOT / ".venv/bin/python"
if not PYTHON.exists():
  PYTHON = pathlib.Path(__file__).with_name("python").resolve()

SCHEMA = "tinygrad.nv_pdl_phase_c_driver.v1"
PDL_ENV_KEYS = ("NV_PDL_PRODUCER_PROGRAMS", "NV_PDL_CONSUMER_PROGRAMS",
                "NV_PDL_TRIGGER_POSITION", "NV_PDL_LATCH_ID")


def _run_child(cmd: list[str], env: dict[str, str], arm: str) -> dict:
  wrapped = ["timeout", "600", "flock", "-w", "120", LOCK, "env", *cmd]
  run = subprocess.run(wrapped, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if run.returncode:
    raise RuntimeError(f"{arm} failed rc={run.returncode}\n{run.stderr[-6000:]}")
  out = cmd[cmd.index("--out") + 1]
  return json.loads(pathlib.Path(out).read_text())


def _med(rows: list[dict], key: str) -> float:
  return round(statistics.median(r[key] for r in rows), 3)


def _summarize(payload: dict) -> dict:
  return {
    "reps": payload.get("reps"),
    "checksum_correct_all": payload.get("checksum_correct_all"),
    "median": payload.get("median"),
    "binary_path": payload.get("binary_path"),
    "qmd_latch_fields": payload.get("qmd_latch_fields"),
    "source_sha256": payload.get("source_sha256"),
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--reps", type=int, default=12)
  ap.add_argument("--warmup", type=int, default=1)
  ap.add_argument("--evidence-dir", type=pathlib.Path,
                  default=ROOT / "docs/task_workflow/evidence/nv-split-phase-pdl-causal-design-review-20260820")
  ap.add_argument("--out", type=pathlib.Path, required=True)
  ap.add_argument("--backend", choices=("cuda", "native", "both"), default="both")
  ap.add_argument("--workdir", type=pathlib.Path, default=pathlib.Path("/tmp/nv-pdl-phase-c"))
  args = ap.parse_args()

  args.evidence_dir.mkdir(parents=True, exist_ok=True)
  cuda_order = ("no_pdl", "pdl_end", "pdl_start", "pdl_prologue", "no_pdl")
  native_order = ("no_pdl", "pdl_end", "pdl_start", "pdl_prologue", "qmd_latch", "no_pdl")

  results: dict[str, dict] = {}
  base_env = dict(os.environ)
  for key in PDL_ENV_KEYS:
    base_env.pop(key, None)
  base_env["DEV"] = "NV"

  if args.backend in ("cuda", "both"):
    for i, arm in enumerate(cuda_order):
      out = args.evidence_dir / f"phase_c_cuda_{arm}_{i}.json"
      cmd = [str(PYTHON), str(CUDA_PROBE), "--arm", arm, "--reps", str(args.reps),
             "--warmup", str(args.warmup),
             "--workdir", str(args.workdir), "--out", str(out)]
      results[f"cuda_{arm}_{i}"] = _run_child(cmd, base_env, f"cuda:{arm}:{i}")

  if args.backend in ("native", "both"):
    for i, arm in enumerate(native_order):
      out = args.evidence_dir / f"phase_c_native_{arm}_{i}.json"
      cmd = [str(PYTHON), str(NATIVE_PROBE), "--arm", arm, "--reps", str(args.reps),
             "--warmup", str(args.warmup),
             "--out", str(out)]
      results[f"native_{arm}_{i}"] = _run_child(cmd, base_env, f"native:{arm}:{i}")

  # Phase C is a semantic gate, not an endpoint bracket.  Report the
  # discriminator observables and leave the hypothesis verdicts to the output
  # report, which must reconcile these with the Phase B construction census.
  def first(prefix):
    return next(v for k, v in results.items() if k.startswith(prefix))

  def cuda(key):
    return first(f"cuda_{key}_")

  def native(key):
    return first(f"native_{key}_")

  gates: dict = {"checksum": {}}
  if args.backend in ("cuda", "both"):
    gates.update({
      "cuda_pdl_fired": cuda("pdl_start")["median"]["overlap_us"] > 0,
      "cuda_trigger_position_moves_launch": (
        abs(cuda("pdl_start")["median"]["trigger_shadow_us"]
            - cuda("pdl_end")["median"]["trigger_shadow_us"]) > 1.0),
      "cuda_wait_placement_moves_wait": (
        abs(cuda("pdl_prologue")["median"]["wait_us"]
            - cuda("pdl_end")["median"]["wait_us"]) > 1.0),
    })
    gates["checksum"].update({
      "cuda_no_pdl": bool(cuda("no_pdl")["checksum_correct_all"]),
      "cuda_pdl_end": bool(cuda("pdl_end")["checksum_correct_all"]),
      "cuda_pdl_start": bool(cuda("pdl_start")["checksum_correct_all"]),
      "cuda_pdl_prologue": bool(cuda("pdl_prologue")["checksum_correct_all"]),
    })
  if args.backend in ("native", "both"):
    gates.update({
      "native_qmd_latch_overlaps": native("qmd_latch")["median"]["overlap_us"] > 0,
      "native_in_kernel_pdl_overlaps": native("pdl_start")["median"]["overlap_us"] > 0,
    })
    gates["checksum"].update({
      "native_no_pdl": bool(native("no_pdl")["checksum_correct_all"]),
      "native_pdl_end": bool(native("pdl_end")["checksum_correct_all"]),
      "native_pdl_start": bool(native("pdl_start")["checksum_correct_all"]),
      "native_pdl_prologue": bool(native("pdl_prologue")["checksum_correct_all"]),
      "native_qmd_latch": bool(native("qmd_latch")["checksum_correct_all"]),
    })

  payload = {
    "schema": SCHEMA,
    "reps": args.reps,
    "warmup": args.warmup,
    "cuda_order": list(cuda_order),
    "native_order": list(native_order),
    "results": {k: _summarize(v) for k, v in results.items()},
    "gates": gates,
    "evidence_dir": str(args.evidence_dir),
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps(gates, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
