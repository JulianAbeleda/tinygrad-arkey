"""Canonical, deliberately fail-closed benchmark record surface for tinygrad.llm.

This module provides a small public control execution path and its provenance
record. It does not turn execution into a performance claim.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, json, os, pathlib, platform, subprocess, sys
from datetime import datetime, timezone
from typing import Any, Sequence

SCHEMA_VERSION = 1
RECORD_TYPE = "tinygrad.llm.benchmark"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str | None:
  """Return a file's SHA-256, or None when it is not an ordinary file."""
  candidate = pathlib.Path(path)
  if not candidate.is_file(): return None
  digest = hashlib.sha256()
  with candidate.open("rb") as handle:
    for chunk in iter(lambda: handle.read(chunk_size), b""): digest.update(chunk)
  return digest.hexdigest()


def git_state(cwd: str | os.PathLike[str] = REPO_ROOT) -> dict[str, Any]:
  """Best-effort repository identity; unknown is explicit rather than guessed."""
  def run(*args: str) -> str | None:
    try:
      value = subprocess.run(["git", "-C", os.fspath(cwd), *args], check=True, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip()
      return value or None
    except (OSError, subprocess.SubprocessError): return None
  commit, status = run("rev-parse", "HEAD"), run("status", "--porcelain")
  return {"commit": commit, "dirty": None if status is None else bool(status),
          "state": "ok" if commit is not None and status is not None else "unknown"}


def device_facts() -> dict[str, Any]:
  """CPU-control facts only: never open, enumerate, or probe an accelerator."""
  return {"platform": platform.platform(), "python": platform.python_version(), "backend": "CPU",
          "driver": None, "state": "cpu_control", "probe": "disabled_no_accelerator_probe"}


def route_trace(route_ids: Sequence[str]) -> list[dict[str, Any]]:
  # A route name alone is not evidence that a generated artifact was used.
  return [{"route_id": route_id, "status": "unproven", "plan_hash": None, "artifact_hash": None}
          for route_id in (route_ids or ["generic"])]


def build_record(args: argparse.Namespace, argv: Sequence[str] | None = None) -> dict[str, Any]:
  model_path = pathlib.Path(args.model).expanduser() if args.model else None
  model_hash = sha256_file(model_path) if model_path else None
  model_status = "present" if model_hash else ("not_requested" if model_path is None else "missing_or_not_file")
  return {
    "schema_version": SCHEMA_VERSION, "record_type": RECORD_TYPE,
    "recorded_at": datetime.now(timezone.utc).isoformat(), "git": git_state(),
    "model": {"path": None if model_path is None else str(model_path), "sha256": model_hash, "status": model_status},
    "target": {"requested": args.target, "effective": "CPU", "kind": "cpu_smoke"}, "device": device_facts(),
    "command": {"argv": list(sys.argv if argv is None else argv),
                "config": {"warmups": args.warmups, "samples": args.samples, "route_ids": args.route_id,
                           "phase": args.phase, "context": args.context, "control": "generic_fp16", "target": "CPU"}},
    "routes": route_trace(args.route_id),
    "execution": {"status": "not_run", "control": "generic_fp16", "trace": []},
    "correctness": {"status": "not_run", "detail": "metadata-only record; no model execution occurred"},
    "measurement": {"status": "not_measured", "warmups": args.warmups, "samples": args.samples,
                    "throughput_tokens_per_s": None},
    "authority": {"status": "unverified", "throughput_authoritative": False,
                  "reason": "exact generated route plans and artifacts are not verified by this surface"},
}


@contextlib.contextmanager
def _generic_fp16_control():
  """Force public prefill/decode selectors away from optimized routes."""
  keys = ("TINYGRAD_PREFILL_ROUTE", "TINYGRAD_DECODE_ROUTE")
  old = {key:os.environ.get(key) for key in keys}
  for key in keys: os.environ[key] = "fp16"
  try: yield
  finally:
    for key, value in old.items():
      if value is None: os.environ.pop(key, None)
      else: os.environ[key] = value


@contextlib.contextmanager
def _cpu_generic_control():
  """Pin model load and its one control inference to CPU without changing defaults."""
  from tinygrad.helpers import Context
  with _generic_fp16_control(), Context(DEV="CPU"):
    yield


def run_control(args: argparse.Namespace, record: dict[str, Any], *, loader=None, tensor_factory=None) -> dict[str, Any]:
  """Load a supplied GGUF and execute one ordinary tinygrad control dispatch.

  ``loader`` and ``tensor_factory`` make this boundary unit-testable without a
  model or accelerator. No elapsed time or throughput is reported.
  """
  if not args.model: raise ValueError("--model is required for --execute")
  if record["model"]["status"] != "present": raise ValueError("--model must name an ordinary local file")
  if args.phase == "decode" and args.context != 1: raise ValueError("--phase decode requires --context 1")
  if loader is None:
    from tinygrad.llm.model import Transformer
    loader = lambda path: Transformer.from_gguf(path, max_context=max(args.context, 1))
  if tensor_factory is None:
    from tinygrad import Tensor
    tensor_factory = lambda tokens: (Tensor([tokens], device="CPU"), Tensor([0.0], device="CPU"))
  trace = [{"event": "control_selected", "control": "generic_fp16"}]
  from tinygrad.llm.model import generic_llm_control
  with _cpu_generic_control(), generic_llm_control():
    model, _kv = loader(str(pathlib.Path(args.model).expanduser()))
    trace.append({"event": "model_loaded", "status": "completed"})
    tokens, temperature = tensor_factory([0] * args.context)
    output = model(tokens, 0, temperature)
    if hasattr(output, "realize"): output.realize()
    trace.append({"event": "dispatch_completed", "phase": args.phase, "context": args.context})
  record["execution"] = {"status": "completed", "control": "generic_fp16", "trace": trace}
  record["routes"] = [{"route_id":"generic_fp16", "status":"generic", "plan_hash":None, "artifact_hash":None}]
  record["correctness"] = {"status": "not_checked", "detail": "control dispatch completed; no reference comparison was supplied"}
  record["measurement"]["status"] = "not_measured"
  return record


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
  """Small stable schema gate for consumers; never accepts an unsupported claim."""
  required = {"schema_version", "record_type", "git", "model", "target", "device", "command", "routes",
              "correctness", "measurement", "authority", "execution"}
  missing = required - record.keys()
  if missing: raise ValueError(f"benchmark record missing keys: {sorted(missing)}")
  if record["schema_version"] != SCHEMA_VERSION or record["record_type"] != RECORD_TYPE: raise ValueError("unsupported benchmark record")
  if record["target"] != {"requested":"CPU", "effective":"CPU", "kind":"cpu_smoke"}: raise ValueError("public control target must be CPU")
  if record["authority"].get("throughput_authoritative") is not False: raise ValueError("unverified benchmark surface cannot authorize throughput")
  if record["measurement"].get("throughput_tokens_per_s") is not None: raise ValueError("unverified benchmark surface cannot report throughput")
  if record["execution"].get("status") not in ("not_run", "completed"): raise ValueError("invalid control execution status")
  for route in record["routes"]:
    if not route.get("route_id") or route.get("status") not in ("unproven", "generic"): raise ValueError("invalid route trace")
  return record


def parser() -> argparse.ArgumentParser:
  result = argparse.ArgumentParser(description="Emit a versioned, fail-closed tinygrad LLM benchmark record.")
  result.add_argument("--model", help="Local GGUF path. Required with --execute; never fetched or downloaded.")
  result.add_argument("--target", choices=("CPU",), default="CPU", help="Public control target; CPU only.")
  result.add_argument("--phase", choices=("prefill", "decode"), default="prefill", help="Control dispatch phase (default: prefill).")
  result.add_argument("--context", type=int, default=1, help="Token count for prefill; decode requires exactly 1.")
  result.add_argument("--warmups", type=int, default=0, help="Requested warmup count recorded in metadata (default: 0).")
  result.add_argument("--samples", type=int, default=0, help="Requested sample count recorded in metadata (default: 0).")
  result.add_argument("--route-id", action="append", default=[], help="Route identifier to trace; repeatable.")
  result.add_argument("--metadata-only", "--dry-run", action="store_true", help="Emit provenance only; do not load the model.")
  result.add_argument("--execute", action="store_true", help="Run one forced generic-fp16 control dispatch; no performance result is reported.")
  result.add_argument("--output", help="Write JSON record to this path instead of stdout.")
  return result


def main(argv: Sequence[str] | None = None) -> int:
  args = parser().parse_args(argv)
  if args.warmups < 0 or args.samples < 0 or args.context <= 0: parser().error("--warmups, --samples, and --context must be valid")
  if args.metadata_only and args.execute: parser().error("choose only one of --metadata-only or --execute")
  if not args.metadata_only and not args.execute: parser().error("choose --metadata-only or --execute")
  record = build_record(args, argv)
  if args.execute:
    try: record = run_control(args, record)
    except ValueError as exc: parser().error(str(exc))
  record = validate_record(record)
  encoded = json.dumps(record, sort_keys=True, indent=2) + "\n"
  if args.output: pathlib.Path(args.output).write_text(encoded)
  else: sys.stdout.write(encoded)
  return 0


if __name__ == "__main__": raise SystemExit(main())
