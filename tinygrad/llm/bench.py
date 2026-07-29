"""Canonical, deliberately fail-closed benchmark record surface for tinygrad.llm.

This module records the provenance needed to compare a future measured run.  It
does not turn metadata collection into a performance claim: execution is not
enabled until a verified generated-route/artifact authority is supplied.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, platform, subprocess, sys
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
  """Collect portable facts first; tinygrad probe failures remain visible."""
  facts: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version(),
                           "backend": os.environ.get("TINYGRAD_DEVICE"), "driver": None,
                           "state": "unknown"}
  try:
    # This is a core tinygrad helper, intentionally not a generated-route import.
    from tinygrad.llm.device_facts import scan_device_facts
    scanned = scan_device_facts()
    facts["tinygrad"] = scanned.to_json()
    facts["backend"] = scanned.backend or facts["backend"]
    facts["state"] = scanned.state
  except Exception as exc: facts["probe_error"] = f"{type(exc).__name__}: {exc}"
  return facts


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
    "device": device_facts(),
    "command": {"argv": list(sys.argv if argv is None else argv),
                "config": {"warmups": args.warmups, "samples": args.samples, "route_ids": args.route_id}},
    "routes": route_trace(args.route_id),
    "correctness": {"status": "not_run", "detail": "metadata-only record; no model execution occurred"},
    "measurement": {"status": "not_measured", "warmups": args.warmups, "samples": args.samples,
                    "throughput_tokens_per_s": None},
    "authority": {"status": "unverified", "throughput_authoritative": False,
                  "reason": "exact generated route plans and artifacts are not verified by this surface"},
  }


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
  """Small stable schema gate for consumers; never accepts an unsupported claim."""
  required = {"schema_version", "record_type", "git", "model", "device", "command", "routes",
              "correctness", "measurement", "authority"}
  missing = required - record.keys()
  if missing: raise ValueError(f"benchmark record missing keys: {sorted(missing)}")
  if record["schema_version"] != SCHEMA_VERSION or record["record_type"] != RECORD_TYPE: raise ValueError("unsupported benchmark record")
  if record["authority"].get("throughput_authoritative") is not False: raise ValueError("unverified benchmark surface cannot authorize throughput")
  if record["measurement"].get("throughput_tokens_per_s") is not None: raise ValueError("unverified benchmark surface cannot report throughput")
  for route in record["routes"]:
    if not route.get("route_id") or route.get("status") not in ("unproven", "generic"): raise ValueError("invalid route trace")
  return record


def parser() -> argparse.ArgumentParser:
  result = argparse.ArgumentParser(description="Emit a versioned, fail-closed tinygrad LLM benchmark record.")
  result.add_argument("--model", help="Local GGUF/model path to identify and SHA-256; it is never loaded.")
  result.add_argument("--warmups", type=int, default=0, help="Requested warmup count recorded in metadata (default: 0).")
  result.add_argument("--samples", type=int, default=0, help="Requested sample count recorded in metadata (default: 0).")
  result.add_argument("--route-id", action="append", default=[], help="Route identifier to trace; repeatable.")
  result.add_argument("--metadata-only", "--dry-run", action="store_true", help="Explicit no-execution mode (currently the only mode).")
  result.add_argument("--output", help="Write JSON record to this path instead of stdout.")
  return result


def main(argv: Sequence[str] | None = None) -> int:
  args = parser().parse_args(argv)
  if args.warmups < 0 or args.samples < 0: parser().error("--warmups and --samples must be non-negative")
  if not args.metadata_only: parser().error("--metadata-only is required until verified generated-route execution lands")
  record = validate_record(build_record(args, argv))
  encoded = json.dumps(record, sort_keys=True, indent=2) + "\n"
  if args.output: pathlib.Path(args.output).write_text(encoded)
  else: sys.stdout.write(encoded)
  return 0


if __name__ == "__main__": raise SystemExit(main())
