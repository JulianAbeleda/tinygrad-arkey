#!/usr/bin/env python3
"""Keep one Qwen model and TinyGPU AMD device resident while sampling provider health."""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, signal, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.usbgpu.tests.qualify import (DEFAULT_APP, MONOTONIC_FIELDS, QualificationError, atomic_json, sha256,
                                        status_command, validate_environment, validate_power_continuity,
                                        validate_power_status, validate_status)

SCHEMA = "tinygrad.egpu.persistent-model-residency.v1"
RESOURCE_FIELDS = ("active_workload_leases", "active_bar_mappings", "active_dma_allocations")


class ResidencyInterrupted(QualificationError):
  def __init__(self, signum:int):
    self.signum = signum
    super().__init__(f"interrupted by signal {signum}")


def _idle_view(status:dict) -> dict:
  return status | {key:0 for key in RESOURCE_FIELDS}


def validate_loaded_sample(status:dict, power:dict, *, generation:int|None=None,
                           previous_status:dict|None=None, previous_power:dict|None=None) -> int:
  """Validate the normal healthy policy while requiring live workload resources."""
  validate_status(_idle_view(status))
  validate_power_status(power)
  if status["active_workload_leases"] != 1: raise QualificationError("loaded run requires exactly one workload lease")
  if status["active_bar_mappings"] == 0: raise QualificationError("loaded run lost its workload BAR mappings")
  if status["active_dma_allocations"] == 0: raise QualificationError("loaded run lost its DMA/VRAM allocations")
  current_generation = status["provider_generation"]
  if power["provider_generation"] != current_generation: raise QualificationError("loaded status provider generations differ")
  if generation is not None and current_generation != generation: raise QualificationError("provider generation changed during loaded residency")
  if power["last_canary_identity_dword"] != status["last_identity_dword"] or \
     power["last_canary_success_monotonic_ns"] < status["last_success_monotonic_ns"]:
    raise QualificationError("power-residency canary does not cover loaded keepalive sample")
  if previous_status is not None:
    if any(status[key] < previous_status[key] for key in MONOTONIC_FIELDS): raise QualificationError("loaded keeper counter regressed")
    if status["failures"] != previous_status["failures"]: raise QualificationError("loaded keeper failure count changed")
    if status["successes"] <= previous_status["successes"]: raise QualificationError("loaded keeper did not advance")
  if previous_power is not None: validate_power_continuity(previous_power, power, require_canary_advance=True)
  return current_generation


def _token_digest(token_ids:list[int]) -> str:
  return hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()


def run(model_path:pathlib.Path, out_path:pathlib.Path, *, duration_s:float=600, decode_interval_s:float=2,
        status_interval_s:float=30, max_context:int=1024, clock=time.monotonic, sleeper=time.sleep,
        status_reader=None, power_reader=None) -> int:
  if duration_s <= 0 or decode_interval_s <= 0 or status_interval_s <= 0: raise ValueError("durations must be positive")
  if max_context < 256: raise ValueError("max context must be at least 256")
  model_path = model_path.expanduser().resolve(); out_path = out_path.expanduser().resolve()
  if not model_path.is_file(): raise QualificationError("model does not exist")
  status_reader = status_reader or (lambda:status_command([str(DEFAULT_APP), "keepalive", "status"]))
  power_reader = power_reader or (lambda:status_command([str(DEFAULT_APP), "power", "status"]))
  environment = validate_environment()
  lock_snapshot = {key:os.environ.get(key) for key in ("TINYGRAD_GPU_LOCK_PATH", "TINYGRAD_GPU_LOCK_NONCE")}
  model_identity = {"path":str(model_path), "size_bytes":model_path.stat().st_size, "sha256":sha256(model_path)}
  started_unix_ns, process_started_monotonic_ns = time.time_ns(), time.monotonic_ns()
  samples, token_ids, first_failure = [], [], None
  loaded_started_monotonic_ns = loaded_ended_monotonic_ns = None
  generator = None
  previous_handlers = {}
  try:
    for sig in (signal.SIGINT, signal.SIGTERM):
      previous_handlers[sig] = signal.signal(sig, lambda signum, _frame: (_ for _ in ()).throw(ResidencyInterrupted(signum)))

    # Import tinygrad only after the exact AMD environment has been admitted.
    from tinygrad import Device
    from tinygrad.llm.generate import load_model_and_tokenizer

    device = Device[Device.DEFAULT]
    model, tokenizer = load_model_and_tokenizer(model_path, max_context, seed=20260617)
    prompt = (tokenizer.prefix() if hasattr(tokenizer, "prefix") else []) + \
             tokenizer.encode("The quick brown fox jumps over the lazy dog. TinyGPU loaded residency. " * 8)
    prompt = prompt[:32]
    if not prompt: raise QualificationError("tokenizer produced an empty prompt")
    generator = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
    token_ids.append(int(next(generator)))
    device.synchronize()
    loaded_started_monotonic_ns = clock()
    deadline = loaded_started_monotonic_ns + duration_s
    next_decode = loaded_started_monotonic_ns + decode_interval_s
    next_status = loaded_started_monotonic_ns
    generation = previous_status = previous_power = None

    def take_sample(label:str) -> None:
      nonlocal generation, previous_status, previous_power
      keepalive, power = status_reader(), power_reader()
      generation = validate_loaded_sample(keepalive, power, generation=generation,
                                          previous_status=previous_status, previous_power=previous_power)
      sample = {"label":label, "unix_ns":time.time_ns(), "elapsed_s":clock()-loaded_started_monotonic_ns,
                "token_count":len(token_ids), "keepalive":keepalive, "power":power}
      samples.append(sample); previous_status, previous_power = keepalive, power
      print(f"loaded-residency {sample['elapsed_s']:.1f}s tokens={len(token_ids)} generation={generation} "
            f"bars={keepalive['active_bar_mappings']} dma={keepalive['active_dma_allocations']}", file=sys.stderr, flush=True)

    while True:
      now = clock()
      if now >= next_status:
        take_sample("loaded-start" if not samples else "loaded-periodic")
        next_status += status_interval_s
        while next_status <= now: next_status += status_interval_s
        now = clock()
      if now >= deadline: break
      if now >= next_decode:
        decode_started = now
        token_ids.append(int(next(generator)))
        device.synchronize()
        next_decode = decode_started + decode_interval_s
        continue
      sleeper(min(next_decode, next_status, deadline) - now)

    if not samples or samples[-1]["elapsed_s"] < duration_s:
      take_sample("loaded-final")
    loaded_ended_monotonic_ns = clock()
    if loaded_ended_monotonic_ns - loaded_started_monotonic_ns < duration_s: raise QualificationError("loaded interval ended early")
    if len(token_ids) < 100: raise QualificationError("loaded interval produced fewer than 100 decode tokens")
  except BaseException as exc:
    first_failure = {"type":type(exc).__name__, "message":str(exc)}
    loaded_ended_monotonic_ns = clock()
  finally:
    if generator is not None:
      try: generator.close()
      except BaseException as exc:
        first_failure = first_failure or {"type":type(exc).__name__, "message":f"generator close: {exc}"}
    for sig, handler in previous_handlers.items(): signal.signal(sig, handler)
    ended_unix_ns, process_ended_monotonic_ns = time.time_ns(), time.monotonic_ns()
    artifact = {
      "schema":SCHEMA, "status":"passed" if first_failure is None else "failed", "first_failure":first_failure,
      "model":model_identity, "environment":environment, "lock":lock_snapshot,
      "requested_duration_s":duration_s, "decode_interval_s":decode_interval_s, "status_interval_s":status_interval_s,
      "max_context":max_context, "prompt_tokens":len(prompt) if "prompt" in locals() else None,
      "started_unix_ns":started_unix_ns, "ended_unix_ns":ended_unix_ns,
      "process_elapsed_s":(process_ended_monotonic_ns-process_started_monotonic_ns)/1e9,
      "loaded_started_monotonic_ns":loaded_started_monotonic_ns, "loaded_ended_monotonic_ns":loaded_ended_monotonic_ns,
      "loaded_elapsed_s":None if loaded_started_monotonic_ns is None else loaded_ended_monotonic_ns-loaded_started_monotonic_ns,
      "token_count":len(token_ids), "token_ids_sha256":_token_digest(token_ids), "token_ids":token_ids,
      "samples":samples,
    }
    atomic_json(out_path, artifact)
  print(out_path)
  return 0 if first_failure is None else (128 + first_failure.get("signal", 0) if first_failure.get("signal") else 1)


def main(argv:list[str]|None=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", type=pathlib.Path, required=True)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  ap.add_argument("--duration-s", type=float, default=600)
  ap.add_argument("--decode-interval-s", type=float, default=2)
  ap.add_argument("--status-interval-s", type=float, default=30)
  ap.add_argument("--max-context", type=int, default=1024)
  args = ap.parse_args(argv)
  return run(args.model, args.out, duration_s=args.duration_s, decode_interval_s=args.decode_interval_s,
             status_interval_s=args.status_interval_s, max_context=args.max_context)


if __name__ == "__main__": raise SystemExit(main())
