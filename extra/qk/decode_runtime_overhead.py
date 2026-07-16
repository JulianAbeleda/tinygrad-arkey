#!/usr/bin/env python3
"""Genuine fixed-depth decode authority.

Each measurement starts from an independent generation request. The prompt is
prefilled through ``Transformer.generate`` to exactly ``ctx`` tokens before
decode timing begins; JIT capture is warmed in a separate request.

  W = production generate path, including one token ``item`` sync per step.
  D = the same production-selected model JITs, queued without per-token sync,
      followed by one final device synchronization (diagnostic only; it is not
      assumed to be an upper bound).
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, statistics, sys, tempfile, time

from extra.qk.decode_harness import DEFAULT_MODEL, csv_ints, decode_run_profile

SCHEMA = "tinygrad.decode.fixed_depth.v2"


def _atomic_json(path:pathlib.Path, payload:dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
  try:
    with os.fdopen(fd, "w") as f:
      json.dump(payload, f, indent=2, sort_keys=True)
      f.write("\n")
      f.flush()
      os.fsync(f.fileno())
    os.replace(temporary, path)
  except BaseException:
    try: os.unlink(temporary)
    except FileNotFoundError: pass
    raise


def _model_identity(path:str) -> dict:
  resolved = pathlib.Path(path).expanduser().resolve()
  stat = resolved.stat()
  identity = {"path": str(resolved), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
  identity["identity_sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
  return identity


def _token_evidence(tokens:list[int]) -> dict:
  encoded = ",".join(map(str, tokens)).encode()
  return {"count": len(tokens), "sha256": hashlib.sha256(encoded).hexdigest(), "first_token_ids": tokens[:16]}


def _host_residual(w_ms:float, d_ms:float) -> tuple[float|None, float|None]:
  """Only subtract D from W when D is empirically an upper-bound candidate."""
  if d_ms > w_ms: return None, None
  residual = w_ms - d_ms
  return residual, 100 * residual / w_ms


def _captured_program_count(jit) -> int | None:
  """Count concrete PROGRAM launches inside graphed or ungraphed JIT calls."""
  from tinygrad.uop.ops import Ops
  captured = getattr(jit, "captured", None)
  if captured is None: return None
  def count_target(target) -> int:
    if target.op is Ops.PROGRAM: return 1
    if target.op is Ops.CUSTOM_FUNCTION and target.arg == "graph" and target.src:
      return sum(count_target(call.src[0]) for call in target.src[0].src)
    return 0
  return sum(count_target(call.src[0]) for call in captured.linear.src)


def _make_prompt(ids:list[int], depth:int) -> list[int]:
  if depth < 1: raise ValueError("fixed decode depth must be positive")
  if not ids: raise ValueError("tokenizer produced no deterministic prompt tokens")
  return (ids * (1 + depth // len(ids)))[:depth]


def _reset(model) -> None:
  reset = getattr(model, "reset_generation_state", None)
  if reset is None: raise TypeError("decode authority requires Transformer.reset_generation_state()")
  reset()


def _prefill(model, prompt:list[int], chunk_size:int):
  """Populate exact prompt KV through production generate and return its first sampled token."""
  gen = model.generate(prompt.copy(), chunk_size=chunk_size, temperature=0.0)
  first = int(next(gen))
  return gen, first


def _route(model, start_pos, token_extent:int) -> bool:
  from tinygrad.llm.route_policy import should_use_flash_decode
  return bool(model.config.flash_decode and should_use_flash_decode(start_pos, token_extent))


def _warm_depth(model, prompt:list[int], chunk_size:int, warmup_decode:int) -> None:
  _reset(model)
  gen, _ = _prefill(model, prompt, chunk_size)
  try:
    for _ in range(warmup_decode): next(gen)
  finally: gen.close()


def _measure_w(model, dev, prompt:list[int], chunk_size:int, nmeas:int) -> tuple[float, list[float], list[int]]:
  _reset(model)
  gen, _ = _prefill(model, prompt, chunk_size)
  latencies, generated = [], []
  try:
    dev.synchronize()
    for _ in range(nmeas):
      started = time.perf_counter()
      generated.append(int(next(gen)))
      latencies.append(time.perf_counter() - started)
  finally: gen.close()
  return sum(latencies), latencies, generated


def _measure_d(model, dev, prompt:list[int], chunk_size:int, nmeas:int, max_context:int):
  from tinygrad import Tensor, UOp
  _reset(model)
  gen, first = _prefill(model, prompt, chunk_size)
  gen.close()
  start = len(prompt)
  v_sp = UOp.variable("start_pos", 0, max_context - 1)
  temp = Tensor([0.0])
  out = Tensor([[first]], dtype="int32").contiguous()
  routes = []
  dev.synchronize()
  started = time.perf_counter()
  for i in range(nmeas):
    use_flash = _route(model, v_sp.bind(start + i), 1)
    routes.append("flash" if use_flash else "sdpa")
    out = model(out, v_sp.bind(start + i), temp, use_flash=use_flash)
  out.realize()
  dev.synchronize()
  elapsed = time.perf_counter() - started
  final_token = int(out.item())
  return elapsed, routes, final_token


def main(argv:list[str] | None=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", default=os.environ.get("QK_MODEL", DEFAULT_MODEL), help="GGUF path")
  ap.add_argument("--ckpts", default=os.environ.get("QK_CKPTS"), help="comma-separated fixed decode depths")
  ap.add_argument("--max-context", type=int, default=int(os.environ.get("QK_MAX_CONTEXT", 4608)))
  ap.add_argument("--nmeas", type=int, default=int(os.environ.get("QK_NMEAS", 40)), help="decode tokens per repetition")
  ap.add_argument("--reps", type=int, default=int(os.environ.get("QK_REPS", 5)))
  ap.add_argument("--warmup-decode", type=int, default=int(os.environ.get("QK_WARMUP_DECODE", 3)))
  ap.add_argument("--chunk-size", type=int, default=int(os.environ.get("QK_CHUNK_SIZE", 32)))
  ap.add_argument("--out", required=True, help="unique output JSON for this invocation")
  args = ap.parse_args(argv)
  if args.reps < 1: raise ValueError("reps must be positive")
  if args.warmup_decode < 2: raise ValueError("warmup-decode must be at least 2 to capture the production TinyJit")
  if args.chunk_size < 1: raise ValueError("chunk-size must be positive")
  profile = decode_run_profile(ckpts=csv_ints(args.ckpts) if args.ckpts else None,
                               max_context=args.max_context, nmeas=args.nmeas)

  from tinygrad import Device
  from extra.llm.generate import load_model_and_tokenizer

  dev = Device[Device.DEFAULT]
  model, tokenizer = load_model_and_tokenizer(args.model, profile.max_context, seed=20260617)
  base_ids = (tokenizer.prefix() if hasattr(tokenizer, "prefix") else []) + \
             tokenizer.encode("the quick brown fox jumps. " * 800)

  rows = []
  for depth in profile.ckpts:
    prompt = _make_prompt(base_ids, depth)
    _warm_depth(model, prompt, args.chunk_size, args.warmup_decode)
    w_reps, d_reps = [], []
    route_reps, token_reps = [], []
    for rep in range(args.reps):
      w_elapsed, per_token, generated = _measure_w(model, dev, prompt, args.chunk_size, profile.nmeas)
      d_elapsed, routes, final_token = _measure_d(model, dev, prompt, args.chunk_size, profile.nmeas, profile.max_context)
      w_reps.append({"rep": rep, "elapsed_s": w_elapsed, "tok_s": profile.nmeas / w_elapsed,
                     "per_token_ms": [x * 1e3 for x in per_token]})
      d_reps.append({"rep": rep, "elapsed_s": d_elapsed, "tok_s": profile.nmeas / d_elapsed,
                     "final_token_id": final_token})
      route_reps.append(routes)
      token_reps.append(_token_evidence(generated))

    w_tok_s = [r["tok_s"] for r in w_reps]
    d_tok_s = [r["tok_s"] for r in d_reps]
    w_ms = 1e3 / statistics.median(w_tok_s)
    d_ms = 1e3 / statistics.median(d_tok_s)
    route_set = sorted({route for routes in route_reps for route in routes})
    jits = [model.rollout_jit_flash if route == "flash" else model.rollout_jit for route in route_set]
    programs = {route: _captured_program_count(jit) for route, jit in zip(route_set, jits)}
    host_ms, host_pct = _host_residual(w_ms, d_ms)
    row = {"ctx": depth, "fixed_depth": depth, "decode_tokens": profile.nmeas, "reps": args.reps,
           "route_sequence": route_reps[0], "route_sequences_identical": all(x == route_reps[0] for x in route_reps),
           "routes": route_set, "programs_per_token_by_route": programs,
           "wall_ms_W": w_ms, "dispatch_ms_D": d_ms, "host_sync_residual_ms": host_ms,
           "host_sync_pct_of_wall": host_pct, "tok_s_W": statistics.median(w_tok_s),
           "tok_s_D_diagnostic": statistics.median(d_tok_s),
           "D_interpretation": ("upper_bound_candidate" if host_ms is not None else
                                "not_an_upper_bound; host/runtime subtraction refused"),
           "W_reps": w_reps, "D_reps": d_reps,
           "prompt_evidence": _token_evidence(prompt), "generated_token_evidence": token_reps,
           "generated_reps_identical": len({x["sha256"] for x in token_reps}) == 1}
    row["flash"] = route_set == ["flash"]
    row["route"] = route_set[0] if len(route_set) == 1 else "mixed"
    rows.append(row)
    print(f"ctx {depth:5}: W {w_ms:6.2f}ms ({row['tok_s_W']:.2f} tok/s) | "
          f"D {d_ms:6.2f}ms ({row['tok_s_D_diagnostic']:.2f} tok/s) | "
          f"host-sync {f'{host_ms:.2f}ms ({host_pct:.1f}%)' if host_ms is not None else 'N/A (D slower than W)'} | "
          f"{','.join(route_set)}",
          file=sys.stderr, flush=True)

  valid_host = [row["host_sync_pct_of_wall"] for row in rows if row["host_sync_pct_of_wall"] is not None]
  median_host = statistics.median(valid_host) if valid_host else None
  identity = _model_identity(args.model)
  artifact = {"schema": SCHEMA, "artifact_version": 2, "created_unix_ns": time.time_ns(),
              "model": identity, "model_identity": identity, "model_id": pathlib.Path(args.model).stem,
              "tool": {"path": str(pathlib.Path(__file__).resolve()), "argv": list(sys.argv if argv is None else argv)},
              "device": {"tinygrad_device": Device.DEFAULT, "runtime_type": type(dev).__name__},
              "hardware": f"{Device.DEFAULT} / {type(dev).__name__}", "ckpts": list(profile.ckpts),
              "nmeas": profile.nmeas, "reps": args.reps, "max_context": profile.max_context,
              "workload": {"ckpts": list(profile.ckpts), "max_context": profile.max_context,
                           "decode_tokens": profile.nmeas, "reps": args.reps, "warmup_decode": args.warmup_decode,
                           "chunk_size": args.chunk_size, "temperature": 0.0, "seed": 20260617},
              "runtime_settings": {"kv_cache": "int8+fp16_scale" if model.config.kv_quant else "fp16",
                                   "flash_decode_capable": bool(model.config.flash_decode),
                                   "flash_decode_mode": os.environ.get("FLASH_DECODE", "auto"),
                                   "flash_decode_threshold": int(os.environ.get("FLASH_DECODE_THRESHOLD", "512")),
                                   "ring": bool(model.config.ring)},
              "method": "genuine prompt prefill; W=production generate item/token; D=same model JITs with final sync",
              "rows": rows, "median_host_sync_pct": median_host}
  out_path = pathlib.Path(args.out).expanduser().resolve()
  _atomic_json(out_path, artifact)
  print(f"artifact: {out_path}", file=sys.stderr)
  print("@@DONE@@")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
