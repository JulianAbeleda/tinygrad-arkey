#!/usr/bin/env python3
import argparse, csv, json, os, pathlib, sys, time
from math import prod

from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.helpers import GlobalCounters
from tinygrad.llm.gguf import ggml_data_to_tensor

from extra.qk.layout import (
  GGML_Q4_K, GGUFInfo, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, model_shape_targets, pick_tensor, q4_k_reference,
  q4_k_weight_bytes, read_metadata, tensor_shape,
)
from extra.qk.modes import PrimitiveMode, primitive_mode_choices

def correctness_gate(raw_slice:Tensor, n:int, info:GGUFInfo) -> None:
  ref = q4_k_reference(raw_slice, n).reshape(*tensor_shape(info)).contiguous().realize()
  got = ggml_data_to_tensor(raw_slice, n, info.typ).reshape(*tensor_shape(info)).contiguous().realize()
  # Tensor equality is enough here: the gate is bit-exact relative to the frozen current expression.
  ok = (got == ref).all().numpy().item()
  if not ok: raise AssertionError(f"Q4_K correctness gate failed for {info.name}")

def bench(label:str, iters:int, q4_bytes:int, fn) -> dict[str, float|int|str|None]:
  fn().realize()
  GlobalCounters.reset()
  st = time.perf_counter()
  for _ in range(iters): fn().realize()
  dt = (time.perf_counter() - st) / iters
  dev_dt = GlobalCounters.time_sum_s / iters
  return {"name": label, "iters": iters, "ms": dt*1000, "per_s": 1/dt, "kernels": GlobalCounters.kernel_count / iters,
          "device_ms": dev_dt*1000 if dev_dt > 0 else None,
          "device_q4_eff_gbs": q4_bytes / dev_dt / 1e9 if dev_dt > 0 else None,
          "global_mem_mb": GlobalCounters.global_mem / iters / 1e6, "q4_weight_mb": q4_bytes / 1e6,
          "q4_eff_gbs": q4_bytes / dt / 1e9}

def emit(results:list[dict], fmt:str):
  if fmt == "json":
    print(json.dumps(results, indent=2, sort_keys=True))
  elif fmt == "csv":
    writer = csv.DictWriter(sys.stdout, fieldnames=sorted({k for r in results for k in r.keys()}))
    writer.writeheader()
    writer.writerows(results)
  else:
    for r in results:
      dev_eff = f"{r['device_q4_eff_gbs']:.2f} GB/s" if r["device_q4_eff_gbs"] is not None else "n/a"
      print(f"{r['tensor']} {r['shape']} {r['name']}: {r['ms']:.3f} ms ({r['per_s']:.2f}/s) "
            f"q4_eff={r['q4_eff_gbs']:.2f} GB/s device_q4_eff={dev_eff} "
            f"kernels={r['kernels']:.1f} mem={r['global_mem_mb']:.2f} MB")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Baseline GGUF Q4_K decode/matvec benchmark")
  parser.add_argument("gguf", type=pathlib.Path)
  parser.add_argument("--tensor", help="exact tensor name to benchmark")
  parser.add_argument("--device", default=None, help="tinygrad device, for example AMD or CPU")
  parser.add_argument("--iters", type=int, default=5)
  parser.add_argument("--seq-len", type=int, default=1, help="input rows for matmul; 1 is decode, >1 is prefill-shaped")
  parser.add_argument("--all-shapes", action="store_true", help="benchmark representative Q4_K decode GEMV shapes from model metadata")
  parser.add_argument("--max-shapes", type=int, default=None, help="cap representative shapes")
  parser.add_argument("--no-correctness", action="store_true", help="skip the mandatory correctness gate, for debugging only")
  parser.add_argument("--activation", choices=("random", "ones"), default="random", help="activation vector used by matvec benches")
  parser.add_argument("--seed", type=int, default=1337, help="seed for random activations")
  parser.add_argument("--primitive", action="store_true", help="also run the custom Q4_K GEMV primitive")
  parser.add_argument("--primitive-mode", choices=primitive_mode_choices(), default="partial")
  parser.add_argument("--primitive-parts", type=int, default=1)
  parser.add_argument("--primitive-row-group", type=int, default=1)
  parser.add_argument("--primitive-schedule", choices=("none", "auto"), default="none")
  parser.add_argument("--primitive-opt", action="append", default=None, help="explicit primitive opt OP:AXIS:ARG, can repeat")
  parser.add_argument("--primitive-unpack-check-rows", type=int, default=2)
  parser.add_argument("--primitive-tol", type=float, default=1e-2)
  parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
  parser.add_argument("--list", action="store_true", help="list Q4_K tensors and exit")
  args = parser.parse_args()
  if args.seq_len < 1: raise ValueError("--seq-len must be >= 1")
  if args.primitive and args.seq_len != 1: raise ValueError("--primitive only supports decode GEMV (--seq-len 1)")
  primitive_default_opts = [] if args.primitive_mode == PrimitiveMode.TILE_CUSTOM.value else ["LOCAL:0:32"]
  primitive_opts = tuple(args.primitive_opt if args.primitive_opt is not None else primitive_default_opts)

  meta = read_metadata(args.gguf)
  if args.list:
    for info in meta.infos:
      if info.typ == GGML_Q4_K: print(f"{info.name} dims={tensor_shape(info)} off={info.off}")
    raise SystemExit(0)

  targets = model_shape_targets(meta.infos, meta.kv, args.max_shapes) if args.all_shapes else [pick_tensor(meta.infos, args.tensor)]
  if args.format == "text":
    arch = meta.kv.get("general.architecture", "")
    cfg = {k: meta.kv.get(f"{arch}.{k}") for k in ("embedding_length", "feed_forward_length", "block_count", "attention.head_count",
                                                   "attention.head_count_kv") if f"{arch}.{k}" in meta.kv}
    print(f"model_config: arch={arch} {cfg}")
    print(f"variant: GGUF_Q4K_WIDE={os.getenv('GGUF_Q4K_WIDE', '0')} device={args.device or 'default'} "
          f"seq_len={args.seq_len} activation={args.activation}")
    if args.primitive:
      print(f"primitive: mode={args.primitive_mode} parts={args.primitive_parts} row_group={args.primitive_row_group} "
            f"schedule={args.primitive_schedule} opts={list(primitive_opts)}")

  raw = Tensor(args.gguf)
  if args.primitive:
    from extra.qk.quant.q4_k_gemv_primitive import (
      parse_opt, q4k_gemv_grouped_partial_kernel, q4k_gemv_kernel,
      q4k_gemv_packed_load_partial_kernel,
      q4k_gemv_partial_kernel, q4k_gemv_tile_custom_partial_kernel, q4k_gemv_vector_load_partial_kernel,
      q4k_unpack_kernel,
    )
    parsed_primitive_opts = tuple(parse_opt(x) for x in primitive_opts)
    raw_words = Tensor(args.gguf, dtype=dtypes.uint32)
  Tensor.manual_seed(args.seed)
  results = []
  for info in targets:
    n, shape, q4_bytes = prod(info.dims), tensor_shape(info), q4_k_weight_bytes(info)
    byte_start = meta.data_start + info.off
    raw_slice = raw[byte_start:byte_start+q4_bytes].to(args.device).contiguous().realize()
    if not args.no_correctness:
      correctness_gate(raw_slice, n, info)
      if args.format == "text": print(f"correctness: PASS {info.name}")

    decoded = ggml_data_to_tensor(raw_slice, n, info.typ).reshape(*shape).cast(dtypes.float16).contiguous().realize()
    x = (Tensor.randn((args.seq_len, decoded.shape[-1]), device=args.device, dtype=decoded.dtype) if args.activation == "random"
         else Tensor.ones((args.seq_len, decoded.shape[-1]), device=args.device, dtype=decoded.dtype)).realize()

    @TinyJit
    def matvec():
      return x.matmul(decoded.transpose())

    @TinyJit
    def decode_matvec():
      return x.matmul(ggml_data_to_tensor(raw_slice, n, info.typ).reshape(*shape).cast(dtypes.float16).transpose())

    base = {"tensor": info.name, "shape": "x".join(map(str, shape)), "ggml_type": info.typ,
            "device": args.device or "default", "seq_len": args.seq_len, "activation": args.activation, "q4_bytes": q4_bytes}
    results += [{**base, **bench("matmul_decoded", args.iters, q4_bytes, matvec)},
                {**base, **bench("decode_q4_k_plus_matmul", args.iters, q4_bytes, decode_matvec)}]
    if args.primitive:
      rows, k = shape
      if k % Q4_K_BLOCK_ELEMS != 0: raise ValueError(f"{info.name} K={k} is not Q4_K block aligned")
      if byte_start % 4 != 0: raise ValueError(f"{info.name} byte offset is not uint32 aligned: {byte_start}")
      row_bytes = k // Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES
      parts = min(args.primitive_parts, k // Q4_K_BLOCK_ELEMS)
      words = raw_words[byte_start//4:byte_start//4+q4_bytes//4].to(args.device).contiguous().realize()
      out = Tensor.empty(rows, dtype=dtypes.float32, device=args.device)
      partials = Tensor.empty(rows, parts, dtype=dtypes.float32, device=args.device)
      vector_partials = Tensor.empty(rows, parts, dtype=dtypes.float32, device=args.device)

      unpack_rows = min(args.primitive_unpack_check_rows, rows)
      if unpack_rows > 0:
        unpack_words = raw_words[byte_start//4:byte_start//4+(unpack_rows*row_bytes)//4].to(args.device).contiguous().realize()
        unpack_out = Tensor.empty(unpack_rows, k, dtype=dtypes.float32, device=args.device)
        unpack_got = unpack_out.custom_kernel(unpack_words, fxn=q4k_unpack_kernel(unpack_rows, k))[0].realize()
        unpack_ref = q4_k_reference(Tensor(args.gguf)[byte_start:byte_start+unpack_rows*row_bytes].to(args.device), unpack_rows*k).reshape(unpack_rows, k).realize()
        unpack_max_abs = (unpack_got - unpack_ref).abs().max().item()
        if args.format == "text": print(f"primitive_unpack_correctness: PASS {info.name} rows={unpack_rows} max_abs={unpack_max_abs:.6g}")
        if unpack_max_abs != 0: raise AssertionError(f"Q4_K primitive unpack correctness failed for {info.name}")

      @TinyJit
      def primitive_gemv():
        x_vec = x.reshape(k)
        if args.primitive_mode == PrimitiveMode.SERIAL.value:
          return out.custom_kernel(words, x_vec, fxn=q4k_gemv_kernel(rows, k, args.primitive_schedule, parsed_primitive_opts))[0]
        if args.primitive_mode == PrimitiveMode.PACKED_LOAD.value:
          partial = partials.custom_kernel(
            words, x_vec, fxn=q4k_gemv_packed_load_partial_kernel(rows, k, parts, args.primitive_schedule, parsed_primitive_opts))[0]
          return partial.sum(axis=1)
        if args.primitive_mode == PrimitiveMode.HOIST_SCALE_MIN.value:
          raise ValueError("hoist_scale_min mode was removed (benchmarked performance regressed vs packed_load)")
        if args.primitive_mode == PrimitiveMode.VECTOR_LOAD.value:
          partial = vector_partials.custom_kernel(
            words, x_vec, fxn=q4k_gemv_vector_load_partial_kernel(rows, k, parts, args.primitive_schedule, parsed_primitive_opts))[0]
          return partial.sum(axis=1)
        if args.primitive_mode == PrimitiveMode.GROUPED.value:
          partial = partials.custom_kernel(
            words, x_vec,
            fxn=q4k_gemv_grouped_partial_kernel(rows, k, parts, args.primitive_row_group, args.primitive_schedule, parsed_primitive_opts))[0]
        elif args.primitive_mode == PrimitiveMode.TILE_CUSTOM.value:
          partial = partials.custom_kernel(
            words, x_vec, fxn=q4k_gemv_tile_custom_partial_kernel(rows, k, parts, args.primitive_schedule, parsed_primitive_opts))[0]
        else:
          partial = partials.custom_kernel(words, x_vec, fxn=q4k_gemv_partial_kernel(rows, k, parts, args.primitive_schedule, parsed_primitive_opts))[0]
        return partial.sum(axis=1)

      ref_out = matvec().reshape(rows).realize()
      got = primitive_gemv().realize()
      primitive_max_abs = (got - ref_out).abs().max().item()
      if args.format == "text": print(f"primitive_gemv_correctness: PASS {info.name} max_abs={primitive_max_abs:.6g}")
      if primitive_max_abs > args.primitive_tol:
        raise AssertionError(f"Q4_K primitive GEMV correctness failed for {info.name}: max_abs={primitive_max_abs}")
      primitive_base = {**base, "primitive_mode": args.primitive_mode, "primitive_parts": parts,
                        "primitive_row_group": args.primitive_row_group,
                        "primitive_schedule": args.primitive_schedule, "primitive_opts": " ".join(primitive_opts),
                        "primitive_gemv_max_abs": primitive_max_abs}
      results.append({**primitive_base, **bench("q4k_primitive_gemv", args.iters, q4_bytes, primitive_gemv)})
  emit(results, args.format)
