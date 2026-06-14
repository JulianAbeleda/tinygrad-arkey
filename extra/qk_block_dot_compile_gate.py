#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pathlib, subprocess, sys, time
from typing import Any

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt import Opt
from tinygrad.helpers import GlobalCounters, cdiv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp

from extra.q4_k_gemv_primitive import parse_opt, q4k_gemv_partial_kernel, q4k_unpack_kernel
from extra.qk_layout import (
  GGML_Q4_K, Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, pick_tensor, q4_k_reference, read_metadata, tensor_shape,
)
from extra.qk_packed_tile_closeout_diagnostic import parse_debug7_log

DEFAULT_ARTIFACT = pathlib.Path("bench/qk-block-dot-compile-gate-20260613")
DEFAULT_MODEL = pathlib.Path("~/models/Qwen3-8B-Q4_K_M.gguf")
DEFAULT_TENSOR = "blk.0.ffn_gate.weight"
ROWS, K, PARTS = 64, 4096, 1
V1_KERNEL = f"q4k_gemv_partial_{ROWS}_{K}_{PARTS}"


def qk_kernel_name(rows:int, k:int, parts:int) -> str:
  return f"q4k_block_dot_partial_{rows}_{k}_{parts}"


QK_KERNEL = qk_kernel_name(ROWS, K, PARTS)


def _kernel_info(name:str, opts:tuple[Opt, ...]) -> KernelInfo:
  return KernelInfo(name=name, opts_to_apply=opts if opts else ())


def qk_block_dot_source() -> str:
  return r"""({{
  // QK_BLOCK_DOT: block-local Q4_K load/decode/dot. Row/K loops remain in tinygrad UOps.
  const unsigned int *qk_words = (const unsigned int*)({0});
  const _Float16 *xv = (const _Float16*)({1});
  typedef unsigned int tg_uint4 __attribute__((ext_vector_type(4)));
  unsigned int fp = qk_words[0];
  float d = (float)(__builtin_bit_cast(_Float16, (unsigned short)(fp & 65535u)));
  float dmin = (float)(__builtin_bit_cast(_Float16, (unsigned short)((fp >> 16u) & 65535u)));
  float total = 0.0f;
  for (int pair = 0; pair < 4; pair++) {{
    int even = pair * 2;
    int odd = even + 1;
    unsigned int sb0 = (qk_words[1 + even/4] >> (8u * (unsigned int)(even%4))) & 255u;
    unsigned int mb0 = (qk_words[1 + (4+even)/4] >> (8u * (unsigned int)((4+even)%4))) & 255u;
    unsigned int sb1 = (qk_words[1 + odd/4] >> (8u * (unsigned int)(odd%4))) & 255u;
    unsigned int mb1 = (qk_words[1 + (4+odd)/4] >> (8u * (unsigned int)((4+odd)%4))) & 255u;
    unsigned int sc_even, mn_even, sc_odd, mn_odd;
    if (even < 4) {{
      sc_even = sb0 & 63u; mn_even = mb0 & 63u;
    }} else {{
      unsigned int slo = (qk_words[1 + (even-4)/4] >> (8u * (unsigned int)((even-4)%4))) & 255u;
      unsigned int mlo = (qk_words[1 + (4+even-4)/4] >> (8u * (unsigned int)((4+even-4)%4))) & 255u;
      unsigned int high = (qk_words[1 + (8+even-4)/4] >> (8u * (unsigned int)((8+even-4)%4))) & 255u;
      sc_even = (high & 15u) | ((slo >> 6u) << 4u);
      mn_even = (high >> 4u) | ((mlo >> 6u) << 4u);
    }}
    if (odd < 4) {{
      sc_odd = sb1 & 63u; mn_odd = mb1 & 63u;
    }} else {{
      unsigned int slo = (qk_words[1 + (odd-4)/4] >> (8u * (unsigned int)((odd-4)%4))) & 255u;
      unsigned int mlo = (qk_words[1 + (4+odd-4)/4] >> (8u * (unsigned int)((4+odd-4)%4))) & 255u;
      unsigned int high = (qk_words[1 + (8+odd-4)/4] >> (8u * (unsigned int)((8+odd-4)%4))) & 255u;
      sc_odd = (high & 15u) | ((slo >> 6u) << 4u);
      mn_odd = (high >> 4u) | ((mlo >> 6u) << 4u);
    }}
    for (int lane_vec = 0; lane_vec < 2; lane_vec++) {{
      tg_uint4 qv = *((const tg_uint4*)(qk_words + 4 + pair * 8 + lane_vec * 4));
      for (int lane = 0; lane < 4; lane++) {{
        unsigned int w = qv[lane];
        int pos_base = lane_vec * 16 + lane * 4;
        for (int nib = 0; nib < 4; nib++) {{
          unsigned int byte = (w >> (8u * (unsigned int)nib)) & 255u;
          unsigned int q_even = byte & 15u;
          unsigned int q_odd = byte >> 4u;
          float x_even = (float)xv[even*32 + pos_base + nib];
          float x_odd = (float)xv[odd*32 + pos_base + nib];
          total += (d * (float)sc_even * (float)q_even - dmin * (float)mn_even) * x_even;
          total += (d * (float)sc_odd * (float)q_odd - dmin * (float)mn_odd) * x_odd;
        }}
      }}
    }}
  }}
  total;
}})"""


def q4k_block_dot_partial_kernel(rows:int, k:int, parts:int, opts:tuple[Opt, ...]):
  k_blocks = k // Q4_K_BLOCK_ELEMS
  blocks_per_part = cdiv(k_blocks, parts)
  block_dot = qk_block_dot_source()

  def kernel(partials:UOp, words:UOp, x:UOp) -> UOp:
    row = UOp.range(rows, 0)
    part = UOp.range(parts, 1)
    blk_part = UOp.range(blocks_per_part, 2, axis_type=AxisType.REDUCE)
    blk = part * blocks_per_part + blk_part
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    qk_ptr = words.index(base, ptr=True)
    x_ptr = x.index(blk * Q4_K_BLOCK_ELEMS, ptr=True)
    contrib = UOp(Ops.QK_BLOCK_DOT, dtypes.float32, (qk_ptr, x_ptr), arg=block_dot)
    if k_blocks % parts != 0:
      raise ValueError("QK_BLOCK_DOT compile gate only supports exact split-K partitions")

    acc = partials[row, part].set(0.0)
    acc = partials[row, part].set(acc.after(blk_part)[row, part] + contrib, end=blk_part)
    return acc.end(row, part).sink(arg=_kernel_info(qk_kernel_name(rows, k, parts), opts))

  return kernel


def _sanitize_log(text:str, *, repo:pathlib.Path, model:pathlib.Path) -> str:
  out = text.replace(str(repo.resolve()), ".")
  if model.exists():
    home = pathlib.Path.home().resolve()
    try:
      out = out.replace(str(model.resolve()), "~/" + str(model.resolve().relative_to(home)))
    except ValueError:
      out = out.replace(str(model.resolve()), model.name)
  return "\n".join(line.rstrip() for line in out.splitlines()) + "\n"


def _portable(path:pathlib.Path, repo:pathlib.Path) -> str:
  try:
    return str(path.resolve().relative_to(repo.resolve()))
  except ValueError:
    return str(path)


def _primitive(mode:str, out:Tensor, partials:Tensor, words:Tensor, x:Tensor, rows:int, k:int, parts:int, opts:tuple[Opt, ...]) -> Tensor:
  if mode == "v1_partial":
    partial = partials.custom_kernel(words, x, fxn=q4k_gemv_partial_kernel(rows, k, parts, "none", opts))[0]
  elif mode == "qk_block_dot":
    partial = partials.custom_kernel(words, x, fxn=q4k_block_dot_partial_kernel(rows, k, parts, opts))[0]
  else:
    raise ValueError(f"unknown mode {mode!r}")
  return partial.sum(axis=1)


def run_one(args:argparse.Namespace) -> int:
  model = args.model.expanduser()
  meta = read_metadata(model)
  info = pick_tensor(meta.infos, args.tensor)
  if info.typ != GGML_Q4_K: raise ValueError(f"{info.name} is ggml_type={info.typ}, expected Q4_K")
  shape = tensor_shape(info)
  if len(shape) != 2: raise ValueError(f"{info.name} is not a matrix: shape={shape}")
  rows, k = min(args.rows, shape[0]), shape[1]
  if rows != ROWS or k != K:
    raise ValueError(f"compile gate is fixed to shape ({ROWS},{K}); got ({rows},{k})")
  parts = min(args.parts, k // Q4_K_BLOCK_ELEMS)
  opts = tuple(parse_opt(x) for x in args.opt)
  byte_start = meta.data_start + info.off
  row_bytes = k // Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES
  q4_bytes = rows * row_bytes
  nwords = q4_bytes // 4
  print(f"mode={args.mode} tensor={info.name} shape=({rows},{k}) parts={parts} opts={[str(x) for x in opts]} q4_bytes={q4_bytes}")

  raw_words = Tensor(model, dtype=dtypes.uint32)
  words = raw_words[byte_start//4:byte_start//4+nwords].to(args.device).contiguous().realize()
  Tensor.manual_seed(args.seed)
  x = Tensor.randn(k, dtype=dtypes.float16, device=args.device).realize()
  out = Tensor.empty(rows, dtype=dtypes.float32, device=args.device)
  partials = Tensor.empty(rows, parts, dtype=dtypes.float32, device=args.device)

  unpack_words = raw_words[byte_start//4:byte_start//4+row_bytes//4].to(args.device).contiguous().realize()
  unpack_out = Tensor.empty(1, k, dtype=dtypes.float32, device=args.device)
  unpack_got = unpack_out.custom_kernel(unpack_words, fxn=q4k_unpack_kernel(1, k))[0].realize()
  unpack_ref = q4_k_reference(Tensor(model)[byte_start:byte_start+row_bytes].to(args.device), k).reshape(1, k).realize()
  unpack_max_abs = (unpack_got - unpack_ref).abs().max().item()
  print(f"unpack_correctness: max_abs={unpack_max_abs:.6g}")
  if unpack_max_abs != 0: raise AssertionError("Q4_K unpack correctness failed")

  raw_u8 = Tensor(model)[byte_start:byte_start+q4_bytes].to(args.device)
  decoded = q4_k_reference(raw_u8, rows*k).reshape(rows, k).cast(dtypes.float16).realize()
  ref = (decoded.cast(dtypes.float32) * x.reshape(1, k).cast(dtypes.float32)).sum(axis=1).realize()
  got = _primitive(args.mode, out, partials, words, x, rows, k, parts, opts).realize()
  max_abs = (got - ref).abs().max().item()
  print(f"correctness: max_abs={max_abs:.6g}")
  if max_abs > 1e-2:
    print("got", got.numpy())
    print("ref", ref.numpy())
    raise AssertionError("QK_BLOCK_DOT GEMV correctness failed")

  fn = lambda: _primitive(args.mode, out, partials, words, x, rows, k, parts, opts).realize()
  fn()
  GlobalCounters.reset()
  st = time.perf_counter()
  for _ in range(args.iters): fn()
  wall_dt = (time.perf_counter() - st) / args.iters
  dev_dt = GlobalCounters.time_sum_s / args.iters
  print(f"bench: wall_ms={wall_dt*1000:.4f} wall_q4_gbs={q4_bytes/wall_dt/1e9:.2f} "
        f"device_ms={dev_dt*1000:.4f} device_q4_gbs={(q4_bytes/dev_dt/1e9 if dev_dt > 0 else 0):.2f}")
  return 0


def run_debug7_logs(*, repo:pathlib.Path, model:pathlib.Path, outdir:pathlib.Path, device:str, python:str,
                    tensor:str=DEFAULT_TENSOR) -> dict[str, pathlib.Path]:
  outdir.mkdir(parents=True, exist_ok=True)
  env = os.environ.copy()
  env["DEV"] = device
  env["DEBUG"] = "7"
  env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
  modes = {
    "v1_partial": ["--mode", "v1_partial", "--opt", "LOCAL:0:32"],
    "qk_block_dot": ["--mode", "qk_block_dot", "--opt", "LOCAL:0:32"],
  }
  logs = {}
  for mode, mode_args in modes.items():
    cmd = [
      python, "extra/qk_block_dot_compile_gate.py", "run-one", "--model", str(model), "--device", device,
      "--tensor", tensor, "--rows", str(ROWS), "--parts", str(PARTS), "--iters", "1", *mode_args,
    ]
    result = subprocess.run(cmd, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
    path = outdir / f"{mode}-debug7.log"
    path.write_text(_sanitize_log(result.stdout.decode("utf-8", errors="replace"), repo=repo, model=model))
    if result.returncode != 0:
      raise RuntimeError(f"{mode} DEBUG=7 run failed with exit {result.returncode}; see {path}")
    logs[mode] = path
  return logs


def summarize_gate(parsed:dict[str, dict[str, Any]]) -> dict[str, Any]:
  v1, qk = parsed["v1_partial"], parsed["qk_block_dot"]
  wide_loads = qk["global_load_b128"] > 0
  preserves_parallelism = qk["workgroup_size"] >= v1["workgroup_size"] and qk["workgroup_size"] > 1
  body_ok = qk["instruction_count"] <= v1["instruction_count"] * 2
  source_ok = qk["source_has_vector_type"] and qk["source_has_tg_uint4_load"]
  if not source_ok:
    decision = "qk_block_dot_compile_gate_rejected_source"
  elif not wide_loads:
    decision = "qk_block_dot_compile_gate_rejected_no_target_wide_load"
  elif not preserves_parallelism:
    decision = "qk_block_dot_compile_gate_rejected_scheduler_shape"
  elif not body_ok:
    decision = "qk_block_dot_compile_gate_rejected_target_body_size"
  else:
    decision = "qk_block_dot_compile_gate_passed_compile_shape"
  return {
    "decision": decision,
    "run_microbench": decision == "qk_block_dot_compile_gate_passed_compile_shape",
    "run_full_decode": False,
    "wide_loads": wide_loads,
    "source_ok": source_ok,
    "preserves_parallelism": preserves_parallelism,
    "target_body_size_ok": body_ok,
    "reason": {
      "wide_loads": "requires target global_load_b128 evidence",
      "scheduler_shape": "requires workgroup size >1 and at least v1 workgroup size",
      "target_body_size": "requires target instruction count <= 2x v1 before any microbench promotion",
      "full_decode": "full decode is never run from compile-gate evidence alone",
    },
  }


def build_report(logs:dict[str, pathlib.Path], *, repo:pathlib.Path) -> dict[str, Any]:
  parsed = {
    "v1_partial": parse_debug7_log(logs["v1_partial"].read_text(errors="replace"), kernel=V1_KERNEL, mode="v1_partial"),
    "qk_block_dot": parse_debug7_log(logs["qk_block_dot"].read_text(errors="replace"), kernel=QK_KERNEL, mode="qk_block_dot"),
  }
  return {
    "kind": "qk_block_dot_compile_gate",
    "schema_version": 1,
    "artifact": _portable(logs["v1_partial"].parents[1], repo),
    "shape": {"tensor": DEFAULT_TENSOR, "rows": ROWS, "k": K, "parts": PARTS},
    "modes": parsed,
    "summary": summarize_gate(parsed),
  }


def report_markdown(report:dict[str, Any]) -> str:
  lines = [
    "# QK_BLOCK_DOT Compile Gate",
    "",
    f"Decision: `{report['summary']['decision']}`",
    "",
    "This is a compile-shape gate only. It does not add runtime integration,",
    "full-decode measurement, or a promoted policy family.",
    "",
    "| mode | workgroup | group ids | local ids | source vector | target inst | mem inst | global_load_b128 | global_load_b32 | global_load_b64 | last DEBUG time |",
    "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for mode in ("v1_partial", "qk_block_dot"):
    row = report["modes"][mode]
    last = row["last_profile_time_us"]
    last_text = "n/a" if last is None else f"{last:.2f} us"
    lines.append(
      f"| `{mode}` | `{row['workgroup_size']}` | `{row['group_counts']}` | `{row['local_counts']}` | "
      f"`{row['source_has_vector_type']}` | `{row['instruction_count']}` | `{row['memory_instruction_count']}` | "
      f"`{row['global_load_b128']}` | `{row['global_load_b32']}` | `{row['global_load_b64']}` | `{last_text}` |"
    )
  lines += [
    "",
    "## Gate Interpretation",
    "",
    f"- source vector evidence: `{report['summary']['source_ok']}`",
    f"- target wide-load evidence: `{report['summary']['wide_loads']}`",
    f"- preserves scheduler parallelism: `{report['summary']['preserves_parallelism']}`",
    f"- target body size within gate: `{report['summary']['target_body_size_ok']}`",
    f"- run repeated microbench next: `{report['summary']['run_microbench']}`",
    f"- run full decode next: `{report['summary']['run_full_decode']}`",
    "",
  ]
  return "\n".join(lines)


def write_artifact(report:dict[str, Any], artifact:pathlib.Path) -> None:
  artifact.mkdir(parents=True, exist_ok=True)
  (artifact / "compile-gate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  md = report_markdown(report)
  (artifact / "compile-gate.md").write_text(md)
  (artifact / "README.md").write_text(md)


def main() -> int:
  parser = argparse.ArgumentParser(description="QK_BLOCK_DOT compile-shape gate")
  sub = parser.add_subparsers(dest="cmd")
  run = sub.add_parser("run-one")
  run.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
  run.add_argument("--tensor", default=DEFAULT_TENSOR)
  run.add_argument("--device", default="AMD")
  run.add_argument("--rows", type=int, default=ROWS)
  run.add_argument("--parts", type=int, default=PARTS)
  run.add_argument("--iters", type=int, default=1)
  run.add_argument("--mode", choices=("v1_partial", "qk_block_dot"), required=True)
  run.add_argument("--opt", action="append", default=[])
  run.add_argument("--seed", type=int, default=1337)

  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  parser.add_argument("--artifact", type=pathlib.Path, default=DEFAULT_ARTIFACT)
  parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--tensor", default=DEFAULT_TENSOR)
  parser.add_argument("--python", default=sys.executable)
  parser.add_argument("--reuse", action="store_true")
  args = parser.parse_args()

  if args.cmd == "run-one": return run_one(args)

  repo = args.repo.resolve()
  artifact = (repo / args.artifact).resolve() if not args.artifact.is_absolute() else args.artifact.resolve()
  source_dir = artifact / "source"
  model = args.model.expanduser().resolve()
  if args.reuse:
    logs = {mode: source_dir / f"{mode}-debug7.log" for mode in ("v1_partial", "qk_block_dot")}
    missing = [str(path) for path in logs.values() if not path.exists()]
    if missing: raise FileNotFoundError(f"--reuse requested but logs are missing: {missing}")
  else:
    if not model.exists(): raise FileNotFoundError(f"model not found: {model}")
    logs = run_debug7_logs(repo=repo, model=model, outdir=source_dir, device=args.device, python=args.python, tensor=args.tensor)
  report = build_report(logs, repo=repo)
  write_artifact(report, artifact)
  print(report_markdown(report))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
