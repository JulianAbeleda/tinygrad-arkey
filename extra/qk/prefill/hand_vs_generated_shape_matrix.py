#!/usr/bin/env python3
"""Compare generated native-ISA DBUF candidates against hand LDS2 shapes.

This is a thin matrix harness over existing authorities:
- generated timing/correctness: extra.qk.prefill_v2_schedule_search._run_config
- hand timing/correctness: extra.qk.prefill.wmma.build_gemm_lds2 launched via Tensor.custom_kernel

The shape label is (wm, wn) for hand LDS2 and (u0, u1) for generated.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time, traceback
from contextlib import contextmanager
from typing import Iterable

import numpy as np

sys.path.insert(0, os.getcwd())

from tinygrad import Tensor, Device, TinyJit, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.engine.realize import Estimates
from tinygrad.helpers import colored
from tinygrad.uop.ops import KernelInfo, Ops, UOp

from extra.qk.prefill.wmma import LDS2WaitPolicy, build_gemm_lds2
from extra.qk.prefill_v2_schedule_search import _compile_native_program, _run_config
from extra.qk.timing_harness import add_clock_pin_arg, pinned_peak_from_env, set_clock_pin_env


DEFAULT_DBUF_ENV = {
  "DEV": "AMD:ISA",
  "AMD_ISA_WMMA_B128_FRAG": "1",
  "AMD_ISA_REG_ACCUM": "1",
  "AMD_ISA_WAITCNT_TARGETED": "1",
  "PREFILL_TC_LOCAL_STAGE": "both",
  "PREFILL_TC_LOCAL_STAGE_WITH_LOCAL": "1",
  "PREFILL_TC_LOCAL_STAGE_B_TILEKEY": "1",
  "PREFILL_TC_LOCAL_STAGE_POST": "1",
  "PREFILL_LDS_PACK_WITHLOCAL_B128": "1",
  "PREFILL_DBUF": "1",
  "PREFILL_DBUF_LDS_CONST_IMM": "1",
  "PREFILL_DBUF_LDS_INDEX_SPLIT": "1",
  "PREFILL_DBUF_LDS_STORE_BASE_SPLIT": "1",
  "PREFILL_DBUF_DIRECT_B128_CHAIN": "1",
  "PREFILL_DBUF_LDS_ADDR_USE_DEP": "1",
  "REGALLOC_ADDR_REMAT": "1",
}


@contextmanager
def _patched_env(env: dict[str, str]):
  old = {k: os.environ.get(k) for k in env}
  os.environ.update(env)
  try:
    yield
  finally:
    for k, v in old.items():
      if v is None: os.environ.pop(k, None)
      else: os.environ[k] = v


def _parse_shapes(raw: str) -> list[tuple[int, int]]:
  out = []
  for item in raw.split(";"):
    if not item.strip(): continue
    a, b = item.split(",", 1)
    out.append((int(a), int(b)))
  return out


def _lds2_resource(m: int, n: int, k: int, wm: int, wn: int, waves_m: int, waves_n: int, bk: int, pad: int, dbuf: int) -> dict:
  threads = waves_m * waves_n * 32
  bm, bn = waves_m * wm * 16, waves_n * wn * 16
  cpr = bk // 8
  rstride = threads // cpr
  loads_a, loads_b = bm // rstride, bn // rstride
  sa = bk * 2 + pad
  sb = bk * 2 + pad
  bufsz = sa * bm + sb * bn
  lds_bytes = bufsz * (2 if dbuf else 1)
  fa = 10
  fb = fa + wm * 8
  accb = fb + wn * 8
  cta = accb + wm * wn * 8
  ctb = cta + loads_a * 4
  scr = ctb + loads_b * 4
  return {
    "tile_m": bm, "tile_n": bn, "threads": threads, "lds_bytes": lds_bytes,
    "scratch_end_vgpr": scr + 2, "acc_vgpr": wm * wn * 8,
    "loads_a": loads_a, "loads_b": loads_b,
  }


def _mn(inst) -> str:
  return str(inst).split("(", 1)[0]


def _inst_bytes(insts) -> int:
  total = 0
  for inst in insts:
    if isinstance(inst, tuple): continue
    total += len(inst.to_bytes())
  return total


def _structure_metrics(insts) -> dict:
  counts: dict[str, int] = {}
  inst_count = 0
  for inst in insts:
    if isinstance(inst, tuple): continue
    inst_count += 1
    counts[_mn(inst)] = counts.get(_mn(inst), 0) + 1
  wmma = counts.get("v_wmma_f32_16x16x16_f16", 0)
  def per(name: str) -> float:
    return round(counts.get(name, 0) / wmma, 3) if wmma else 0.0
  memory_ops = sum(counts.get(x, 0) for x in (
    "global_load_b128", "global_load_u16", "global_store_b16",
    "ds_store_b128", "ds_store_b64", "ds_store_b32", "ds_store_b16",
    "ds_load_b128",
  ))
  return {
    "instruction_count": inst_count,
    "byte_count": _inst_bytes(insts),
    "wmma_count": wmma,
    "waitcnt_count": counts.get("s_waitcnt", 0),
    "global_load_b128_count": counts.get("global_load_b128", 0),
    "ds_store_b128_count": counts.get("ds_store_b128", 0),
    "ds_load_b128_count": counts.get("ds_load_b128", 0),
    "barrier_count": counts.get("s_barrier", 0),
    "memory_op_count": memory_ops,
    "inst_per_wmma": round(inst_count / wmma, 3) if wmma else 0.0,
    "bytes_per_wmma": round(_inst_bytes(insts) / wmma, 3) if wmma else 0.0,
    "waitcnt_per_wmma": per("s_waitcnt"),
    "global_b128_per_wmma": per("global_load_b128"),
    "ds_store_b128_per_wmma": per("ds_store_b128"),
    "ds_load_b128_per_wmma": per("ds_load_b128"),
    "barrier_per_wmma": per("s_barrier"),
    "memops_per_wmma": round(memory_ops / wmma, 3) if wmma else 0.0,
  }


def _generated_structure(m: int, n: int, k: int, u0: int, u1: int, loc: int, unr: int) -> dict:
  env = {**os.environ, "STRUCTURE_WORKER": "1", "MM": str(m), "OUTF": str(n), "INF": str(k),
         "U0": str(u0), "U1": str(u1), "LOC": str(loc), "UNR": str(unr), "PYTHONPATH": os.getcwd()}
  p = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, timeout=180)
  for ln in p.stdout.splitlines():
    if ln.startswith("STRUCTURE_RESULT "): return json.loads(ln[len("STRUCTURE_RESULT "):])
  return {"structure_status": "no-result", "structure_stdout_tail": p.stdout.splitlines()[-8:],
          "structure_stderr_tail": p.stderr.splitlines()[-8:]}


def _generated_wmma_chain_trace(m: int, n: int, k: int, u0: int, u1: int, loc: int, unr: int, max_rows: int) -> dict:
  env = {**os.environ, "WMMA_CHAIN_TRACE_WORKER": "1", "MM": str(m), "OUTF": str(n), "INF": str(k),
         "U0": str(u0), "U1": str(u1), "LOC": str(loc), "UNR": str(unr), "TRACE_MAX_ROWS": str(max_rows),
         "PYTHONPATH": os.getcwd()}
  p = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, timeout=180)
  for ln in p.stdout.splitlines():
    if ln.startswith("WMMA_CHAIN_TRACE_RESULT "): return json.loads(ln[len("WMMA_CHAIN_TRACE_RESULT "):])
  return {"trace_status": "no-result", "trace_stdout_tail": p.stdout.splitlines()[-8:],
          "trace_stderr_tail": p.stderr.splitlines()[-8:]}


def _generated_structure_worker() -> None:
  from tinygrad.uop.ops import Ops
  from extra.qk.prefill import native_isa_l4_stream_probe as sp
  m, n, k = int(os.environ["MM"]), int(os.environ["OUTF"]), int(os.environ["INF"])
  u0, u1, loc, unr = int(os.environ["U0"]), int(os.environ["U1"]), int(os.environ["LOC"]), int(os.environ["UNR"])
  prg = _compile_native_program(m, n, k, u0, u1, loc, unr)
  lin_uop = next(u for u in prg.src if u.op is Ops.LINEAR)
  final_uops = sp._final_stream(Device[Device.DEFAULT].renderer, lin_uop.src)
  insts = sp._insts_from_uops(final_uops)
  print("STRUCTURE_RESULT", json.dumps(_structure_metrics(insts)))


def _generated_wmma_chain_trace_worker() -> None:
  from tinygrad.uop.ops import Ops
  from extra.qk.prefill import native_isa_l4_stream_probe as sp
  m, n, k = int(os.environ["MM"]), int(os.environ["OUTF"]), int(os.environ["INF"])
  u0, u1, loc, unr = int(os.environ["U0"]), int(os.environ["U1"]), int(os.environ["LOC"]), int(os.environ["UNR"])
  max_rows = int(os.environ.get("TRACE_MAX_ROWS", "0"))
  prg = _compile_native_program(m, n, k, u0, u1, loc, unr)
  lin_uop = next(u for u in prg.src if u.op is Ops.LINEAR)
  final_uops = sp._final_stream(Device[Device.DEFAULT].renderer, lin_uop.src)
  insts = sp._insts_from_uops(final_uops)
  trace = sp.wmma_chain_trace(insts, max_rows=max_rows or None)
  trace.update({"trace_status": "ok", "shape": f"{u0}x{u1}", "m": m, "n": n, "k": k, "loc": loc, "unr": unr})
  print("WMMA_CHAIN_TRACE_RESULT", json.dumps(trace))


def _hand_kernel(m: int, n: int, k: int, wm: int, wn: int, waves_m: int, waves_n: int, bk: int, pad: int, dbuf: int,
                 wait_policy: LDS2WaitPolicy | None = None, plrab: int = 0):
  insts = build_gemm_lds2(m, n, k, waves_m, waves_n, wm, wn, bk, pad, dbuf, PLRAB=plrab, wait_policy=wait_policy)
  res = _lds2_resource(m, n, k, wm, wn, waves_m, waves_n, bk, pad, dbuf)
  lds_bytes = max(res["lds_bytes"], 65536 // 8)
  grid = (n // res["tile_n"], m // res["tile_m"], 1)
  name = f"hand_lds2_matrix_{m}_{n}_{k}_{wm}x{wn}"

  def asm_kernel(A, Bt, C):
    lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=lds_bytes, addrspace=AddrSpace.LOCAL), (), "lds")
    g = [UOp.special(grid[0], "gidx0"), UOp.special(grid[1], "gidx1")]
    sink = UOp.sink(A.base, Bt.base, C.base, lds, *g, UOp.special(res["threads"], "lidx0"),
                    arg=KernelInfo(name=colored(name, "cyan"),
                                   estimates=Estimates(ops=m*n*k*2, mem=(m*k+n*k+m*n)*2)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                                 UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))
  return asm_kernel, res | _structure_metrics(insts) | {"lds_alloc_bytes": lds_bytes}


def _run_hand(m: int, n: int, k: int, wm: int, wn: int, waves_m: int, waves_n: int, bk: int, pad: int, dbuf: int,
              reps: int, iters: int, wait_policy: LDS2WaitPolicy | None = None, plrab: int = 0) -> dict:
  row = {"shape": f"{wm}x{wn}", "wm": wm, "wn": wn, "status": "?", "tflops": 0.0}
  try:
    asm_kernel, resource = _hand_kernel(m, n, k, wm, wn, waves_m, waves_n, bk, pad, dbuf, wait_policy=wait_policy,
                                        plrab=plrab)
    row.update(resource)
    rng = np.random.default_rng(0)
    a_np = (rng.standard_normal((m, k)) * 0.1).astype(np.float16)
    b_np = (rng.standard_normal((n, k)) * 0.1).astype(np.float16)
    ref = a_np.astype(np.float32) @ b_np.astype(np.float32).T
    refn = np.sqrt(np.mean(ref**2)) + 1e-9
    a, b = Tensor(a_np), Tensor(b_np)
    c = Tensor.empty(m, n, dtype=dtypes.half, device=a.device).contiguous()
    out = Tensor.custom_kernel(a, b, c, fxn=asm_kernel)[2]
    got = out.float().numpy()
    rr = float(np.sqrt(np.mean((got - ref)**2)) / refn)
    row["rel_rmse"] = rr
    if not np.isfinite(rr) or rr > 2e-2:
      row["status"] = f"WRONG rr={rr:.1e}"
      return row
    j = TinyJit(lambda: Tensor.custom_kernel(a, b, c, fxn=asm_kernel)[2].realize())
    dev = Device[Device.DEFAULT]
    with pinned_peak_from_env() as pin_prov:
      if pin_prov is not None: row["clock_pin"] = pin_prov
      for _ in range(5): j()
      dev.synchronize()
      ts = []
      for _ in range(reps):
        dev.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): j()
        dev.synchronize()
        ts.append((time.perf_counter() - t0) / iters * 1e3)
    row["ms_min"] = round(min(ts), 4)
    row["tflops"] = round(2*m*n*k/min(ts)*1e-12*1e3, 2)
    row["status"] = "ok"
  except Exception as e:
    row["status"] = type(e).__name__
    row["message"] = str(e)
    row["traceback_tail"] = traceback.format_exc().splitlines()[-8:]
    try: row.update(_lds2_resource(m, n, k, wm, wn, waves_m, waves_n, bk, pad, dbuf))
    except Exception: pass
  return row


def _run_generated(m: int, n: int, k: int, u0: int, u1: int, loc: int, unr: int, pin_clock: bool,
                   wmma_chain_trace: bool=False, trace_max_rows: int=0) -> dict:
  row = _run_config(m, n, k, u0, u1, loc, unr, pin_clock=pin_clock)
  row["shape"] = f"{u0}x{u1}"
  try:
    row.update(_generated_structure(m, n, k, u0, u1, loc, unr))
  except Exception as e:
    row["structure_status"] = type(e).__name__
    row["structure_message"] = str(e)
  if wmma_chain_trace:
    try:
      row["wmma_chain_trace"] = _generated_wmma_chain_trace(m, n, k, u0, u1, loc, unr, trace_max_rows)
    except Exception as e:
      row["wmma_chain_trace"] = {"trace_status": type(e).__name__, "trace_message": str(e)}
  return row


def main(argv: Iterable[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--shapes", default="2,2;4,2;2,4",
                  help="semicolon-separated wm,wn/u0,u1 list. 4,4 is parked on gfx1100 and must be requested explicitly.")
  ap.add_argument("--m", type=int, default=512)
  ap.add_argument("--n", type=int, default=5120)
  ap.add_argument("--k", type=int, default=5120)
  ap.add_argument("--loc", type=int, default=2, help="generated _run_config LOC")
  ap.add_argument("--unr", type=int, default=2, help="generated _run_config UNR")
  ap.add_argument("--waves-m", type=int, default=1)
  ap.add_argument("--waves-n", type=int, default=1)
  ap.add_argument("--bk", type=int, default=32)
  ap.add_argument("--pad", type=int, default=0)
  ap.add_argument("--dbuf", type=int, default=1)
  ap.add_argument("--plrab", type=int, default=0)
  ap.add_argument("--hand-reps", type=int, default=3)
  ap.add_argument("--hand-iters", type=int, default=15)
  ap.add_argument("--skip-hand", action="store_true")
  ap.add_argument("--skip-generated", action="store_true")
  ap.add_argument("--generated-env", choices=("current", "dbuf-safe"), default="dbuf-safe")
  ap.add_argument("--wmma-chain-trace", action="store_true", help="diagnostic-only: dump no-LDS generated WMMA chain spans/origins/stores")
  ap.add_argument("--trace-max-rows", type=int, default=0, help="limit WMMA/store rows in --wmma-chain-trace JSON; 0 means all")
  ap.add_argument("--json", action="store_true")
  add_clock_pin_arg(ap)
  args = ap.parse_args(list(argv) if argv is not None else None)

  set_clock_pin_env(os.environ, args.pin_clock)
  shapes = _parse_shapes(args.shapes)
  env = {"PYTHONPATH": os.getcwd()}
  if args.generated_env == "dbuf-safe": env.update(DEFAULT_DBUF_ENV)
  rows = []
  with _patched_env(env):
    for wm, wn in shapes:
      rows.append({
        "shape": f"{wm}x{wn}",
        "generated": {} if args.skip_generated else
                     _run_generated(args.m, args.n, args.k, wm, wn, args.loc, args.unr, args.pin_clock,
                                    args.wmma_chain_trace, args.trace_max_rows),
        "hand_lds2": {} if args.skip_hand else _run_hand(args.m, args.n, args.k, wm, wn, args.waves_m, args.waves_n,
                                                         args.bk, args.pad, args.dbuf, args.hand_reps, args.hand_iters,
                                                         plrab=args.plrab),
      })

  payload = {"m": args.m, "n": args.n, "k": args.k, "loc": args.loc, "unr": args.unr,
             "waves_m": args.waves_m, "waves_n": args.waves_n, "bk": args.bk, "pad": args.pad, "dbuf": args.dbuf,
             "plrab": args.plrab, "generated_env": args.generated_env, "rows": rows}
  if args.json:
    print(json.dumps(payload, indent=2))
  else:
    print(f"shape matrix M={args.m} N={args.n} K={args.k} generated_env={args.generated_env} loc={args.loc} unr={args.unr}")
    print("| shape | generated status | generated TFLOPS | hand LDS2 status | hand TFLOPS | hand resources |")
    print("|---|---:|---:|---:|---:|---|")
    for r in rows:
      g, h = r["generated"], r["hand_lds2"]
      hres = f"lds={h.get('lds_bytes')} scratch_end={h.get('scratch_end_vgpr')} inst={h.get('instruction_count')}"
      print(f"| {r['shape']} | {g.get('status')} | {g.get('tflops', 0.0)} | {h.get('status')} | {h.get('tflops', 0.0)} | {hres} |")
    print("\n| shape | path | inst/WMMA | wait/WMMA | memops/WMMA | gld.b128/WMMA | ds.store/WMMA | ds.load/WMMA | barrier/WMMA |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
      for label, row in (("generated", r["generated"]), ("hand", r["hand_lds2"])):
        print(f"| {r['shape']} | {label} | {row.get('inst_per_wmma', 0.0)} | {row.get('waitcnt_per_wmma', 0.0)} | "
              f"{row.get('memops_per_wmma', 0.0)} | {row.get('global_b128_per_wmma', 0.0)} | "
              f"{row.get('ds_store_b128_per_wmma', 0.0)} | {row.get('ds_load_b128_per_wmma', 0.0)} | "
              f"{row.get('barrier_per_wmma', 0.0)} |")
  return 0


if __name__ == "__main__":
  if os.environ.get("WMMA_CHAIN_TRACE_WORKER") == "1":
    _generated_wmma_chain_trace_worker()
    raise SystemExit(0)
  if os.environ.get("STRUCTURE_WORKER") == "1":
    _generated_structure_worker()
    raise SystemExit(0)
  raise SystemExit(main())
