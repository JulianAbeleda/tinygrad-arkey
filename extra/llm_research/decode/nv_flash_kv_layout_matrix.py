#!/usr/bin/env python3
"""Research-only K/V layout and late-predication matrix for wide Flash.

Builds exact tinygrad UOp variants, captures their native NV programs, and
measures each from a standardized target-hot state with and without the same
96-MiB read-only conditioner used by the production-conditioning probe.  No
model route or runtime policy is changed.
"""
from __future__ import annotations

import argparse, json, pathlib, statistics, subprocess, sys
from dataclasses import dataclass

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from nv_l2_eviction_decisive import _compile_stream_read
from nv_r_residual_cache_dispatch_probe import _alloc, _make_queue

Hq, Hkv, Hd, Tc, W = 32, 8, 128, 512, 130
STREAM_MIB, STREAM_BLOCK = 96, 256
SCHEMA = "tinygrad.nv_flash_kv_layout_matrix.v1"


@dataclass(frozen=True)
class Variant:
  name: str
  maxc: int
  splits: int
  guard: bool = False
  separate: bool = False
  token_bound: int|None = None


VARIANTS = (
  Variant("combined_m1024_s8", 1024, 8),
  Variant("combined_m1024_s8_guard", 1024, 8, guard=True),
  Variant("combined_m1024_s4_tb512", 1024, 4, token_bound=512),
  Variant("combined_m1024_s5_tb640", 1024, 5, token_bound=640),
  Variant("combined_m1024_s6_tb768", 1024, 6, token_bound=768),
  Variant("combined_m1024_s7_tb896", 1024, 7, token_bound=896),
  Variant("combined_m512_s4", 512, 4),
  Variant("combined_m640_s5", 640, 5),
  Variant("combined_m768_s6", 768, 6),
  Variant("combined_m896_s7", 896, 7),
  Variant("separate_m1024_s8", 1024, 8, separate=True),
  Variant("separate_m1024_s8_guard", 1024, 8, guard=True, separate=True),
)


def _data(maxc:int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  # Deterministic, bounded values keep every variant on the same logical K/V
  # contents while making accidental invalid-tail contributions observable.
  q = (((np.arange(Hq*Hd, dtype=np.int32) * 17 + 3) % 127) - 63).astype(np.float32).reshape(Hq, Hd) / 256.0
  ix = np.arange(Hkv*maxc*Hd, dtype=np.int64).reshape(Hkv, maxc, Hd)
  k = ((((ix * 13 + 5) % 251) - 125) / 512.0).astype(np.float16)
  v = ((((ix * 19 + 7) % 241) - 120) / 512.0).astype(np.float16)
  return q, k, v


def _compile_variant(v:Variant) -> dict:
  import tinygrad.runtime.ops_nv as ops_nv
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.llm.flash_decode_attention import flash_vec_llama_score_pv_kernel
  from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program
  from tinygrad.uop.ops import UOp

  captured = {}
  orig_call = ops_nv.NVProgram.__call__
  def patched_call(self, *bufs, global_size=(1, 1, 1), local_size=(1, 1, 1), vals=(), wait=False, timeout=None):
    if self.name.startswith("flash_vec_llama_score_pv") and not captured:
      captured.update(program=self, bufs=tuple(bufs), vals=tuple(vals), global_size=tuple(global_size),
                      local_size=tuple(local_size), buf_sizes=[int(x.size) for x in bufs])
    return orig_call(self, *bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait, timeout=timeout)

  q_np, k_np, v_np = _data(v.maxc)
  emitter = flash_vec_llama_score_pv_kernel(Hd, Hq, Hkv, v.maxc, v.splits, UOp.const(dtypes.int, Tc),
    wide_kv=True, wide_q=False, token_bound=v.token_bound, guard_kv_loads=v.guard, separate_kv=v.separate)
  program = KernelProgram("research.nv_flash_kv_layout_matrix", v.name, KernelProgramProvenance.RESEARCH_ONLY,
    emitter, output_spec=OutputSpec((Hq*v.splits*W,), dtypes.float32))
  q = Tensor(q_np.reshape(-1), device="NV").contiguous().realize()
  if v.separate:
    kt = Tensor(k_np, device="NV").contiguous().realize()
    vt = Tensor(v_np, device="NV").contiguous().realize()
    inputs = (q, Tensor(kt.uop.bitcast(dtypes.uint32)), Tensor(vt.uop.bitcast(dtypes.uint32)))
  else:
    cache = Tensor(np.stack((k_np, v_np)), device="NV").contiguous().realize()
    inputs = (q, Tensor(cache.uop.bitcast(dtypes.uint32)))
  out = Tensor.empty(Hq*v.splits*W, dtype=dtypes.float32, device="NV")
  ops_nv.NVProgram.__call__ = patched_call
  try:
    result = execute_research_program(out, *inputs, program=program).realize()
    Device["NV"].synchronize()
  finally:
    ops_nv.NVProgram.__call__ = orig_call
  if not captured: raise RuntimeError(f"failed to capture {v.name}")
  captured.update(name=v.name, variant=v, raw_output=np.asarray(result.numpy()).copy(), holders=(out, q, inputs, result),
                  cache_bytes=np.stack((k_np, v_np)).tobytes())
  return captured


def _combined_output(raw:np.ndarray, splits:int) -> np.ndarray:
  x = raw.reshape(Hq, splits, W).astype(np.float64)
  pv, den, mx = x[:, :, :Hd], x[:, :, Hd], x[:, :, Hd+1]
  gmax = np.max(mx, axis=1)
  weights = np.exp(mx - gmax[:, None])
  return np.sum(pv * weights[:, :, None], axis=1) / np.sum(den * weights, axis=1)[:, None]


def _exec(q, rec):
  q.exec(rec["program"], rec["program"].fill_kernargs(rec["bufs"], vals=rec["vals"]),
         rec["global_size"], rec["local_size"])


def _measure(dev, rec:dict, conditioner, n:int, warmup:int, timeout_s:float) -> dict:
  vals = []
  for _ in range(n):
    q = _make_queue(dev)
    _exec(q, rec)
    if conditioner is not None:
      prg, bufs, grid, block = conditioner
      q.exec(prg, prg.fill_kernargs(bufs), grid, block)
    st, en = dev.new_signal(), dev.new_signal()
    q.timestamp(st); _exec(q, rec); q.timestamp(en)
    q.signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
    dev.synchronize(timeout=int(timeout_s*1000))
    vals.append(float(en.timestamp-st.timestamp))
  vals = vals[warmup:]
  return {"median_us":round(statistics.median(vals), 3), "mean_us":round(statistics.mean(vals), 3),
          "min_us":round(min(vals), 3), "max_us":round(max(vals), 3),
          "samples_us":[round(x, 3) for x in vals]}


def _timed_row(dev, rec:dict, conditioner, n:int, warmup:int, timeout_s:float) -> dict:
  hot_a = _measure(dev, rec, None, n, warmup, timeout_s)
  cold = _measure(dev, rec, conditioner, n, warmup, timeout_s)
  hot_c = _measure(dev, rec, None, n, warmup, timeout_s)
  hot = statistics.median((hot_a["median_us"], hot_c["median_us"]))
  return {"hot_a":hot_a, "cold_96mib":cold, "hot_c":hot_c, "hot_midpoint_us":round(hot, 3),
          "conditioning_penalty_us":round(cold["median_us"]-hot, 3)}


def _color_records(dev, control:dict, offsets:tuple[int, ...]) -> tuple[list[dict], list[object]]:
  cache_slot, cache_size = 2, control["buf_sizes"][2]
  records, holders = [], []
  for off in offsets:
    base = _alloc(dev, cache_size + off + 4096)
    view = base.offset(offset=off, size=cache_size)
    dev.allocator._copyin(view, memoryview(control["cache_bytes"]))
    bufs = list(control["bufs"]); bufs[cache_slot] = view
    records.append({**control, "name":f"color_{off}", "bufs":tuple(bufs), "color_offset":off,
                    "cache_va":int(view.va_addr), "cache_va_mod_2m":int(view.va_addr % (2*1024*1024))})
    holders.append(base)
  dev.synchronize()
  return records, holders


def main() -> int:
  global Tc
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--n", type=int, default=32); ap.add_argument("--warmup", type=int, default=8)
  ap.add_argument("--tc", type=int, default=Tc); ap.add_argument("--skip-colors", action="store_true")
  ap.add_argument("--timeout-s", type=float, default=180.0); ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()
  if args.n <= args.warmup: raise SystemExit("--n must exceed --warmup")
  if not 1 <= args.tc <= 1024: raise SystemExit("--tc must be in [1, 1024]")
  Tc = args.tc

  from tinygrad import Device
  from tinygrad.runtime.ops_nv import NVProgram
  from extra.llm_research.decode.qk_norm_rope_wall_bracket import _gpu_state

  dev = Device["NV"]
  stream_buf, stream_out = _alloc(dev, STREAM_MIB*1024*1024), _alloc(dev, 4)
  dev.allocator._copyin(stream_buf, memoryview(bytearray(stream_buf.size)))
  dev.allocator._copyin(stream_out, memoryview(bytearray(stream_out.size)))
  words = STREAM_MIB*1024*1024//4
  stream = NVProgram(dev, "nv_l2_stream_flash_layout_96mib", _compile_stream_read(dev, words, "flash_layout_96mib"))
  conditioner = (stream, (stream_buf, stream_out), ((words+STREAM_BLOCK-1)//STREAM_BLOCK, 1, 1), (STREAM_BLOCK, 1, 1))

  compiled = {v.name:_compile_variant(v) for v in VARIANTS}
  control = compiled["combined_m1024_s8"]
  reference = _combined_output(control["raw_output"], control["variant"].splits)
  rows = {}
  for v in VARIANTS:
    rec = compiled[v.name]
    combined = _combined_output(rec["raw_output"], v.splits)
    semantic = {"combined_max_abs":float(np.max(np.abs(combined-reference))),
                "combined_allclose_2e_4":bool(np.allclose(combined, reference, rtol=2e-4, atol=2e-4)),
                "raw_bit_exact_to_control":bool(np.array_equal(rec["raw_output"], control["raw_output"]))
                  if v.splits == control["variant"].splits else None}
    rows[v.name] = {"maxc":v.maxc, "splits":v.splits, "guard":v.guard, "separate":v.separate,
                    "token_bound":v.token_bound,
                    "program_name":rec["program"].name, "buf_sizes":rec["buf_sizes"], "semantic":semantic,
                    **_timed_row(dev, rec, conditioner, args.n, args.warmup, args.timeout_s)}
    print(json.dumps({v.name:{"semantic":semantic, "hot":rows[v.name]["hot_midpoint_us"],
      "cold":rows[v.name]["cold_96mib"]["median_us"], "penalty":rows[v.name]["conditioning_penalty_us"]}}, sort_keys=True), flush=True)

  offsets = (0, 4096, 16384, 65536, 131072, 262144, 524288, 1048576, 1572864)
  color_recs, color_holders = _color_records(dev, control, offsets) if not args.skip_colors else ([], [])
  colors = {}
  for rec in color_recs:
    colors[str(rec["color_offset"])] = {"cache_va":rec["cache_va"], "cache_va_mod_2m":rec["cache_va_mod_2m"],
      **_timed_row(dev, rec, conditioner, args.n, args.warmup, args.timeout_s)}
    print(json.dumps({f"color_{rec['color_offset']}":colors[str(rec["color_offset"])]}, sort_keys=True), flush=True)

  payload = {"schema":SCHEMA, "commit":subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "gpu_state":_gpu_state(), "shape":{"Hq":Hq,"Hkv":Hkv,"Hd":Hd,"Tc":Tc},
    "method":"exact UOp programs on native NV HCQ; target reheat before every sample; only final target timestamped",
    "n_per_arm":args.n-args.warmup, "conditioner_mib":STREAM_MIB, "rows":rows, "colors":colors,
    "control":"combined_m1024_s8"}
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
  return 0


if __name__ == "__main__": raise SystemExit(main())
