#!/usr/bin/env python3
from __future__ import annotations

import argparse, contextlib, functools, hashlib, json, os, pathlib, statistics, subprocess, time
from typing import Any

import numpy as np

from tinygrad import GlobalCounters, Tensor
from tinygrad.device import Device
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.codegen import to_program
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from tinygrad.renderer.amd.dsl import NULL, s, v
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  ds_bpermute_b32, ds_load_b32, ds_store_b32, global_load_b32, global_store_b32, s_barrier, s_endpgm, s_load_b128,
  s_load_b64, s_waitcnt, v_add_f32_e32, v_add_nc_u32_e32, v_and_b32_e32, v_cmp_ne_u32_e32, v_cndmask_b32_e32,
  v_cvt_f32_u32_e32, v_dot4_i32_iu8, v_lshlrev_b32_e32, v_lshrrev_b32_e32, v_mov_b32_e32, v_xor_b32_e32,
)
from extra.q8_ffn_asm_fullrow_reduce import HIDDEN, Q4_WORDS, Q8_BYTES, build_fullrow_reduce
from extra.q8_ffn_asm_schedule_audit import main as schedule_audit_main
from extra.q8_ffn_codegen_transfer_audit import GROUPS
from extra.q8_ffn_fast_artifact_probe import HIP_MMVQ_GATEUP_SOURCE, compile_hipcc_linked
from extra.q8_ffn_comgr_fused_gateup_probe import COMGR_MMVQ_GATEUP_SOURCE

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/q8-ffn-dynamic-scheduler-observability"

def pctile(xs:list[float], p:float) -> float:
  ys = sorted(xs)
  return ys[min(len(ys)-1, max(0, round((len(ys)-1)*p)))]

def stats_ms(xs:list[float]) -> dict[str, Any]:
  return {
    "samples_ms": [round(x, 6) for x in xs],
    "min_ms": round(min(xs), 6),
    "median_ms": round(statistics.median(xs), 6),
    "mean_ms": round(statistics.fmean(xs), 6),
    "p10_ms": round(pctile(xs, 0.10), 6),
    "p90_ms": round(pctile(xs, 0.90), 6),
    "max_ms": round(max(xs), 6),
  }

def sh(cmd:list[str], **kwargs) -> subprocess.CompletedProcess:
  return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)

def git_commit() -> str:
  try:
    sha = sh(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], timeout=10).stdout.strip()
    dirty = sh(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--"], timeout=10).returncode != 0
    return sha + ("-dirty" if dirty else "")
  except Exception:
    return "unknown"

def hbytes(data:bytes|None) -> str|None:
  return hashlib.sha256(data).hexdigest()[:16] if data else None

def read_json(path:pathlib.Path) -> dict[str, Any] | None:
  try:
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else None
  except Exception:
    return None

class HCQCollector:
  def __init__(self): self.rows:list[dict[str, Any]] = []

  def record(self, prg:Any, bufs:tuple[Any, ...], kwargs:dict[str, Any], host_ms:float, ret:Any) -> None:
    self.rows.append({
      "program_name": getattr(prg, "name", type(prg).__name__),
      "runtime_class": type(prg).__name__,
      "code_hash": hbytes(getattr(prg, "lib", None)),
      "launch": {"global_size": tuple(kwargs.get("global_size", (1, 1, 1))),
                 "local_size": tuple(kwargs.get("local_size", (1, 1, 1)))},
      "metadata": {
        "kernargs_alloc_size": getattr(prg, "kernargs_alloc_size", None),
        "kernargs_segment_size": getattr(prg, "kernargs_segment_size", None),
        "group_segment_size": getattr(prg, "group_segment_size", None),
        "private_segment_size": getattr(prg, "private_segment_size", None),
        "wave32": getattr(prg, "wave32", None),
        "rsrc1": getattr(prg, "rsrc1", None),
        "rsrc2": getattr(prg, "rsrc2", None),
        "rsrc3": getattr(prg, "rsrc3", None),
        "buffer_count": len(bufs),
        "vals_count": len(tuple(kwargs.get("vals", ()))),
      },
      "timing": {"host_ms": round(host_ms, 6), "device_ms": round(float(ret)*1000.0, 6) if isinstance(ret, (float, int)) else None},
      "sync": {"wait": bool(kwargs.get("wait", False))},
    })

@contextlib.contextmanager
def collect_hcq(c:HCQCollector):
  import tinygrad.runtime.support.hcq as hcq
  orig = hcq.HCQProgram.__call__
  @functools.wraps(orig)
  def wrapped(self, *bufs, **kwargs):
    st = time.perf_counter()
    ret = orig(self, *bufs, **kwargs)
    c.record(self, bufs, kwargs, (time.perf_counter()-st)*1000.0, ret)
    return ret
  hcq.HCQProgram.__call__ = wrapped
  try: yield
  finally: hcq.HCQProgram.__call__ = orig

def build_variant(kind:str, dst:UOp, q4:UOp, q8:UOp) -> UOp:
  gidxs = [UOp.special(n, f"gidx{i}") for i, n in enumerate((HIDDEN, 2, 1))]
  lidxs = [UOp.special(n, f"lidx{i}") for i, n in enumerate((128, 1, 1))]
  lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=16, addrspace=AddrSpace.LOCAL), (), f"lds_{kind}")
  insts = [
    s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0, soffset=NULL),
    s_load_b64(sdata=s[18:19], sbase=s[0:1], offset=0x10, soffset=NULL),
    s_waitcnt(simm16=0),
  ]
  if kind == "reduction_only":
    insts += [v_cvt_f32_u32_e32(vdst=v[10], src0=v[0])]
  elif kind == "dot_synthetic":
    insts += [v_mov_b32_e32(vdst=v[4], src0=0), v_mov_b32_e32(vdst=v[5], src0=0)]
    for _ in range(8):
      insts += [
        v_dot4_i32_iu8(vdst=v[4], src0=0x01020304, src1=0x01010101, src2=v[4], neg=2),
        v_dot4_i32_iu8(vdst=v[5], src0=0x01010101, src1=0x01010101, src2=v[5], neg=2),
      ]
    insts += [v_cvt_f32_u32_e32(vdst=v[10], src0=v[4])]
  elif kind == "load_wait_only":
    insts += [
      v_lshrrev_b32_e32(vdst=v[20], src0=3, vsrc1=v[0]),
      v_and_b32_e32(vdst=v[21], src0=7, vsrc1=v[0]),
      v_lshlrev_b32_e32(vdst=v[23], src0=4, vsrc1=v[20]),
      v_lshlrev_b32_e32(vdst=v[24], src0=5, vsrc1=v[21]),
      v_add_nc_u32_e32(vdst=v[23], src0=v[24], vsrc1=v[23]),
      v_lshlrev_b32_e32(vdst=v[24], src0=3, vsrc1=v[20]),
      v_add_nc_u32_e32(vdst=v[24], src0=v[21], vsrc1=v[24]),
      v_lshlrev_b32_e32(vdst=v[24], src0=2, vsrc1=v[24]),
      v_mov_b32_e32(vdst=v[10], src0=0),
    ]
    for _ in range(8):
      insts += [
        global_load_b32(vdst=v[8], addr=v[23], saddr=s[4:5]),
        global_load_b32(vdst=v[9], addr=v[24], saddr=s[18:19]),
        s_waitcnt(simm16=0),
        v_xor_b32_e32(vdst=v[10], src0=v[8], vsrc1=v[10]),
        v_xor_b32_e32(vdst=v[10], src0=v[9], vsrc1=v[10]),
        v_add_nc_u32_e32(vdst=v[23], src0=4, vsrc1=v[23]),
        v_add_nc_u32_e32(vdst=v[24], src0=4, vsrc1=v[24]),
      ]
    insts += [v_cvt_f32_u32_e32(vdst=v[10], src0=v[10])]
  elif kind == "wait_grouped_load_only":
    insts += [
      v_lshrrev_b32_e32(vdst=v[20], src0=3, vsrc1=v[0]),
      v_lshlrev_b32_e32(vdst=v[23], src0=4, vsrc1=v[20]),
      v_lshlrev_b32_e32(vdst=v[24], src0=5, vsrc1=v[20]),
      v_mov_b32_e32(vdst=v[10], src0=0),
    ]
    for i in range(8):
      insts += [
        global_load_b32(vdst=v[8+i%4], addr=v[23], saddr=s[4:5]),
        global_load_b32(vdst=v[12+i%4], addr=v[24], saddr=s[18:19]),
        v_add_nc_u32_e32(vdst=v[23], src0=4, vsrc1=v[23]),
        v_add_nc_u32_e32(vdst=v[24], src0=4, vsrc1=v[24]),
      ]
    insts += [s_waitcnt(simm16=0)]
    for i in range(4): insts += [v_xor_b32_e32(vdst=v[10], src0=v[8+i], vsrc1=v[10]), v_xor_b32_e32(vdst=v[10], src0=v[12+i], vsrc1=v[10])]
    insts += [v_cvt_f32_u32_e32(vdst=v[10], src0=v[10])]
  else:
    raise ValueError(kind)

  insts += [v_and_b32_e32(vdst=v[50], src0=31, vsrc1=v[0])]
  for off in [16, 8, 4, 2, 1]:
    insts += [
      v_xor_b32_e32(vdst=v[51], src0=off, vsrc1=v[50]),
      v_lshlrev_b32_e32(vdst=v[51], src0=2, vsrc1=v[51]),
      ds_bpermute_b32(vdst=v[52], addr=v[51], data0=v[10]),
      s_waitcnt(simm16=0),
      v_add_f32_e32(vdst=v[10], src0=v[52], vsrc1=v[10]),
    ]
  insts += [
    v_lshrrev_b32_e32(vdst=v[53], src0=5, vsrc1=v[0]),
    v_lshlrev_b32_e32(vdst=v[53], src0=2, vsrc1=v[53]),
    ds_store_b32(addr=v[53], data0=v[10]),
    s_waitcnt(simm16=0),
    s_barrier(),
    v_mov_b32_e32(vdst=v[54], src0=0), ds_load_b32(vdst=v[10], addr=v[54]),
    v_mov_b32_e32(vdst=v[54], src0=4), ds_load_b32(vdst=v[11], addr=v[54]),
    v_mov_b32_e32(vdst=v[54], src0=8), ds_load_b32(vdst=v[12], addr=v[54]),
    v_mov_b32_e32(vdst=v[54], src0=12), ds_load_b32(vdst=v[13], addr=v[54]),
    s_waitcnt(simm16=0),
    v_add_f32_e32(vdst=v[10], src0=v[11], vsrc1=v[10]),
    v_add_f32_e32(vdst=v[12], src0=v[13], vsrc1=v[12]),
    v_add_f32_e32(vdst=v[10], src0=v[12], vsrc1=v[10]),
    v_lshlrev_b32_e32(vdst=v[2], src0=2, vsrc1=v[0]),
    v_mov_b32_e32(vdst=v[3], src0=0),
    global_store_b32(addr=v[2], data=v[10], saddr=s[4:5]),
    s_endpgm(),
  ]
  sink = UOp.sink(dst.base, q4.base, q8.base, lds, *gidxs, *lidxs, arg=KernelInfo(name=f"q8_dso_{kind}"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))

def time_linear(linear, warmups:int, iters:int) -> dict[str, Any]:
  samples = []
  for i in range(warmups + iters):
    GlobalCounters.reset()
    st = time.perf_counter()
    run_linear(linear)
    wall_ms = (time.perf_counter() - st) * 1000.0
    dev_ms = GlobalCounters.time_sum_s * 1000.0
    if i >= warmups: samples.append(dev_ms if dev_ms > 0 else wall_ms)
  return stats_ms(samples)

def run_variant(kind:str, warmups:int, iters:int) -> dict[str, Any]:
  dst = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  q4 = Tensor(np.arange(Q4_WORDS, dtype=np.uint32), dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8 = Tensor((np.arange(Q8_BYTES, dtype=np.uint8) * 13).astype(np.uint8), dtype=dtypes.uint8, device="AMD").contiguous().realize()
  out, *_ = Tensor.custom_kernel(dst, q4, q8, fxn=functools.partial(build_variant, kind))[:1]
  prg_uop = build_variant(kind, dst.uop, q4.uop, q8.uop)
  insts = [u.arg for u in prg_uop.src[2].src]
  mnems = [str(i).split("(", 1)[0] for i in insts]
  grouped = {k: sum(1 for m in mnems if any(m.startswith(p) for p in prefs)) for k, prefs in GROUPS.items()}
  timing = time_linear(out.schedule_linear(), warmups, iters)
  return {"variant": kind, "timing": timing, "instruction_count": len(insts), "grouped_counts": grouped}

def run_dso(args:argparse.Namespace) -> dict[str, Any]:
  OUT.mkdir(parents=True, exist_ok=True)
  os.environ.setdefault("DEV", "AMD")

  # DSO-0: refresh S0 in-place through the existing authority script.
  old_argv = os.sys.argv
  try:
    os.sys.argv = ["q8_ffn_asm_schedule_audit.py", "--out", str(ROOT/"bench/q8-ffn-codegen-transfer/asm_schedule_audit.json")]
    schedule_audit_main()
  finally:
    os.sys.argv = old_argv
  s0 = read_json(ROOT/"bench/q8-ffn-codegen-transfer/asm_schedule_audit.json") or {}

  preflight = {
    "commit": git_commit(),
    "device": os.environ.get("DEV", "AMD"),
    "authority_us": {"asm": 166.649, "comgr": 146.88, "target": 60.0, "hipcc_lld_lifecycle": 114.12},
    "s0_path": "bench/q8-ffn-codegen-transfer/asm_schedule_audit.json",
    "s0_verdict": s0.get("verdict"),
    "profile_env": {k: os.environ.get(k) for k in ("PROFILE", "PMC", "SQTT", "PMC_COUNTERS", "SQTT_BUFFER_SIZE")},
  }
  (OUT/"preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")

  # DSO-1/2: direct HCQ rows for runtime objects, plus metadata.
  dev = Device["AMD"]
  c = HCQCollector()
  with collect_hcq(c):
    hip = dev.runtime("q8_dso_hipcc_lld_gateup", compile_hipcc_linked(HIP_MMVQ_GATEUP_SOURCE, args.arch))
    comgr = dev.runtime("q8_dso_comgr_gateup", dev.compiler.compile(COMGR_MMVQ_GATEUP_SOURCE))
    # Empty buffers are enough for launch metadata and timing is not used for correctness here. Use tiny rows count for safety.
    bufs = [dev.allocator.alloc(sz, options=None) for sz in (HIDDEN*4, HIDDEN*4, Q4_WORDS*4, Q4_WORDS*4, Q8_BYTES)]
    for prg in (hip, comgr):
      for _ in range(args.metadata_launches):
        prg(*bufs, global_size=(HIDDEN, 2, 1), local_size=(32, 4, 1), wait=True)
  hcq_rows = {"rows": c.rows, "summary": {"row_count": len(c.rows)}}
  (OUT/"hcq_rows.json").write_text(json.dumps(hcq_rows, indent=2) + "\n")

  resources = {
    "hipcc_lld": c.rows[0]["metadata"] if c.rows else {},
    "comgr": next((r["metadata"] for r in c.rows if "comgr" in r["program_name"]), {}),
    "tinygrad_asm": {
      "minimal_elf_note": "tinygrad assemble_linear ELF is accepted by AMDProgram but LLVM tools reject it",
      "static": (s0.get("objects") or {}).get("tinygrad_asm_gateup_full", {}).get("disasm", {}),
    },
    "s0_summary": s0.get("summary"),
    "s0_deltas": s0.get("deltas", {}),
  }
  (OUT/"resource_audit.json").write_text(json.dumps(resources, indent=2) + "\n")

  variants = [run_variant(k, args.warmups, args.iters) for k in ("reduction_only", "dot_synthetic", "load_wait_only", "wait_grouped_load_only")]
  asm_ms = 0.166649
  for vrow in variants:
    vrow["relative_to_full_asm"] = round(vrow["timing"]["median_ms"] / asm_ms, 4)
  variant_ladder = {"variants": variants, "authority_full_asm_ms": asm_ms}
  (OUT/"variant_ladder.json").write_text(json.dumps(variant_ladder, indent=2) + "\n")

  pmc_attempt = {"attempted": args.try_pmc, "verdict": "SKIPPED", "reason": "not requested"}
  if args.try_pmc:
    env = {**os.environ, "DEV": "AMD", "PROFILE": "1", "PMC": "1", "SQTT": "0"}
    proc = sh(["python3", "extra/q8_ffn_asm_gateup_full.py", "--warmups", "1", "--iters", "2",
               "--out", str(OUT/"pmc_q8_gateup_full.json")], cwd=ROOT, env=env, timeout=120)
    pmc_attempt = {"attempted": True, "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:],
                   "target": "q8_ffn_asm_gateup_full", "verdict": "PASS_Q8_PROFILE_RUN" if proc.returncode == 0 else "BLOCKED_OR_FAILED"}
  (OUT/"pmc_attempt.json").write_text(json.dumps(pmc_attempt, indent=2) + "\n")

  load_variant = next(v for v in variants if v["variant"] == "load_wait_only")
  grouped_variant = next(v for v in variants if v["variant"] == "wait_grouped_load_only")
  reduction_variant = next(v for v in variants if v["variant"] == "reduction_only")
  body_insensitive = reduction_variant["timing"]["median_ms"] >= asm_ms * 0.75 and load_variant["timing"]["median_ms"] >= asm_ms * 0.75
  if body_insensitive:
    label = "wait_scheduler_bound"
  elif grouped_variant["timing"]["median_ms"] <= load_variant["timing"]["median_ms"] - 0.015:
    label = "wait_scheduler_bound"
  elif load_variant["timing"]["median_ms"] >= asm_ms * 0.55 and reduction_variant["timing"]["median_ms"] < asm_ms * 0.50:
    label = "load_shape_bound"
  elif reduction_variant["timing"]["median_ms"] >= asm_ms * 0.50:
    label = "reduction_bound"
  else:
    label = "closed_project_level"
  if pmc_attempt["verdict"] == "BLOCKED_OR_FAILED" and label == "closed_project_level": label = "unobservable_l4_required"

  result = {
    "date": "2026-06-19",
    "phase": "DSO-0_to_DSO-5",
    "preflight": preflight,
    "hcq_rows_path": "bench/q8-ffn-dynamic-scheduler-observability/hcq_rows.json",
    "resource_audit_path": "bench/q8-ffn-dynamic-scheduler-observability/resource_audit.json",
    "variant_ladder_path": "bench/q8-ffn-dynamic-scheduler-observability/variant_ladder.json",
    "pmc_attempt_path": "bench/q8-ffn-dynamic-scheduler-observability/pmc_attempt.json",
    "classifier": label,
    "decision": "do_not_reopen_q8_native_ownership" if label in {"closed_project_level", "unobservable_l4_required", "wait_scheduler_bound"} else "scope_bounded_followup_before_reopen",
    "summary": {
      "asm_full_ms": asm_ms,
      "variant_medians_ms": {v["variant"]: v["timing"]["median_ms"] for v in variants},
      "s0_global_load_delta": ((s0.get("deltas") or {}).get("tinygrad_asm_minus_hipcc_lld_grouped") or {}).get("global_load"),
      "s0_valu_delta": ((s0.get("deltas") or {}).get("tinygrad_asm_minus_hipcc_lld_grouped") or {}).get("valu"),
      "hcq_rows": len(c.rows),
      "pmc_verdict": pmc_attempt.get("verdict"),
      "body_insensitive_variant_ladder": body_insensitive,
    },
  }
  (OUT/"result.json").write_text(json.dumps(result, indent=2) + "\n")
  return result

def main() -> int:
  ap = argparse.ArgumentParser(description="DSO q8 dynamic scheduler observability")
  ap.add_argument("--arch", default="gfx1100")
  ap.add_argument("--warmups", type=int, default=5)
  ap.add_argument("--iters", type=int, default=12)
  ap.add_argument("--metadata-launches", type=int, default=3)
  ap.add_argument("--try-pmc", action="store_true")
  args = ap.parse_args()
  result = run_dso(args)
  print(json.dumps({"out": str(OUT/"result.json"), "classifier": result["classifier"], "summary": result["summary"]}, indent=2))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
