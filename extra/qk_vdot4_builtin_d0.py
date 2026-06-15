#!/usr/bin/env python3
"""Phase D0 (reframed): does the SCHEDULABLE udot4 BUILTIN beat the asm-volatile v_dot4 BARRIER?

Phase D concluded "DP4A is the wrong lever" -- but its v_dot4 was emitted via `asm volatile`, a hard
scheduling barrier, and was the SLOWEST variant (35 Q4-GB/s). The renderer maps to HIP C++, where the
clean path is `__builtin_amdgcn_udot4` (a compiler builtin the scheduler can move). gfx1100 gates the
SIGNED sdot4 (needs dot1-insts) but the UNSIGNED udot4 compiles with `__attribute__((target("dot-insts")))`
on the kernel and emits v_dot4_u32_u8 -- and Q4_K already uses the unsigned dot + bias correction.

This builds the SAME Q4_K x q8_1 GEMV two ways (asm-volatile vs builtin), runs both on identical random
data (cross-validating the builtin's correctness against asm), times both on device, and disassembles
both. If the builtin is faster, the instruction-count headroom (consolidated doc: fp 4.06 -> DP4A ~1.35
VALU/weight) is realizable and Phase D's negative was an asm-volatile artifact. If equal, the barrier
wasn't the issue and the negative stands.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_vdot4_builtin_d0.py
"""
from __future__ import annotations
import json, pathlib, statistics, sys
import numpy as np
from tinygrad.device import Device, Buffer
from tinygrad.dtype import dtypes, _to_np_dtype
from extra.q4_k_gemv_primitive import _q4k_scale_min_expr, _u8_sum_expr, Q4K_WORDS_PER_BLOCK

ART = pathlib.Path("bench/amd-decode-flywheel-proof-20260614/dp4a-d0")
ROWS, K = 4096, 4096
KB = K // 256  # Q4_K blocks per row (256 elems/block)
RW = KB * Q4K_WORDS_PER_BLOCK  # uint32 words per row
LOCAL = 64  # threads/workgroup (full-occupancy launch: each thread does one row)


def gen_source(name: str, builtin: bool) -> str:
  # the dot: builtin (schedulable) vs asm volatile (barrier). udot4 = unsigned, matches the q8+128 bias.
  if builtin:
    dot_line = "      dot = __builtin_amdgcn_udot4(q8w, q4w, dot, false);"
    attr = '__attribute__((target("dot-insts"))) '
  else:
    dot_line = '      asm volatile("v_dot4_u32_u8 %0, %1, %2, %0" : "+v"(dot) : "v"(q8w), "v"(q4w));'
    attr = ""
  L = []
  L.append('extern "C" __attribute__((device, const)) unsigned long __ockl_get_group_id(unsigned int);')
  L.append('extern "C" __attribute__((device, const)) unsigned int __ockl_get_local_id(unsigned int);')
  L.append(f'extern "C" __attribute__((global)) {attr}void __attribute__((amdgpu_flat_work_group_size(1,{LOCAL}))) {name}(')
  L.append("    float* out, unsigned int* words, unsigned int* q8, float* xscales) {")
  L.append(f"  unsigned int row = __ockl_get_group_id(0) * {LOCAL} + __ockl_get_local_id(0);")
  L.append(f"  unsigned int* W = words + row * {RW};")
  L.append("  float total = 0.0f;")
  L.append(f"  for (int blk = 0; blk < {KB}; blk++) {{")
  L.append(f"    int base = blk * {Q4K_WORDS_PER_BLOCK};")
  L.append("    unsigned int fp = W[base];")
  L.append("    float d = (float)(__builtin_bit_cast(_Float16, (unsigned short)(fp & 65535u)));")
  L.append("    float dmin = (float)(__builtin_bit_cast(_Float16, (unsigned short)((fp >> 16u) & 65535u)));")
  for grp in range(8):
    scale, mn = _q4k_scale_min_expr("W", grp)
    shift = 4 if grp % 2 else 0
    qbase = 4 + (grp // 2) * 8
    L.append(f"    {{ unsigned int sc = {scale}; unsigned int mn = {mn};")
    L.append("      unsigned int dot = 0u; int q4sum = 0; int q8sum = 0;")
    for lane4 in range(8):
      L.append(f"      {{ unsigned int q4w = ((W[base+{qbase+lane4}] >> {shift}u) & 0x0f0f0f0fu);")
      L.append(f"      unsigned int q8w = q8[blk*64+{grp*8+lane4}];")
      L.append(dot_line)
      L.append(f"      q4sum += {_u8_sum_expr('q4w')}; q8sum += {_u8_sum_expr('q8w')}; }}")
    L.append("      int dot_signed = ((int)dot) - q4sum * 128; int q8_signed_sum = q8sum - 4096;")
    L.append(f"      float xscale = (float)xscales[blk*8+{grp}];")
    L.append("      total += xscale * (d * (float)sc * (float)dot_signed - dmin * (float)mn * (float)q8_signed_sum); }")
  L.append("  }")
  L.append("  out[row] = total;")
  L.append("}")
  return "\n".join(L)


def _np(buf):
  return np.frombuffer(buf.as_memoryview().cast(buf.dtype.base.fmt), dtype=_to_np_dtype(buf.dtype.base)).copy()


def main():
  dev = Device["AMD"]
  rng = np.random.default_rng(20260615)
  words = rng.integers(0, 2**32, size=ROWS*RW, dtype=np.uint32)
  # each block's base word holds (d, dmin) as two fp16; random bits -> nan/inf. Set valid: d=1.0, dmin=0.5.
  words = words.reshape(ROWS, KB, Q4K_WORDS_PER_BLOCK)
  words[:, :, 0] = np.uint32(0x38003C00)  # low half=0x3C00 (1.0), high half=0x3800 (0.5)
  words = words.reshape(-1)
  q8 = rng.integers(0, 2**32, size=KB*64, dtype=np.uint32)
  xscales = rng.standard_normal(KB*8).astype(np.float32)

  def mkbuf(arr):
    b = Buffer("AMD", arr.size, dtypes.uint32 if arr.dtype == np.uint32 else dtypes.float32).ensure_allocated()
    b.copyin(memoryview(arr)); return b
  wb, qb, xb = mkbuf(words), mkbuf(q8), mkbuf(xscales)
  q4_bytes = ROWS * RW * 4

  results = {}
  outs = {}
  for builtin in (False, True):
    tag = "builtin_udot4" if builtin else "asm_volatile"
    src = gen_source(tag, builtin)
    lib = dev.compiler.compile(src)
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): dev.compiler.disassemble(lib)
    asm = buf.getvalue()
    n_vdot = sum(1 for l in asm.splitlines() if "v_dot4" in l)
    n_valu = sum(1 for l in asm.splitlines() if "\tv_" in l or l.strip().startswith("v_"))
    ob = Buffer("AMD", ROWS, dtypes.float32).ensure_allocated()
    prg = dev.runtime(tag, lib)
    prg(ob._buf, wb._buf, qb._buf, xb._buf, global_size=(ROWS//LOCAL,1,1), local_size=(LOCAL,1,1), wait=True)  # warmup
    tms = [prg(ob._buf, wb._buf, qb._buf, xb._buf, global_size=(ROWS//LOCAL,1,1), local_size=(LOCAL,1,1), wait=True) for _ in range(20)]
    t = statistics.median(tms)
    outs[tag] = _np(ob)
    results[tag] = {"q4_gbs": round(q4_bytes/t/1e9, 1), "device_us": round(t*1e6, 1),
                    "n_vdot4": n_vdot, "n_valu": n_valu}
    print(f"{tag}: {results[tag]['q4_gbs']} Q4-GB/s ({results[tag]['device_us']} us)  "
          f"v_dot4={n_vdot} valu={n_valu}", file=sys.__stdout__)

  # cross-correctness: builtin must equal asm (same computation)
  a, b = outs["asm_volatile"], outs["builtin_udot4"]
  max_rel = float(np.max(np.abs(a-b) / (np.abs(a)+1e-6)))
  match = max_rel < 1e-4
  speedup = round(results["builtin_udot4"]["q4_gbs"] / results["asm_volatile"]["q4_gbs"], 2)
  print(f"cross_correctness: max_rel={max_rel:.2e} match={match}  builtin/asm speedup={speedup}x", file=sys.__stdout__)
  out = {"kind": "qk_vdot4_builtin_d0", "rows": ROWS, "k": K, "variants": results,
         "builtin_equals_asm": match, "max_rel_err": max_rel, "builtin_vs_asm_speedup": speedup,
         "fp_baseline_gbs": 173, "intdot_gbs": 242, "asm_vdot_phaseD_gbs": 35}
  ART.mkdir(parents=True, exist_ok=True)
  (ART / "builtin_vs_asm_result.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
