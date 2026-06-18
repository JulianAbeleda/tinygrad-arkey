#!/usr/bin/env python3
"""MMVQ lowering scaffold proof harness (Q4_K ffn_gate/up).

Answers the scaffold's core question: can a MINIMAL representation give tinygrad's linearizer a *schedulable*
llama-class MMVQ inner loop (packed extract + dot4 + qsum + per-group scale), or is dot4 only reachable via
opaque CUSTOM bodies (the 52% custom_kernel ceiling)?

Decisive test: render the PURE-UOp int8xint8->int32 reduce (q4k_q8_1_intdot_partial_kernel -- the only
linearizer-visible representation, no CUSTOM/asm) and check whether tinygrad auto-lowers it to native dot4 or
scalarizes it. Then compare microkernel %HBM-peak across the representations.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_mmvq_lowering_scaffold.py
"""
from __future__ import annotations
import io, json, pathlib, re, contextlib
import numpy as np

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+.+?\s+arg\s+\d+\s+mem\s+[\d.]+\s+GB\s+tm\s+([\d.]+)us")
PEAK = 900.0

def main():
  from tinygrad import Tensor, Context, GlobalCounters, dtypes
  from extra.llm_generate import load_model_and_tokenizer
  from extra.q4_k_gemv_primitive import (q4k_gemv_partial_kernel, q4k_coop_partial_kernel,
    q4k_q8_1_intdot_partial_kernel, q4k_q8_1_vdot_builtin_partial_kernel, q8_1_bias_pack_u32_kernel)
  from extra.qk_layout import q8_1_quantize
  m, _ = load_model_and_tokenizer("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 2048, seed=1)
  lin = next(l for l in m._q4k_linears.linears if type(l).__name__ == "Q4KPrimitiveLinear"
             and l.out_features == 12288 and l.in_features == 4096 and l.parts == 1)
  OUT, IN = lin.out_features, lin.in_features
  words = lin.q4k_storage.words.realize(); MB = OUT * IN * 4.5 / 8 / 1e6
  x = Tensor(np.random.default_rng(5).standard_normal((IN,)).astype(np.float32)).realize()
  q, sc = q8_1_quantize(x); q = q.realize(); sc = sc.realize()
  qbias = Tensor.empty(IN // 4, dtype=dtypes.uint32).custom_kernel(q, fxn=q8_1_bias_pack_u32_kernel(IN))[0].realize()
  art = pathlib.Path("bench/qk-mmvq-lowering-scaffold"); art.mkdir(parents=True, exist_ok=True)

  # --- Phase 2: generated-source proof of the linearizer-visible representation (pure-UOp int reduce) ---
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=4):
    Tensor.empty(OUT, 1, dtype=dtypes.float32).custom_kernel(words, q, sc,
      fxn=q4k_q8_1_intdot_partial_kernel(OUT, IN, 1, "none", ()))[0].realize()
  src = buf.getvalue()
  kern = next((k for k in re.split(r'extern "C"', src) if "intdot" in k), src)
  (art / "generated_source.txt").write_text(kern)
  scheck = {"native_dot4_emitted": len(re.findall(r"v_dot4|sdot4|udot4|_dp4a|dot4", kern)),
            "scalar_nibble_mac": bool(re.search(r"\(int\)\(\(\(?val\d.*?>>.*?&\s*15u?\)\).*?\*\s*\(int\)", kern)),
            "source_lines": kern.count("\n")}
  scheck["verdict"] = ("SCALARIZED (no auto-dot4 lowering)" if scheck["native_dot4_emitted"] == 0
                       else "dot4 present")
  (art / "source_check.json").write_text(json.dumps(scheck, indent=2))

  # --- Phase 4: microkernel perf across representations ---
  def tm(f, warm=3, n=6):
    for _ in range(warm): f().realize()
    best = 1e9
    for _ in range(n):
      b = io.StringIO()
      with contextlib.redirect_stdout(b), Context(DEBUG=2):
        GlobalCounters.reset(); f().realize()
      best = min(best, sum(float(mm.group(1)) for l in b.getvalue().splitlines() if (mm := _LINE.search(_ANSI.sub("", l)))))
    return best
  def pk(us): return round(MB / (us / 1e6) / 1e3 / PEAK * 100, 1)
  xf = x.cast(dtypes.float16).realize()
  runs = {
    "base_fp": tm(lambda: Tensor.empty(OUT,1,dtype=dtypes.float32).custom_kernel(words,xf,fxn=q4k_gemv_partial_kernel(OUT,IN,1,"none",lin.opts))[0].sum(1)),
    "fp_coop": tm(lambda: Tensor.empty(OUT,8,dtype=dtypes.float32).custom_kernel(words,xf,fxn=q4k_coop_partial_kernel(OUT,IN,16))[0].sum(1)),
    "intdot_pure_uop_linearizer_visible": tm(lambda: Tensor.empty(OUT,1,dtype=dtypes.float32).custom_kernel(words,q,sc,fxn=q4k_q8_1_intdot_partial_kernel(OUT,IN,1,"none",()))[0].sum(1)),
    "dp4a_udot4_schedulable_in_CUSTOM": tm(lambda: Tensor.empty(OUT,1,dtype=dtypes.float32).custom_kernel(words,qbias,sc,fxn=q4k_q8_1_vdot_builtin_partial_kernel(OUT,IN,1,"none",()))[0].sum(1)),
  }
  perf = {k: {"us": round(v,1), "pct_peak": pk(v)} for k, v in runs.items()}
  perf["opaque_asm_signed_dot4_coalesced"] = {"us": None, "pct_peak": 52, "note": "prior measurement (custom_kernel, removed)"}
  perf["llama_READRAW_reference"] = {"pct_peak": 70}
  (art / "perf.json").write_text(json.dumps(perf, indent=2))
  print(json.dumps({"source_check": scheck, "perf_pct_peak": {k: (v.get("pct_peak")) for k, v in perf.items()}}, indent=2))

if __name__ == "__main__":
  main()
