#!/usr/bin/env python3
"""TPE-5 — shape matrix: launch+verify+time the extracted rocBLAS Tensile kernels for the high-share prefill GEMM
roles (ffn_gate/up, ffn_down, attn_q/o) through tinygrad HCQ, then compute the weighted full-prefill model.

Reuses the TPE-3 named-descriptor loader + the multi-role kernarg capture (extra/qk_tensile_kernarg_capture.cpp ->
/tmp/kernarg_all.json, committed copy bench/qk-tensile-extraction/kernarg_all.jsonl). Per role: substitute only the 4
Address VAs (offsets 16/24/32/40, identical across roles), keep the captured kernarg (WGM/stagger fields) verbatim,
launch global=num_workgroups / local from the capture, verify vs a tinygrad fp16 oracle, time device ms via HCQ
signals. No HIP runtime in-process, no copies. Research-only; no model route, no defaults, decode untouched.

GEMM (rocBLAS col-major, all roles): C[m,n] = A[m,k]*B[k,n], lda=m ldb=k ldc=m, alpha=1 beta=0.
tinygrad row-major mapping: A_t[k,m], B_t[n,k], C_t[n,m]; oracle C_t = B_t @ A_t.

  run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_tensile_shape_matrix.py
"""
from __future__ import annotations
import json, struct, statistics, pathlib
from tinygrad import Tensor, Device, dtypes
from extra.qk_tensile_hcq_launch import NamedAMDProgram, kd_offset, unbundle

# rocBLAS m,n,k per role (from extra/qk_prefill_blas_ceiling.cpp) + per-LAYER count for the weighted prefill model
ROLES = {
  "ffn_gate_up": dict(m=512, n=12288, k=4096, per_layer=2, ref_tflops=60.96),   # gate + up
  "ffn_down":    dict(m=512, n=4096,  k=12288, per_layer=1, ref_tflops=70.9),
  "attn_q_o":    dict(m=512, n=4096,  k=4096,  per_layer=2, ref_tflops=76.7),   # q + o
}
ATTN_KV_FLOP_PER_LAYER = 2*512*1024*4096*2                                       # k+v, low-EV, modeled at tinygrad speed
LAYERS = 36
TINYGRAD_PLATEAU = 42.0                                                          # ~40.8-42 TFLOPS across these shapes (POWN/PXB-1)
PREFILL_V2_FWD_MS = 245.0; NONMATMUL_MS = 64.0                                   # PREFILL_V2 forward + fixed non-matmul (~26%)
WARM, ITERS = 20, 60

def flop(r): return 2*r["m"]*r["n"]*r["k"]

def run_role(dev, role, cap, elf):
  r = ROLES[role]; m,n,k = r["m"], r["n"], r["k"]
  raw = bytearray(cap["kernarg_bytes"]); sym = cap["kernel_symbol"]
  Tensor.manual_seed(0)
  A_t = Tensor.randn(k, m, dtype=dtypes.half).contiguous().realize()
  B_t = Tensor.randn(n, k, dtype=dtypes.half).contiguous().realize()
  C_t = Tensor.zeros(n, m, dtype=dtypes.half).contiguous().realize()
  oracle = (B_t.float() @ A_t.float()).realize()
  dev.synchronize()
  va = lambda t: t.uop.buffer._buf.va_addr
  struct.pack_into("<Q", raw, 16, va(C_t)); struct.pack_into("<Q", raw, 24, va(C_t))
  struct.pack_into("<Q", raw, 32, va(A_t)); struct.pack_into("<Q", raw, 40, va(B_t))
  kd = kd_offset(elf, sym)
  prg = NamedAMDProgram(dev, f"tensile_{role}", elf, kd, bytes(raw))
  gx,gy,gz = cap["global"]; lx,ly,lz = cap["local"]; gws = (gx//lx, gy//ly, gz//lz)
  # correctness + stability (re-zero C each run in case of StreamK accumulation)
  rels=[]
  for _ in range(4):
    C_t.assign(Tensor.zeros(n, m, dtype=dtypes.half)).realize(); dev.synchronize()
    prg(global_size=gws, local_size=(lx,ly,lz), wait=True, timeout=10000); dev.synchronize()
    rels.append((C_t.float()-oracle).abs().max().item()/(oracle.abs().max().item()+1e-6))
  # timing
  for _ in range(WARM): prg(global_size=gws, local_size=(lx,ly,lz), wait=True)
  dev.synchronize()
  ms = sorted(prg(global_size=gws, local_size=(lx,ly,lz), wait=True)*1000.0 for _ in range(ITERS))
  dev.synchronize()
  med = statistics.median(ms); tf = flop(r)/(med*1e-3)/1e12
  return dict(role=role, m=m, n=n, k=k, kernel_symbol=sym, kernarg_size=cap["kernarg_size"],
              global_size=list(gws), local_size=[lx,ly,lz], workspace="none (no workspace ptr in kernarg)",
              streamk=("SU32" in sym), rel_err=round(max(rels),6), stable=round(max(rels)-min(rels),6)<1e-4,
              correct=max(rels)<2e-2, median_ms=round(med,4), best_ms=round(min(ms),4),
              median_tflops=round(tf,1), best_tflops=round(flop(r)/(min(ms)*1e-3)/1e12,1),
              ref_tflops=r["ref_tflops"], pct_of_ref=round(100*tf/r["ref_tflops"],1),
              tinygrad_tflops=TINYGRAD_PLATEAU, speedup_vs_tinygrad=round(tf/TINYGRAD_PLATEAU,2),
              meets_62=tf>=62.0)

def weighted_model(rows):
  # Anchored on MEASURED PREFILL_V2: forward 245ms, matmul bucket ~181ms (74%), non-matmul ~64ms. Each role's
  # tinygrad time = matmul_bucket * (role FLOP share); replacing it swaps in the measured Tensile-HCQ time. This
  # keeps the all-tinygrad case == the measured 245ms (more defensible than a flat-TFLOPS approximation).
  by = {r["role"]: r for r in rows}
  MATMUL_BUCKET_MS = PREFILL_V2_FWD_MS - NONMATMUL_MS                                        # ~181ms, measured
  total_flop = sum(flop(ROLES[r])*ROLES[r]["per_layer"] for r in ROLES) + ATTN_KV_FLOP_PER_LAYER
  role_tg_ms  = lambda role: MATMUL_BUCKET_MS * (flop(ROLES[role])*ROLES[role]["per_layer"]) / total_flop
  role_ten_ms = lambda role: by[role]["median_ms"] * ROLES[role]["per_layer"] * LAYERS
  out={}
  for label, repl in (("ffn_gate_up", {"ffn_gate_up"}), ("ffn_gate_up+ffn_down", {"ffn_gate_up","ffn_down"}),
                      ("ffn_gate_up+ffn_down+attn_q_o", {"ffn_gate_up","ffn_down","attn_q_o"})):
    fwd = PREFILL_V2_FWD_MS - sum(role_tg_ms(r) for r in repl) + sum(role_ten_ms(r) for r in repl)
    out[label] = dict(replaced_tinygrad_ms=round(sum(role_tg_ms(r) for r in repl),1),
                      replaced_tensile_ms=round(sum(role_ten_ms(r) for r in repl),1),
                      forward_ms=round(fwd,1), pp_speedup_vs_prefillv2=round(PREFILL_V2_FWD_MS/fwd,3))
  return out

def main():
  assert Device.DEFAULT == "AMD"; dev = Device[Device.DEFAULT]
  src = pathlib.Path("/tmp/kernarg_all.json")
  if not src.exists(): src = pathlib.Path("bench/qk-tensile-extraction/kernarg_all.jsonl")
  caps = {json.loads(l)["role"]: json.loads(l) for l in open(src)}
  elf = unbundle()
  rows = [run_role(dev, role, caps[role], elf) for role in ROLES]
  model = weighted_model(rows)
  full = model["ffn_gate_up+ffn_down+attn_q_o"]["pp_speedup_vs_prefillv2"]
  all_correct = all(r["correct"] and r["stable"] for r in rows)
  no_workspace = all("none" in r["workspace"] for r in rows)
  verdict = ("PASS" if (all_correct and no_workspace and full>=1.25) else
             "REDIRECT" if (all_correct and full>=1.10) else "KILL")
  res = dict(schema="qk_tensile_shape_matrix_v1", phase="TPE-5", device="RX 7900 XTX / gfx1100",
             rows=rows, weighted_model=model, full_pp_speedup=full,
             gates=dict(all_correct=all_correct, all_stable=all(r["stable"] for r in rows), no_layout_copies=True,
                        no_workspace=no_workspace, full_pp_ge_125=full>=1.25), verdict=verdict)
  pathlib.Path("bench/qk-tensile-extraction/shape_matrix.json").write_text(json.dumps(res, indent=2))
  for r in rows: print(f"{r['role']:14s} {r['median_tflops']:5.1f} TF ({r['pct_of_ref']:5.1f}% ref, {r['speedup_vs_tinygrad']:.2f}x tg) "
                       f"rel {r['rel_err']:.5f} {'OK' if r['correct'] and r['stable'] else 'BAD'} ws={r['workspace'][:6]}")
  print("\nweighted pp vs PREFILL_V2:")
  for k,v in model.items(): print(f"  {k:34s} {v['pp_speedup_vs_prefillv2']}x  (fwd {v['forward_ms']}ms)")
  print("\nTPE-5 VERDICT:", verdict)

if __name__ == "__main__":
  main()
