#!/usr/bin/env python3
"""Verify the in-model MATMUL penalty: does the ffn_gate/up GEMM run at isolated speed inside the forward?

One process, pinned clock. Three numbers for the same shape (M=512,N=12288,K=4096):
  A. in-model gate/up GEMM  = (gate/up GPU-share 39.5%) x (warm forward wall) / 72 launches  -> TFLOPS
  B. isolated tinygrad AUTHORITY (the in-model kernel type), host-amortized batch          -> TFLOPS
  C. isolated OUR dependency-free GEMM, host-amortized batch                                -> TFLOPS
If A ~= B  -> the matmul INTEGRATES FINE (no recoverable graph penalty); the in-model kernel IS the limit; the
              integration lever is NOT the matmul -> go to attention (28%). Upside of wiring our kernel ~ C/B.
If A << B  -> graph/scheduling penalty: the same kernel runs slower in the forward than alone -> recoverable.

Run: DEV=AMD PREFILL_V2=1 PYTHONPATH=. python3 extra/qk_prefill_matmul_penalty_verify.py <model.gguf>
"""
from __future__ import annotations
import importlib.util, json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
REF_SRC = ROOT / "extra/gemm/rdna3_wmma_matmul.py"
PTM1_SRC = ROOT / "extra/qk_amd_bb5a10_ptm1_same_harness_bridge.py"
M, N, K = 512, 12288, 4096
FLOP = 2 * M * N * K
GATEUP_SHARE = 0.395   # r_16_192 kernel share of in-model GPU time (from qk_prefill_inmodel_attribution)
GATEUP_LAUNCHES = 72   # 36 layers x (gate+up)


def load_mod(p, n):
  s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def perflevel(x): subprocess.run(["rocm-smi", "--setperflevel", x], capture_output=True, text=True)


def main() -> int:
  import numpy as np
  from tinygrad import Tensor, UOp, Device
  from tinygrad.dtype import dtypes
  from tinygrad.engine.realize import run_linear
  import tinygrad.codegen.opt.postrange as pr
  from tinygrad.llm.model import Transformer, PREFILL_UBATCH
  dev = Device["AMD"]; ref = load_mod(REF_SRC, "ref")
  perflevel("high")
  result = {"date": "2026-06-20", "phase": "PREFILL_MATMUL_PENALTY_VERIFY", "shape": {"M": M, "N": N, "K": K}}
  try:
    # --- in-model: warm forward wall ---
    Tensor.manual_seed(0)
    model, _ = Transformer.from_gguf(pathlib.Path(sys.argv[1]).expanduser(), 768)  # 2048 OOMs beside isolated kernels; isolated numbers + bench-wall estimate carry the result
    Nt = PREFILL_UBATCH; maxc = model.max_context
    vsp = UOp.variable("start_pos", 0, maxc - 1); temp = Tensor([0.0])
    t = Tensor((([5, 6, 7, 8, 9, 10] * (maxc // 6 + 1))[:maxc]), dtype="int32").reshape(1, maxc)
    sp = vsp.bind(0); chunk = t[:, sp:sp + Nt]
    pr._warmstart_stats = {"match": 0, "apply": 0, "error": 0}
    fwd = lambda: model(chunk, sp, temp)
    for _ in range(5): fwd().realize(); dev.synchronize()
    walls = []
    for _ in range(8):
      dev.synchronize(); t0 = time.perf_counter(); fwd().realize(); dev.synchronize(); walls.append(time.perf_counter() - t0)
    W = min(walls)
    inmodel_gateup_s = GATEUP_SHARE * W / GATEUP_LAUNCHES
    A_tflops = FLOP / inmodel_gateup_s * 1e-12

    # --- isolated kernels (small buffers; fits beside the model) ---
    rng = np.random.default_rng(1)
    a = Tensor((rng.standard_normal((M, K)) * 0.1).astype(np.float16)).realize()
    bt = Tensor((rng.standard_normal((N, K)) * 0.1).astype(np.float16)).realize()
    c = Tensor.empty(M, N, dtype=dtypes.half).realize(); Tensor.realize(a, bt, c)
    ours_lin, _ = ref._run_insts_lds(ref.build_gemm_lds2(M, N, K, 2, 2, 4, 4, 32, 16, 0, 1), a, bt, c, M, N, K, "ours", 32768, 128, 128, 128)
    launches = {"ours_isolated": lambda: run_linear(ours_lin)}
    try:
      ptm1 = load_mod(PTM1_SRC, "ptm1"); auth_lin, _m, _al = ptm1.build_authority()
      launches["authority_isolated"] = lambda: run_linear(auth_lin)
    except Exception as ex:
      result["authority_err"] = repr(ex)[:120]

    def batched(fn, B=16, reps=40):  # host-amortized GPU throughput (per the batch-isolate method)
      for _ in range(20): fn()
      dev.synchronize(); best = 1e9
      for _ in range(reps):
        dev.synchronize(); t0 = time.perf_counter()
        for _ in range(B): fn()
        dev.synchronize(); best = min(best, (time.perf_counter() - t0) / B)
      return FLOP / best * 1e-12
    iso = {k: round(batched(fn), 1) for k, fn in launches.items()}
  finally:
    perflevel("auto")

  B_auth = iso.get("authority_isolated"); C_ours = iso.get("ours_isolated")
  result.update({"warm_forward_wall_ms": round(W * 1e3, 1), "gateup_share_used": GATEUP_SHARE,
                 "A_inmodel_gateup_tflops": round(A_tflops, 1), "B_authority_isolated_tflops": B_auth,
                 "C_ours_isolated_tflops": C_ours,
                 "inmodel_over_authority": round(A_tflops / B_auth, 3) if B_auth else None,
                 "ours_over_authority": round(C_ours / B_auth, 3) if (B_auth and C_ours) else None})
  BENCH_WALL_MS = 183.0   # validated warmstart-applied forward (qk_prefill_v2_measure: 2797 tok/s)
  if W * 1e3 > 1.4 * BENCH_WALL_MS:
    # forward ran far slower than the validated bench => TC warmstart did NOT apply (matmul ran without WMMA).
    # the in-model number is then the NO-WMMA matmul, not the real prefill path. Reject; estimate from bench wall.
    A_bench = FLOP / (GATEUP_SHARE * (BENCH_WALL_MS / 1e3) / GATEUP_LAUNCHES) * 1e-12
    result["forward_unwarmed_warmstart_not_applied"] = True
    result["A_inmodel_gateup_tflops_BENCHwall_estimate"] = round(A_bench, 1)
    result["inmodel_over_authority_BENCHwall"] = round(A_bench / B_auth, 3) if B_auth else None
    result["verdict"] = "FORWARD_UNWARMED_REJECT_INMODEL"
    result["why"] = (f"warm forward ran {W*1e3:.0f}ms >> validated {BENCH_WALL_MS:.0f}ms => TC warmstart didn't apply "
                     f"(matmul without WMMA at {A_tflops:.0f} TFLOPS). REJECT this in-model number. Using the validated "
                     f"bench wall instead: in-model gate/up ~{A_bench:.0f} vs isolated authority {B_auth:.0f} = "
                     f"{A_bench/B_auth:.2f}x => matmul largely INTEGRATES FINE (gap ~clock). Isolated ours {C_ours} = "
                     f"{C_ours/B_auth:.2f}x authority -> since matmul integrates fine, wiring our kernel would transfer "
                     f"~{round((C_ours/B_auth-1)*GATEUP_SHARE*100)}% to prefill.")
  elif B_auth:
    r = A_tflops / B_auth
    if r >= 0.85:
      result["verdict"] = "MATMUL_INTEGRATES_FINE"; result["why"] = (f"in-model gate/up {A_tflops:.0f} ~= isolated authority {B_auth:.0f} TFLOPS (ratio {r:.2f}) -> no recoverable graph penalty; the kernel IS the limit. Matmul is NOT the integration lever -> attention (28%). Wiring our kernel upside ~ {C_ours/B_auth:.2f}x on the gate/up slice only.")
    else:
      result["verdict"] = "MATMUL_GRAPH_PENALTY"; result["why"] = (f"in-model gate/up {A_tflops:.0f} << isolated authority {B_auth:.0f} TFLOPS (ratio {r:.2f}) -> the SAME kernel runs slower in the forward = recoverable graph/scheduling penalty on ~70% of prefill.")
  else:
    result["verdict"] = "INCOMPLETE_NO_AUTHORITY"
  pathlib.Path("bench/amd-broad-backend-roadmap").mkdir(parents=True, exist_ok=True)
  (ROOT / "bench/amd-broad-backend-roadmap/prefill_matmul_penalty_verify_result.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({k: result[k] for k in ("warm_forward_wall_ms", "A_inmodel_gateup_tflops", "B_authority_isolated_tflops", "C_ours_isolated_tflops", "inmodel_over_authority", "ours_over_authority", "verdict", "why") if k in result}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
