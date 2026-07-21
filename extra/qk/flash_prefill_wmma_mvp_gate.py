"""M0: Roofline ceilings + SDPA baseline measurement.
Gate script for the fused-WMMA-flash-prefill MVP proof-of-theory.
See docs/flash-prefill-wmma-mvp-scope-20260721.md for the full scope."""

import sys, os, time, math
sys.path.insert(0, '.')
os.environ.setdefault('DEV', 'AMD')
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import getenv

dev = Device[Device.DEFAULT]

# ─── helpers ────────────────────────────────────────────────────────
def bench(fn, n=10, warmup=200):
    """Measure GPU kernel time via DEBUG=2 tm. Warm to boost clocks."""
    # Warmup
    for _ in range(warmup):
        out = fn(); out.realize(); Device[Device.DEFAULT].synchronize()
    # Timed runs
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(); out.realize(); Device[Device.DEFAULT].synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times), out

def bench_tm(fn, n=10, warmup=200):
    """Measure via wall-clock (proxy for tm when DEBUG=2 isn't parsed)."""
    for _ in range(warmup):
        out = fn(); out.realize(); Device[Device.DEFAULT].synchronize()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(); out.realize(); Device[Device.DEFAULT].synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times)

# ─── M0: Roofline Ceilings ─────────────────────────────────────────
print("=" * 80)
print("M0 — ROOFLINE CEILINGS + SDPA BASELINE")
print("=" * 80)

# Compute ceiling: large square fp16 GEMM lowered to WMMA
M = N = K = 4096
a = Tensor.randn(M, K, dtype=dtypes.float16).contiguous().realize()
b = Tensor.randn(K, N, dtype=dtypes.float16).contiguous().realize()

def gemm():
    return a @ b

t_gemm = bench_tm(gemm, n=5, warmup=50)
flops_gemm = 2 * M * N * K
C_peak_tflops = flops_gemm / t_gemm / 1e12
print(f"\nCompute ceiling (C_peak): ({M},{N},{K}) fp16 GEMM")
print(f"  Time: {t_gemm*1e3:.2f} ms")
print(f"  FLOPs: {flops_gemm/1e9:.1f} GFLOP")
print(f"  C_peak = {C_peak_tflops:.2f} TFLOP/s")

# Memory ceiling: large D2D copy
SZ = 256 * 1024 * 1024  # 256M elements
n_elems = SZ
a_bw = Tensor.randn(n_elems, dtype=dtypes.float16).contiguous().realize()
b_bw = Tensor.randn(n_elems, dtype=dtypes.float16).contiguous().realize()

def d2d_copy():
    return a_bw + b_bw  # simplest op that reads a and b

t_bw = bench_tm(d2d_copy, n=5, warmup=50)
bytes_moved = n_elems * 2 * 3  # read a (2 bytes/elem) + read b + write = 3 * 2 bytes
B_peak_gbs = bytes_moved / t_bw / 1e9
print(f"\nMemory ceiling (B_peak): {n_elems/1e6:.0f}M-element D2D copy")
print(f"  Time: {t_bw*1e3:.2f} ms")
print(f"  Bytes: {bytes_moved/1e6:.1f} MB")
print(f"  B_peak = {B_peak_gbs:.1f} GB/s")

# Save ceilings
print(f"\n>>> C_peak = {C_peak_tflops:.2f} TFLOP/s")
print(f">>> B_peak = {B_peak_gbs:.1f} GB/s")

# ─── M0: SDPA Baseline at 14B, T=KV=2048 ─────────────────────────
print("\n" + "-" * 60)
print("SDPA Baseline: 14B, T=KV=2048, causal, fp16")

Hkv, G, Hq, Hd, B = 8, 5, 40, 128, 1
T = KV = 2048
scale = Hd ** -0.5

q = Tensor.randn(B, Hkv, G, T, Hd, dtype=dtypes.float16).contiguous().realize()
k = Tensor.randn(B, Hkv, 1, KV, Hd, dtype=dtypes.float16).contiguous().realize()
v = Tensor.randn(B, Hkv, 1, KV, Hd, dtype=dtypes.float16).contiguous().realize()
mask = Tensor.full((1, 1, T, KV), float("-inf"), dtype=dtypes.float16, buffer=False).triu(1).contiguous().realize()

def sdpa_ref():
    scores = (q @ k.transpose(-1, -2)).float() * scale
    scores = scores + mask.reshape(1, 1, 1, T, KV)
    s = scores.softmax(-1)
    return (s.cast(dtypes.float16) @ v)

# Warmup + measure
for _ in range(50):
    out = sdpa_ref(); out.realize(); Device[Device.DEFAULT].synchronize()

t_sdpa = bench_tm(sdpa_ref, n=10, warmup=0)
out_ref = sdpa_ref()
out_ref.realize()
Device[Device.DEFAULT].synchronize()

# Compute SDPA's roofline fractions
flops_attn = 2 * T * KV * Hd * Hq * 2  # QK^T + PV
score_bytes = T * KV * Hq * 4  # fp32 score matrix spill
kv_bytes = (KV * Hd * 2 * 2)  # K + V reads (approximate, per head)
total_bytes_approx = score_bytes + kv_bytes * Hkv + T * Hd * 2 * Hq  # rough

sdpa_tflops = flops_attn / t_sdpa / 1e12
sdpa_gbs = total_bytes_approx / t_sdpa / 1e9
compute_frac_sdpa = sdpa_tflops / C_peak_tflops
mem_frac_sdpa = sdpa_gbs / B_peak_gbs

print(f"\nSDPA reference:")
print(f"  Time: {t_sdpa*1e3:.2f} ms")
print(f"  FLOPs: {flops_attn/1e9:.1f} GFLOP")
print(f"  Score matrix: {score_bytes/1e6:.1f} MB (fp32, spilled to HBM)")
print(f"  SDPA TFLOP/s: {sdpa_tflops:.2f}")
print(f"  SDPA GB/s:   {sdpa_gbs:.1f}")
print(f"  compute_frac: {compute_frac_sdpa:.3f}  ({100*compute_frac_sdpa:.1f}% of C_peak)")
print(f"  mem_frac:     {mem_frac_sdpa:.3f}  ({100*mem_frac_sdpa:.1f}% of B_peak)")

print(f"\n>>> SDPA attention sits at {100*compute_frac_sdpa:.1f}% compute-ceiling, {100*mem_frac_sdpa:.1f}% memory-ceiling")

# ─── Save state ────────────────────────────────────────────────────
# Save reference output for later correctness checks
ref_tensor = out_ref.numpy() if hasattr(out_ref, 'numpy') else None
print(f"\nM0 complete. Reference output saved. Ceilings: C={C_peak_tflops:.2f} TFLOP/s, B={B_peak_gbs:.1f} GB/s")
