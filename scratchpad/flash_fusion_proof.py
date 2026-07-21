"""Prove whether flash attention fusion moves the needle.

Sweeps context sizes on tinygrad's materialized attention,
measures the matmul-only bound (the fundamental FLOP cost),
and estimates what a fused flash kernel could actually recover.

The key number: recoverable_ms at each context size.
"""
import os, time, math
os.environ.setdefault("DEV", "AMD")
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import Timing

dev = Device[Device.DEFAULT]

# Qwen2.5-14B attention config (GQA)
Hkv = 8
G = 5
Hq = Hkv * G  # 40
Hd = 128
B = 1

results = []

def make_inputs(T):
    """T = query tokens. KV = total context (set = T for prefill-like, since this is the regress region)."""
    KV = T  # worst case: full self-attention prefill
    q = Tensor.randn(B, Hkv, G, T, Hd, dtype=dtypes.float16).contiguous().realize()
    k = Tensor.randn(B, Hkv, 1, KV, Hd, dtype=dtypes.float16).contiguous().realize()
    v = Tensor.randn(B, Hkv, 1, KV, Hd, dtype=dtypes.float16).contiguous().realize()
    mask = Tensor.full((1, 1, T, KV), float("-inf"), dtype=dtypes.float16, buffer=False).triu(1).contiguous().realize()
    return q, k, v, mask

def bench(fn, n=10, warmup=3):
    for _ in range(warmup):
        out = fn()
        out.realize()
        Device[Device.DEFAULT].synchronize()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        out.realize()
        Device[Device.DEFAULT].synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times), out

for T in [512, 1024, 2048, 4096]:
    qg, kg, vg, mask = make_inputs(T)
    KV = T
    scale = Hd ** -0.5

    # Full materialized attention (current path)
    def full_attn():
        scores = (qg @ kg.transpose(-1, -2)).float() * scale
        scores = scores + mask.reshape(1, 1, 1, T, KV)
        s = scores.softmax(-1)
        return (s.cast(dtypes.float16) @ vg)

    t_full, _ = bench(full_attn)

    # QK^T matmul only
    def qkT():
        return (qg @ kg.transpose(-1, -2)).float() * scale

    t_qkT, _ = bench(qkT)

    # PV matmul only (precompute softmax scores)
    scores = (qg @ kg.transpose(-1, -2)).float() * scale
    scores = scores + mask.reshape(1, 1, 1, T, KV)
    s_fixed = scores.softmax(-1).cast(dtypes.float16).contiguous().realize()
    def pv():
        return s_fixed @ vg
    t_pv, _ = bench(pv)

    matmul_baseline = t_qkT + t_pv  # if flash could fuse perfectly (no overhead)
    reclaimable = t_full - matmul_baseline
    reclaimable_pct = 100 * reclaimable / t_full

    # FLOP accounting
    flops = 2 * T * KV * Hd * Hq * 2  # QK^T + PV, MAC = 2 FLOPs each
    score_bytes = T * KV * Hq * 4  # fp32 materialized scores

    # Realistic flash estimate:
    # A fused kernel saves the score matrix traffic BUT has overhead:
    #   - online softmax ops (not free, but cheap)
    #   - LDS staging efficiency loss vs global GEMM
    #   - Wave occupancy constraints
    # From llama's own numbers: flash-off is 2.3x flash-on for attention.
    # For tinygrad, estimate: fused recovers ~70% of the reclaimable time
    # (the other 30% is LDS overhead, softmax pipe, wave constraints)
    flash_efficiency_factor = 0.70
    estimated_flash_savings = reclaimable * flash_efficiency_factor
    estimated_flash_time = matmul_baseline + (reclaimable * (1 - flash_efficiency_factor))
    estimated_flash_pct = 100 * estimated_flash_time / t_full

    # Estimate whole-model impact (40 layers)
    n_layers = 40
    model_full = t_full * n_layers * 1e3
    model_flash = estimated_flash_time * n_layers * 1e3
    model_savings = model_full - model_flash

    results.append({
        'T': T, 'KV': KV,
        't_full_ms': t_full * 1e3,
        't_qkT_ms': t_qkT * 1e3,
        't_pv_ms': t_pv * 1e3,
        'matmul_baseline_ms': matmul_baseline * 1e3,
        'reclaimable_ms': reclaimable * 1e3,
        'reclaimable_pct': reclaimable_pct,
        'est_flash_ms': estimated_flash_time * 1e3,
        'est_flash_pct_of_full': estimated_flash_pct,
        'model_full_ms': model_full,
        'model_flash_ms': model_flash,
        'model_savings_ms': model_savings,
        'flops_g': flops / 1e9,
        'score_MB': score_bytes / 1e6,
        'tflops_full': flops / t_full / 1e12,
        'tflops_matmul': flops / matmul_baseline / 1e12,
    })
    print(f"  T={T:4d} done")

# Print results table
print("\n" + "=" * 120)
print("TINYGRAD ATTENTION: Scalability breakdown + Flash Fusion Impact Estimate")
print("=" * 120)

# Determine max widths
header_fmt = "{:<6} {:<7} {:<10} {:<10} {:<12} {:<12} {:<10} {:<10} {:<12} {:<12} {:<12} {:<12}"
fmt = "{:<6} {:<7} {:<10.2f} {:<10.2f} {:<12.2f} {:<12.2f} {:<10.1f} {:<10.2f} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f}"

print(header_fmt.format("T", "KV", "Full ms", "Matmul ms", "Reclaim ms", "Reclaim %", 
                          "Score MB", "Flash ms", "Flash %full", "ModelFull", "ModelFlash", "Savings"))
print("-" * 120)

for r in results:
    print(fmt.format(r['T'], r['KV'], r['t_full_ms'], r['matmul_baseline_ms'],
                     r['reclaimable_ms'], r['reclaimable_pct'],
                     r['score_MB'], r['est_flash_ms'], r['est_flash_pct_of_full'],
                     r['model_full_ms'], r['model_flash_ms'], r['model_savings_ms']))

print("-" * 120)

# Summary
print("\n=== KEY TAKEAWAY ===")
largest = results[-1]
print(f"At T=KV={largest['T']} (worst case prefill):")
print(f"  Materialized attention:     {largest['t_full_ms']:.2f} ms/layer  ->  {largest['model_full_ms']:.0f} ms model (40 layers)")
print(f"  Matmul-only bound:          {largest['matmul_baseline_ms']:.2f} ms/layer  ->  {largest['matmul_baseline_ms']/largest['t_full_ms']*100:.1f}% of full")
print(f"  Reclaimable (materialize overhead):  {largest['reclaimable_pct']:.1f}%")
print(f"  Estimated flash fused:      {largest['est_flash_ms']:.2f} ms/layer  ->  {largest['model_flash_ms']:.0f} ms model (40 layers)")
print(f"  Estimated savings:          {largest['model_savings_ms']:.0f} ms  ({100-largest['est_flash_pct_of_full']:.1f}% reduction)")
print(f"\n  Score matrix size at T=KV={largest['T']}: {largest['score_MB']:.0f} MB (fp32 materialized)")
print(f"  Matmul efficiency: {largest['tflops_matmul']:.1f} vs {largest['tflops_full']:.1f} TFLOP/s (full path)")
print(f"  Materialized path loses {largest['tflops_matmul']-largest['tflops_full']:.1f} TFLOP/s to memory pressure")
