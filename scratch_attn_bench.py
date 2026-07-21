import os, time
os.environ.setdefault("DEV", "AMD")
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import Timing

dev = Device[Device.DEFAULT]

B = 1
Hkv = 8
G = 5
Hq = Hkv * G  # 40
Hd = 128
T = 512
start_pos = 3584
KV = start_pos + T  # 4096
scale = Hd ** -0.5

print(f"Config: B={B} Hq={Hq} Hkv={Hkv} G={G} Hd={Hd} T={T} KV={KV} start_pos={start_pos}")

def make_inputs():
    q = Tensor.randn(B, Hkv, G, T, Hd, dtype=dtypes.float16).contiguous().realize()
    k = Tensor.randn(B, Hkv, 1, KV, Hd, dtype=dtypes.float16).contiguous().realize()
    v = Tensor.randn(B, Hkv, 1, KV, Hd, dtype=dtypes.float16).contiguous().realize()
    mask = Tensor.full((1, 1, T, KV), float("-inf"), dtype=dtypes.float16, buffer=False).triu(start_pos + 1).contiguous().realize()
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

qg, kg, vg, mask = make_inputs()

# Stage A: QK^T matmul only
def stage_qkT():
    return (qg @ kg.transpose(-1, -2)).float() * scale

# Stage A+B: QK^T + scale + mask
def stage_qkT_mask():
    scores = (qg @ kg.transpose(-1, -2)).float() * scale
    scores = scores + mask.reshape(1, 1, 1, T, KV)
    return scores

# Stage A+B+C: + softmax
def stage_softmax():
    scores = (qg @ kg.transpose(-1, -2)).float() * scale
    scores = scores + mask.reshape(1, 1, 1, T, KV)
    s = scores.softmax(-1)
    return s

# Stage A+B+C+D: full attention (matches model.py exactly)
def stage_full():
    scores = (qg @ kg.transpose(-1, -2)).float() * scale
    scores = scores + mask.reshape(1, 1, 1, T, KV)
    s = scores.softmax(-1)
    attn = (s.cast(dtypes.float16) @ vg)
    return attn

# isolate PV matmul alone (given a precomputed softmax'd s)
s_fixed = stage_softmax().cast(dtypes.float16).contiguous().realize()
def stage_pv():
    return s_fixed @ vg

print("\nWarming up + timing (min of 10, steady-state)...")

t_qkT, _ = bench(stage_qkT)
t_qkT_mask, _ = bench(stage_qkT_mask)
t_softmax, _ = bench(stage_softmax)
t_full, _ = bench(stage_full)
t_pv, _ = bench(stage_pv)

# derive isolated stage costs
qkT_only = t_qkT
mask_only = max(t_qkT_mask - t_qkT, 0)
softmax_only = max(t_softmax - t_qkT_mask, 0)
pv_only = t_pv
matmul_total_isolated = qkT_only + pv_only

print("\n=== Per-stage timings (ms), min-of-10 steady state ===")
print(f"QK^T matmul (isolated):        {qkT_only*1e3:.4f} ms")
print(f"+scale+mask add (cumulative):  {t_qkT_mask*1e3:.4f} ms   (delta={mask_only*1e3:.4f} ms)")
print(f"+softmax (cumulative):         {t_softmax*1e3:.4f} ms   (delta={softmax_only*1e3:.4f} ms)")
print(f"PV matmul (isolated):          {pv_only*1e3:.4f} ms")
print(f"FULL fused sequence (A+B+C+D): {t_full*1e3:.4f} ms")

# whole-model estimate: 40 layers
n_layers = 40
total_model_ms = t_full * n_layers * 1e3
print(f"\n=== Whole-model attention-growth estimate (x{n_layers} layers) ===")
print(f"{total_model_ms:.2f} ms  (cross-check vs BoltBeam's ~1039ms)")

# FLOP accounting
# QK^T: for each of Hq heads, T x KV x Hd MACs => 2 * T*KV*Hd flops
flops_qkT = 2 * T * KV * Hd * Hq
flops_pv = 2 * T * KV * Hd * Hq
flops_fundamental = flops_qkT + flops_pv
print(f"\n=== FLOPs (per layer) ===")
print(f"QK^T FLOPs: {flops_qkT/1e9:.3f} GFLOP")
print(f"PV   FLOPs: {flops_pv/1e9:.3f} GFLOP")
print(f"Fundamental total: {flops_fundamental/1e9:.3f} GFLOP")

achieved_tflops_full = flops_fundamental / t_full / 1e12
achieved_tflops_matmul_isolated = flops_fundamental / matmul_total_isolated / 1e12
print(f"\nAchieved TFLOP/s using FULL time (materialized attn): {achieved_tflops_full:.3f} TFLOP/s")
print(f"Achieved TFLOP/s using ISOLATED matmul-only time:      {achieved_tflops_matmul_isolated:.3f} TFLOP/s")

# gfx1100 peak fp16 (with fp32 accum, matmul via WMMA) - RDNA3 7900XTX class peak ~ 122.8 TFLOPS FP16 (dense, no sparsity)
gfx1100_peak_fp16_tflops = 122.8
print(f"gfx1100 peak (approx, dense FP16 WMMA): {gfx1100_peak_fp16_tflops} TFLOP/s")
print(f"Efficiency (full time): {100*achieved_tflops_full/gfx1100_peak_fp16_tflops:.2f}%")
print(f"Efficiency (matmul isolated): {100*achieved_tflops_matmul_isolated/gfx1100_peak_fp16_tflops:.2f}%")

# HBM traffic for materialized scores
score_bytes = T * KV * Hq * 2  # fp16 (well actually stage stores float32 intermediate then casts -- but scores is .float() = fp32 first)
score_bytes_fp32 = T * KV * Hq * 4
print(f"\n=== Materialized score-matrix traffic (per layer) ===")
print(f"T*KV*Hq*2bytes (fp16-equiv): {score_bytes/1e6:.2f} MB")
print(f"T*KV*Hq*4bytes (fp32, since scores=.float()): {score_bytes_fp32/1e6:.2f} MB")
# write + read for scale, +mask, +softmax = up to 3 round trips of fp32 tensor + one more for cast to fp16
approx_roundtrips = 3  # write after scale*mask, read+write after mask add, read+write after softmax (rough)
print(f"Approx materialization HBM traffic (write+read x stages): ~{score_bytes_fp32*2*approx_roundtrips/1e6:.2f} MB (rough, {approx_roundtrips} round trips)")

print("\n=== Reclaimable vs fundamental split ===")
matmul_frac_of_full = matmul_total_isolated / t_full
reclaimable_frac = 1 - matmul_frac_of_full
print(f"Matmul (QK^T+PV) isolated time: {matmul_total_isolated*1e3:.4f} ms = {100*matmul_frac_of_full:.1f}% of full")
print(f"Reclaimable (mask+softmax+materialization): {100*reclaimable_frac:.1f}% of full")

recoverable_ms_per_layer = (t_full - matmul_total_isolated) * 1e3
recoverable_total_ms = recoverable_ms_per_layer * n_layers
print(f"\nRecoverable time if flash eliminated all non-matmul overhead: {recoverable_ms_per_layer:.4f} ms/layer -> {recoverable_total_ms:.2f} ms total (x{n_layers})")
