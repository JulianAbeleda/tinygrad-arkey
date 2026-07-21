import os, time
os.environ.setdefault("DEV", "AMD")
from tinygrad import Tensor, dtypes, Device
from tinygrad.engine.jit import TinyJit

B = 1
Hkv = 8
G = 5
Hq = Hkv * G
Hd = 128
T = 512
start_pos = 3584
KV = start_pos + T
scale = Hd ** -0.5
n_layers = 40

print(f"Config: B={B} Hq={Hq} Hkv={Hkv} G={G} Hd={Hd} T={T} KV={KV}")

qg = Tensor.randn(B, Hkv, G, T, Hd, dtype=dtypes.float16).contiguous().realize()
kg = Tensor.randn(B, Hkv, 1, KV, Hd, dtype=dtypes.float16).contiguous().realize()
vg = Tensor.randn(B, Hkv, 1, KV, Hd, dtype=dtypes.float16).contiguous().realize()
mask = Tensor.full((1, 1, T, KV), float("-inf"), dtype=dtypes.float16, buffer=False).triu(start_pos + 1).contiguous().clone().realize()

def dev_sync(): Device[Device.DEFAULT].synchronize()

def timeit_jit(fn, n=20, warmup=5):
    jfn = TinyJit(fn)
    for _ in range(warmup):
        out = jfn(qg, kg, vg, mask)
        out.realize(); dev_sync()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = jfn(qg, kg, vg, mask)
        out.realize(); dev_sync()
        times.append(time.perf_counter() - t0)
    return min(times), out

def f_qkT(q,k,v,m): return (q @ k.transpose(-1,-2)).float() * scale
def f_qkT_mask(q,k,v,m):
    s = (q @ k.transpose(-1,-2)).float() * scale
    return s + m.reshape(1,1,1,T,KV)
def f_softmax(q,k,v,m):
    s = (q @ k.transpose(-1,-2)).float() * scale
    s = s + m.reshape(1,1,1,T,KV)
    return s.softmax(-1)
def f_full(q,k,v,m):
    s = (q @ k.transpose(-1,-2)).float() * scale
    s = s + m.reshape(1,1,1,T,KV)
    p = s.softmax(-1)
    return (p.cast(dtypes.float16) @ v)

t_qkT,_ = timeit_jit(f_qkT)
t_qkT_mask,_ = timeit_jit(f_qkT_mask)
t_softmax,_ = timeit_jit(f_softmax)
t_full,_ = timeit_jit(f_full)

# PV isolated: use fixed precomputed s
s_fixed = f_softmax(qg,kg,vg,mask).cast(dtypes.float16).contiguous().realize()
def f_pv(q,k,v,m): return s_fixed @ v
t_pv,_ = timeit_jit(f_pv)

mask_delta = max(t_qkT_mask - t_qkT, 0)
softmax_delta = max(t_softmax - t_qkT_mask, 0)
matmul_isolated = t_qkT + t_pv

print("\n=== JIT steady-state per-stage timings (ms), min-of-20 ===")
print(f"QK^T matmul (isolated):        {t_qkT*1e3:.4f} ms")
print(f"+scale+mask (cumulative):      {t_qkT_mask*1e3:.4f} ms  (delta={mask_delta*1e3:.4f} ms)")
print(f"+softmax (cumulative):         {t_softmax*1e3:.4f} ms  (delta={softmax_delta*1e3:.4f} ms)")
print(f"PV matmul (isolated):          {t_pv*1e3:.4f} ms")
print(f"FULL fused sequence:           {t_full*1e3:.4f} ms")

total_model_ms = t_full * n_layers * 1e3
print(f"\nWhole-model attn-growth estimate x{n_layers}: {total_model_ms:.2f} ms (cross-check BoltBeam ~1039ms)")

flops_qkT = 2*T*KV*Hd*Hq
flops_pv = 2*T*KV*Hd*Hq
flops_fund = flops_qkT+flops_pv
ach_full = flops_fund/t_full/1e12
ach_mm = flops_fund/matmul_isolated/1e12
peak = 122.8
print(f"\nFundamental FLOPs/layer: {flops_fund/1e9:.2f} GFLOP")
print(f"Achieved TFLOP/s (full):   {ach_full:.3f}  ({100*ach_full/peak:.2f}% of {peak} peak)")
print(f"Achieved TFLOP/s (mm-only):{ach_mm:.3f}  ({100*ach_mm/peak:.2f}% of {peak} peak)")

mm_frac = matmul_isolated/t_full
print(f"\nMatmul fraction of full: {100*mm_frac:.1f}%  | Reclaimable fraction: {100*(1-mm_frac):.1f}%")
recov_per_layer_ms = (t_full-matmul_isolated)*1e3
print(f"Recoverable ms/layer: {recov_per_layer_ms:.4f} -> total x{n_layers}: {recov_per_layer_ms*n_layers:.2f} ms")
