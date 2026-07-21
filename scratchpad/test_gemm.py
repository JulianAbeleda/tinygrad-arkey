import os
os.environ.setdefault("DEV", "AMD")

from tinygrad import Tensor, dtypes, Device
import time

PEAK_TFLOPS = 122.8  # FP16 peak
WARMUP = 5
ITERS = 10

shapes = [
    # Q/K/V projections: (1, T, 5120) @ (5120, 5120)
    ("Q/K/V proj (T=512)",   512,  5120, 5120),
    ("Q/K/V proj (T=1024)", 1024,  5120, 5120),
    ("Q/K/V proj (T=2048)", 2048,  5120, 5120),
    ("Q/K/V proj (T=4096)", 4096,  5120, 5120),
    # Gate/Up projection: (1, T, 5120) @ (5120, 13824)
    ("gate/up proj (T=512)",   512,  5120, 13824),
    ("gate/up proj (T=1024)", 1024,  5120, 13824),
    ("gate/up proj (T=2048)", 2048,  5120, 13824),
    ("gate/up proj (T=4096)", 4096,  5120, 13824),
    # Down projection: (1, T, 13824) @ (13824, 5120)
    ("down proj (T=512)",   512, 13824, 5120),
    ("down proj (T=1024)", 1024, 13824, 5120),
    ("down proj (T=2048)", 2048, 13824, 5120),
    ("down proj (T=4096)", 4096, 13824, 5120),
    # LM head: (1, T, 5120) @ (5120, 256000)
    ("lm_head (T=512)",   512,  5120, 256000),
    ("lm_head (T=1024)", 1024,  5120, 256000),
    ("lm_head (T=2048)", 2048,  5120, 256000),
    ("lm_head (T=4096)", 4096,  5120, 256000),
]


def gflops(M, N, K):
    return 2 * M * N * K / 1e9


def benchmark(T, K, N):
    dev = Device[Device.DEFAULT]
    A = Tensor.rand(1, T, K, dtype=dtypes.float16, device=Device.DEFAULT)
    B = Tensor.rand(K, N, dtype=dtypes.float16, device=Device.DEFAULT)

    # Warmup: force realization so kernels compile and cache
    for _ in range(WARMUP):
        C = A.matmul(B).realize()
        dev.synchronize()

    times = []
    for _ in range(ITERS):
        dev.synchronize()
        t0 = time.perf_counter()
        C = A.matmul(B).realize()
        dev.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    t_min = min(times)
    flops = gflops(T, N, K)  # M=T, N=N, K=K
    tflops = flops / t_min     # GFLOPS / ms = TFLOP/s
    pct = tflops / PEAK_TFLOPS * 100
    return t_min, tflops, pct


if __name__ == "__main__":
    dev = Device[Device.DEFAULT]
    dev.synchronize()

    print(f"{'Shape':<28} {'T':>6} {'TFLOP/s':>9} {'%peak':>7}")
    print("-" * 54)

    results = []
    total_gflops = 0.0
    total_time_ms = 0.0

    for name, T, K, N in shapes:
        t_ms, tflops, pct = benchmark(T, K, N)
        flops = gflops(T, N, K)
        print(f"{name:<28} {T:>6} {tflops:>9.2f} {pct:>6.1f}%")
        results.append((name, T, K, N, t_ms, tflops, pct, flops))
        total_gflops += flops
        total_time_ms += t_ms

    print()

    # Weighted-average efficiency: total GFLOPS / total time -> TFLOPS
    total_tflops = total_gflops / total_time_ms
    avg_pct = total_tflops / PEAK_TFLOPS * 100

    print(f"--- Big number ---")
    model_ms = 4721.0
    gemm_ms = 3635.0  # 77% of model
    print(f"Model total: {model_ms:.0f} ms, GEMM portion: {gemm_ms:.0f} ms (77%)")
    print(f"Weighted-avg GEMM efficiency: {avg_pct:.1f}% of peak ({PEAK_TFLOPS} TFLOPS)")

    # Each +1 percentage-point GEMM efficiency improvement:
    # actual_time = ideal_time / (eff/100), so at eff+1: new_time = ideal_time / ((eff+1)/100)
    # ratio = eff/(eff+1) -> time_saved = actual_time * (1 - ratio) = actual_time / (eff+1)
    save_per_pp = gemm_ms / (avg_pct + 1) if avg_pct > 0 else float('inf')
    print(f"Each +1pp GEMM efficiency saves: {save_per_pp:.1f} ms "
          f"(model goes from {model_ms:.0f} to {model_ms - save_per_pp:.0f} ms)")

    print()
    print(f"--- Room for improvement (lowest %peak = most headroom) ---")
    results.sort(key=lambda r: r[6])
    for i, (name, T, K, N, t_ms, tflops, pct, flops) in enumerate(results):
        bar = "\u2588" * int(pct / 5) + "\u2591" * (20 - int(pct / 5))
        print(f"  {i+1:>2}. {name:<28} {pct:>5.1f}%  {bar}")
