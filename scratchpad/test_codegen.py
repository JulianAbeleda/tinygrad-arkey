#!/usr/bin/env python3
"""
Codegen efficiency benchmark: tinygrad attention matmuls vs llama reference.

Measures QK^T and PV matmuls in isolation across context sizes,
computes achieved TFLOP/s, and quantifies the codegen gap relative
to llama's achieved ~6.7 TFLOP/s on these shapes.

Run:
    cd /home/ubuntu/tinygrad-arkey && python3 -c "import sys; sys.path.insert(0,'.'); exec(open('scratchpad/test_codegen.py').read())"
"""
import os, time
os.environ.setdefault("DEV", "AMD")
from tinygrad import Tensor, dtypes, Device

# fixed shape params
B   = 1
Hkv = 8
G   = 5
Hq  = Hkv * G   # 40
Hd  = 128

# peak & reference
GFX1100_PEAK_FP16_TFLOPS = 122.8   # RDNA3 WMMA dense FP16
LLAMA_REF_TFLOPS         =   6.7   # llama.cpp achieves ~6.7 TFLOP/s on these shapes
WARMUP_RUNS = 5
BENCH_RUNS  = 20

dev = Device[Device.DEFAULT]


def sync():
    dev.synchronize()


def bench(fn, warmup=WARMUP_RUNS, n=BENCH_RUNS):
    """Return min-of-n steady-state wall-clock seconds."""
    for _ in range(warmup):
        out = fn()
        out.realize()
        sync()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        out.realize()
        sync()
        times.append(time.perf_counter() - t0)
    return min(times)


def run_one_context(T):
    """Benchmark QK^T and PV matmuls at a single context size KV=T."""
    KV = T

    # build inputs
    q = Tensor.randn(B, Hkv, G, T,  Hd, dtype=dtypes.float16).contiguous().realize()
    k = Tensor.randn(B, Hkv, 1, KV, Hd, dtype=dtypes.float16).contiguous().realize()
    v = Tensor.randn(B, Hkv, 1, KV, Hd, dtype=dtypes.float16).contiguous().realize()

    # QK^T matmul isolated
    def qkt():
        # Match the attention path: matmul in fp16, cast to float32 for
        # downstream scale/softmax, but we only measure the matmul itself.
        return (q @ k.transpose(-1, -2)).float()

    t_qkt = bench(qkt)

    # PV matmul isolated
    # Precompute the softmax scores so PV timing is clean.
    s_fixed = qkt().softmax(-1).cast(dtypes.float16).contiguous().realize()

    def pv():
        return s_fixed @ v

    t_pv = bench(pv)

    # FLOP accounting
    # Each matmul: 2 * M * N * K  (MAC -> 2 FLOP)
    #  QK^T: (Hq*T) x Hd x (Hq*KV)  ->  2 * T * KV * Hd * Hq
    #  PV:   same dimensions
    flops_per_mm = 2 * T * KV * Hd * Hq
    flops_total  = flops_per_mm * 2

    tflops_qkt = flops_per_mm / t_qkt / 1e12
    tflops_pv  = flops_per_mm / t_pv  / 1e12
    tflops_total = flops_total / (t_qkt + t_pv) / 1e12

    eff_pct_total = 100.0 * tflops_total / GFX1100_PEAK_FP16_TFLOPS
    gap_vs_llama  = LLAMA_REF_TFLOPS / tflops_total if tflops_total > 0 else float("inf")
    gap_qkt       = LLAMA_REF_TFLOPS / tflops_qkt  if tflops_qkt  > 0 else float("inf")
    gap_pv        = LLAMA_REF_TFLOPS / tflops_pv   if tflops_pv   > 0 else float("inf")

    return {
        "T": T,
        "KV": KV,
        "t_qkt_ms":     t_qkt    * 1e3,
        "t_pv_ms":      t_pv     * 1e3,
        "t_total_ms":   (t_qkt + t_pv) * 1e3,
        "tflops_qkt":   tflops_qkt,
        "tflops_pv":    tflops_pv,
        "tflops_total": tflops_total,
        "eff_pct":      eff_pct_total,
        "gap_vs_llama": gap_vs_llama,
        "gap_qkt":      gap_qkt,
        "gap_pv":       gap_pv,
    }


def main():
    contexts = [512, 1024, 2048, 4096]
    results = []

    print("=" * 96)
    print(f"  Codegen efficiency benchmark - gfx1100 peak = {GFX1100_PEAK_FP16_TFLOPS:.1f} TFLOP/s  |  llama ref = {LLAMA_REF_TFLOPS:.1f} TFLOP/s")
    print(f"  Shapes: B={B} Hkv={Hkv} G={G} Hq={Hq} Hd={Hd}")
    print("=" * 96)

    for T in contexts:
        print(f"\n  Running T={T} ...", end="", flush=True)
        r = run_one_context(T)
        results.append(r)
        print(f" done ({r['t_total_ms']:.2f} ms, {r['tflops_total']:.2f} TFLOP/s)")

    # output table
    print()
    print("-" * 96)
    print(f"{'T':>6}  {'QK^T ms':>9}  {'PV ms':>8}  {'Total ms':>9}  "
          f"{'TFLOP/s':>9}  {'Eff %':>7}  {'llama gap':>10}  {'QK gap':>8}  {'PV gap':>8}")
    print("-" * 96)
    for r in results:
        print(f"{r['T']:>6}  {r['t_qkt_ms']:>9.2f}  {r['t_pv_ms']:>8.2f}  {r['t_total_ms']:>9.2f}  "
              f"{r['tflops_total']:>9.2f}  {r['eff_pct']:>6.2f}%  {r['gap_vs_llama']:>10.2f}x  "
              f"{r['gap_qkt']:>8.2f}x  {r['gap_pv']:>8.2f}x")
    print("-" * 96)

    # summary
    try:
        width = min(int(os.popen("tput cols").read().strip()), 96)
    except Exception:
        width = 72

    print(f"\n{' Summary '.center(width, '-')}")
    print(f"  peak:      {GFX1100_PEAK_FP16_TFLOPS:>6.1f} TFLOP/s  (gfx1100 RDNA3 FP16 dense)")
    print(f"  llama ref: {LLAMA_REF_TFLOPS:>6.1f} TFLOP/s  (observed on these shapes)")
    print()
    for r in results:
        print(f"  T={r['T']:<5}  achieved {r['tflops_total']:.2f} TFLOP/s  "
              f"({r['eff_pct']:.1f}% peak)  "
              f"{r['gap_vs_llama']:.1f}x slower than llama")

    avg_gap = sum(r["gap_vs_llama"] for r in results) / len(results)
    avg_eff = sum(r["eff_pct"] for r in results) / len(results)
    print(f"\n  Average codegen gap vs llama: {avg_gap:.2f}x")
    print(f"  Average efficiency:           {avg_eff:.2f}%")
    print("-" * width)


if __name__ == "__main__":
    main()
