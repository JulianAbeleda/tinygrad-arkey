# Increment 2 Result: flash prefill kernel — CORRECT, but v1 is 15× too slow → not shipped (rest at Increment 0)

Date: 2026-06-20. Repo `/home/ubuntu/tinygrad-arkey`. gfx1100, Qwen3-8B. Scope:
`docs/prefill-increment0-and-flash-execution-scope-20260620.md` (Part B). Kernel: `extra/qk_prefill_flash.py`.

## What was built

A fused causal GQA online-softmax flash attention kernel for PREFILL, **no score materialization**: one workgroup
per (head h, query row q), 128 threads = head_dim d, online softmax over the causal key range [0, start_pos+q]
with a cooperative q·k dot (LDS tree reduction over Hd), accumulating `acc[d] += p·v[t,d]`, output O[h,q,d]=acc/l.
Causal via the `t≤qpos` bound; GQA via `kv=h/G`. Compiled via the same `dev.runtime(dev.compiler.compile(src))`
path as `qk_flash_decode.py` (LDS = `__attribute__((shared,aligned(16)))`, barrier = the amdgcn fence/s_barrier
sequence from the HIP renderer).

## Correctness — PASS (essentially exact)

Standalone vs a causal-SDPA numpy reference, across shapes incl. the full T=512 chunk and causal start_pos:

| T | start_pos | KV | max_err | rel_rmse |
|---:|---:|---:|---:|---:|
| 64 | 0 | 64 | 9.5e-7 | 1.7e-7 |
| 128 | 384 | 512 | 5.7e-7 | 5.4e-7 |
| 512 | 0 | 512 | 1.3e-6 | 2.8e-7 |
| 256 | 512 | 768 | 6.6e-7 | 6.4e-7 |

rel_rmse ~1e-7 (fp32 accumulation) — a correct flash kernel.

## Performance — FAIL (v1 is ~15× slower than the shipped fused path)

Per-layer kernel time (clock pinned high), and the bar to beat (the shipped concrete-fusion attention is ~5% of a
~153 ms forward ≈ 0.2 ms/layer):

| T | start_pos | KV | flash v1 ms/layer | ×36 = ms/forward |
|---:|---:|---:|---:|---:|
| 512 | 0 | 512 | 3.27 | 117.6 |
| 512 | 512 | 1024 | 10.17 | 366 |
| 512 | 1536 | 2048 | 24.08 | 867 |
| 512 | 3072 | 3584 | 45.10 | 1624 |

At KV=512 flash v1 is **~15× slower** than the current fused attention, and it scales linearly with KV (worse at
long context). Cause: 16384 tiny workgroups each running a serial online-softmax loop with ~9 LDS barriers per
key — barrier-serialized, poor ILP, no tensor cores. **Not wired into the model (it would regress).**

## Verdict — rest prefill at the shipped state

- The flash kernel is **correct**, but a naive (non-TC, barrier-per-key) flash loses badly to tinygrad's existing
  fused attention codegen. Beating the fused path needs the real optimizations — **key-tiling + WMMA fragments**
  for Q@Kᵀ/P@V + far fewer barriers — a multi-day kernel build.
- **The marginal value of that build is now low:** Increment 0 (shipped) already puts prefill at **73–111% of
  llama** across contexts, and the structural advantage flash was meant to provide (avoiding the Hq×T×KV score
  materialization) **doesn't translate to a win** — the concrete fusion path materializes scores and is *still*
  73% of llama at KV=3584 (234 ms forward). Materialization is not the bottleneck at the contexts that matter.
- This is the **"isolated kernel wins don't transfer / integration is the bottleneck"** pattern once more: a
  hand-rolled flash kernel is structurally elegant but loses to the integrated fused path.

**Recommendation: rest prefill.** The shipped combination — Branch B fusion (default-on) + Increment 0 concrete-KV
precompile (opt-in) — achieves llama parity on prefill throughput, byte-identical. Keep `extra/qk_prefill_flash.py`
as a correct, reusable starting point if a WMMA-flash build is ever justified (e.g. extreme context where
materialization VRAM, not time, forces it).

## If ever resumed (v2 sketch)
Key-tiling (Tk keys/iter, the 128 threads cooperatively load a K-tile to LDS), WMMA fragments for the per-tile
Q@Kᵀ and P@V (RDNA3 `v_wmma`; layout per `extra/gemm/rdna3_wmma_matmul.py`), online-softmax rescale across tiles,
one barrier per tile not per key. Symbolic-length via the flash-decode bound/unbound twins. Each step its own gate.
