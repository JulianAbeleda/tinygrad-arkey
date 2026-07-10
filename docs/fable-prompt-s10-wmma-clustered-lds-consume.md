# Fable Prompt: S10 WMMA Clustered LDS Consume

We need design advice, not generic GPU advice.

## Big Picture

We are trying to replace a fast hand/backend-atom LDS2 WMMA lifecycle with a generated/compiler-owned S10 path in
`tinygrad-arkey`.

The role is 8B prefill `ffn_gate_up` / bounded active shape:

```text
M=512, N=5120, K=5120 for the bounded probe
active generated shape = 2x2
target = AMD:ISA:gfx1100
```

S9 hand LDS2 is fast because it amortizes LDS loads and waits across WMMA clusters. S10 generated is correct but slow.

## The Math

RDNA3 fp16 WMMA useful work:

```text
1 WMMA = 16 * 16 * 16 FMAs = 8192 FLOPs
useful_flops = wmma_count * 8192
flops_per_overhead = useful_flops / overhead_count
```

Measured:

| Route | P8 | WMMA | useful FLOPs | waits/WMMA | max burst | ds_load/WMMA | inst/WMMA | FLOPs/wait |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S9 hand LDS2 `2x2` | pass | 64 | 524288 | 0.406 | 4 | 2.0 | 9.547 | 20165 |
| S10 generated DBUF `2x2` | fail | 16 | 131072 | 3.312 | 1 | 4.0 | 39.062 | 2473 |
| S10 K-major `2x2` | fail | 16 | 131072 | 2.875 | 3 | 2.0 | 34.625 | 2849 |
| S10 K-major + clustered wait | fail | 16 | 131072 | 2.562 | 4 | 2.0 | 34.312 | ~3197 |

K-major fixed LDS load reuse:

```text
ds_load/WMMA: 4.0 -> 2.0
max burst:    1 -> 3
```

But it did not fix wait amortization:

```text
S9 target: 0.406 waits/WMMA
S10 now:   2.562-2.875 waits/WMMA
```

## Current Generated Shape

Roughly:

```python
for phase in phases:
    # loads are somewhat grouped after K-major, but not enough
    ds_load fragments
    wait
    wmma
    wait
    wmma
    ds_load/reload fragments
    wmma
```

The desired shape:

```python
for cluster in phase:
    ds_load all A/B fragments needed by 4 WMMAs
    s_waitcnt lgkmcnt(...)

    wmma()
    wmma()
    wmma()
    wmma()
```

## Failed Small Tests

1. `AMD_ISA_WMMA_CLUSTER_LGKM_WAIT=1`

For WMMA consumers only, coalesce the targeted LDS wait to `lgkmcnt(0)`.

Result:

```text
K-major waits: 46 -> 41
wait/WMMA: 2.875 -> 2.562
max burst: 3 -> 4
TFLOPS: 12.24 -> 11.88
```

This is structurally positive but slower. It is too blunt: it waits for more, but does not improve load placement.

2. `PREFILL_WMMA_CLUSTERED_LDS_CONSUME=1`

Inside `_try_wmma_kmajor_phase`, we tried to materialize all A/B packs for a phase first and add them as dependencies
to the phase's WMMAs.

Result:

```text
wait/WMMA: 2.812
max burst: 3
ds_load/WMMA: 2.0
```

Combined with clustered wait:

```text
wait/WMMA: 2.562
max burst: 4
TFLOPS: 11.55
```

So dependency-only preloading is not enough. It does not change the actual final DS-load/WMMA/wait structure enough.

## Relevant Code

Key path:

```text
tinygrad/renderer/isa/amd.py
  _try_wmma_kmajor_phase(...)
  _pack_frag_tile(...)
  _frag_b128_loads(...)
  _build_wmma_from_packs(...)
  AMDISARenderer._insert_waitcnt(...)
```

Existing flags:

```text
PREFILL_WMMA_KMAJOR_PHASE=1
PREFILL_WMMA_AB_PROOF_KEY=1
PREFILL_WMMA_AB_PHASE_SCOPED_KEY=1
PREFILL_WMMA_AB_PROOF_FROM_LDS_DESC=1
```

## Question

What is the primitive design that can make the generated path naturally satisfy the S9 amortization math?

Specifically:

```text
waits/WMMA <= 1.0, target ~0.4
max_wmma_burst >= 4
ds_load_b128/WMMA <= 2.0
inst/WMMA materially closer to 9-12 than 34+
correctness preserved
```

I suspect the answer is not wait tuning and not dependency-only preloading. It may require a true cluster object:

```python
Cluster {
  wmma_nodes: [4 adjacent WMMA nodes],
  resident_A_fragments: fixed VGPR spans,
  resident_B_fragments: fixed VGPR spans,
  lds_windows: exact byte windows,
  barrier_epoch: int,
}

emit_cluster(cluster):
    emit all needed ds_load_b128 into resident VGPRs
    emit one wait
    emit all WMMAs before those VGPRs are reused
```

Please review that thesis and propose the smallest implementable primitive in this codebase.

