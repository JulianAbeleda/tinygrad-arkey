# NV decode theory-to-evidence handoff for third-party audit (2026-08-22)

Status: analysis only. No runtime, renderer, scheduler, or model code changed.
Purpose: give an independent reviewer the full ledger as pseudocode so they can
re-run our theories instead of re-deriving our history.

Conventions: "measured" and "inferred" are different labels. A projected
ceiling is never a promotion. All times are settled reverse-bracket medians
unless stated otherwise.

## 0. Locked baseline

Same machine, same model, same depth (d512), RTX 5090, Qwen3-8B-Q4_K_M.

| side | tok/s | wall us/token | source |
| --- | ---: | ---: | --- |
| llama.cpp (`ac4cddeb0`, CUDA) | 246.37 | 4058.9 | nsys CUPTI, 762 nodes/replay |
| tinygrad-arkey (NV HCQ) | 208.84 | 4788.3 | native node ledger, 594 nodes/token |
| gap | | +729.4 | |

This is the exact, zero-residual account
(`nv-240-exact-wall-account-20260817.md`). A later locked ledger
(`nv-weighted-inter-anchor-ledger-20260820.json`) puts the same gap at +717.505
us; the mechanism is identical.

## 1. The story as pseudocode

This is the whole story in one block. Every later section is just this block
filled in with measurements.

```text
# one token's traced kernels and their dependency DAG
K = [(start, end, bytes, flops, deps) for each kernel]

mass[i]       = end[i] - start[i]
node_sum      = sum(mass[i])                 # total kernel residence time
union         = measure(union(intervals))    # GPU busy time, counted once
overlap       = node_sum - union             # time with >1 kernel resident
span          = max(end) - min(start)
critical_path = longest dependency-weighted path
host_gap      = wall - union                 # launch / submission / sync
wall          = union + host_gap

# H1 correction: not every resident microsecond is useful work.
spin[i]       = time kernel i waits on a dependency/launch signal
useful[i]     = mass[i] - spin[i]
useful_body   = sum(useful[i])               # the durable quantity
useful_overlap = time >=2 kernels are useful simultaneously

union == useful_body - useful_overlap
```

The central mistake, stated as an objective:

```text
# we optimized this:
objective_ours = minimize(node_sum)

# hardware actually charges this:
objective_real = minimize(wall)
               = minimize(useful_body - useful_overlap + host_gap)
```

`node_sum` counts llama's spin as if it were work, so "we do less work" was a
measurement artifact, not a wall win.

## 2. The lifecycle in pseudocode

Both routes compute the same layer. The difference is not the bodies; it is
which operations are `folded` or `overlapped` versus `exposed`.

```text
tinygrad_layer(x):
    nx = rmsnorm(x)                    # S0: reduce + scale, exposed
    q  = gemv(Wq, nx)                  # ANCHOR Q, memory bound
    q  = q_norm(q); q = rope(q)        # S1: exposed, nobody hides it
    k  = gemv(Wk, nx)                  # off spine but exposed here
    v  = gemv(Wv, nx)                  # off spine but exposed here
    kv = kv_store(k, v)                # exposed
    s  = flash_score(q, kv)            # exposed
    a  = flash_combine(s, kv)          # exposed
    o  = gemv(Wo, a, resadd=x)         # ANCHOR O, residual folded
    nh = rmsnorm(o)                    # S2: reduce + scale, exposed
    z  = fused_gate_up(nh)             # ANCHOR gate/up, SiLU folded
    d  = gemv(Wd, z, resadd=o)         # ANCHOR down, residual folded
    return d

llama_layer(x):
    q8 = quantize_q8_1(x)              # shared activation, overlapped
    q  = mmvq(Wq, q8)                  # ANCHOR Q
    qn = rms_norm(q)                   # overlapped
    qr = rope(qn)                      # overlapped
    k  = mmvq(Wk, q8); v = mmvq(Wv, q8) # overlapped
    s  = flash_attn(qr, kv)            # overlapped
    o  = mmvq(Wo, s, resadd=x)         # residual folded
    z  = mmvq(Wg|Wu, q8, swiglu)       # one fused anchor
    d  = mmvq(Wd, q8(z), resadd=x)     # residual folded
    return d
```

Measured S1 interval (Q anchor end to O anchor start):

```text
tinygrad S1 = 1152.250 us = 1042.500 body + 109.750 dead
llama   S1 =  517.916 us =  844.050 body - 326.134 overlap
```

Llama does not remove the S1 work; it packs the same work into a shorter window
and hides more behind its anchors.

## 3. The two competing views as pseudocode

These two views are internally consistent but point at different levers. The
third party must adjudicate them, not assume one is right.

```text
# View A: the gap is overlap.
assert overlap_llama  == 1125.1
assert overlap_tinygrad == 0.0
# so llama's union is 1125 us shorter than its node_sum, and ours is not.
leverage_A = overlap_llama - overlap_tinygrad   # ~1125 us of wall

# View B: that overlap is mostly spin, not useful parallelism.
measured shadow_share = 0.919 .. 0.954          # H1 wait-exit measurement
useful_overlap_llama = overlap_llama * (1 - shadow_share)
                     = 52 .. 91 us
# so overlap parity recovers ~5 tok/s, not ~40 tok/s.
leverage_B = useful_body_tinygrad - useful_body_llama   # +760..+800 us
```

The reconciliation problem:

```text
# View B says our anchors are +319.7..+345.2 us behind.
# The per-node table says the opposite on Q/K/V/O and parity on gate/up:
llama_anchor_alone   = 2604.3 .. 2629.7
llama_rope_quant_kv  = rope + quant + kv = 71 + 197 + 31 ~= 299
llama_anchor_total   = llama_anchor_alone + llama_rope_quant_kv ~= 2903..2936
tinygrad_anchor      = 2949.44      # rope/quant folded into this row
real_anchor_delta    = tinygrad_anchor - llama_anchor_total ~= +13..+46 us
```

If the classification is corrected, the real anchor deficit is ~13-46 us, not
~320-345 us, and the clean open rows become reduce, residual, and flash_score.

## 4. Theory ledger (what we tested)

### FUSE camp

| theory | why it should work | measured | verdict | evidence |
| --- | --- | --- | --- | --- |
| Fold RMSNorm into gate/up GEMV epilogue | remove 36 norm kernels | +82.08 us/token | NO_GO_WALL | nv-bounded-fusion-m1-result-20260821.md |
| Compute norm once instead of twice in-kernel | remove redundant epilogue | +10.8 us/launch, 1.478x | NO_GO | nv-both-directions-feasibility-20260822.md |
| Stage normalized fp16 in shared memory | run scale once, read fp16 from LDS | +38.3% over control | CLOSED_REFUTED | nv-w1w3-norm-smem-result-20260822.md |
| Copy-free native fp16 RMSNorm | remove fp32 intermediate | +12.5..+17.1 us/token | NO_GO | nv-weighted-inter-anchor-causal-gap-result-20260820.md |
| Fused w1+w3 producer, fp16 store (M2a) | remove fp32->fp16 cast | -57.2 us/token | LANDED | decode-q4k-w1w3-fp16-store-route-policy.json |
| Fold FFN-down residual (M2b) | remove residual kernel | landed | LANDED | decode-ffn-down-resadd policy |
| Four-warp fp16 geometry Q4 down | occupancy 38.8% -> 66.3% | -100.3 us, +2.01% | WALL_PASS | nv-q4-down-fp16-geometry-promotion-20260815.md |
| Four-warp fp16 geometry Q6 down | same geometry fix | -39.0 us, +0.79% | WALL_PASS | nv-q6k-ffn-down-four-warp-fp16-promotion-scope-20260816.md |
| Four-warp fp16 geometry Q6 V | replace split-K parts route | -147.35 us, +3.07% | WALL_PASS | nv-q6k-v-four-warp-fp16-promotion-20260816.md |
| DP4A, separate Q8 provider | 4x MAC/instr | -25.6 us, rel L2 2.98e-3 | NO_GO precision | nv-q4-down-dp4a-resadd-18block-gate-20260814.md |
| DP4A, producer-folded Q8_1 | no extra provider node | +45.0 us, -0.87% | NO_GO_WALL | /tmp/q4k_scalar_q8_all18_timing.json |
| Fold vocab top-1 argmax tail | remove 5 tail kernels | hidden mass, 2-3 us ceiling | NO_GO | nv-fuse-hide-eliminate-ledger-20260818.md |

### HIDE camp

| theory | why it should work | measured | verdict | evidence |
| --- | --- | --- | --- | --- |
| Two compute queues | overlap support with anchor | +113 us/token | partial, 1/6 of gap | nv-first-principles-reaudit-20260819.md |
| Reuse-lane arena coloring | remove false WAR edge | +6 tok/s, ~140 us | LANDED | nv-overlap-substrate-reuse-lanes evidence |
| PDL launch-ahead | start consumers at producer last wave | -8..-12 us/token | NO_GO | nv-pdl-queue-theories-test-20260820.md |
| Overlap is the whole 1125 us lever | View A | 92-95% shadow, 52-91 us useful | REFUTED bounded | nv-llama-useful-body-h1-result-20260821.md |
| Merge 5 graph replays into 1 | remove host submit cost | +112.9 us/token | NO_GO | nv-internal-gap-first-principles-20260818.md |

### ELIMINATE camp

| theory | why it should work | measured | verdict | evidence |
| --- | --- | --- | --- | --- |
| Search flash score tile geometry | isolated body beat 4.19 us | wall-neutral, faster variants break token gate | NO_GO | nv-flash-geometry-ab-clean evidence |
| Q4 FFN-down quad u128 smem load | faster standalone load | 34.48 vs 26.24 us in-loop | NO_GO | nv-q4-down-quad-re-census-20260813 |
| Q4/Q6 K four-warp | occupancy fix | -14%/launch isolated, -1.81 us wall | wall-neutral | nv-q4k-k-four-warp-route-result-20260822.md |
| DP4A datapath in isolation | dp4a is 4x fma.rn.f32 | 122 vs 30.6 TMAC/s | real but does not survive included cost | nv-decode-datapath-fma-vs-dp4a-measurement-20260814.md |

## 5. Why our approach should have worked, and where the theory broke

Four load-bearing mistakes, each now measured, in pseudocode.

```text
# 1. wrong objective
win_if  = node_sum_ours < node_sum_llama     # we satisfied this: -496 us
win_real = wall_ours < wall_llama            # we failed this: +729 us
# because node_sum counts llama's spin as work.

# 2. fusion is not hiding
fuse(anchor, support):
    node_sum -= mass(support)                # good
    if support was off the critical path:
        critical_path += mass(support)       # bad: now serial
hide(anchor, support):
    if independent(anchor, support) and not contend(anchor, support):
        union -= useful_overlap              # node_sum unchanged, wall falls

# 3. M=1 GEMV is not bandwidth-idle
assumed: fuse norm ALU into GEMV costs 0 (memory stalls hide it)
measured: control 22.63 us -> norm_once 33.85 us (1.48x)
because: anchor at ~60% DRAM peak is instruction-issue/latency bound

# 4. activation quantization is amortized, not free
quantize(x) costs O(K); reuse(x) = number of consumers of x
adopt_dp4a(x) wins iff reuse(x) * save_per_dot > quantize_cost(x)
FFN_down: reuse(z) = 1  -> loses (measured +45 us)
attention: reuse(x) >= 3 -> untested at wall, higher prior
```

## 5b. Where the approach should have worked, mapped to the hotspot

Each break in section 5 lands on one concrete useful-body row. This is the
"we should have won here" map. The reduce + residual rows are the single
cleanest hotspot: llama folds both into epilogues, we exposed them as separate
kernels, and our FUSE work never touched that boundary.

| theory that should have worked | hotspot where it should have won | measured excess us | probe |
| --- | --- | ---: | --- |
| fusion == hiding (break 2) | reduce + residual, folded at the norm/output boundary like llama | +234.3 + +218.4 | 1 |
| GEMV is bandwidth-idle (break 3) | anchor Q/K/V/O dots, flash score body | +13..46 anchors, +109..112 flash | 2, 3 |
| quant is amortized (break 4) | gemv_kv + flash_combine + vocab | +78..82, +47..49, +12.3..12.5 | 4, 5 |
| minimize node_sum (break 1) | every row above, re-scored as wall | +760..800 total | all |

```text
# the same map as pseudocode, so it can be re-checked by hand
should_have_won = {
    "fusion==hiding":  ("reduce", "residual"),      # fold where llama folds
    "gemv_is_idle":    ("anchor", "flash_score"),   # ALU was never free
    "quant_amortized": ("gemv_kv", "flash_combine", "vocab"),
    "minimize_node_sum": ALL_ROWS,                  # objective error, meta
}
# the one we are most confident about:
hotspot_primary = reduce + residual   # +452 us, both llama folds into epilogues
```

## 6. Hot areas as probes

Ranked by measured useful-body delta and the size of the open construction.
Each is written as the pseudocode a third party should run.

```text
# 1. reduce +234 us and residual +218 us (largest clean rows)
probe_reduce_residual():
    control   = two_kernel_norm(x)       # current: sumsq pass, then scale pass
    candidate = llama_rms_norm_f32(x)    # one pass: sumsq + normalize + fp16 write
    candidate = fold_residual(candidate) # at norm/output boundary, NOT in GEMV
    gate(exact_tokens or semantic_l2 <= 1e-3)
    measure(wall(control), wall(candidate))
    # not yet measured cleanly at current HEAD

# 2. is the anchor deficit real or a labeling artifact?
probe_anchor_mapping():
    assert llama_anchor_total ~= llama_anchor + llama_rope + llama_quant + llama_kv
    real_delta = tinygrad_anchor - llama_anchor_total   # expected ~13..46 us
    # decides whether any dot-speed work remains after four-warp promotions

# 3. flash_score +109..+112 us: launch vs body
probe_flash_split():
    assert isolated_score_body_at_parity  # 4.16..4.19 vs 4.10 us
    wall_delta - body_delta = launch/L2/overlap residual
    # stop searching tiles; measure the residual

# 4. overlap useful vs shadow (H1 is an inference)
probe_overlap_split():
    useful = sum(mass[i] - spin[i])
    shadow = overlap - useful
    assert shadow_share == 0.919..0.954  # verify with DRAM counters

# 5. effective bandwidth gap (bytes are accounting, not counters)
probe_bandwidth():
    B_min   = packed_weight_bytes_per_token
    B_route = B_min + intermediate_roundtrip_bytes
    measure(B_route) with hardware DRAM counters   # ~5.04 vs 4.70 GB is an estimate

# 6. host gap +100.6 us (5 replays vs 1)
probe_host_gap():
    assert one_replay_was_slower  # +112.9 us measured
    candidate = submit_ahead_or_multi_replay_handoff()
    measure(wall(control), wall(candidate))
    # removes 4 extra submissions without one-huge-replay GPU penalty
```

## 7. Evidence index

- Exact wall account: docs/task_workflow/input/nv-240-exact-wall-account-20260817.md
- Side-by-side: docs/task_workflow/input/nv-us-vs-llama-side-by-side-20260818.md
- Fuse/hide/eliminate ledger: docs/task_workflow/input/nv-fuse-hide-eliminate-ledger-20260818.md
- Locked S1 ledger: docs/task_workflow/output/nv-ledger-roofline-pseudocode-brief-20260820.md
- Useful-body H1: docs/task_workflow/output/nv-llama-useful-body-h1-result-20260821.md
- Useful-body reconciliation: docs/task_workflow/output/nv-useful-body-reconciliation-result-20260821.md
- DP4A datapath: docs/task_workflow/input/nv-decode-datapath-fma-vs-dp4a-measurement-20260814.md
- DP4A wall verdicts: /tmp/q4k_scalar_q8_all18_timing.json (NO_GO_WALL) and
  /tmp/q4k_fp16_geom_timing_20260815.json (WALL_PASS)
