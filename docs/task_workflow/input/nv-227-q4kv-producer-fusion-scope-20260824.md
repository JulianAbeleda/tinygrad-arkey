# NV 227 tok/s push: Q4/Q4 K/V producer fusion scope

Date: 2026-08-24
Repo: `/home/ubuntu/tinygrad-arkey`
Base checkpoint: `a8f153e1d`
Target: Qwen3-8B-Q4_K_M decode on RTX 5090 (`NV sm_120`), depth 512

## Starting ledger and target

The conservative installed endpoint is `4.5564175625 ms/token`, or
`219.4706667 tok/s`. The 227 tok/s boundary is `4.4052863436 ms/token`, so
this phase must recover `151.1312189 us/token` (`+7.5293333 tok/s`).

The installed device profile is nearly serialized:

```text
node sum      4301.744 us/token
device union  4298.500 us/token
overlap          4.106 us/token
```

Consequently, an earlier producer timestamp is not a performance result.
Moving the `67.360 us/token` K terminal body earlier cannot reduce wall time
unless the construction also creates measured overlap or removes device work.
The previous `93.875 us/token` producer-to-flash ready gap is likewise not
bookable: it contains scheduled useful kernels, not demonstrated idle time.

## First bounded construction

Fuse each Q4_K attention-K/Q4_K attention-V projection pair into one
dual-output producer. The construction reuses the same fp16 activation and
streams two independent Q4_K matrices in one launch. Each output preserves
the installed vector-load dot-product association exactly; only launch and
stream topology change.

The current kernel census has 18 Q4/Q4 K/V pairs (36 projection launches) and
18 Q4/Q6 pairs. Nine Q4/Q4 pairs use the ordinary vector-Q4 route and nine sit
behind the shared-Q8 route. The Q4/Q4 population is tested first because it has one exact
emitter grammar on both sides and therefore isolates the fusion mechanism.

### Phase A: native microgate

Control:

```text
q4k_g3_lanemap_gemv_vec_1024_4096(K)
q4k_g3_lanemap_gemv_vec_1024_4096(V)
```

Candidate:

```text
q4k_g3_lanemap_gemv_pair_vec_1024_4096(K, V)
```

Required gates:

- all 2048 fp32 outputs are bit-identical;
- candidate uses one launch and keeps the exact vector per-lane accumulation;
- locked-clock CUDA-event timing compares the two-launch control pair with
  the one-launch candidate;
- no model route is wired before this gate passes.

Advance to production when the hot microgate is faster and the estimated
18-pair recovery is at least `15 us/token`. This is an investigation threshold,
not a booked gain.

### Phase B: one-block structural lease

Install a closed research admission on one exact ordinary Q4/Q4 block. It must:

- replace exactly two K/V projection launches with one dual-output launch;
- introduce no materialization, cast, completion, or transport kernel;
- preserve the full token stream;
- retain the installed producer-owned cache sink unchanged initially.

This phase tests projection work removal independently of direct cache writes.

### Phase C: all nine ordinary Q4/Q4 blocks

Run a locked depth-512 profile and a control/candidate/control wall bracket.
The candidate advances only if node sum and union both fall, token hashes are
identical, and candidate wall beats both controls. Any timestamp-only
reordering with flat device sum is a no-go.

### Phase D: shared-Q8 pairs and direct V cache output

Only after ordinary projection fusion pays, build the distinct shared-Q8 pair
emitter for the other nine Q4/Q4 blocks, then test writing fused V output directly
to the V cache half while K remains the output consumed by the exact K
RMSNorm+RoPE cache producer. This second construction must remove device work
or create measured overlap. It is not bundled into Phase A-C, so a cache-view
or dependency bug cannot obscure whether K/V projection fusion itself pays.

## Relationship to 227

This row cannot be assumed to cover the complete `151.131 us/token` target.
The current Q4/Q4 K/V population is a clean producer-side rate/launch lever;
after its measured wall result, the ledger will determine whether the next
surface is the nine mixed Q4/Q6 K/V pairs, projection output transport, or a
different body-reduction row. No theoretical ceiling is converted to tok/s
without a same-session wall bracket.
