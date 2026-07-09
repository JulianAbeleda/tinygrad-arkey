# 8B Prefill Lifecycle Compression Audit

Date: 2026-07-09.

## Question

Can we audit the major lifecycle compression step before doing more e2e work?

Yes. The useful audit stack is:

| Layer | Question | Tool |
|---|---|---|
| L0 route/shape | Which route and tile shape are we comparing? | `prefill_route_census.py`, `hand_vs_generated_shape_matrix.py` |
| L1 pre-isel ownership | Does the compiler still know LDS stage/slot/producer identity? | `prefill_stage_owner_audit.py` |
| L2 final stream density | What instructions did we actually emit? | `kernel_lifecycle_trace.py` |
| L3 bounded timing | Does the structural change move TFLOPS? | `hand_vs_generated_shape_matrix.py` |
| L4 e2e transfer | Does whole prefill move? | `prefill_whole_synced.py --require-route` |

This audit stops at L3. The result says the next blocker is still inside generated lifecycle compression, so another e2e
run would not be the first useful move.

## Shape

Representative comparison:

```text
m=512, n=5120, k=5120
generated shape: u0=2,u1=2,loc=2,unr=2
hand oracle:     wm=2,wn=2,waves_m=1,waves_n=1,bk=64,dbuf=1
target:          AMD:ISA:gfx1100
```

## Final Stream Density

| variant | TFLOPS | status | inst/WMMA | wait/WMMA | global/WMMA | ds_store/WMMA | ds_load/WMMA | barrier/WMMA | D3 | max WMMA cluster |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| generated baseline DBUF | 9.75 | ok | 39.062 | 3.312 | 2.0 | 2.0 | 4.0 | 0.125 | false | 1 |
| generated K-major | 12.51 | ok | 34.625 | 2.875 | 2.0 | 2.0 | 2.0 | 0.125 | false | 3 |
| generated K-major + D3 stage steal | 10.33 | ok | 42.500 | 4.562 | 3.125 | 3.125 | 2.0 | 1.062 | true | 3 |
| hand LDS2 oracle | structural only | structural only | 9.547 | 0.406 | 1.0 | 1.0 | 2.0 | 0.062 | true | 4 |

Interpretation:

- K-major proves the generated path can reach hand-like LDS reload density: `ds_load_b128/WMMA = 2.0`.
- K-major still lacks body DBUF cadence: `D3=false`.
- K-major + D3 stage steal proves the combined property exists, but it is too expensive:
  - `global/WMMA` rises from `2.0` to `3.125`,
  - `ds_store/WMMA` rises from `2.0` to `3.125`,
  - `barrier/WMMA` rises from `0.125` to `1.062`,
  - bounded TFLOPS drops from `12.51` to `10.33`.
- The hand oracle is still much denser:
  - generated K-major has about `3.6x` more instructions per WMMA,
  - about `7.1x` more waits per WMMA,
  - about `2x` global/LDS-store traffic per WMMA,
  - the same LDS-load density only after K-major.

## Pre-Isel Ownership

`prefill_stage_owner_audit.py` shows:

| boundary | stage_count | store_count | WMMA operands | tagged stores | result |
|---|---:|---:|---:|---:|---|
| postrange | 2 | 0 | 2 | 0 | Stage buffers are still visible and tagged. |
| full | 0 | 0 | 32 | 0 | WMMA carriers are untagged stacks; stage identity is no longer directly visible. |

This is the upstream reason slot-only suppression is dangerous. By the late lowering point where prologue stores are
suppressed, the original store key is gone and only the absolute LDS slot is available.

## Suppression Finding

The existing broad suppress path:

```text
PREFILL_WMMA_KMAJOR_STAGE_STEAL=1
PREFILL_WMMA_KMAJOR_STAGE_STEAL_MEMO=1
PREFILL_WMMA_KMAJOR_STAGE_STEAL_SUPPRESS=1
```

reduced traffic, but produced wrong output:

```text
global/WMMA: 3.125 -> 1.625
store/WMMA:  3.125 -> 1.625
status:      WRONG rr=nan
```

Audit reason:

```text
The suppress set records stolen windows, but late stores mostly report key=None and only an absolute LDS slot.
The same LDS slot is reused across K phases.
Slot-only suppression deletes a phase-0 producer needed before the first WMMA.
```

## Can This Get To 58 TFLOPS?

The hand oracle proves the hardware/shape class can be much faster, but the generated path is not yet in the same
lifecycle-density class.

Current bounded generated best in this pass:

```text
K-major: 12.51 TFLOPS
hand-class target discussed previously: about 58 TFLOPS
missing: about 4.6x
```

That missing factor is not a small waitcnt tweak. The trace says the generated stream still has:

```text
~3.6x instruction density
~7.1x wait density
~2.0x global/store density
weaker WMMA clustering
```

So yes, we can trace before e2e, and the trace says the 58-TFLOPS gap is already lost in the final generated lifecycle.

## Next Primitive

The primitive is not "add D3" anymore. The existing D3 marker can do that.

The next primitive is:

```text
epoch-aware stage movement / suppression
```

Required behavior:

```text
move/reuse next-slot stage work in the body
preserve K-major fragment reuse
suppress only the duplicate stage store for the same producer epoch
never suppress the phase-0 producer for the same LDS slot
```

Done signal:

```text
D3=true
ds_load_b128/WMMA <= 2.0
max_cluster >= 3
global_b128/WMMA close to 2.0, not 3.125
ds_store_b128/WMMA close to 2.0, not 3.125
barriers close to 2 total, not 17
bounded TFLOPS > 12.51
```

Only after that should we run whole-prefill again.
