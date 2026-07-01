# LDS Staging Primitive — LDS0 Capability Audit

Date: 2026-07-01. Model: Qwen3-14B-Q4_K_M, gfx1100. Target: E_49152_32_3 (6.69% GPU at ctx512).

Follows EB0-EB5 track. EB5 recorded `decode_bypass_kv_slice_lds` as PRIMITIVE_MISSING for LDS-alloc UOp.
This audit checks whether that classification is correct.

## Correction: PRIMITIVE_MISSING was wrong

EB5's claim that LDS-alloc is PRIMITIVE_MISSING is **incorrect**. Searching the codebase:

```
grep -rn "DEFINE_LOCAL\|AddrSpace.LOCAL" tinygrad/ extra/qk_flash_decode.py
```

Finds:
- `Ops.DEFINE_LOCAL` in `tinygrad/uop/__init__.py:23` — the UOp exists
- `AddrSpace.LOCAL` with AMD lowering in `tinygrad/renderer/cstyle.py`, `tinygrad/renderer/amd/`
- **Already used in production**: `extra/qk_flash_decode.py:214,253` — K staging in
  `flash_pall_lds_crosslane_fdot2_score_whole_cache_kernel`

The K-staging pattern already in this file:
```python
klds = UOp.placeholder((Hd,), dtypes.half, 174, addrspace=AddrSpace.LOCAL)
kstage = klds[e].store(cache[...].cast(dtypes.half), in_r).end(r)
bar = UOp.barrier(UOp.group(kstage))
# ... use klds.after(bar)[...] in compute
```

`Ops.BARRIER` lowering to `__builtin_amdgcn_fence + __builtin_amdgcn_s_barrier` is also confirmed
in `tinygrad/renderer/cstyle.py:370`.

**LDS primitives: REACHABLE_NOW.** The blocker is structural, not primitive-missing.

## Three angles for V staging

### Angle 1 — Cross-kernel LDS staging

**GRAPH_LIFETIME_BLOCKED.**

AMD HIP LDS (Local Data Share) is workgroup-scoped. It is allocated per-CU per-kernel and does not
persist across kernel launches. E_49152_32_3 and flash_partial_coop_vec are separate kernel launches;
there is no mechanism to pass LDS data between them. This is an AMD hardware constraint, not a tinygrad
limitation.

### Angle 2 — Per-token intra-kernel V staging (inside the j loop)

**REACHABLE_NOW but NOT_BENEFICIAL.**

The K-staging pattern stages K[kv, t, :] into LDS per j iteration. Applying the same pattern to V:
```python
vlds = UOp.placeholder((Hd,), dtypes.half, X, addrspace=AddrSpace.LOCAL)
vstage = vlds[d_safe].store(cache[((1*Hkv+kvh)*MAXC+t_safe)*Hd+d_safe].cast(dtypes.half)).end(???)
bar = UOp.barrier(UOp.group(vstage))
```

K-staging is beneficial because q.k requires ALL Hd K elements accessed by ALL 32 lanes (cross-lane
reduction). Each lane reads R = Hd/32 elements; LDS staging enables all lanes to share the full Hd K row.

V access in flash_partial is fundamentally different: thread d reads V[kv, t, d] — **one element per
thread per token**. There is no cross-thread sharing. Staging into LDS adds a write + barrier for zero
read reuse. Not worth implementing.

### Angle 3 — Full-split intra-kernel V staging (prologue + main reduce)

**EMITTER_BLOCKED.** This is the only viable angle. Classification update: `PRIMITIVE_MISSING → EMITTER_BLOCKED`.

**What would work:** Stage all L=128 tokens of V (one split) into a 32KB LDS buffer before the main j
REDUCE loop begins. Structure:

```
Phase 1 (prologue): All W=129 threads cooperatively load V[kv, s*L..s*L+L, :] into LDS
                    Each thread loads ceil(L*Hd/W) ≈ 127 elements via an inner loop
                    Barrier after prologue completes
Phase 2 (reduce):  j reduce loop reads vlds[j * Hd + d] instead of global cache[...]
```

This eliminates E_49152_32_3 (6.69% GPU time) by providing equivalent or better data locality
(LDS is on-chip; E_49152's L2 warming is slower).

**Why it's EMITTER_BLOCKED:** The current UOp kernel DSL has no "prologue range" primitive. A REDUCE range
over j drives the accumulation loop. A pre-stage requires a SEPARATE loop over j_load (same dimension, same
count) that runs BEFORE the main j REDUCE, with a barrier between. tinygrad's scheduler does not support
two independent sequential loops sharing a REDUCE dimension feeding an intermediate buffer, followed by
the main accumulation. The lifecycle kernel (`flash_pall_score_state_pv_lifecycle`) also reads V directly
from global (no LDS staging), confirming this pattern is currently unexpressed.

**LDS sizing feasibility:**
- Per workgroup: L=128 × Hd=128 × 2 bytes = 32KB
- gfx1100 LDS per CU: 64KB
- Max occupancy with 32KB LDS: 2 workgroups/CU (vs. 4+ without)
- But: eliminating E_49152_32_3 (6.69%) and cold V global reads likely outweighs occupancy reduction

## Summary

| angle | verdict | reason |
|-------|---------|--------|
| Cross-kernel LDS staging | GRAPH_LIFETIME_BLOCKED | LDS doesn't persist across kernel launches on AMD HIP |
| Per-token V staging (inside j loop) | NOT_BENEFICIAL | No data reuse; adds overhead |
| Full-split V staging (prologue + reduce) | EMITTER_BLOCKED | Needs "prologue range" UOp before main REDUCE |

**Overall LDS0 verdict: EMITTER_BLOCKED**

EB5's PRIMITIVE_MISSING claim is corrected. The primitives (DEFINE_LOCAL, barrier) exist and are already
used in qk_flash_decode.py. The missing piece is a structural UOp kernel pattern: a pre-stage loop
before the main accumulation reduce.

## Reopen condition (precise)

A "prologue range" UOp primitive that allows an indexed pre-stage loop before the main REDUCE, writing
a tensor tile into LDS, with an implicit post-stage barrier, in the same kernel scope. Once expressible:
1. Modify `flash_partial_coop_vec_whole_cache_kernel` (or a new variant) to add the V prologue stage
2. Replace the j-loop global V read with an LDS V read
3. Remove E_49152_32_3 from the graph (the cooperatively-loaded LDS provides better warming than L2)
4. Gate behind `DECODE_FLASH_V_LDS_STAGE=0`
5. Verify correctness (rel_rmse ≈ 0, same token sequence)
6. W==D: projected upper bound 6.69% at ctx512 (E_49152_32_3 eliminated)
