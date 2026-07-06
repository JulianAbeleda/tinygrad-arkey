# Phase-0 Grid-Wide Barrier — Draft v2 (post-Fable-review)

Date: 2026-07-06
Status: **NEEDS-REVISION → revised.** v1 was reviewed by Fable 5 (verdict: right approach, wrong on two
engineering specifics + one strategic gap). This v2 folds those in. Still design-only — not built.
Subordinate to `docs/generated-megakernel-decode-scope-20260706.md` (Phase 0, primitive #1) and
`docs/pure-machine-search.md` (generated-not-handwritten contract).

## TL;DR of what the review changed

- **NOT impossible.** Fable: "the right long-term primitive." Every finding is a fixable spec/staging
  bug, not a wall. The architecture (renderer primitive, monotonic counter, static occupancy on Navi31,
  generated-not-handwritten) all survived.
- **Blocker fixed:** the cache-coherence sequence was wrong (waited `vmcnt` where RDNA3 needs `vscnt`;
  put the invalidate on the counter side, not before the payload read). Corrected in §Coherence.
- **Prerequisite surfaced:** the backend has **no vscnt store-drain at all** (`_insert_waitcnt`,
  `amd.py:747-792`). That infra must be added *before* the barrier is buildable. Now Step 0-infra.
- **Staging fixed:** labels/branches are post-regalloc, `ctx.vreg` is pre-regalloc — one isel rule can't
  span both. Restructured as a two-stage **distinct `Ops.GRID_BARRIER`** (isel-tag → post-regalloc-lower),
  not an overloaded `Ops.BARRIER` (which `amd.py:606` would silently flatten to a workgroup barrier).
- **Strategic gate added (most important):** measure the HCQ-graph decode baseline and a within-layer-only
  (workgroup-barrier) fusion FIRST. The grid barrier only earns its reboot risk if it's on the decode
  critical path — and decode is weight-HBM-bandwidth-bound, which a megakernel does not reduce.

## What this is

The one piece the megakernel needs that has **zero precedent** in the codebase: a **device-scope
(grid-wide) barrier** so layer N+1 sees all of layer N's writes across every workgroup. Everything else
in the megakernel is additive composition of existing UOp op-emitters (verified: decode emitters emit
body-as-`.sink()`, grid/prologue synthesized downstream). The barrier is the only primitive with real
correctness risk and the only one that, if wrong, **wedges the MES ring → reboot**.

## Step 0 (STRATEGIC GATE) — measure before you reboot-risk

Do this before writing any barrier code. Fable's biggest hit: v1 never quantified the win.

- Decode per-token cost is dominated by **weight HBM streaming** (GBs/token) — a megakernel does **not**
  reduce this. It only removes launch overhead, pipeline bubbles, and *activation* round-trips
  (activations are KBs/token, negligible vs weights).
- Launch overhead is **already amortized**: the HCQ graph (`runtime/graph/hcq.py`) replays the whole
  decode as one submit — there is no per-op CPU round-trip today.
- Therefore the grid barrier only earns its keep fusing *across* global-reduction boundaries
  (attention/FFN), and the weight-bandwidth floor caps even that gain.

**Gate:** measure (a) current HCQ-graph decode latency, and (b) a within-layer-only fusion that uses
only the existing **workgroup `s_barrier`** (no grid barrier, no persistent grid, no reboot risk). Only if
a device-scope boundary is demonstrably on the critical path after (b) do we proceed to build the grid
barrier. The cheaper 80% (norm/elementwise/within-layer chains) needs no new primitive at all.

## The "no handwritten kernel" line — where this sits (PASS, confirmed by review)

Per the scope doc: *authoring a spec + emitter + primitive is allowed; authoring a kernel body is
forbidden.* Adding an `Ops`/`AMDOps` enum member + isel/post-regalloc rules keyed on a UOp op is squarely
compiler infra — same category as `isel_special`/`isel_index`/`lower_range` (`amd.py:190,211,579`).
Instructions come out of a matcher rule keyed on an op, **not** a route-local function returning
`list[Inst]` next to a route (the forbidden `build_gemm_*`/`wmma.py` style).

- **Watch-item (from review):** the throwaway probe uses `Tensor.custom_kernel(fxn=...)` composing only
  UOp-space methods (`.grid_barrier`, `.set`, `.range`) — compliant, but confirm `pure_kernel_surface_audit`
  whitelists a `custom_kernel` whose `fxn` only calls UOp methods, or it will false-positive.

## Prerequisite infra (Step 0-infra) — RDNA3 vscnt store-drain

The barrier's correctness depends on draining prior **stores** to L2, which RDNA3 tracks in a separate
**VS_CNT**. The backend today models only vmcnt/lgkm/expcnt: `_insert_waitcnt` (`amd.py:747-792`) always
emits `s_waitcnt(simm16=0)` and tracks `global_store` as `vm_store` drained by that same op (line 790) —
there is **no `s_waitcnt_vscnt` emission anywhere**. The encoder helper *does* exist
(`s_waitcnt_vscnt`, `ins.py:1126`); the waitcnt *pass* has to learn to emit it.

**Task:** teach `_insert_waitcnt` to emit `s_waitcnt_vscnt 0` at store→(atomic/release) boundaries; prove
every existing kernel still passes (this is a latent gap the barrier merely exposes, not barrier-specific).

## Coherence — the corrected release/acquire (was the blocker)

The barrier must make the **payload** writes (probe: `scratch[g]`; real megakernel: a whole layer's HBM
activations) visible cross-CU — not just the counter. gfx11 agent-scope fences, matching LLVM:

```
# ---- RELEASE (producer side, BEFORE the arrive) ----
s_waitcnt_vscnt 0                    # drain payload STORES to L2  (NOT vmcnt — vmcnt is loads)
if wi_id == 0:
    global_atomic_add_u32 [counter], 1, glc   # arrival; glc => device scope
    s_waitcnt vmcnt(0) / vscnt 0     # the atomic itself completed
s_barrier                            # whole wg reaches the wait together

# ---- WAIT (all lanes spin on the counter) ----
top:
    global_load_b32 v_seen, [counter]
    s_waitcnt vmcnt(0)
    buffer_gl0_inv                   # re-fetch the counter each iter (weak model)
    s_cmp_lt_u32 v_seen, expected    # SCC = seen < (phase+1)*grid_size
    s_cbranch_scc0 top               # spin while less-than
    # (bounded-spin guard, see below)

# ---- ACQUIRE (consumer side, AFTER counter satisfied, BEFORE any payload read) ----
buffer_gl0_inv ; buffer_gl1_inv      # invalidate so the payload load hits L2, not a stale L0/L1 line
s_barrier                            # resume wg in lockstep
# ...only now may the body issue payload loads (scratch[neighbor], next-layer activations)
```

Two hard requirements the review pinned:

1. **vscnt, not vmcnt, for the release** — `vmcnt` waits on loads; store-to-L2 completion is `vscnt`.
   (Depends on Step 0-infra.)
2. **The acquire invalidate must DOMINATE the payload read.** The gl0/gl1-inv belongs *between* barrier
   exit and the payload load — not on the counter side. And the scheduler must not hoist a payload load
   above it: `buffer_gl*_inv` is a MUBUF op **not** in the scheduler's `MEM` tuple (`amd.py:688`), so add
   it there (or attach an explicit waitcnt/sched edge) or the acquire can be reordered and corrupt data.

Proof obligation: the permutation probe **with the negative control** on gfx1100 is the only real gate
for this (see §Test). If (1) or (2) is wrong, the counter is ordered and the data is stale — silent
corruption, exactly the megakernel's worst failure mode.

## Semantics — monotonic counter + phase target (correct, per review)

No mid-kernel reset. The counter increases monotonically for the whole kernel; barrier `p` waits for
`counter >= (p+1) * grid_size`. Each WG contributes +1 per barrier. A fast WG that laps only makes the
counter *larger*, so a slow WG's `seen >= (p+1)*grid_size` still holds — sound across N barriers (Fable
tried to break it; couldn't).

**HARD INVARIANT (not an aside): per-decode-step re-zero by the launcher.** Increments/token ≈ grid_size
(~1e3) × barriers/token (~1e2) ≈ 1e5–1e6, vs u32 max 4.3e9 — one step is fine, but across many tokens it
overflows without re-zero. The launcher zeroes the 1-elem u32 counter between decode-step launches (kernel
idle → no reset race). This is required, not optional.

## Emit — two-stage `Ops.GRID_BARRIER` (was staged wrong in v1)

v1's "one ~30-line isel rule" cannot work: labels/branches are emitted **post-regalloc**
(`lower_range`, `amd.py:603`); `ctx.vreg` (for the `seen` reg) is **pre-regalloc**. Mirror how RANGE is
done — two stages — and use a **distinct op**, because `post_regalloc_matcher` at `amd.py:606` lowers
*every* `Ops.BARRIER` unconditionally to a single `s_barrier` (an overloaded grid-BARRIER would be
silently flattened, dropping the spin).

- **New op:** `Ops.GRID_BARRIER` (distinct from `Ops.BARRIER`).
- **UOp constructor** (`tinygrad/uop/ops.py`, next to `.barrier()`):
  ```python
  def grid_barrier(self, counter:UOp, grid_size:int, phase:int, **kw):
      # self = dependency sink (writes that must be globally visible first)
      # counter = INDEX into a 1-elem u32 HBM scratch, per-step-zeroed by the launcher
      return UOp(Ops.GRID_BARRIER, dtypes.void, (self, counter), arg=(grid_size, phase), **kw)
  ```
- **Stage A — isel-tag** (pre-regalloc): reserve the spin VGPR + a **distinct** reserved exec SGPR
  (NOT `_S[5]` — that's the gated-store exec save at `amd.py:560`; reusing it collides if a barrier ever
  nests in a gated-store scope), resolve the counter address via the existing `isel_index` path, and stamp
  a unique label id via `_next_loop_label` (`amd.py:126`). Emit a `GRID_BARRIER`-tagged marker carrying
  {reserved_vgpr, reserved_sgpr, counter_addr, label_id, grid_size, phase}.
- **Stage B — post-regalloc lower** (a rule in `post_regalloc_matcher` beside `lower_range`): expand the
  marker into the release/spin/acquire `Ops.INS` sequence above, using `_label`/`_branch` (`amd.py:577`)
  for the spin, the reserved regs (hard-coded — regalloc has already run), and the existing exec
  save/restore pattern (`s_and_saveexec_b32`/`s_mov_b32 EXEC`, `amd.py:559-563`) with the *distinct*
  reserved SGPR.

Encoders needed — all present (review-confirmed): `global_atomic_add_u32` (`ins.py:659`, with `glc`
bitfield `ins.py:38/43`), `buffer_gl0_inv`/`buffer_gl1_inv` (`ins.py:830-831`), `s_cmp_lt_u32`
(`ins.py:1067`), `s_waitcnt_vscnt` (`ins.py:1126`), `s_barrier`, `s_and_saveexec_b32`. Add one `AMDOps`
member for the atomic; the rest are import + wire.

## Occupancy invariant — statically knowable on Navi31 (per review)

gfx1100 = fixed 48 CUs; co-residency is computable from post-regalloc per-WG VGPR/LDS/waveslot budget.
`grid_size = num_CU × max_coresident_wg_per_CU`. **Must UNDER-subscribe** — any co-tenant (display/compute)
steals CUs, and a single non-resident WG never arrives ⇒ permanent deadlock ⇒ reboot. Assert at emit time,
using the *actual post-regalloc* VGPR/LDS numbers (which is another reason the reg-sensitive parts live in
Stage B, not isel).

## Reboot-risk mitigation — bounded spin

A logic bug in the spin condition must surface as wrong output, not an infinite spin (MES-ring wedge →
reboot). Cap the spin at a large constant (e.g. 1<<24) via a second scalar counter + `s_cmp`; on timeout
fall through to the acquire (leaving a sentinel in `out` so the harness sees "timeout" vs "wrong data").
One `s_add`/`s_cmp` per iter — effectively free. Ship the megakernel with this bound.

## Phase-0 test (generated, not a hand kernel)

Permutation cross-read that ONLY passes if the barrier globally orders writes:

```python
# extra/qk/grid_barrier_probe.py  (role: scratch; deleted or promoted after Phase 0)
def emit_barrier_probe(n):
    def kernel(out, scratch, counter):
        g = UOp.range(n, 0, axis_type=AxisType.GLOBAL)     # one persistent workgroup per gid
        w = scratch[g].set(g)                              # payload write
        w = w.grid_barrier(counter[0], grid_size=n, phase=0)
        r = scratch[(g + 1) % n]                           # cross-wg read: neighbor's payload
        return out[g].set(r.after(w)).end(g).sink(arg=KernelInfo(name=f"grid_bar_probe_{n}"))
    return kernel
# driver: Tensor.custom_kernel(out, scratch, counter, fxn=emit_barrier_probe(n))[0].realize()
# assert out == [(i+1)%n for i in range(n)]
```

Gates (per `extra/qk/README.md` one-rule → a `GateSpec` in `gate_registry.py`, `needs_gpu=True`):

1. **Permutation passes on gfx1100** at n ∈ {2, 8, 32, grid_max}. THE test — broken barrier → stale/garbage.
2. **Negative control**: same kernel, barrier removed → must FAIL on gfx1100 (proves the test exercises
   cross-wg ordering, isn't trivially green).
3. **Repeated barriers** (phase 0,1,2 in one kernel) pass → proves monotonic/phase scheme, no reset race.
4. **DEV=PYTHON first** (proves emit/graph/indexing; can't wedge — necessary but NOT sufficient, since
   PYTHON runs WGs serially and a broken barrier still "passes"), THEN gfx1100 once occupancy-assert +
   bounded-spin are in.

## Revised build order

0. **(strategic)** Measure HCQ-graph decode baseline + within-layer workgroup-barrier fusion. Proceed only
   if a device-scope boundary is on the critical path.
0-infra. Add RDNA3 `vscnt` store-drain to `_insert_waitcnt`; prove existing kernels pass.
1. Add `global_atomic_add_u32` to `AMDOps` + wiring; unit-test encoding bytes vs known-good disasm (no GPU).
2. Add `Ops.GRID_BARRIER` + `UOp.grid_barrier` + Stage-A isel-tag + Stage-B post-regalloc lower; add
   `buffer_gl*_inv` to the scheduler `MEM` set (`amd.py:688`).
3. Emit the probe; run **DEV=PYTHON** — emit lowers, permutation math right.
4. Add occupancy assert (post-regalloc VGPR/LDS) + bounded spin. Run **DEV=AMD** n∈{2,8,32}: permutation
   passes, negative control fails, repeated-barrier passes.
5. Register the gate; `pure_kernel_surface_audit` classifies generated/primitive; confirm the UOp-only
   `custom_kernel` whitelist (no route-local `list[Inst]` barrier body).

Done (Phase 0): grid barrier proven on gfx1100, emitted by the substrate, coherence-correct
(vscnt-release + inv-acquire dominating payload reads), zero op-math duplicated, no handwritten kernel —
**and** demonstrated to be on the decode critical path before it was built.

## Review trail

- v1 verdict (Fable 5): NEEDS-REVISION. Blocker: coherence used vmcnt where RDNA3 needs vscnt + inv on
  wrong side (silent corruption). Major: emit staged across pre/post-regalloc; `Ops.BARRIER` overload
  flattened at `amd.py:606`; invented `_leader_predicate`/`_restore_exec` helpers + `_S[5]` collision.
  Strategic: no measured baseline proving the barrier is on the critical path. All folded into v2 above.
