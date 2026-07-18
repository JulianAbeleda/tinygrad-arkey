# Qwen3-14B generated-prefill Claude handoff

## 0.5 Current audit status 2026-07-17 (head `7b863aaec`)

Read this section before the historical status sections below. The repository head is now
`7b863aaec` (`[test][qk] honor ProgramInfo global buffer indices`), twelve commits ahead of
`origin/master`; the worktree is clean. This includes the allocator lease fix at `23eaf693b` and the new
fail-closed harness under `extra/qk/mmq_llama_five_buffer_gpu_harness.py`.

The corrected phase-major graph and allocator lease regression coverage are structurally healthy:

- `PYTHONPATH=. PYTHONHASHSEED=0 pytest -q test/unit/test_mmq_llama_five_buffer_full_kernel.py`: **13 passed**.
- The focused rematerialization/end-liveness/span suites: **23 passed**.
- `test/unit/test_amd_isa_wmma.py`: **36 passed, 4 failed**. The four 16-subtile multi-output failures are the
  same spill-free-gate failures reproduced on the pre-change `423f6ff83` baseline; they are historical and not a
  regression from `23eaf693b`.
- The new fail-closed GPU harness tests: **3 passed**. A matched-environment run reached a real AMD dispatch but
  synchronization reported `MMU fault: 0x7F8827C51000 | NotPresent=1`; no numerical verdict was captured. The harness
  preserves this structured blocker rather than treating it as a pass.

The full K=256 `to_program` path now **emits successfully** under the matched environment
(`PYTHONHASHSEED=0 REGALLOC_ADDR_REMAT=1 REGALLOC_END_NO_SOURCE_LIVE=1`). Graph/codegen selection reports
`10,101` UOps and a pre-allocation peak of `312` live virtuals at UOp `5938`; the full allocator run then reports
`REGALLOC_SPILLS: count=0 stack_size=0`, and `compile_llama_five_buffer_full_kernel` returns
`emitted=True program=True blocker=''`. The CPU-only compile takes several minutes in the long register-selection
pass, so this is a valid zero-spill emission result but not a GPU correctness or timing result. The `23eaf693b`
change reserves the serialized high A/B fragment lease and has a focused unit test; the end-to-end run above is the
evidence for the full-grid gate.

K=512 exact FP32 state-carry and phase-major lifecycle tests pass structurally. Its exact compile fails before
regalloc in `_register_stage_leases`: 128 independent C roots request low-accumulator leases `[v8, v1032)`, over
the 256-VGPR file, because epoch-aware A/B fragment provenance is not yet proven. K=256 has since been launched on
the AMD device, but synchronization reported an MMU `NotPresent=1` fault before output comparison. No shape has been
numerically compared with llama or performance-measured; the dispatch fault is the next acceptance gate.

Date: 2026-07-16  
Repository: `/home/ubuntu/tinygrad-arkey`  
Branch: `master`  
Handoff revision: `412d7998f` (`[amd] recover propagated progressive C roots`)  
Remote state at handoff: `master == origin/master`, clean worktree

## 0. Status update 2026-07-17: bounded emission gate CLOSED

The §15 assignment (spill-free bounded emission) is complete and pushed. Sections 1, 4 and 7 below describe the
pre-`5982d83c1` state and are retained for history; read this section first.

Accepted commits, oldest to newest:

- `8a8ffc6ae [amd] select native fp16 multiply for half2 metadata`
- `00a79b247 [codegen] fail fast on ops that reach regalloc unselected`
- `5982d83c1 [amd] guard chain-head A/B loads on the previous chain release`

The 112 spills were **two** stacked blockers, which is why the single-blocker framing in §7 misleads:

1. **A/B chain-head lifetime.** A multi-output-tile kernel has one chain per subtile and every chain reloads the same
   physical high A/B pair (`ab_key = "wmma_ab"`, `amd.py`). Accumulate tiles WAR-guard that shared run
   (`dep = (prev.src[0],)`); a chain head has no prior tile and takes only `_wmma_carrier_order_deps(tile.src[2])`, so
   eight heads opened simultaneously onto one capacity-1 run. `_serialize_progressive_c_drains` already ordered each
   head WMMA after the previous chain's release frontier, but the head's `DS_LOAD_B128` operands carried no such edge
   and floated ahead of it: the ordering edge was on the wrong node. Guarding the loads closed it.
2. **Missing typed half-MUL selection.** isel selected `MUL` for float32 and ints only, so the half2 metadata
   recurrence's genuine fp16 multiply survived unselected. Its children lowered to raw-bit machine ops, leaving a
   mixed `ushort x half` node. Post-isel this *looks* like malformed oracle algebra; it is not. Both source graphs are
   clean (bounded 3,239 nodes / 346 MULs, full-grid 7,802 / 966, zero mixed-dtype). Do not "fix" it upstream.

Exact bounded oracle, matched environment (`PYTHONHASHSEED=0 REGALLOC_ADDR_REMAT=1 REGALLOC_END_NO_SOURCE_LIVE=1`):

| metric | at `412d7998f` | at `5982d83c1` |
|---|---:|---:|
| spill requests | 112 (`v0` at 2580) | **0** |
| stack | 448 B | **0** |
| emitted instructions | none (fail-closed) | **8,287, all encodable** |
| `v_wmma_i32_16x16x16_iu8` | 128 | **128** |
| `ds_load_b128` | 256 | **256** |
| `v_mul_f16_e32` | 0 (unselected) | **12** |
| post-selection UOps / peak live | 9,481 / 158 @ 5908 | **9,481 / 158 @ 5908** |

No scratch or buffer traffic. The bounded probe still sets no full-grid claim.

Two traps recorded so they are not re-entered:

- **`regalloc_rewrite` indexes `ctx.uops` BY POSITION** via `i = next(ctx.idx)`. Any op with no ISA rule is never
  visited, shifts every later index by one, and surfaces as a `KeyError` on an unrelated vreg — the allocator blaming
  itself for a backend gap. `00a79b247` makes this fail fast and name the op. If you see a mystery `KeyError` in
  regalloc, suspect selection first.
- **gfx11 VOP f16 literals live in the low 16 bits.** `_vop2_f`'s `float()` path encodes an fp32 pattern
  (`0.3f -> -0.0027h`). Inline constants survive it and everything still encodes, so "the program encodes" does not
  prove literal correctness. `_vop2_h` owns the fp16 path.

### Next owning problem: full-grid is NOT gated on spills

**Superseded by §0.1: both SPEC gates are now closed and full-grid reaches regalloc. The GEP/AFTER diagnosis below is
wrong about the verifier and is retained only as history.** The spill gate was only one of the listed full-grid
blockers. §13.1 was still closed at that point, and its remaining blockers were
graph/spec-shaped: `compile_llama_five_buffer_full_kernel(build_llama_five_buffer_full_kernel(128,128,256))` fails
**before** register allocation, in SPEC type verification. Verify each `BLOCKER` line independently rather than
assuming the resource one was the last.

Two of these have now been walked. Reproduce with:

```bash
env PYTHONHASHSEED=0 REGALLOC_ADDR_REMAT=1 REGALLOC_END_NO_SOURCE_LIVE=1 python3 -c "
from extra.qk.mmq_llama_five_buffer_full_kernel import (build_llama_five_buffer_full_kernel,
                                                        compile_llama_five_buffer_full_kernel)
compile_llama_five_buffer_full_kernel(build_llama_five_buffer_full_kernel(128,128,256))"
```

**Cleared at `e5efb9707`** — `AFTER(ADD, STORE)`. Do not re-diagnose this:

```text
UOp verification failed at 3842 on Ops.AFTER dtypes.float 2
  [(Ops.ADD, dtypes.float, None), (Ops.STORE, dtypes.void, None)]
```

The source sink is legal (7,802 nodes; `AFTER` src0 is only `DEFINE_LOCAL`/`INDEX`/`BITCAST`/`STACK`; zero
`AFTER(ADD, STORE)`). The illegal node is manufactured *during codegen rewriting*: a bare `STACK(float.vec(8))` is
spec-legal as written but expands away, dropping its effect order onto the scalar FP32 update. Fix was the no-op
typed BITCAST carrier the bounded release already documents. **Lesson: the source graph being clean does not mean the
verified graph is; check post-rewrite before blaming the oracle.**

**Current exact victim** — a different verifier blocker, further down:

```text
UOp verification failed at 3327 on Ops.GEP dtypes.float 1
  [(Ops.BITCAST, dtypes.float.vec(8), None)] (0,)
```

Known so far, so it is not re-derived:

- That node **passes `spec_program` in isolation** (`GEP(BITCAST(STACK(float.vec(8))))`, arg `(0,)` -> `True`). So the
  real node differs in a field `type_verify`'s message does not print. `shape` is the first suspect: `spec.py:244`
  (`False if x.dtype.count > 1 and (x.dtype.count,) != x.shape`) is shape-sensitive.
- Note that rule would fire on the vec(8) `BITCAST`, not the `GEP`, so the reported node may not be the guilty one.
- Method: build the sink, run `full_rewrite_to_sink` with `SPEC=0`, walk the real post-rewrite nodes, and diff the
  failing node against the synthetic one that passes. Do not reason from the error string alone.
- **Do not widen `spec_shared`.** It is shared across backends, and the bounded probe proves the graph-side idiom
  works. Prefer a graph fix at the producer.

Unchanged and still true: the four 16-subtile AMD tests, two `test_current_prefill_execution_adapter` rows, and
`test_q4k_wmma_tiled_no_hand_scan_is_clean` fail identically at `412d7998f` and at `5982d83c1`. They are historical,
not regressions. Do not xfail them.

## 0.1 Status update 2026-07-17 (later): both SPEC gates CLOSED, full-grid reaches regalloc

Accepted and pushed: `4b153b8e9 [qk][mmq] order full-grid stores through the pointer, not a no-op value bitcast`.

The §13.1 verifier blocker is closed. It was **one** defect seen through two different verifiers, not two:

- `_full_grid_sink` built `value = UOp(Ops.BITCAST, store.src[1].dtype, (store.src[1],))` — a BITCAST to the dtype it
  already has. Codegen folds that no-op away and the `.after()` it carried lands on the scalar FP32 update, so
  `spec_program` rejected `AFTER(ADD, STORE)` at `codegen/__init__.py:196`.
- `e5efb9707` read that as a missing movement carrier and wrapped the accumulator vectors in a typed BITCAST. That
  stopped the AFTER only by keeping a `vec(8)` alive across the writeback, which blocks the `GEP(STACK, i) -> src[i]`
  fold and left 64 `GEP(BITCAST(...))` in the **source** sink — rejected by `spec_tensor` at `codegen/__init__.py:77`.

**The two error strings came from two different `type_verify` call sites.** `spec_tensor` has no general GEP rule (only
GEP over `WMMA`/`SHAPED_WMMA`/`LOAD`, `spec.py:131,143,147`); the general rule at `spec.py:250` is `spec_program`-only.
The handoff's "that node passes `spec_program` in isolation" was true and irrelevant — it was never checked against
`spec_tensor`. **Always confirm WHICH verifier failed before diagnosing the node.** `shape` was never involved.

Fix: revert the accumulator carrier and drop the value-side edge. The pointer's `INDEX` is a real movement value and
already totally orders the 64 stores. No arithmetic, rounding, or ABI change.

### Current exact blocker: full-grid register pressure (genuinely resource-shaped)

Matched env (`PYTHONHASHSEED=0 REGALLOC_ADDR_REMAT=1 REGALLOC_END_NO_SOURCE_LIVE=1` + `REGALLOC_DEBUG`/`_PRESSURE`/`_SPILLS`):

```text
REGALLOC_DEBUG: 13547 uops, PEAK 452 live vregs @ uop 5147, pool=255
REGALLOC_PRESSURE: spill_request=v0 at=3014 pool=255
REGALLOC_SPILLS: count=96 stack_size=381
```

Spill victims: 80 `V_CVT_I2F`, 11 `V_MUL`, 4 `V_ADD`, 1 `V_CMPLT_I`. Peak composition: 184 `V_CVT_I2F`, 129 `V_IMUL`,
64 `GLOBAL_LOAD`, then a thin tail. Bounded oracle is unregressed at `4b153b8e9`: 9,481 UOps, peak 158 @ 5908, 0 spills,
0 stack.

The decisive measurement — **the full-grid math is identical to the bounded probe**:

| metric | bounded | full-grid (128,128,256) |
|---|---:|---:|
| `v_wmma_i32_16x16x16_iu8` | 128 | 128 |
| `ds_load_b128` | 256 | 256 |
| total UOps | 9,481 | 13,547 |
| peak live vregs | **158** | **452** |
| `V_CVT_I2F` live at peak | **8** | **184** |

Same WMMA and LDS counts, same 512 total `V_CVT_I2F`. Full-grid adds 4,066 UOps and 64 `GLOBAL_STORE` (at 11,568..13,543).
So the extra 294 live virtuals are **not** new math — they come from the writeback seam.

All 184 drains live at the peak are defined in UOps 2,841..5,321 and every one is consumed by a `V_MUL` at 5,930..8,935.
The `V_MUL`/`V_ADD` pairs there are the exact serial recurrence (`previous + scale*C`), consumed one term at a time, so
each drain is held for thousands of UOps waiting its turn. WMMAs are not all hoisted first (they span 2,818..13,089) —
the pile-up is local to that region.

**REFUTED at `8124ddaa9` — see §0.2. The real cause was a missing ordering edge in the full-grid producer, and
`_accumulator_vectors` is not a contributor at all. Retained only as history; do not revive.** Leading hypothesis: `_accumulator_vectors` transposes eight scalar WMMA
lanes across eight oracle subtiles (vector `e` = `STACK(lane_0..lane_7)` with `subtile -> e`). That transposition
requires all eight subtiles' drains co-live to assemble any one vector, whereas the bounded probe drains each subtile
independently and keeps only 8 `V_CVT_I2F` live. Note 184 != 64, so the transposition alone does not obviously account
for the whole peak — do not assume it is the only contributor.

Next probe: map each of the 184 peak-live `V_CVT_I2F` back to its subtile/lane and its consuming `V_MUL`, and check
whether the co-liveness is forced by the transposition or is a scheduling choice `pressure_schedule` could undo.

**Rails for this blocker:** the recurrence order and rounding are fixed authority — do not reassociate
`(previous + scale*C) + bias` or introduce FMA/MULACC to shorten the chain. Do not enable AMD scratch.

### Measurement caveat (new, important)

Full-grid spill/peak numbers are **environment-sensitive and load-sensitive**:

- with `REGALLOC_DEBUG_PRESSURE=1`: 452 @ 5147 / 96 spills / 381 B — reproduced 2/2.
- without it: 465 @ 5355 / 95 spills / 377 B — reproduced 3/3.
- one early run under concurrent load reported 444 @ 5098 / 85 spills. Same env as the 452 run, different result.

Each environment is self-consistent when the machine is otherwise idle, but the load-dependent outlier is unexplained
and was not chased. **Run the full-grid reproducer on an idle machine and compare only matched environments.** The
bounded oracle is deterministic (9,481 / 158 @ 5908 / 0) and is the trustworthy comparator.

## 0.2 Status update 2026-07-17 (later still): drain ordering applied, blocker moves to the writeback

Accepted and pushed: `8124ddaa9 [qk][mmq] order full-grid WMMAs behind the preceding lane drain`.

§0.1's hypothesis (`_accumulator_vectors` transposition) was **wrong and is now refuted twice over**. Do not revive it:

1. `_accumulator_vectors` groups by subtile, **not across** subtiles: `vector[e] = STACK([lane[l].substitute(subtile -> e)
   for l in range(8)])` substitutes the *same* `e` into every lane. Assembling one vector needs one subtile's eight
   **lanes**, never all eight subtiles. Measured: `vector[e]`'s weakint consts are `[e]` only.
2. It is structurally inert regardless. `build_wmma_writeback` immediately GEPs each lane back out and
   `GEP(STACK, i) -> src[i]` folds, so the STACK never reaches the machine graph. Measured: `GEP(vector[3], 5)` is
   already `Ops.ADD` and is *identical* to `lanes[5].substitute(subtile -> 3)`.

**The real cause was a missing ordering edge in the full-grid producer.** The oracle keeps the integer WMMA chain and
the eight FP32 lane chains as separate algebraic dependencies, so a legal schedule issues every WMMA before consuming
any C lane. `_bounded_accumulator_drain`'s docstring in `extra/qk/mmq_llama_full_kernel.py` has named this exact
failure all along — the bounded probe has carried the edge for a long time and the full-grid seam simply never got it.

Two facts that discriminated **forced dataflow** from **scheduling choice**, and are worth reusing as a method:

- `pressure_schedule` provably *could not* fix it: it is block-local (`regalloc.py:103-111` splits blocks at
  `BARRIER`/`RANGE`/`END`), the drains were defined in blocks `(2685,4217)`/`(5070,5474)` while their `V_MUL` consumers
  lived in blocks `(5915,6060)`..`(8845,8990)`, with `BARRIER`s between. Barrier indices are identical in its input and
  output — it permutes only *within* blocks and had no legal hoist available.
- But it was not dataflow-forced either, proved by construction: adding the edge changed nothing but the order.

The repair extracts `order_wmma_behind_lane_drain(epochs, tag_prefix)` from `_bounded_accumulator_drain` and applies it
in `_full_grid_sink` before `_accumulator_vectors` substitutes, so edges replicate into all eight subtiles.

| metric | before | after |
|---|---:|---:|
| `V_CVT_I2F` live at peak | 184 | **64** |
| peak live vregs | 465 | **372** |
| spills | 95 | **54** |
| stack | 377 B | **213 B** |

Selected math unchanged (128 `V_WMMA_I8`, 256 `DS_LOAD_B128`, 512 `V_CVT_I2F`, 640 `V_MUL`, 960 `V_ADD`, 64
`GLOBAL_STORE` — identical to pre-repair). Bounded oracle byte-identical (9,481 / 158 @ 5908 / 0 / 0). All 54 focused
tests pass. `test_full_grid_orders_each_wmma_behind_the_preceding_lane_drain` locks the invariant and was confirmed to
fail with the producer fix stashed.

### Current exact blocker: writeback address math and load lifetimes

54 spills still fail closed, so §13.1 stays open, and **the owning problem has moved**. Peak composition inverts:

```text
REGALLOC_DEBUG: 13659 uops, PEAK 372 live vregs @ uop 5049
  132  V_IMUL
   64  GLOBAL_LOAD
   64  V_ADD
   64  V_CVT_I2F
```

`V_IMUL` (132) + `GLOBAL_LOAD` (64) now dominate, not the drains. Spill victims are still mostly `V_CVT_I2F` (the
residual 64 = 8 subtiles x 8 lanes), plus one long-lived `V_CMPLT_I` (v315, range 563..13656 — a whole-program
predicate worth its own look).

Two candidate next moves, in order:

1. **The 132 `V_IMUL` and 64 `GLOBAL_LOAD` at the peak** are the writeback/grid address math. This is the largest
   contributor and is untouched by any repair so far. Start here; `a293391b0 [codegen] localize constrained leases at
   consumers` is the precedent for shrinking def-to-use distance on address carriers.
2. **The residual 64 drains are cross-subtile co-liveness** the per-subtile edge does not cover. Ordering subtile `e`'s
   WMMAs after subtile `e-1`'s updates needs edges injected *post*-substitution. **Unverified and risky:**
   `amd.py:1883-1886` explicitly warns that ordering on FP32 update nodes "can compose into a cycle". Extending
   `_serialize_progressive_c_drains` from the CVT release frontier to the update frontier is the natural candidate —
   note that serializer **does fire** on full-grid (128 machine WMMAs, 64 selected roots, 63 edges applied); it orders
   chain heads on the CVT frontier, which frees the C lease but does not force the FP32 consumer. It is insufficient,
   not absent. Do not re-diagnose it as a bail-out.

Rails unchanged: no reassociation of `(previous + scale*C) + bias`, no FMA/MULACC, no rounding-boundary move, no AMD
scratch, no `spec_shared` widening.

## 0.3 Status update 2026-07-17 (later still): cross-subtile serialization, 9 spills left

Accepted and pushed: `cd989ad0e [qk][mmq] serialize full-grid subtiles behind the preceding subtile's drains`.

§0.2's residual-64 note was right: the binding pressure after the intra-subtile fix was **cross-subtile** C-carrier
co-liveness. The eight subtiles share no integer WMMA node (the subtile index only touches the Q4/A operand), so a
legal schedule still issued all eight WMMA chains before consuming any subtile's drains — 8 subtiles x 8 lanes co-live.
`order_wmma_behind_lane_drain` runs *before* the subtile substitution and cannot see across subtiles.

Fix: `_accumulator_vectors` now takes the epoch-0/group-0 chain head and, per substituted element `e>0`, orders `e`'s
chain head behind element `e-1`'s eight drains. The chain head reaches every lane of its subtile (second WMMA takes
first as its accumulator input, every lane chains through `second.gep(i)`), so one edge serializes the whole subtile.
Strictly increasing element index => acyclic (verified by a cycle-detecting toposort in review).

| metric | before | after |
|---|---:|---:|
| `V_CVT_I2F` live at peak | 64 | **8** |
| peak live vregs | 372 | **273** |
| spills | 54 | **9** |
| stack | 213 B | **29 B** |

`V_CVT_I2F` at peak (8) now equals the bounded probe. Selected math unchanged (128 `V_WMMA_I8`, 256 `DS_LOAD_B128`,
512 `V_CVT_I2F`, 640 `V_MUL`, 960 `V_ADD`, 64 `GLOBAL_STORE`); op-count diff vs parent is only +21 `AFTER` (7x3) and
+9 `BITCAST`. Bounded oracle byte-identical (9,481 / 158 @ 5908 / 0 / 0). A Fable review audited math-neutrality, cycle
safety, per-subtile reach, and the substitution-target liveness — all SAFE with direct evidence.

**Method note that keeps paying off:** for these three full-grid repairs, the diagnosis that held was always (a) show the
would-be fixing layer *cannot legally act* — `pressure_schedule` is block-local (`regalloc.py:103-111`) — and (b) prove
non-forcedness *by construction* by adding the edge and measuring. Reason from the string and you get the 132-`V_IMUL`
red herring; the `V_IMUL` are fully rematerialized by `REGALLOC_ADDR_REMAT` and cost 0 spills.

### Current exact blocker: residual 9 spills, mixed mechanism

Matched env: `PEAK 273 @ 4384`, 9 spills / 29 B. The 9 victims are **3 `V_CVT_I2F`, 2 `GLOBAL_LOAD`, 2 `V_MUL`, 1
`V_ADD`, 1 `V_CMPLT_I`** — no longer a single class, so no single edge will close it. Peak composition is now `128
V_IMUL` (remat, 0 spills) + `64 GLOBAL_LOAD` + a thin tail; real physical demand is ~264-273 against a 255 pool, i.e.
only ~9-18 over. Candidate probes, unverified:

- **`v315` (`V_CMPLT_I`, range 563..13656)** is a whole-program integer predicate reused across every epoch/load, so it
  is live end to end and independent of the drains (it survived both drain fixes). It originates in the cooperative DS4
  panel-staging lane map, not the writeback. A recompute-at-uses localization (analogous to the address remat, cf.
  `a293391b0`) is the natural fix for this one register. Low individual payoff but clean.
- **The 2 `GLOBAL_LOAD` + 2 `V_MUL` + 1 `V_ADD`** sit near the peak at ~4384; inspect their ranges with
  `REGALLOC_DEBUG_DETAIL` and check whether a localization or a small ordering nudge frees them.
- **The 3 residual `V_CVT_I2F`** are the tail of the drain serialization; a milder cross-subtile edge (order behind
  `e-1`'s CVT frontier rather than its full accumulator — the `prior_drains` knob in `_accumulator_vectors`) might trade
  the last few without over-constraining. Try this only if the cheaper localizations above do not reach zero.

Because the peak is only ~9-18 over pool and the victims are mixed, the endgame is likely two or three small
localizations, not one more big ordering edge. Rails unchanged (no reassociation, FMA/MULACC, rounding move, scratch,
or `spec_shared` widening).

## 0.4 Status update 2026-07-17 (final for this session): 9 spills traced, one dead end burned

No new code commit — this section is diagnosis only. Head is `6f65ef31c`, worktree clean, 5 stashes intact.

### The 9 spills, traced to source (matched env)

Each spill victim's def / srcs / consumer / live-range, mapped to the oracle. **Matched env** — the same env the
9-spill baseline is defined in. (Do NOT trace under an added `SPILL_TRACE`-style debug flag: it perturbs allocator
traversal to ~16 spills and reweights the classes. An earlier trace did exactly this and mis-ranked the metadata as
dominant; the corrected matched-env ranking is below. **Lesson: trace in the SAME env you measure in.**)

| class | count | def <- srcs | consumer | span | what it is |
|---|---:|---|---|---|---|
| `V_CVT_I2F` | 3 | `<- V_WMMA_I8` | next chain `V_WMMA_I8` | 17-65 | residual WMMA drain tail |
| `V_MUL` | 2 | `<- V_CVT_H2F, V_CVT_H2F` | `V_MUL`/`V_ADD` | 63-236 | half2 `scale`/`bias` metadata (`recurrence.py:184-185`) |
| `GLOBAL_LOAD` | 2 | `<- V_OFFSET, S_LOAD_PTR` | `DS_STORE` | **~9600** | CSE'd Q8 panel, reused across all 8 subtiles' LDS stagings |
| `V_ADD` | 1 | `<- V_MUL, V_MUL` | `V_ADD` | 262 | recurrence accumulation |
| `V_CMPLT_I` | 1 (`v315`) | `<- V_AND, V_CONST, V_CONST` | `END` | ~13100 | whole-program lane predicate |

### REFUTED — do not retry: ordering the half2 metadata

The tempting "symmetric twin" move (order each group's `scale`/`bias` metadata behind the prior group's drain, like the
C-drain fix) **makes it dramatically worse**. Verified by construction (matched env, full-grid 128,128,256):

| variant | peak | spills |
|---|---:|---:|
| baseline `6f65ef31c` | 273 | **9** |
| metadata behind prior drain (new void barriers) | 282 | 27 |
| metadata behind existing release (barrier-free, minimal edge) | 267 | **74** |

There are 128 metadata muls (8 subtiles x 8 groups x 2). They are *schedulable* (the `dm`/`ds` loads depend only on the
phase `publish` barrier, not on drains, so `pressure_schedule` — block-local — cannot move them), but they contribute
only ~6 registers to the peak (273->267) while serializing 128 uniform loads into the tight drain windows explodes
spills 8x. The metadata is NOT the dominant class and NOT the lever. Cross it off.

### For the next owner (Codex): two honest options, and the recommendation

**The cheap ordering-edge move is exhausted and now fights itself.** The residual is dominated by the `V_CVT_I2F` drain
tail (3/9) and the two long-span Q8 `GLOBAL_LOAD`s (2/9) — and those loads are span ~9600 *because* the cross-subtile
serialization (`cd989ad0e`) pins the shared Q8 panel live across all 8 subtiles. More serialization lengthens that
span. So closing the last 9 is a genuine trade, not another `.after()`:

1. **Grind the spill gate to 0.** Candidate levers, each UNVERIFIED, each needs measure-by-construction:
   - The Q8 `GLOBAL_LOAD` (biggest single spans): decide whether the panel is re-staged per subtile (redundant DS_STOREs
     that could share) or genuinely 8 chunks. If shareable, stage once; if not, the tension with serialization is real
     and may need a re-load-vs-hold decision (registers vs LDS/mem traffic). This is a design call, not a mechanical edge.
   - `v315`: recompute-at-uses localization (analogous to address remat, cf. `a293391b0`) — one clean register.
   - The 3 `V_CVT_I2F`: milder cross-subtile edge via the `prior_drains` knob in `_accumulator_vectors` (order behind
     `e-1`'s CVT frontier, not its full accumulator). Risk: over-constraint blew up the metadata; measure every step.
   - Peak is only ~9-18 over a 255 pool, so 2-3 small wins plausibly reach 0 — but none are proven and the Q8 one is a
     design decision.
2. **Stand up the GPU correctness harness FIRST, before grinding to 0.** This is the RECOMMENDED path. Rationale below.

**Why correctness-first is the better use of effort.** Everything to date is CPU-side compile / register allocation
(`DEV=PYTHON`, no execution). Zero output has ever been checked against the frozen llama comparator. Closing the spill
gate proves the kernel *compiles*, not that it is *correct* or *fast* — those are separate, larger, unmeasured gates
(§13.1 correctness, §13.2 performance = the actual promotion authority). The dominant uncertainty in any ETA is what the
FIRST GPU run surfaces (writeback layout `col*N+row`, five-buffer ABI offsets, launch geometry) — none of which the spill
work touches. One correct number collapses far more uncertainty than 9->0 spills. The compile foundation is solid for
this: clean source graphs, math-neutral ordering edges (op counts identical, rounding boundary untouched), asserted ABI
in `__post_init__` — so a wrong number will point at the writeback/ABI logic, not at allocator hacks.

**What has NOT been done, so no one over-claims:** never run on GPU; never numerically validated; only the smallest full
grid (128,128,256) compiled — real roles are 512x1024x5120 up to 512x5120x17408, where more epochs = more `consumer_seam`
ordering and the aligned-tail/dynamic-offset paths first get exercised; performance entirely unmeasured. The 7 historical
failing tests (four 16-subtile, two prefill-adapter, one q4k-tiled) still fail identically on a clean tree — verify before
blaming any change.

Rails unchanged (no reassociation of `(previous + scale*C) + bias`, no FMA/MULACC, no rounding-boundary move, no AMD
scratch, no `spec_shared` widening, no model/VRAM/GPU/route branching).

## 0.5 Status update 2026-07-17 (Codex): full-grid GPU proof, R5 win, R6/R7 fail-closed

The recommended correctness-first path now has a concrete bounded result. Commits `472ca6da9`, `72e58d322`,
`ef9fb08c8`, `e78551257`, `f6fa66b96`, `76b721184`, `1b7748e26`, `1a26a721a`, `2643551d4`, `48c11957d`, and
`661aea564`, `fea30a7ee`, `1b525a2a5`, `1b7b249cd`, `1c01325de`, `26edaa8f2`, and `78101b7c9`
are pushed on `master`.

* The emitted full-grid 128x128x256 PROGRAM passed an in-process AMD probe: 0/16,384 output mismatches against the
  fp16-rounded Q8 DS4 oracle, max absolute error 3.05e-5, mean 1.12e-8. Resource evidence is vgpr=256, LDS=57,856,
  scratch=0, wave32, with distinct source/binary hashes. This is the `q4k_q8_1_mmq_amd_isa_full_grid_v0` backend.
* Same-session timing on identical quantized inputs is full-grid 0.287 ms minimum versus direct-packed 8.980 ms
  minimum (about 31.2x minimum speedup). This is R5 bounded evidence only; it is not a production route claim.
* R5 ranking now recognizes an emitted full-grid win even when the oracle row ranks faster. `promotion_eligible` stays
  false and R6 remains `BLOCKED_ROLE_SHAPE_INTEGRATION`; no role was changed and direct-packed remains default.
* The exact R6 target is `ffn_gate_up` 512x17408x5120. The probe covers only 128x128x256. The route-shape artifact
  records 20 required K-epoch launches over the full 4x136 M/N grid, plus Q4/DS4 repacking and accumulation (the
  full-grid sink already owns/scatters all M/N tiles in each launch).
  A manifest-scoped smoke artifact now passes the negative-role (`attn_qo`, `ffn_down`, `attn_kv`) and direct-packed
  rollback/default checks; these subgates are true, but they do not substitute for target-shape execution.
* A monolithic K=512 compile is concretely blocked by `NotImplementedError: vgpr lease exceeds virtual pool`. An
  explicit per-store LOAD+ADD accumulation sink was prototyped, but its two-launch 128x128x512 probe exceeded the hard
  six-minute compile deadline with no structured result. The safe follow-up delegates accumulation to tinygrad
  elementwise add over fresh partial outputs and **passes** the bounded 128x128x512 proof: 0/16,384 mismatches,
  max abs 2.44e-4, vgpr=256, LDS=57,856, scratch=0. This is adapter evidence only; production repack, 20-launch
  scheduling, role census, and no-hidden-fallback proof remain absent. R7 rows carry both the per-store timeout and
  the bounded elementwise result instead of claiming source-clone conversion.
* The lazy owner manifest now lets the actual 14B shape `(512,17408,256)` build and emit one K-epoch PROGRAM in
  182.23s: vgpr=256, LDS=57,856, scratch=0, owner count 8,912,896, source SHA `b8923985…`, binary SHA
  `21908e0b…`. This closes the target-shape compile/resource slice, not the 20-epoch GPU correctness/health gate.
  A reduced 256x256x256 one-epoch dispatch then exposed the next blocker: 65,425/65,536 mismatches (max abs 840.9)
  despite the same zero-scratch resources; 128x128 remains the only numerically passing grid. The target 20-epoch
  run also hit an HCQ wait timeout (signal 9 not set, current 8). Treat this as a grid-scaling address/writeback
  defect, not a promotion or GPU-health pass. The diagnostic launch is explicitly `global_size=[2,2,1]`,
  `local_size=[256,1,1]`; all four 128x128 tiles fail, so the issue is not isolated to a nonzero tile origin.

The through-line is unchanged: the first real GPU result collapsed more uncertainty than further spill grinding, but
the 31x bounded win is not evidence for a 14B route. Never run a production role on this candidate until the K-tiled
adapter has exact numerical evidence, resource/health evidence, negative-role tests, and a no-hidden-fallback census.

## 1. Executive state

The project is building a generated tinygrad prefill route for non-fitting quantized models, using Qwen3-14B as the
proof workload. The immediate goal is generated Q4_K/Q8_1 MMQ parity or better against the frozen llama.cpp comparator.
The model is selected by the user, but compiler route selection must remain a function of workload and hardware facts,
never a model-name, fixed-VRAM, or fixed-GPU branch.

The architectural substrate is approximately 90% complete. The full project is approximately 60-65% complete. The
active blocker is still before GPU performance work: the source-pinned generated oracle does not emit a final
spill-free AMD binary.

Current exact compiler progress:

| checkpoint | peak live virtuals | spill fallback | stack | first request |
|---|---:|---:|---:|---:|
| matched pre-marker baseline | 184 | 525 | 2,100 B | UOp 2532 |
| carrier marker recovery, `5e9a5c5dc` | 150 | 185 | 740 B | UOp 2532 |
| propagated marker recovery, `412d7998f` | 158 | 112 | 448 B | UOp 2580 |

Only matched runs under the same environment are comparable. Older reports of 406 spills used a different diagnostic
environment and should not be numerically compared with the 525/185/112 sequence.

At the current revision:

- Post-selection structure remains 9,481 UOps.
- The capture sees 128 selected machine WMMA nodes. The source oracle has 16 logical WMMAs; selection/expansion is why
  both numbers appear in notes. Do not treat 128 as a changed mathematical kernel.
- All 112 spill requests are `AMDOps.DS_LOAD_B128` base carriers.
- There are no remaining `V_WMMA_I8` spill requests in the retained exact run.
- The first residual spill is an A/B fragment pair whose four-VGPR bases are constrained to the shared fragment runs.
- Compilation still correctly fails closed because AMD scratch spilling is forbidden.

This is meaningful progress, not completion. No new 14B end-to-end or performance claim is authorized yet.

## 2. User intent and non-negotiable constraints

The user wants the work continued through the logical endpoint, but with these boundaries:

1. Stop before performance autoscan. Autoscan is allowed only after a manually selected generated route proves beyond
   llama parity.
2. The model remains user-selected. GPU capability, VRAM, memory admission, and eventually candidate choice should be
   discovered from facts, not flags or hardcoded model/GPU cases.
3. Do not create an `8B` route and a `14B` route. The architectural distinction is fitting versus non-fitting resource
   conditions.
4. Do not permit hidden dense dequantization, scratch spilling, or a handwritten/oracle kernel to satisfy a generated
   route claim.
5. Preserve exact llama recurrence ordering and rounding. Do not introduce FMA/MULACC, reassociate
   `(previous + scale*C) + bias`, or move fp16/fp32 conversion boundaries without a new correctness authority.
6. Keep commits coherent and small. Hooks are enabled. Push accepted commits to `origin/master`.
7. Preserve unrelated user changes and stashes. Never rewrite or delete user stashes.
8. Temporary authored-code budget is 35,000 lines. After parity proof, perform another prune and restore the 30,000-line
   target. Do not add speculative frameworks when an existing primitive can be reused.
9. Decode and prefill are separate benchmark authorities. The current work is generated prefill; decode must later pass
   a regression gate.
10. GPU may be used now, but do not consume it while another explicitly reported benchmark run owns it. The active
    compiler reproducer is CPU-only.

## 3. Canonical objective and promotion authority

The canonical scope remains:

- `docs/qwen3-14b-generated-prefill-completion-scope-20260714.md`

The final promotion authority is whole-prefill synchronized tokens/second at prompt/context lengths 512, 1,024, 2,048,
and 4,096. Llama and tinygrad must run sequentially in at least three alternating pinned-clock sessions with raw samples
retained.

Required final gates:

- No accepted context below 98% of llama.
- Every declared context median exceeds llama.
- Paired aggregate 95% lower confidence bound exceeds 1.00.
- Aggregate geometric-mean target is at least 105% of llama.
- Project target is at least 2,000 tok/s.
- Decode remains correct and within the declared regression tolerance.
- If any context or aggregate misses, run a fresh Boltbeam timing pass on the exact candidate revision and reconcile
  candidate roles, activation preparation, attention, launch/synchronization, and residual work.

Kernel TFLOP/s and Boltbeam are attribution evidence, not promotion authority.

## 4. Current phase ledger

| phase | state | closing condition |
|---|---|---|
| Scope and architecture | complete | Canonical decision tree and no-model-branch rules are documented. |
| Six-row evidence and attribution | complete | Four Q4 and two Q6 workload rows are exact and identity-qualified. |
| Source-pinned oracle foundation | complete | Exact llama structure is adapted to the five-buffer ABI and full-grid seam. |
| Spill-free generated emission | **in progress** | Complete bounded oracle emits a real AMD program with zero spills/scratch. |
| Q4/Q6 role policy | blocked on emission | Every row has an exact candidate or declared rollback under one immutable policy. |
| 14B mixed-route end to end | not started | Correctness, route census, memory/resource proof, GPU health, decode regression. |
| Matched llama/Boltbeam qualification | not started | Multi-context statistical parity/beyond-parity gates pass. |
| Autoscan and post-proof prune | deliberately paused | Start only after beyond-parity proof. |

## 5. Target architecture

```text
loaded GGUF tensor facts + user-selected model
  -> typed workload inventory
  -> target capability + memory admission
  -> candidate identity and exact packed ABI
  -> Q8 activation producer
  -> generated Q4_K/Q8_1 MMQ contraction
  -> resource/correctness evidence
  -> immutable manual six-row policy
  -> 14B end-to-end proof
  -> matched llama comparison + Boltbeam attribution
  -> only then performance autoscan
```

The current source-pinned MMQ authority is:

- Tile: `128x128x256`.
- Eight waves, 256 threads.
- 57,856 bytes LDS.
- Persistent decoded Q4 panel.
- Two K128 Q8 phases.
- Eight K32 correction groups / sixteen signed i8 WMMAs per K256 epoch.
- Exact half2 metadata recurrence.
- Split five-buffer ABI:
  - slot 0: fp32 output;
  - slot 1: Q4 packed `uint32`;
  - slot 2: Q8 values `int8`, physical `[K/128, M, 128]`;
  - slot 3: Q8 scales `fp32`, physical `[K/128, M, 4]`;
  - slot 4: Q8 original sums `fp32`, physical `[K/128, M, 4]`.

Scale and sum are independently converted to fp16 only at LDS half2 staging, preserving llama rounding.

## 6. Relevant code map

### Oracle and generated full-grid seam

- `extra/qk/mmq_llama_candidate_plan.py`
  - Source-pinned physical schedule and identity.
- `extra/qk/mmq_llama_oracle_epoch.py`
  - Exact epoch construction.
- `extra/qk/mmq_llama_oracle_recurrence.py`
  - Exact K32 recurrence and rounding authority.
- `extra/qk/mmq_llama_full_kernel.py`
  - Bounded callback graph and release seams.
- `extra/qk/mmq_llama_record_producers.py`
  - Typed producer witnesses.
- `extra/qk/mmq_llama_five_buffer_graph.py`
  - Five-buffer composition across epochs.
- `extra/qk/mmq_llama_five_buffer_full_kernel.py`
  - Full-grid ownership, dynamic offsets, aligned-tail policy, row-major writeback, fail-closed compilation.

### AMD selection, scheduling, and allocation

- `tinygrad/renderer/isa/amd.py`
  - `_progressive_c_assignment`: symbolic root path cover.
  - `_acc_base`: physical C-fragment allocation/reuse.
  - `_try_wmma_kmajor_phase`: alternate K-major selection path. A conditional chain-major experiment here was proven
    inactive for the exact workload and reverted; do not assume this function owns the residual path.
  - `_selected_wmma_roots`: retained at `412d7998f`; recovers one-step carrier markers and propagated multi-step
    machine-chain markers, walking to the earliest machine head.
  - `_serialize_progressive_c_drains`: finds each chain tail, collects eight conversion drains, and orders reuse heads.
  - `_post_isel_structural_lifetimes`: composes store chaining, progressive-C serialization, and address localization.
  - `_build_wmma_from_packs` / `_build_wmma_tile`: fixed A/B/C machine fragment construction.
- `tinygrad/codegen/late/regalloc.py`
  - `pressure_schedule` / `_pressure_schedule_block`: generic pre-regalloc pressure-aware topological ordering.
  - `LinearScanRegallocContext`: live ranges, rematerialization, spill diagnostics, fail-closed no-spill allocation.
- `tinygrad/codegen/__init__.py`
  - Ordering of pressure scheduling, backend cleanup, allocation, and post-regalloc lowering.

### Policy/evidence

- `extra/qk/runtime_specs.py`
- `extra/qk/prefill/six_row_policy_artifact.py`
- `extra/qk/prefill/q4k_q8_five_buffer_role_gate.py`
- `bench/prefill-pure-full-kernel/`

## 7. Current exact blocker

After `412d7998f`, the retained exact run reports:

```text
REGALLOC_DEBUG: 9481 uops, PEAK 158 live vregs @ uop 5908
REGALLOC_PRESSURE: spill_request=v0 at=2580
REGALLOC_SPILLS: count=112 stack_size=448
```

Peak composition:

```text
65 GLOBAL_LOAD
64 V_ADD
 8 V_CONST
 8 V_CVT_I2F
 6 V_IMUL
 2 WI_ID
 1 ParamArg
 1 V_LSHR
 1 S_LOAD_PTR
 1 lidx0
 1 V_IADD
```

The extra eight `V_CVT_I2F` relative to the 150-virtual checkpoint are intentional: they are the real release frontier
that now remains live long enough to protect progressive C reuse.

Every one of the 112 fallback spills is a `DS_LOAD_B128` base carrier. The first region has seven future A/B pairs
opened before their consumers:

```text
A/B pair ranges begin around UOps 2507..2520
matching consumers occur at 2580, 2633, 2686, 2739, 2792, 2845, 2898
```

The hardware meaning is:

- A fragment: four consecutive VGPRs.
- B fragment: four consecutive VGPRs.
- C/D accumulator: eight in-place VGPRs.
- Only one A/B pair should be resident for the active WMMA lease unless a proven double-buffer schedule assigns a
  distinct physical run.

The allocator representation uses one constrained base definition plus fixed aliases for the remaining fragment lanes.
Do not replay an LDS load in regalloc: it can violate memory/barrier semantics. Do not allow scratch as a workaround.

The next owning problem is therefore selected A/B fragment lifetime/order: pair each constrained A/B load with its
actual WMMA consumer, and prevent future pairs using the same physical run from opening until the preceding consumer has
released that run.

## 8. Required first reproduction

Run from repository root:

```bash
env \
  PYTHONHASHSEED=0 \
  REGALLOC_ADDR_REMAT=1 \
  REGALLOC_END_NO_SOURCE_LIVE=1 \
  REGALLOC_DEBUG=1 \
  REGALLOC_DEBUG_PRESSURE=1 \
  REGALLOC_DEBUG_SPILLS=1 \
  python3 -m pytest -q -s \
  test/unit/test_mmq_llama_full_kernel.py::test_bounded_wave_probe_drains_symbolic_subtiles_without_claiming_a_full_grid
```

Expected current result:

- Pytest reports `1 passed` because the test currently expects the fail-closed `NotImplementedError`.
- The diagnostic must report 112 spills and a 448-byte stack.
- A green pytest alone is **not** success.

When genuine zero-spill emission is achieved, update the bounded test so it no longer expects the spill exception and
instead asserts a real program/resource proof. Do not change the test early merely to make it fail or pass differently.

Fast structural suite:

```bash
python3 -m pytest -q \
  test/unit/test_amd_isa_structural_lifetimes.py \
  test/unit/test_regalloc_pressure_schedule.py \
  test/unit/test_regalloc_end_liveness.py
```

Oracle/full-grid structural suites:

```bash
python3 -m pytest -q \
  test/unit/test_mmq_llama_full_kernel.py \
  test/unit/test_mmq_llama_five_buffer_graph.py \
  test/unit/test_mmq_llama_five_buffer_full_kernel.py
```

Avoid launching duplicate exact-oracle processes. Several previous workers accidentally ran two copies, wasting minutes
and making measurements harder to attribute.

## 9. Next implementation scope

### Step A: prove the selected A/B pair-to-consumer mapping

Before changing scheduling, record for the first spill region:

- each `DS_LOAD_B128` base UOp;
- its fixed physical base and four-lane fragment range;
- its unique numeric WMMA consumer;
- zero-code alias consumers separately;
- its barrier/AFTER ordering dependencies;
- the previous and next load pair using the same physical A/B runs.

Do this post-selection and before regalloc cleanup. Remove temporary instrumentation before committing.

The expected result is seven future pairs opened before their seven consumers. If that is not reproduced, stop and
explain the changed path instead of implementing the plan below from assumption.

### Step B: repair ownership at the AMD selected-lifetime layer

Preferred repair boundary:

- AMD post-selection structural lifetime logic, close to `_serialize_progressive_c_drains` and the selected machine
  fragment graph; or
- the exact AMD selector that emits the residual `DS_LOAD_B128` pair, if the pair order is already wrong at creation.

Required invariant:

```text
previous A/B pair -> matching WMMA consumes pair
                  -> next A/B pair using same physical run may load
                  -> next matching WMMA
```

The ordering must be derived from physical lease identity and actual machine consumers, not from Qwen, 14B, exact
M/N/K, hardcoded UOp ordinals, or a route flag.

If multiple physical A/B runs truly exist, independent runs may overlap. Only pairs mapped to the same constrained run
must serialize.

Potential implementation shape:

1. Discover selected `DS_LOAD_B128` definitions and their fixed base constraints.
2. Follow zero-code fixed aliases but identify the actual machine WMMA consumer.
3. Group load pairs by the physical A/B lease tuple.
4. Order consumers using existing symbolic/machine dependencies; fail closed on ambiguity.
5. Add a zero-code ordering edge from each previous consumer to the next pair's load address/dependency position, while
   preserving the canonical DS-load operand shape and all barrier dependencies.
6. Let `pressure_schedule` place the now-ready pair adjacent to its consumer.

Do not add a generic scheduler rule that assumes every unary one-choice instruction is a zero-code alias. That experiment
was rejected as semantically unsafe. If scheduler lookthrough is needed, use an explicit renderer-owned alias contract.

### Step C: acceptance for the A/B repair

Minimum retained evidence:

- The selected mathematical graph is unchanged.
- Exact WMMA count is unchanged.
- A/B load values, addresses, widths, and fixed bases are unchanged.
- Barriers and release witnesses remain in backward slices.
- At most one A/B pair per shared physical lease is live at a consumer boundary.
- Spill count falls materially from 112.
- No new spill category is introduced.
- Focused AMD and regalloc tests pass.

Closing evidence for the current phase:

- Zero spills.
- Zero scratch/stack.
- Final program emits.
- Resource proof identifies the executed binary.

If a safe repair reduces 112 to a smaller nonzero value, it may be retained as a coherent generic improvement, but the
phase remains open and the next exact victim must be reported.

## 10. Do not repeat these rejected paths

| experiment | result | reason not to revive unchanged |
|---|---|---|
| speculative 12 KiB cooperative descriptor | removed | The source-pinned llama tile is the authority. |
| graph-level Q8 packing | ~232 ms on smallest role | Correct algebra, unusable lifecycle/performance. |
| dequant-once fp16 route | ~4.7 ms on smallest role | Far slower than fused candidate. |
| metadata-only PARAM/SPECIAL carriers | peak worsened in matched A/B | Cosmetic, not owning pressure. |
| atomic eight-register WMMA result span | virtual peak fell, physical saturation remained | Broke UOp invariant and hit unsupported multi-register spilling. |
| vec8 recurrence bundle | reverted | Bundle-side physical improvement was not proven. |
| generic alias-closure scheduler | spills worsened to 516 | Reduced one overlap but delayed other releases; initial alias inference was also unsafe. |
| conditional chain-major `_try_wmma_kmajor_phase` | exact metrics unchanged | The real workload did not select that branch; commit `5913c65c4` was reverted by `63b31be2d`. |
| regalloc replay of `DS_LOAD_B128` | prohibited | Shared-memory loads are not safe rematerialization. |
| enabling AMD scratch | prohibited | Violates performance/resource contract and hides the compiler defect. |

## 11. Accepted compiler repairs and what they proved

Recent accepted commits, oldest to newest:

- `27a910ed6 [amd] preserve wide fragment release ordering`
  - Carries completed FP32 release dependencies through selected LDS loads.
- `272d73af8 [codegen] reduce wide pipeline register pressure`
  - Generic pressure-aware scheduling.
- `c05188028 [codegen] rematerialize renderer-safe register fills`
  - Allows safe V_CONST replay at ordinary uses.
- `a293391b0 [codegen] localize constrained leases at consumers`
  - Shrinks a DS-load-to-WMMA distance from 242 UOps to at most 24 in the measured case.
- `1de98cc7f [codegen] stop END body register liveness`
  - Removes false completed-body machine liveness; matched peak 184 to 180 in its original comparison.
- `0d91841d3 [codegen] prioritize pressure release in late scheduling`
  - Prioritizes immediate release before opening later recurrence generations.
- `5e9a5c5dc [amd] recover progressive C markers from carriers`
  - Activates the existing serializer for reachable marked carriers; matched spills 525 to 185.
- `412d7998f [amd] recover propagated progressive C roots`
  - Handles multi-step chains whose marked carrier is flattened into a tail machine WMMA; walks to earliest machine
    head; removes all retained WMMA-result spill requests and reduces spills 185 to 112.

Review these commits before changing the same mechanisms. Preserve their focused invariants.

## 12. Known measurement and test caveats

1. `peak_candidate_slots=255` is a union of registers that live virtuals may occupy, not an additive physical-VGPR
   requirement. Ordinary scalar values advertise most of the pool. Use assigned constrained-run overlap and spill
   victims as the actionable evidence.
2. A fail-closed bounded test currently passes by expecting the failure.
3. Detailed debug modes alter allocator traversal cost and have produced different absolute spill counts. Compare only
   matched before/after environments.
4. `/tmp/mmq_marker_before.json` and `/tmp/mmq_marker_after.json` were useful local captures but are not versioned
   authorities and may disappear. Reproduce facts from the committed revision.
5. Four older 16-subtile AMD tests fail historically at the same spill-free gate. They predate the recent scheduler
   commits. Do not mark them xfail merely to hide the resource defect.
6. The old ushort-cast test counted every `v_bfe_u32` mnemonic and was fixed by `2ef1174a9` to inspect packed-load-owned
   extracts only.
7. Keep llama and tinygrad benchmark runs sequential. Concurrent residency is not an admissible comparison.

## 13. After zero-spill bounded emission

Do not jump directly to autoscan. Execute these phases in order.

### 13.1 Full-grid compile and correctness

- Compile `mmq_llama_five_buffer_full_kernel` for aligned full-grid ownership.
- Verify row-major writeback remains `col*N + row` for the oracle's A/B orientation.
- Prove exact five-buffer ABI and launch geometry.
- Verify final ISA has no scratch/spills and resource allocation is within target limits.
- Run full-output correctness, immutable input checks, output guards, timeout isolation, and post-dispatch GPU health.
- Retain the exact candidate identity and binary/ISA proof.

### 13.2 Performance qualification

Fresh Q4 role budgets:

| role | shape MxNxK | current legacy candidate | direct-packed comparator |
|---|---|---:|---:|
| `attn_kv` | 512x1024x5120 | ~5.9996 ms | ~0.4571 ms |
| `attn_qo` | 512x5120x5120 | ~27.927 ms | ~0.920 ms |
| `ffn_down` | 512x5120x17408 | ~67.731 ms | ~3.107 ms |
| `ffn_gate_up` | 512x17408x5120 | ~66.872 ms | ~2.762 ms |

Exact `attn_kv` stage attribution:

- Q8 producer: ~0.0875 ms.
- MMQ: ~5.9160 ms.
- Total: ~6.0070 ms.
- About 98.5% of the tax is MMQ, not activation production.

The new oracle must be timed as the whole primitive, including activation preparation. Q4 target is at least 60 aggregate
logical TFLOP/s unless a newer whole-model role budget proves an equivalent result.

### 13.3 Six-row policy

- Four Q4 rows must bind exact correctness/performance-qualified candidates.
- Two Q6 rows already have qualified direct-packed fallback evidence:
  - `attn_kv` max error 0.015625;
  - `ffn_down` max error 0.015625.
- Build one immutable six-row manual policy.
- Missing evidence, identity drift, and unknown workload must fail closed.
- Keep production promotion false until the full end-to-end gate passes.

### 13.4 Phase 6 end to end

- Execute the manual mixed route on Qwen3-14B.
- Record route census proving every intended packed role fired.
- Reconcile memory/admission and final resources.
- Verify GPU health and output correctness.
- Run decode correctness/performance regression separately.
- Prove one-change rollback to direct packed.

### 13.5 Phase 7 parity and Boltbeam

- Run matched llama/tinygrad prefill at contexts 512, 1,024, 2,048, and 4,096.
- Use at least three alternating pinned-clock sessions.
- Retain raw samples and immutable revisions.
- If any gate misses, run Boltbeam immediately on that exact candidate and optimize the largest measured residual.
- Only after beyond-parity proof may performance autoscan begin.

## 14. Commit and handoff discipline

- Inspect `git status --short --branch` before and after every change.
- Do not bundle diagnostics, compiler repair, policy identity update, and benchmark artifact into one commit.
- Remove temporary logging before committing.
- Run `git diff --check`.
- Preserve user and unrelated changes.
- Push accepted commits.
- If an experiment is ineffective, revert its worktree edits before moving on. Avoid adding a speculative default-off flag
  merely to preserve a failed experiment.
- Record exact before/after measurements near the owning scope document when a blocker class changes.

## 15. Definition of this handoff's completion

Claude's immediate assignment is complete only when:

1. The complete source-pinned bounded oracle emits a real program.
2. Final AMD evidence reports zero spills and zero scratch/stack.
3. The selected mathematical graph, packed ABI, WMMA family/count, barriers, and rounding authority remain unchanged.
4. Focused AMD/regalloc tests and the updated bounded emission test pass.
5. The accepted repair is committed and pushed with an exact evidence note.

That closes only the current compiler-emission gate. Continue through full-grid correctness, performance, six-row policy,
Phase 6, and Phase 7 as described above, then stop before autoscan.

## 16. Recommended first 90 minutes

1. Read this document and the canonical completion scope.
2. Inspect `412d7998f`, especially `_selected_wmma_roots` and `_serialize_progressive_c_drains`.
3. Run the exact reproduction once and confirm 112 `DS_LOAD_B128` spills.
4. Add temporary post-selection diagnostics for the first seven residual A/B pairs and their seven WMMA consumers.
5. Determine the exact selector/lifetime function that opens those pairs early. Do not assume `_try_wmma_kmajor_phase`;
   that path was already changed experimentally with zero effect on the exact workload.
6. Add a focused structural test expressing one-A/B-pair-per-shared-lease ownership.
7. Implement the smallest AMD-local, ownership-derived ordering repair.
8. Run focused tests, then one exact oracle run.
9. Keep the patch only if it is semantically safe and materially reduces the 112 residual spills; continue until zero.
