# M1f — diffing the emitted store loop, AMD (correct) vs Metal (writes 18.745%)

Repo `exp` @ `6d13d617f` (HEAD is a docs-only commit about an unrelated M4 FMA microbench; the
files this task touches — `tinygrad/codegen/opt/kernel_lds.py`, `postrange.py`,
`tinygrad/renderer/cstyle.py`, `extra/llm_research/`— are unchanged since `769c81b17`, verified by
`git log --oneline 769c81b17..HEAD -- <those paths>`). Compile-only, no GPU: no `Device[...]`
instantiation, no `.synchronize()`, no dispatch. Same dispatch M1b/M1c/M1d/M1e qualified: Q4_K,
`ffn_gate_up`, shape `(512,12288,4096)`, geometry `(256,64,32,8,1,1)` (`tm,tn,tk,wm,wn,bc`) — AMD's
own promoted tuple from `PACKED_WMMA_ROUTES`, the exact geometry M1b/M1c/M1e measured the Metal
failure at.

This is a differential read, not a hypothesis test. Four prior hypotheses (lane permutation,
C-fragment width overcount, multi-wave decomposition, device-blind admission — M1c/M1d/M1e) were
each a guess about mechanism, tested and refuted or left inconclusive. Nobody had read the two
emitted sources side by side. This task does that instead.

## Method

Reused `scratchpad/m1d_confirm_c_fragment.py` verbatim (proven technique: build the Tensor AST via
`.schedule_linear()`, call `to_program(ast, renderer)` with a renderer built directly from
`Target.parse(...)`, which needs no `Device[...]` and lets AMD's real cross-compiler
(`amd_comgr`, stubbed only at the native-crash `do_compile` step, unrelated to codegen) run on this
Mac with no AMD hardware). Re-ran it fresh at current HEAD; both targets compiled clean
(`"error": None` for both), and the diagnostic numbers matched M1d's exactly:

- **METAL**: 129 `__WMMA` occurrences, 1 `simdgroup_multiply_accumulate` (inside the one wrapper,
  called 128 times), source length 27905 bytes.
- **AMD**: 17 `__WMMA_16_16_16_half_float(` occurrences (1 macro `#define` + 16 call sites), 0
  `simdgroup_multiply_accumulate`, source length 21642 bytes.

Neither source contains the string `simdgroup_matrix` (only `simdgroup_multiply_accumulate` and
`simdgroup_half8x8`/`simdgroup_float8x8`), confirming this run avoided the prior mis-conclusion the
task warned about.

Full sources: `/tmp/m1d_amd_source.c` (369 lines), `/tmp/m1d_metal_source.c` (496 lines) — not
committed, reproducible by re-running `scratchpad/m1d_confirm_c_fragment.py` (or
`scratchpad/m1f_store_address_diff.py`, which calls it as step 1 before its own analysis).

## 1. Store counts and address expressions

Both kernels emit exactly one un-guarded, unconditional store region, immediately after the
`for (int Ridx0 = 0; Ridx0 < 128; Ridx0++)` reduction loop closes. Neither store region is itself
inside a loop in the emitted text — the compiler unrolled the store fully — and neither is gated by
an `if`.

**AMD** (`/tmp/m1d_amd_source.c:305-369`): **64 scalar stores**, one `half` each, one line per
`buf0` slot, index `0..63`. Base address (line 305):

```c
int alu158 = ((gidx1*3145728)+(lidx1*393216)+((lidx0>>4)*12288)+(gidx0<<6)+alu5);
```

(`alu5 = (lidx0&15)`, defined earlier at line 31.) Representative store lines:

```c
*(data0_6291456+(alu158+16)) = ((half)((*(buf0+16))));
*(data0_6291456+(alu158+24576)) = ((half)((*(buf0+1))));
...
*(data0_6291456+alu158) = ((half)((*(buf0+0))));
```

**METAL** (`/tmp/m1d_metal_source.c:463-495`): **32 vector stores**, each a `half2` covering 2
adjacent `buf0` slots (64 slots total, matching Metal's `elements_per_thread[2]=2`). Base address
(line 463):

```c
int alu161 = ((gidx1*3145728)+(lidx1*393216)+((lidx0>>4)*49152)+((alu0&1)*24576)+(((lidx0>>1)&1)*12288)+(gidx0<<6)+(((lidx0>>3)&1)<<2)+((lidx0&1)<<1));
```

(`alu0 = (lidx0>>2)`, defined at line 17.) Representative store lines:

```c
*((device half2*)((data0_6291456+(alu161+8)))) = half2(((half)((*(buf0+8)))),((half)((*(buf0+9)))));
*((device half2*)((data0_6291456+(alu161+98304)))) = half2(((half)((*(buf0+2)))),((half)((*(buf0+3)))));
...
*((device half2*)((data0_6291456+alu161))) = half2(((half)((*(buf0+0)))),((half)((*(buf0+1)))));
```

Grid dims are read directly off the source's own comments and cross-checked against the compiled
kernel names, both of which begin `r_2_192_32_8_...`: `gidx0` in `[0,192)`, `gidx1` in `[0,2)`,
`lidx1` in `[0,8)`, `lidx0` in `[0,32)` — identical launch grid on both targets (`98304` threads),
confirmed from the emitted kernel-name text, not inferred.

## 2. Do the addresses cover the output tile exactly once?

**Yes, on both targets — a clean bijection, verified by brute force, not by inspection.**

`scratchpad/m1f_store_address_diff.py` transcribes the two base-address formulas and full
offset/`buf0`-index tables above verbatim from the rendered source, then evaluates them with NumPy
over the entire launch grid the kernel's own source declares (`98304` threads × 64 `buf0` slots =
`6,291,456` = `512*12288`, exactly the output tile size) and checks the resulting address set for
gaps and collisions. Result (from a fresh run at current HEAD):

```json
"AMD":   {"unique_addresses": 6291456, "min_addr": 0, "max_addr": 6291455,
          "any_address_hit_ne_1_time": false, "max_hit_count": 1, "min_hit_count": 1,
          "is_clean_bijection": true}
"METAL": {"unique_addresses": 6291456, "min_addr": 0, "max_addr": 6291455,
          "any_address_hit_ne_1_time": false, "max_hit_count": 1, "min_hit_count": 1,
          "is_clean_bijection": true}
```

Every one of the `6,291,456` output cells is targeted by exactly one `(thread, buf0-slot)` pair on
**both** targets; no cell is targeted twice, none is never targeted. **Metal's store-address
computation is a clean bijection onto the output tile, identical in this property to AMD's.** This
directly answers the guidance's question about "which 12 of the 64 accumulator slots get stored" —
**none are structurally excluded; the static source addresses all 64 slots' worth of storage for
every thread, covering the whole tile exactly once.** The 18.745% (12/64) write-coverage failure is
**not explained by a source-level address-coverage gap** on either target.

## 3. Where the two sources structurally diverge in the store region

- **Store vector width.** Metal batches its store into `half2` (32 stores of 2 elements each),
  matching its own `tc.elements_per_thread[2]=2` exactly — every store's two `buf0` indices are the
  `.x`/`.y` of one WMMA's own `float2` result (e.g. `(*(buf0+8))=wmma119.x; (*(buf0+9))=wmma119.y;`
  at lines 406-407, then stored together as one `half2` at line 464). AMD does **not** vectorize its
  final store at all — despite `tc.elements_per_thread[2]=8`, it emits 64 fully scalar `half`
  stores, one per `buf0` slot, never batching by 8 the way its own WMMA's `float8` results would
  allow. This is the single largest textual difference in the store region: 64 scalar statements
  vs 32 vector statements. Since AMD is the known-correct target and still doesn't vectorize this
  store, vector width itself is not shown to matter for correctness here — it reads as an
  independent optimizer choice (whether a store-vectorize pass fired), not a correctness signal.

- **Base-address bit-decomposition of `lidx0` (the intra-wave/intra-simdgroup lane id, 32 lanes,
  5 bits) differs structurally between targets:**
  - AMD: `(lidx0>>4)*12288 + alu5` where `alu5=(lidx0&15)` — only the **top bit** of `lidx0` gets a
    stride (`12288`, one output row); the low 4 bits are added as a single unscaled unit
    (`0..15`), i.e. AMD's formula treats `lidx0` as (1 bit high, 4 bits low) with no further
    interleaving.
  - Metal: `(lidx0>>4)*49152 + ((lidx0>>2)&1)*24576 + ((lidx0>>1)&1)*12288 + (((lidx0>>3)&1)<<2) +
    ((lidx0&1)<<1)` — **all 5 bits of `lidx0` individually get their own stride**
    (`49152, 24576, 12288, 4, 2` for bits 4,2,1,3,0 respectively), a fully bit-interleaved mapping.

  This reflects each target's own hardware lane→matrix-element layout (AMD's `16x16x16` WMMA vs
  Apple's `8x8` `simdgroup_matrix` per-quad layout) rather than a shared convention — the two
  targets do not merely differ in vector width, they use genuinely different bit-permutations of
  the same 32-lane index to compute a store address. The brute-force check in §2 confirms Metal's
  particular permutation is still injective/surjective onto the tile, so this divergence is
  real but not, by itself, shown to be a bug.

- **Barrier idiom** (textual difference, same logical position on both targets — once per
  `Ridx0` iteration, immediately after all `buf1` (LDS) producer writes and immediately before the
  fragment reads from `buf1`):
  - AMD (`/tmp/m1d_amd_source.c:163`):
    `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");__builtin_amdgcn_s_barrier();__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");`
  - Metal (`/tmp/m1d_metal_source.c:157`): `threadgroup_barrier(mem_flags::mem_threadgroup);`

  Both are a single barrier per iteration, in the same place in the loop body, on both targets.
  Neither target has a **second** barrier at the end of the loop body — there is no synchronization
  between "last read of `buf1` for iteration i" and "first write to `buf1` for iteration i+1" on
  **either** target, an identical, target-independent structural property, not a place the two
  sources diverge. I did not determine whether this single-barrier-per-iteration pattern is
  provably safe (e.g. by hardware lockstep guarantees AMD wavefronts may have that Metal's
  independently-scheduled `simdgroup`s within a threadgroup may not), only that it is textually
  the same shape on both targets — it cannot itself be *the* target-differentiating cause, though
  it remains a candidate site for a *target-dependent hazard* under identical source.

- **Loop bounds / guards**: identical on both targets — one `for (int Ridx0 = 0; Ridx0 < 128;
  Ridx0++)` reduction loop, unrolled `buf0` zero-init (64 lines on both), no bounds/edge guard
  anywhere in either kernel (consistent with the tile dividing the output exactly: `192*64=12288=N`,
  `2*256=512=M`).

## Verdict

**The address-coverage hypothesis is refuted for this dispatch, on this compile-only reading.**
Both targets' store loops are, at the source level, a complete, non-overlapping, gap-free bijection
onto the full output tile — verified by exhaustive enumeration over the entire launch grid
(`98304` threads), not sampled or inferred. Per the task's own anticipated negative-result branch:
**this is a genuinely useful negative that points away from address computation and toward
ordering/barriers (or something else not visible in a static per-kernel source read) as the
locus of the fault.** The two sources do structurally diverge — in store vector width and in the
bit-permutation each target uses to map a 32-lane wave index to a store address — but neither
divergence produces a coverage gap or collision in the addresses themselves.

## What I could not establish

- **Whether the *values* stored are correct**, as opposed to the *addresses* being correct. This
  task was scoped to store addresses; M1d already established (tracing `postrange.py`) that the
  WMMA node construction on this branch (`postrange.py:536-540`) is generic and
  target-symmetric (`tc.dtype_out.vec(tc.elements_per_thread[2])`, no hardcoded literal), but I did
  not re-derive or execute the full reduction (`buf1` write → barrier → fragment read → WMMA →
  `buf0` write, 128 iterations) to check whether a *value*-level race (as opposed to an
  *address*-level one) could explain 18.745%-coverage + non-determinism. That is consistent with
  the guidance's own framing: a clean address bijection does not rule out "multiple threads write
  the same location" at a level below what a static address read can see (e.g. a real hazard on
  `buf1`, the shared LDS buffer, given the single-barrier-per-iteration structure noted in §3), nor
  does it rule out "a location is read before it is written."
- **Whether the actual GPU dispatch launches the same grid the source encodes.** The kernel names
  (`r_2_192_32_8_...` on both targets) and the source's own `/* 192 */`-style comments agree with
  each other and with what this task's brute-force check used, but this is still a compile-only
  artifact — I did not check (and this task did not ask me to check, being compile-only) whether
  `run_guarded_execution`'s actual `dispatchThreadgroups`/enqueue call on real Metal hardware
  requests the same `global_size`/`local_size` the compiled `Program` implies, nor whether Metal's
  per-`simdgroup` scheduling (as opposed to AMD's per-wavefront lockstep, if any) could let two
  different `(gidx0,gidx1,lidx1,lidx0)` combinations' *instructions* interleave in an order the
  single mid-loop barrier does not forbid, producing exactly the kind of race the addresses alone
  cannot reveal.
- I did not attempt to re-derive the specific "12 of 64" / wave-4-shaped garbage signature M1c/M1e
  measured against this now-ruled-out mechanism; this task's scope was the store address diff, not
  a new runtime measurement.

## Files

- `scratchpad/m1f_store_address_diff.py` — this task's driver: re-runs
  `scratchpad/m1d_confirm_c_fragment.py` to refresh the two rendered sources, then brute-forces the
  transcribed store-address formulas over the full launch grid with NumPy (no GPU, no tinygrad
  runtime — pure arithmetic on the extracted formulas) and reports unique-address counts, min/max,
  and collision counts for both targets. Re-run directly: `python3
  scratchpad/m1f_store_address_diff.py` (add `--skip-regen` to reuse existing `/tmp/m1d_*_source.c`
  without recompiling).
- `/tmp/m1d_amd_source.c`, `/tmp/m1d_metal_source.c` — full rendered kernel source for each target
  from this run (not committed; local scratch output, reproducible by re-running
  `scratchpad/m1d_confirm_c_fragment.py` or `scratchpad/m1f_store_address_diff.py`).
