# llama.cpp `kernel_mul_mm` vs. our precontract kernel — structural comparison

Date: 2026-07-31

Status: compile-only, read-only on llama.cpp, no GPU work performed. Continues the BUG A / BUG B
scope in `docs/task_workflow/input/metal-precontract-two-bug-scope-20260731.md`.

**Purpose.** llama.cpp reaches ~79% of this M4's measured ALU peak on Metal Q4_K_M prefill. A
known-good kernel for the same operation on the same hardware is on disk. This reads it and
compares its synchronisation/staging structure — not its source text — against what our renderer
actually emits for Q4_K `ffn_gate_up (512,12288,4096)`, geometry `(256,64,32,8,1,1)`, Metal. Goal:
locate the structural properties BUG A and BUG B are missing, not to translate llama's code.

---

## 0. What was rendered, and how

`scratchpad/m1d_confirm_c_fragment.py`'s `render_one("METAL")` was called unmodified (no production
file touched) to build the AST via `.schedule_linear()` and `to_program(ast, MetalRenderer(...))`,
compile-only. Full rendered source written to `/tmp/m1d_metal_source.c` (496 lines, not committed —
reproducible by re-running the script). `wmma_call_count = 129` (1 helper definition +
128 call sites), `simdgroup_multiply_accumulate_count = 1` (inside the `__WMMA_8_8_8_half_float`
helper, called 128×). Confirms `__WMMA`/`simdgroup_multiply_accumulate` per the task's search rule —
`simdgroup_matrix` never appears.

llama's kernel read directly from
`/Users/julianabeleda/env/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` (read-only) plus its host
dispatch code in `ggml-metal-device.cpp` / `ggml-metal-ops.cpp` (also read-only) — no llama.cpp file
was modified.

**Which llama kernel is the M4 one.** `ggml-metal.metal:9381` opens `#ifdef GGML_METAL_HAS_TENSOR`
(Apple's newer tensor-core path — a distinct kernel body using `tensor<>`/`cT.store`, not what an M4
runs) with `#else` at `:9505` and `#endif // GGML_METAL_HAS_TENSOR` at `:9722`. The kernel read here
is the `#else` body, `ggml-metal.metal:9509–9720`, which uses `simdgroup_half8x8` /
`simdgroup_multiply_accumulate` (confirmed at `:9671`) — the non-tensor-core path, i.e. what M4
actually executes. `GGML_METAL_HAS_TENSOR` was not evaluated on this machine; the identification
rests on the source-level type usage (`simdgroup_multiply_accumulate`, matching M4's plain-FMA
`simdgroup_half8x8` ISA) not on a runtime capability check.

For the Q4_K instantiation, the template parameter mapping (from
`ggml-metal.metal:10184`) is: `S0=half, S0_4x4=half4x4, S0_8x8=simdgroup_half8x8, S1=half,
S1_2x4=half2x4, S1_8x8=simdgroup_half8x8, block_q=block_q4_K, nl=QK_NL, dequantize_func=
dequantize_q4_K, T0=float, T0_4x4=float4x4, T1=float, T1_2x4=float2x4`. Because `T0_4x4=float4x4 !=
block_q4_K`, the compile-time check `is_same<T0_4x4, block_q>::value` at `:9575` is false, so **every
Q4_K K-loop iteration takes the `else` (dequantize) branch, `:9592–9614`** — the fast
"no dequantization needed" branch at `:9575–9591` never executes for this type. That resolves an
apparent "three barriers" count down to two per iteration for Q4_K specifically (see §1).

---

## 1. Barrier count and placement per K-loop iteration

### llama (`ggml-metal.metal:9573–9677`, the `for (int loop_k = 0; ...)` reduction loop)

Two `threadgroup_barrier(mem_flags::mem_threadgroup)` calls execute per iteration for Q4_K:

```
9592	        } else {
9593	            S0_4x4 temp_a;
9594	            dequantize_func(x, il, temp_a);
9595	
9596	            threadgroup_barrier(mem_flags::mem_threadgroup);
9597	
9598	            FOR_UNROLL (short i = 0; i < 16; i++) {
...
9612	                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
9613	            }
9614	        }
```

Barrier **#1**, `:9596`, sits **after dequant is computed into a thread-local register
(`temp_a`) but before any store to `sa`** — i.e. at the top of the loop body, before the producer
writes. It has nothing to do with the register-only dequant compute; what it orders is the *previous*
iteration's consumer reads of `sa`/`sb` (the `simdgroup_load` calls at `:9659,9665`, from the
*previous* trip through the loop) against *this* iteration's producer writes to `sa` (`:9598–9613`)
and `sb` (`:9616–9642`, unconditional, no barrier of its own — it relies on barrier #1 already having
fired). **This is exactly the read→next-write ordering our kernel lacks.**

```
9644	        il = (il + 2 < nl) ? il + 2 : il % 2;
9645	        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;
9646	        y += NK;
9647	
9649	        threadgroup_barrier(mem_flags::mem_threadgroup);
9650	
9651	        // load matrices from threadgroup memory and conduct outer products
9652	        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
9653	        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));
9655	        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
9656	            simdgroup_barrier(mem_flags::mem_none);
...
9659	                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
```

Barrier **#2**, `:9649`, sits **after both producer stores (`sa` and `sb`) complete, before any
consumer load** (`simdgroup_load` at `:9659,9665`) — classic write→read ordering within the same
iteration.

So the per-iteration sequence is: **barrier(prev-reads→this-writes) → write sa → write sb →
barrier(this-writes→this-reads) → load+compute (`simdgroup_multiply_accumulate`, `:9671`)**. Two
barriers surround a *single* buffered `sa`/`sb` pair — safety comes from bracketing the reuse window
on both ends, not from having a second copy to write into while the first is still being read.

(There is a third barrier call-site, `:9576`, but it is in the mutually-exclusive `if` branch that
Q4_K's `is_same` check never selects — see §0. It is not part of the Q4_K per-iteration count.)

### Ours (rendered `/tmp/m1d_metal_source.c`)

```
 87	  for (int Ridx0 = 0; Ridx0 < 128; Ridx0++) {
...
147	    *((threadgroup __attribute__((aligned(16))) half4*)((buf1+(alu3+4)))) = half4(val10,val11,val12,val13);
...          [8 producer stores total, lines 147-156]
156	    *((threadgroup __attribute__((aligned(16))) half4*)((buf1+alu3))) = half4(val38,val7,val8,val9);
157	    threadgroup_barrier(mem_flags::mem_threadgroup);
158	    half val39 = (*(buf1+(alu5+1)));
...          [consumer reads begin, lines 158-237]
```

`grep -n "threadgroup_barrier\|simdgroup_barrier" /tmp/m1d_metal_source.c` returns **exactly one
hit, line 157**, in the entire 496-line kernel. It sits after the store phase (`:147–156`) and before
the read phase (`:158+`) — structurally **llama's barrier #2 only**. Llama's barrier #1 (previous
reads → this iteration's writes) is **absent**: nothing separates this iteration's reads (`:158–237`)
from the *next* trip through the `for (Ridx0...)` loop's writes (`:147–156` again, same `buf1`
addresses, since `alu3`/`alu5`/`alu4` depend only on `lidx0`/`lidx1`, not `Ridx0`).

This traces to a specific, already-identified source location, confirmed by direct read:

```python
# tinygrad/codegen/opt/kernel_lds.py:656-659
  producer = UOp.group(*stores)
  barrier = UOp.barrier(producer)
  wave_m, wave_n, lane = threads.wave_m, threads.wave_n, threads.lane
  ordered = allocation.after(barrier)
```

`UOp.barrier(producer)` is built from `producer` (the store group) alone — it can only ever order
stores→loads. There is no complementary barrier built from the fragment-load `UOp`s that would order
loads→next-stores. And the call site that reaches this function forces `pipeline_plan=None`
unconditionally:

```python
# tinygrad/codegen/opt/postrange.py:588-592
              elif not register_mode:
                stage = build_precontract_lds_stage(candidate_geometry, tc=tc, allocation=allocation, operands=operands,
                  threads=thread_axes,k_axis=PrecontractKAxis(outer_k,k_substep,outer_k*candidate_geometry.tile[2],k_substep),
                  subtile_m=subtile_m,subtile_n=subtile_n,contracts=tuple(contracts),pipeline_plan=None,
                  lds_bank_dwords=self.ren.lds_bank_dwords,lds_bank_cycle_lanes=self.ren.lds_bank_cycle_lanes)
```

which forces `slot_base = UOp.const(dtypes.weakint, 0)` for every iteration
(`kernel_lds.py:634`), i.e. the same physical LDS bytes are reused every trip with no rotation.

**Verdict: different, and this is BUG A's exact mechanism, now confirmed in the actual emitted
Metal source, not just in the Python UOp construction.** llama: 2 barriers/iteration, bracketing a
single-buffered window on both ends. Ours: 1 barrier/iteration, bracketing only the write→read half;
the read→next-write half is unenforced.

---

## 2. Threadgroup memory layout

### llama

```c
// ggml-metal.metal:9522-9523
    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);
```

For the Q4_K instantiation `S0=S1=half` (2 bytes). Host-side sizing
(`ggml-metal-device.cpp:737-740`):

```c
        res.nr0 = 64;
        res.nr1 = 32;

        res.smem = bc_out ? 8192 : (4096 + 2048);
```

`sa` occupies bytes `[0, 4096)` = `NR0(64) * NK(32) = 2048` half elements. `sb` occupies bytes
`[4096, 4096+2048=6144)` = `NR1(32) * NK(32) = 1024` half elements. Both are indexed by
`64*ib + 8*ly + lx` (`:9612`, `:9628`/`:9641`) — a stride tied only to the fixed 8×8 simdgroup-tile
shape and the compile-time constants `NR0`/`NR1`/`NK`, never to a runtime "which buffer copy" value.
**Single-buffered**: one `sa` region and one `sb` region, reused unrotated every `loop_k` iteration —
safety comes entirely from the two barriers in §1, not from having a second physical copy. (After the
K-loop, the *same* `shmem` bytes are reinterpreted for bounds-checked output staging,
`:9690-9698`, sized up to 8192 bytes when `bc_out` — that reuse is gated by its own barrier pair,
`:9690` and `:9698`, and is irrelevant to the K-loop hazard.)

### Ours

```c
 16	  threadgroup __attribute__((aligned(16))) half buf1[12800];
```

One flat allocation, `12800` half elements = `25600` bytes. This traces exactly to
`scratchpad/m1d_confirm_c_fragment.py:51-53`'s schedule construction for this geometry (`tm=256,
tn=64`):

```python
  a_end, b_end = g["tm"] * 80, (g["tm"] + g["tn"]) * 80   # 256*80=20480, (256+64)*80=25600
  schedule["lds"]["windows"] = {"a": [0, a_end], "b": [a_end, b_end]}
  schedule["lds"]["strides"] = {"a": 80, "b": 80}
```

`a_end=20480` bytes `= 10240` half elements, `b_end=25600` bytes `= 12800` half elements — exactly
`buf1`'s declared size. So, like llama, this is **one physical allocation split into an "A" window
(`[0,10240)`) and a "B" window (`[10240,12800)`)** — same two-region-in-one-buffer idea as
`sa`/`shmem+4096`-style `sb`. The producer writes (`:147-156`) land in `[alu3, alu3+10244]` with the
last two stores (`:154-155`) at offset `alu3+10240`, i.e. exactly the start of the B window — a
correctly-computed producer side. **This buffer is single, not double**: there is exactly one
`buf1[12800]`, no second array, and (per `kernel_lds.py:626-630`, quoted in §1) the allocation is
sized to `geometry.lds_windows[-1].end` — one slot's worth — precisely because `pipeline_plan is
None`.

**Where the OOB reads actually come from — traced past what the task's BUG B summary states.**
Reading every `buf1`-touching line in the render (`grep -n buf1 /tmp/m1d_metal_source.c`, full
enumeration, 61 lines) shows the read side is not simply "a phantom second copy of the whole buffer
at `+12800`" — it is a **stride mismatch in one specific index term**:

- Producer (`:20`): `int alu3 = ((lidx1*320)+(alu0*40)+alu2);` — the `lidx1` (range `0..7`) term's
  coefficient is **320**.
- Consumer, "A"-fragment read (`:22,158-189`): `int alu5 = ((lidx1*2560)+alu4);` — the same `lidx1`
  term's coefficient is **2560** — exactly **8×** the producer's 320, and exactly equal to the "B"
  window's size (`b_end - a_end = 12800 - 10240 = 2560` half elements, both directly computable from
  `m1d_confirm_c_fragment.py:51-52` above).

Because the read-side `lidx1` coefficient is 8× too large, `alu5`'s value for `lidx1 ∈ {5,6,7}`
(`5*2560=12800`) already equals or exceeds `buf1`'s declared size *before* any of the per-read literal
offsets (`0..1945`, `:158-189`) are added — i.e. for 3 of 8 `lidx1` values (96 of 256 threads), the
"A"-fragment read is entirely or partially out of bounds. Its maximum address, `7*2560 + 600(alu4
max) + 1945(largest literal) = 20465`, is exactly the "20465" this repository's own
`scratchpad/mb0_lds_coverage.py` full-grid enumeration already reported (per commit `92a2f0fdd`'s
message). A second, disjoint read block (`:190-221`, `half val71 = (*(buf1+(alu4+12800)));` and 31
further reads with the same `{0,1,8,9,16,17,24,25}`-sub-pattern at bases `12800,13440,14080,14720`)
is **unconditionally** past the buffer for every thread, since its base literal `12800` already equals
`buf1`'s declared size regardless of `alu4 ∈ [0,600]`. This second block's sub-offset pattern
(stride 640 between four groups of 8, `{0,1,8,9,16,17,24,25}` within each group) is structurally
identical to the legitimate `alu5`-based read block, just anchored at `+12800` instead of at
`lidx1*2560`.

I did **not** trace which specific line of `kernel_lds.py`/`postrange.py` synthesizes the `2560`
(`= b_end - a_end`, the B-window's size) as a per-`lidx1` multiplier in the A-fragment's index
expression — that requires re-running the `sys.settrace` capture from `m1d_confirm_c_fragment.py`
against the UOp construction for this specific index term, which is outside this task's "structural
comparison" scope. What I can state, grounded in the two literal numbers above, is: **the OOB
read is not "an entire second `buf1`-sized buffer" in the naive sense; it is one index term whose
coefficient equals a *window size* (2560, the B window) where the corresponding producer-side term
uses a *tile-shape-derived row stride* (320, `= 8 rows × 40 elements/row`, both directly attested by
`schedule["lds"]["strides"]["a"]=80` bytes `/2=40` elements from the same script lines).**

**Verdict: same idea (one physical buffer, two named windows, single-buffered), different in the
detail that actually breaks: llama's index arithmetic for both `sa` and `sb` is built entirely from
compile-time tile-shape constants (`64`, `NR0`, `NR1`, `NK`) with no dependency on a runtime
window-size value; ours has at least one index term (the A-fragment read's `lidx1` coefficient) equal
to a *different window's byte size* rather than to the tile-shape row stride the matching producer
term uses. This is consistent with — but, on this compile-only reading, not proof of — the "address
formula written for a second/other window that `bc=1` never allocated" hypothesis already recorded
for BUG B; it additionally pins down that the specific quantity being misapplied is the B window's
byte span, not an arbitrary constant.**

---

## 3. Tile shape, threads, simdgroups, accumulator decomposition

### llama

- M/N/K tile per threadgroup: `NR0=64` (rows of `src0`/weight, `ggml-metal.metal:9525`), `NR1=32`
  (columns of `src1`/activation, `:9526`), `NK=32` (K per iteration, `:9528`).
- Threads per threadgroup: dispatched as `(32, nsg, 1)` (`ggml-metal-ops.cpp:2212`), with
  `nsg = N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y = 2*2 = 4`
  (`ggml-metal-device.cpp:743`, constants `ggml-metal-impl.h:14-15`) → **128 threads**
  (`tiitg ∈ [0,128)`).
- Simdgroups per threadgroup: **4** (`sgitg ∈ [0,4)`), arranged as a 2×2 grid over the 64×32 output
  tile (`ggml-metal.metal:9682-9683`: `r0 + 32*(sgitg&1)`, `r1 + 16*(sgitg>>1)`) — each simdgroup owns
  a 32×16 sub-tile.
- Accumulator: `simdgroup_float8x8 mc[8]` per thread (`:9567`) — 8 accumulator tiles of 8×8 per
  simdgroup, indexed `i%4` (4 positions along the 32-wide row dimension) × `i/4` (2 positions along
  the 16-wide column dimension) at store time (`:9686`).

### Ours

Rendered kernel header:
```c
 10	kernel void r_2_192_32_8_2_8_4_128_4_...(device half* data0_6291456, device half* data1_2097152, device uint* data2_7077888, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
 11	  int gidx0 = gid.x; /* 192 */
 12	  int gidx1 = gid.y; /* 2 */
 13	  int lidx0 = lid.x; /* 32 */
 14	  int lidx1 = lid.y; /* 8 */
```

Threads per threadgroup: `32 * 8 = 256` (`lidx0 ∈ [0,32)`, `lidx1 ∈ [0,8)`) → simdgroups per
threadgroup `256/32 = 8`. Tile per threadgroup: `M=256, N=64, K=32` — this is exactly `GEOMETRY =
(tm=256, tn=64, tk=32, wm=8, wn=1, bc=1)` from the task's own parameters (`m1d_confirm_c_fragment.py:35`).
`wm=8, wn=1` matches the 8 simdgroups (`8*1=8`). Accumulator: `float buf0[64]`
(`:15`) per thread, `128` `__WMMA_8_8_8_half_float` calls per K-tile iteration (`wmma_call_count -
1 = 128`, each processing a `float2` = 2 accumulator elements), consistent with `64` accumulator
scalars `/ 2` elements-per-WMMA-call `= 32`... times 4 (unrolled `sx`-groups in the store loop,
`:398-461`, mapping 8 `wmma*` results per output group) — the accumulator is register-resident
(`buf0`), not staged through threadgroup memory during the K-loop (only re-used post-loop, `:463+`,
for the final device-memory store, with no threadgroup output-staging path analogous to llama's
`bc_out` branch visible in this render — this geometry's `nr0/nr1` apparently divide the output
exactly, matching the note already on record in
`docs/qwen3-8b-prefill-metal-numeric-failure-isolation-m1f-20260731.md`: `192*64=12288=N`,
`2*256=512=M`, dividing exactly).

**Verdict: different tile shape (ours is 4× llama's M-tile, 2× its N-tile, same K-tile 32), different
thread/simdgroup count (256 threads/8 simdgroups vs. 128 threads/4 simdgroups), same accumulator
strategy in kind (register-resident per-thread accumulator built from repeated 8×8×8
multiply-accumulate calls) but different in scale (128 WMMA calls/iteration vs. llama's `NK/8=4`
inner-loop trips × 8 `simdgroup_multiply_accumulate` calls = 32/iteration, because our tile is 8×
larger in total M×N area: `256*64=16384` vs. llama's `64*32=2048`, ratio 8, matching `128/32/... `
— consistent with a proportionally larger accumulator). This axis is a geometry/tuning choice, not
one of the two located bugs.**

---

## 4. Dequant staging: where Q4_K bytes are decoded, into what type, per-element or per-tile

### llama

```c
// ggml-metal.metal:681-697
template <typename type4x4>
void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;
    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}
```

Called once per thread per K-loop iteration at `:9594`, `dequantize_func(x, il, temp_a)`, reading
directly from `device const block_q4_K * x` (global memory) and decoding **16 elements per call**
into a **thread-local register** (`S0_4x4 temp_a`, `half4x4`). Only *after* that register is fully
populated is it copied into `sa` (threadgroup memory, `:9598-9613`). So: decode once per tile per
thread, straight from device memory into registers, then the register contents are staged into
shared memory — never re-decoded, never decoded element-at-use from shared memory.

### Ours

The dequant arithmetic is inlined directly into the store expression, computed once per K-tile
iteration from the raw packed `uint`/`uint2` values read from `data2_7077888` (the Q4_K-packed
weight buffer):

```c
142	    uint alu80 = (val4>>((uint)((((alu74+8)&3)<<3))));
143	    float alu81 = (alu73?((float)(((val0>>cast4)&63u))):((float)(((alu80&15u)|((((val2>>((uint)(((alu74&3)<<3))))&255u)>>6u)<<4u)))));
144	    float alu82 = (((float)(as_type<half>((ushort)(((ushort)((val5&65535u)))))))*alu81);
145	    float alu83 = (alu73?((float)(((val1>>cast4)&63u))):((float)((((alu80&255u)>>4u)|((((val3>>((uint)((((alu74+4)&3)<<3))))&255u)>>6u)<<4u)))));
146	    float alu84 = (((float)(as_type<half>((ushort)(((ushort)(((val5>>16u)&65535u)))))))*alu83);
154	    *((threadgroup __attribute__((aligned(16))) half4*)((buf1+(alu3+10240)))) = half4(((half)(((alu82*((float)(((val6.x>>cast3)&15u))))-alu84))),...);
```

`alu82`/`alu84` are the per-tile scale/min (analogous to llama's `dl`/`ml`), computed once from
`val0..val5` (the block's packed scale/min bytes, read at `:90-101`) and then applied inline while
constructing the `half4` store value — i.e. **decoded directly from device-memory bit-packed values
into the threadgroup-memory store expression in one step**, with no separate thread-local
`half4x4`-shaped register standing in between (the intermediate values `alu81..alu84` are scalar
`float`s, not a 4×4 register block). This differs from llama's two-stage "decode into typed register,
then copy register into shared memory" in *mechanics* (no intermediate vector register type), but is
the same in *kind*: decode happens **once per tile per thread**, not once per element at the point of
use inside the compute loop — the dequantized `half` values landing in `buf1` are read back
verbatim by the WMMA-feeding loads (`:158-237`), never re-derived from the packed bits again within
the same iteration.

**Verdict: same in kind (decode-once-per-tile, straight from device memory, before staging to
threadgroup memory — dequant is out of the compute inner loop on both sides), different in mechanics
(llama stages through a named `half4x4` thread-local register with a reusable templated function;
ours inlines the scale/min arithmetic directly into the threadgroup-store expression with named
scalar temporaries, no reusable register-typed intermediate). Neither difference bears on BUG A or
BUG B — both bugs are about LDS synchronisation/addressing, not about how or when the bits get
decoded.**

---

## 5. Bounds handling at matrix edges

### llama

- **Row-tile clamp** (`:9537-9542`): `nr0`/`nr1` are `min(remaining, NR0/NR1)`; `lr0`/`lr1` clamp a
  thread's load-row index to `nr0-1`/`nr1-1` so no thread computes an address past the matrix edge
  in the row dimension — it just redundantly re-reads the last valid row instead.
- **K-dimension guard** (`:9590`, `:9628`): only active when the `FC_mul_mm_bc_inp` function constant
  is set (`bc_inp = op->src[0]->ne[0] % 32 != 0`, `ggml-metal-device.cpp:693`) — i.e. only when K does
  not divide the K-tile evenly. The store is `loop_k + 16*il + i < args.ne00 ? *(...) : 0` — zero-fill
  past the true K extent. For this task's shape, `K=4096` is a multiple of `32`, so `bc_inp` would be
  false and the unconditional store path (`:9630-9642`) is used instead — no per-element guard needed.
- **Output bounds** (`:9679-9719`), gated by `FC_mul_mm_bc_out` (`bc_out = ne0%64!=0 || ne1%32!=0`,
  `ggml-metal-device.cpp:700-702`): when the tile divides the output exactly, direct unguarded device
  writes (`:9681-9687`); otherwise the accumulator is staged through `shmem` (reused after the K-loop)
  and copied out element-by-element up to `nr0`/`nr1` (`:9700-9717`) to avoid writing past the
  destination.

### Ours

No bounds-check branch was observed anywhere in the 496-line render: no clamp on `lidx0`/`lidx1`-derived
row indices, no conditional (`? : 0`) guard on any `buf1` store or device-memory read/store, and a
single unconditional device-memory write block at the end (`:464-495`). This is consistent with —
not proof of, since it wasn't independently re-derived here — the note already on record in
`docs/qwen3-8b-prefill-metal-numeric-failure-isolation-m1f-20260731.md` ("no bounds/edge guard
anywhere in either kernel, consistent with the tile dividing the output exactly: `192*64=12288=N`,
`2*256=512=M`") for this exact shape/geometry pair: with `M=512, N=12288, K=4096` and tile
`(256,64,32)`, every dimension divides its tile exactly, so llama's own `bc_inp`/`bc_out` machinery
would *also* select the unguarded fast paths for this specific shape.

**Verdict: cannot be compared as "different structural capability" — for this specific shape, llama's
own bounds machinery is inert (both `bc_inp` and `bc_out` would be false), so the render shows llama
and ours converging to the same "no guards" behavior for this input, not our kernel lacking a
capability llama's has for this case. Whether our renderer has *any* equivalent partial-tile guard
machinery for shapes where it would matter was not established here — out of scope for this
comparison, which only rendered the one shape given.**

---

## Summary table

| # | Property | llama.cpp (M4 / non-tensor path) | Ours (rendered, this shape/geometry) | Verdict |
|---|---|---|---|---|
| 1 | Barriers/K-iteration | 2 (`:9596` prev-reads→this-writes; `:9649` this-writes→this-reads) | 1 (`:157` this-writes→this-reads only) | **Different** — missing barrier is BUG A's exact mechanism |
| 2 | LDS layout | One `shmem`, two windows (`sa` @ 0, `sb` @ 4096B), single-buffered, index stride = compile-time tile constant (64) only | One `buf1[12800]`, two windows (A @ 0, B @ 10240 half-elem), single-buffered, one read index term's stride = a runtime *window size* (2560) not the matching row stride (320) | **Same in kind (one buffer, two windows, single-buffered); different in the detail that breaks** — traced to a stride mismatch, ties to BUG B |
| 3 | Tile/threads/simdgroups/accumulator | 64×32×32 tile, 128 threads, 4 simdgroups, `simdgroup_float8x8 mc[8]`/thread | 256×64×32 tile, 256 threads, 8 simdgroups, `float buf0[64]`/thread, 128 WMMA calls/iter | **Different** (geometry/tuning choice, not a located bug) |
| 4 | Dequant staging | Decode once/tile into thread-local `half4x4` register, then copy to shared mem | Decode once/tile inline into the threadgroup-store expression (scalar temporaries, no register-typed intermediate) | **Same in kind, different in mechanics**; not bug-relevant |
| 5 | Bounds handling | Row clamp always active; K-guard and output-guard both gated off for this exact shape (K,M,N all divide their tiles) | No guards observed anywhere in the render | **Cannot be compared as a capability gap for this shape** — llama's own guards are inert here too |

---

## What our emitted kernel would need to differ in, to match llama's structure

Stated as properties of the emitted code, not as an implementation plan:

1. **A second `threadgroup_barrier` per K-loop iteration**, positioned *before* the producer stores
   to `buf1` (i.e. between the end of one iteration's consumer-read block, `:158-237`, and the start
   of the next iteration's producer-store block, currently `:147-156`) — matching llama's barrier #1
   at `:9596`. Currently only the write→read barrier (`:157`) exists.
2. **The A-fragment consumer-read address's `lidx1` coefficient would need to equal the same
   tile-shape-derived row stride the producer side uses (320, matching `alu3`'s `lidx1*320` term)
   rather than the B window's byte span (2560, `= b_end - a_end`)** — i.e. the read-side index
   arithmetic for the A window should be built from the same window-local row-stride constant the
   write side already correctly uses, not from a size belonging to a different window.

**Does llama's layout explain BUG B's `+12800`?** Partially, and more precisely than the task's
framing states it. llama's structure shows what a *correct* single-buffered, two-window layout looks
like: both windows' index arithmetic is built purely from compile-time tile-shape constants, with no
runtime "window size" ever appearing as a per-lane stride multiplier. Comparing that against our own
render (§2) shows the `+12800` is not, on this reading, an address formula for "a second copy of the
*entire* buffer" — it is one specific read-side index term (`alu5`'s `lidx1` coefficient) that equals
the *B window's byte span* (2560) instead of the *A window's row stride* (320, the value the matching
producer-side term correctly uses). That 8×-too-large coefficient is what pushes `lidx1 ∈ {5,6,7}`
past the buffer end, and the disjoint `+12800`-based read block (`:190-221`) is a second, structurally
identical instance of the same pattern anchored one buffer-length further out. This is consistent
with — but does not, on a compile-only structural read, prove — the "address formula written for a
window that was never allocated" hypothesis already on record; it narrows *which* runtime quantity
(the B window's size) is being substituted for the value that should be there (the A window's row
stride).

## What could not be established

- The exact `tinygrad/codegen/opt/kernel_lds.py` or `postrange.py` line(s) that synthesize the `2560`
  coefficient in the rendered A-fragment read index. This would require re-running the
  `sys.settrace` capture from `m1d_confirm_c_fragment.py` against that specific UOp's construction,
  which is a follow-on diagnostic task, not a structural comparison against llama.
- Whether our renderer has *any* partial-tile bounds-guard machinery at all (§5) — this shape's own
  dimensions happen to divide every tile exactly, on both llama's and our side, so no shape that would
  exercise such a guard was rendered here.
- GGML_METAL_HAS_TENSOR's actual runtime gating value on this specific M4 — the kernel-body
  identification in §0 rests on source-level ISA usage (`simdgroup_multiply_accumulate` vs. `tensor<>`
  types), not on a queried device capability flag, since no GPU was touched for this task.
- Any performance claim. This is a structural/synchronisation comparison only; no kernel from either
  side was executed.

## Files

- Read (llama.cpp, read-only, unmodified): `ggml/src/ggml-metal/ggml-metal.metal` (lines 681-697,
  9381-9722, 10184), `ggml/src/ggml-metal/ggml-metal-device.cpp` (lines 686-744),
  `ggml/src/ggml-metal/ggml-metal-ops.cpp` (lines 2179-2213), `ggml/src/ggml-metal/ggml-metal-impl.h`
  (lines 8-15).
- Read (this repo): `tinygrad/codegen/opt/kernel_lds.py:598-670`, `tinygrad/codegen/opt/postrange.py:588-592`,
  `scratchpad/m1d_confirm_c_fragment.py` (used unmodified via its `render_one` function).
- Generated (not committed, reproducible): `/tmp/m1d_metal_source.c`, from
  `python3 -c "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'scratchpad'); import
  m1d_confirm_c_fragment as m; m.render_one('METAL')"` run from the repo root.
