# Metal prefill loop-body decomposition — MP0 result

Date: 2026-07-30

Status: MP0 complete. Compile-only; no GPU workload executed. Reproduces
`docs/task_workflow/input/metal-prefill-loop-body-decomposition-scope-20260730.md`'s method for Metal. Does not
name any theory (that is MP1) and does not authorize promotion to `dev`/`master`.

## Headline: this is the reconstructed production kernel, not a fallback proxy — and it is 0% tensor-core

Tracing `tinygrad/llm/prefill_routes.py::route_prefill_linear` and `tinygrad/llm/model.py` shows the exact
production compute graph Metal takes for these three GEMMs. `FULL_RESIDENT_OVERLAY` is memory-infeasible on Metal
(`metal-prefill-schedule-search-scope-20260730.md` 2.3), so `lin._pf16_w` is always `None`, so
`Transformer._build_prefill_v2_warmstart` (`tinygrad/llm/model.py:926-936`) produces an **empty** dict, so
`_WARMSTART_OPTS`/`_prefill_v2_opts` never fires on Metal (`tinygrad/codegen/opt/postrange.py::apply_opts`
falls through to the default `hand_coded_optimizations` heuristic). The only reachable production route is
therefore exactly:

```python
w = lin.weight.cast(dtypes.float16)            # lazy GGUF dequant graph, still lazy
out = x.cast(dtypes.float16).linear(w.transpose(), bias)
```

This probe reconstructs that graph directly — `ggml_data_to_tensor(raw_bytes, n, ggml_type)` (type 12 = Q4_K for
gate/up/qkv, type 14 = Q6_K for down, matching llama.cpp's Q4_K_M mix which keeps `ffn_down`/`attn_v` at Q6_K) —
with uninitialized backing tensors (no checkpoint file needed; only shapes/dtypes affect generated code), then
renders it through the identical `to_program` + default-heuristic pipeline. **The rendered kernel name's
shape/opt-encoding prefix matches the profiled production kernel name exactly, digit for digit, for all three
kernels** (verified at run time by `scratchpad/metal_prefill_loop_body_probe.py`, not asserted by hand):

| role | shape (m,k,n) | quant | rendered name | profiled prefix | match |
| --- | --- | --- | --- | --- | :---: |
| gate/up | 512, 4096, 12288 | Q4_K | `r_16_256_8_16_4_3_16_4_2_8_4_b764c12e...` | `r_16_256_8_16_4_3_16_4_2_8_4` | **yes** |
| down | 512, 12288, 4096 | Q6_K | `r_16_64_8_16_4_4_48_2_2_2_16_2_5f6b8990...` | `r_16_64_8_16_4_4_48_2_2_2_16_2` | **yes** |
| qkv (q_proj) | 512, 4096, 4096 | Q4_K | `r_16_64_8_16_4_4_16_4_2_16_2_5e0fe0b9...` | `r_16_64_8_16_4_4_16_4_2_16_2` | **yes** |

A kernel name encodes the full shape *and* the full applied-opt sequence, so an 11-integer digit-for-digit match
is strong evidence of AST identity, not coincidental resemblance. What was **not** done: an end-to-end call
through `Transformer.__call__` with a real loaded checkpoint and a real `PrefillRouteAttachment`. That is why
this is reported as "reconstructed production kernel, name-prefix-verified" rather than "traced production
call" — but per scope 4.2 ("do not decompose a synthetic proxy unless the production AST proves irreproducible")
this is the production AST, not the weaker PR1-style fallback proxy; MP0's stop condition was not needed.

**The single largest finding, stated plainly per the task's instruction not to force it into a bucket:** none of
the three kernels contain a single `simdgroup_matrix` op, at either the MSL or the AIR level. The "useful work"
bucket (the direct Metal analogue of AMD's WMMA) is **0% in all three kernels, at both evidence levels.** The
2070/676/2183 GFLOPS these kernels run at is produced entirely by scalar dequant-unpack-and-accumulate
arithmetic into a register array (`buf0`), not by Apple's matrix units. This directly explains why
`metal-prefill-schedule-search-scope-20260730.md` 2.1's "clean fp16 GEMM ... ~3400 GFLOPS" reference (which
reaches `simdgroup_matrix`, per that scope's 2.2) sits well above the real fused kernels: the fused Q4_K/Q6_K
dequant chain, as scheduled by the default heuristic, never reaches tensor cores. There is also no
`threadgroup`-qualified memory, no `threadgroup_barrier`, no `simd_shuffle*`, and no transcendental op anywhere
in any of the three kernels, at either level — all four of those buckets are 0/0/0 across every row below.

## Method

`scratchpad/metal_prefill_loop_body_probe.py`, structural copy of `scratchpad/kv_tile_amortization_probe.py`
(same shape: compile, locate the loop body, classify against an `INST_CLASS` regex table, emit a group/count/share
table). Compile-only throughout:

1. Build the lazy dequant→fp16→matmul AST (above) for each of the three shapes.
2. Render MSL via `to_program(ast, MetalRenderer(Target.parse("METAL:METAL:Apple9")))`, extracting source from the
   `Ops.SOURCE` UOp (the TG1 technique, `test/unit/test_warp_shfl_xor_renderer_lowering.py`).
3. Locate the MSL reduce (K) loop by brace-matching from the first `for (int Ridx` line — the only real `for`
   construct MetalRenderer emits for these GEMMs (everything else is straight-line prologue/epilogue), mirroring
   the AMD probe's backward-branch loop detection at C-statement granularity instead of ISA granularity.
4. Classify every non-blank, non-brace-only statement line in that span against an MSL `INST_CLASS` table.
5. **The Metal Toolchain finished downloading mid-task** (`xcrun metal`, `metal-objdump`, `metal-nm` are now on
   PATH). Per the coordinator's instruction this is used as the *primary* evidence: write the MSL source to a
   temp `.metal` file, `xcrun metal -c` it to `.air` (compile-only — the Metal *frontend*, no GPU dispatch), then
   `xcrun metal-objdump --disassemble` it, and classify every LLVM-IR instruction in the function body against a
   second, AIR-level `INST_CLASS` table.
6. Emit both AMD-shaped tables (group, count, share) per kernel, with an explicit coverage/unclassified count at
   each level.

`scratchpad/metal_prefill_loop_body_probe.py` imports the repo's own `tinygrad/` (not a site-packages copy — this
machine has a second, stale `tinygrad` on `site-packages` that a naive `python3 script.py` invocation resolves to
instead of the repo checkout; the probe pins `sys.path[0]` to the repo root exactly as
`kv_tile_amortization_probe.py` pins its own absolute path, otherwise the rendered kernel silently comes from the
wrong tinygrad and the name-match check above fails). Rerunning the probe is deterministic: three separate
invocations from three different working directories produced byte-identical rendered sources and kernel names.

### Scope difference between the two evidence levels (read before comparing them)

The MSL table above is scoped to **the loop body only** (matching the AMD probe's methodology exactly: AMD located
the KV-tile loop span; this locates the K-reduce loop span). The AIR table is scoped to **the whole kernel
function** — LLVM preserves all four nested loop levels of these kernels as real backward-branch loops (verified
by inspecting `br i1 ..., label %N, label %M, !llvm.loop` back-edges in the disassembly), so a single clean
ISA-style "loop body span" is not extractable without per-loop-level CFG analysis beyond this probe's compile-only
budget. **The two tables are therefore not expected to reconcile 1:1 in instruction count — only in which groups
are populated at all** (they agree: `barrier`/`threadgroup_ldst`/`simdgroup_matrix`/`shuffle`/`transcendental` are
all 0 at both levels, for all three kernels). Per scope section 8: MSL is above the compiler's scaffolding — the
AIR table below is what exposes it. AIR's `phi` nodes (loop-carried SSA values across all four nested loop levels)
and its `getelementptr`/`sext`/`zext` address computations have **no MSL-source analogue at all** — they don't
exist as statements in the rendered `.metal` text, only appear once LLVM lowers the C-style loop-carried
reassignment into real SSA form. This is exactly AMD's warning (their `s_waitcnt`/`s_delay_alu` scaffolding, 22.4%
of the AMD table, invisible to a source-level view) reproduced on Metal, though the specific mechanism differs
(SSA-form loop bookkeeping, not a hardware synchronization counter — AIR sits above AGX machine code, so even this
does not show true ISA-level scaffolding such as register allocation or hardware sync; see "Known limitations"
below).

**Where the two levels disagree (scope's explicit ask, "say where they disagree"):** `select/compare (mask)`'s
share drops sharply from MSL to AIR, and `index/address math`'s share grows by almost the same amount, in all
three kernels:

| kernel | MSL select/mask -> AIR select/mask | MSL index/addr -> AIR index/addr |
| --- | ---: | ---: |
| gate/up | 34.4% -> 4.5% | 5.7% -> 41.6% |
| down | 6.0% -> 1.9% | 13.0% -> 41.5% |
| qkv | 39.3% -> 5.6% | 6.4% -> 40.5% |

This is consistent with (not directly observed as) LLVM resolving many of the MSL-level ternaries -- the
Q4_K/Q6_K sub-block-selection conditionals, most of whose conditions are loop-index-derived rather than
data-derived -- into direct `getelementptr`/`phi` address computation rather than a runtime `select`/branch. If
that reading is right, a meaningful fraction of what the MSL table calls "masking" is address arithmetic in
disguise once compiled further; this doc reports the shift as observed and leaves the interpretation open rather
than asserting the LLVM internals as fact.

## Results

### gate/up (`r_16_256_8_16_4_3_16_4_2_8_4`, 36.6% of prefill time, profiled 2070 GFLOPS)

MSL, loop body = source lines 56–216 (157 statements):

| group | statements | share |
| --- | ---: | ---: |
| other arithmetic (dequant unpack + accumulate FMA) | 66 | 42.0% |
| select/compare (mask) | 54 | 34.4% |
| global load | 28 | 17.8% |
| index/address math | 9 | 5.7% |
| threadgroup load/store | 0 | 0.0% |
| barrier | 0 | 0.0% |
| **simdgroup-matrix (the only useful work)** | **0** | **0.0%** |
| cross-lane shuffle | 0 | 0.0% |
| transcendental | 0 | 0.0% |
| unclassified | 0 | 0.0% |
| total | 157 | |

AIR (whole kernel, 942 instructions):

| group | instrs | share |
| --- | ---: | ---: |
| other arithmetic | 453 | 48.1% |
| index/address math (incl. `phi`, `getelementptr`) | 392 | 41.6% |
| global load/store | 55 | 5.8% |
| select/compare (mask) | 42 | 4.5% |
| threadgroup load/store | 0 | 0.0% |
| barrier | 0 | 0.0% |
| **simdgroup-matrix** | **0** | **0.0%** |
| cross-lane shuffle | 0 | 0.0% |
| transcendental | 0 | 0.0% |
| unclassified | 0 | 0.0% |
| total | 942 | |

### down (`r_16_64_8_16_4_4_48_2_2_2_16_2`, 30.5%, profiled 676 GFLOPS)

MSL, loop body = source lines 40–144 (100 statements):

| group | statements | share |
| --- | ---: | ---: |
| other arithmetic | 49 | 49.0% |
| global load | 32 | 32.0% |
| index/address math | 13 | 13.0% |
| select/compare (mask) | 6 | 6.0% |
| threadgroup load/store | 0 | 0.0% |
| barrier | 0 | 0.0% |
| **simdgroup-matrix** | **0** | **0.0%** |
| cross-lane shuffle | 0 | 0.0% |
| transcendental | 0 | 0.0% |
| unclassified | 0 | 0.0% |
| total | 100 | |

AIR (whole kernel, 687 instructions):

| group | instrs | share |
| --- | ---: | ---: |
| other arithmetic | 353 | 51.4% |
| index/address math | 285 | 41.5% |
| global load/store | 36 | 5.2% |
| select/compare (mask) | 13 | 1.9% |
| threadgroup load/store | 0 | 0.0% |
| barrier | 0 | 0.0% |
| **simdgroup-matrix** | **0** | **0.0%** |
| cross-lane shuffle | 0 | 0.0% |
| transcendental | 0 | 0.0% |
| unclassified | 0 | 0.0% |
| total | 687 | |

### qkv / q_proj (`r_16_64_8_16_4_4_16_4_2_16_2`, 11.7%, profiled 2183 GFLOPS)

MSL, loop body = source lines 41–184 (140 statements):

| group | statements | share |
| --- | ---: | ---: |
| select/compare (mask) | 55 | 39.3% |
| other arithmetic | 48 | 34.3% |
| global load | 28 | 20.0% |
| index/address math | 9 | 6.4% |
| threadgroup load/store | 0 | 0.0% |
| barrier | 0 | 0.0% |
| **simdgroup-matrix** | **0** | **0.0%** |
| cross-lane shuffle | 0 | 0.0% |
| transcendental | 0 | 0.0% |
| unclassified | 0 | 0.0% |
| total | 140 | |

AIR (whole kernel, 804 instructions):

| group | instrs | share |
| --- | ---: | ---: |
| other arithmetic | 381 | 47.4% |
| index/address math | 326 | 40.5% |
| global load/store | 52 | 6.5% |
| select/compare (mask) | 45 | 5.6% |
| threadgroup load/store | 0 | 0.0% |
| barrier | 0 | 0.0% |
| **simdgroup-matrix** | **0** | **0.0%** |
| cross-lane shuffle | 0 | 0.0% |
| transcendental | 0 | 0.0% |
| unclassified | 0 | 0.0% |
| total | 804 | |

## Coverage statement (scope 5.3, mandatory)

**100% of every emitted statement/instruction, at both levels, in all three kernels, was classified.
`unclassified` is 0/157, 0/100, 0/140 at MSL and 0/942, 0/687, 0/804 at AIR.** This is not silent dropping: the
regex table's last entry before `unclassified` is a broad catch-all (`other_arith` at MSL: any remaining `=`
statement; `other_arith` at AIR: the standard LLVM arithmetic/cast/call opcode set), the same design AMD's own
`INST_CLASS` used (`valu`/`salu` as the final catch-alls before its own implicit `other`). The catch-all's
contents are reported honestly rather than hidden behind a misleadingly-specific label:

- **MSL "other arithmetic"** (66/49/48 statements) is composed of exactly: the Q4_K/Q6_K scale (`d`/`dmin`)
  float computations, the per-nibble dequant-to-`half` casts, and the `buf0[i] += val * cast` accumulate
  statements — i.e., this bucket *is* the GEMM's actual multiply-accumulate work, expressed as scalar FMA rather
  than `simdgroup_matrix` calls, because no tensor-core op is emitted at all (see Headline).
- **MSL "select/compare (mask)"** is dominated by ternary-gated conditional loads/unpacks (Q4_K/Q6_K sub-block
  selection, e.g. `alu49 ? high_nibble : low_nibble`) plus a handful of plain boolean loop-position comparisons —
  the same precedence AMD gave `v_cndmask`-gated loads (classified as mask, not load).
- **AIR "index/address math"** is dominated by `phi` (loop-carried accumulator/index values across all four
  nested loop levels — no MSL analogue) and `getelementptr` (pointer-offset computation for every load/store).

## Production vs. proxy — explicit statement

**This is the reconstructed production kernel** (see Headline), verified by an exact 11-integer kernel-name-prefix
match for all three target kernels, not the weaker "closest synthetic that only reproduces the name prefix"
fallback that MP0's stop condition allows for. No fallback proxy was needed.

## AIR availability

The Metal Toolchain (`xcrun metal`, `metal-objdump`, `metal-nm`) finished downloading during this task and was
used as the primary/upgraded evidence level per the coordinator's instruction, alongside the MSL table (not
instead of it). Both tables are reported above for all three kernels; see "Scope difference between the two
evidence levels" for where and why they are not directly instruction-count-comparable, and where they agree
(every zero-count group agrees at both levels, for all three kernels).

## Known limitations

- **AIR is IR, not machine code.** It sits above Apple's AGX machine-code backend, so — like the MSL table, only
  less severely — it still cannot show true hardware scaffolding (register allocation, any hardware
  synchronization equivalent to AMD's `s_waitcnt`/`s_delay_alu`, which was AMD's single largest group at 22.4%).
  Nothing in this doc claims those are small or absent; they are simply not observable at either evidence level
  available on this machine. Do not compare total instruction counts between the AMD table and either Metal
  table above — they are measured at different levels of the compilation stack.
- The `select_mask`/`other_arith` MSL buckets are statement counts, not dynamic instruction counts; MSL statements
  do not have a fixed instruction cost (e.g. a ternary compiles to more than one AGX instruction). Share-of-source
  is not share-of-cycles.
- Only `q_proj` was measured for the qkv role; `k_proj`/`v_proj` (out_features=1024) were not separately
  rendered — the profiled `r_16_64_8_16_4_4_16_4_2_16_2` kernel matches the q_proj shape (4096×4096) exactly, and
  k/v (1024×4096, smaller) are not the kernel this profile entry refers to.
- No theory is named here. Per scope section 5/rule 4 ("no theory without a table"), that is MP1's job, informed
  by this table — in particular, the 0% `simdgroup-matrix` finding above and the "other arithmetic"/"select/mask"
  dominance are exactly the kind of evidence MP1 must react to, not conclusions this doc draws for it.
- `test/unit` carries ~114 pre-existing failures unrelated to this work; none were run or touched here.
