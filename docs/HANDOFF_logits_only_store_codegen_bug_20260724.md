# >>> RESOLVED (2026-07-24, commit `ee2fa89c6`) <<<
#
# Fixed. The sharpened "reduce_acc_upcast_fix fail-close" guess below was WRONG (verified on
# GPU). REAL root cause: the **deferred-reduce output projection** for the full 512×151936
# vocab reduce lowered to `STORE(STACK(GEP(LOAD(INDEX(out,addr))) doubled), values)` — an
# UPCAST'd size-2 inner reduce axis mapped 32 value lanes onto 16 distinct output addresses
# (each duplicated), rendered as a `make_floatN(...)` lvalue. `reduce_acc_upcast_fix` correctly
# DECLINES it (not a manual accumulator). The bare-`LOAD(INDEX)` output-projection restoration at
# `codegen/__init__.py:235-244` only matches non-wide, non-doubled lanes and skipped this shape.
# FIX: sibling rule `_devec_output_projection_store` in `pm_distinct_reg_store_devec`
# (`tinygrad/codegen/late/reg_store.py`) restores addressable per-address global stores and
# ADD-reduces each duplicate group. Fail-closed. Validated on gfx1100: `--logits-only` full-logits
# argmax=198 == with-argmax ground truth; shipped path unchanged (8B SDPA==FUSED==198); canonical
# harness compiles+runs. Regression test `test/unit/test_logits_only_reg_store.py`. The devectorizer
# was also modularized first (`e9677b161`, NFC) — `reduce_acc_upcast_fix`/reg-store devec now live
# in `reg_store.py`, so the file/line pointers below that say `devectorizer.py` are pre-refactor.
# >>> END RESOLVED — original (partially-misattributed) handoff follows for the record <<<

# Handoff: `--logits-only` prefill emits an unassignable vector store (CompileError) — 2026-07-24

Scoped for another Claude to fix. Self-contained: symptom, exact repro, root cause with the
offending source, what is ruled out, blast radius, and a concrete fix direction with file/line
pointers. No GPU needed to understand it; a GPU (gfx1100) is needed to repro + verify.

## TL;DR

The prefill harness's `--logits-only` measurement path (`extra/qk/prefill/prefill_whole_synced.py`)
fails to compile. The HIP renderer emits a C++ statement whose **left-hand side is a
`make_float32(...)` constructor call** — an rvalue — so gfx1100's compiler rejects it:

```
/tmp/comgr-*/input/<null>:180:273: error: expression is not assignable
tinygrad.device.CompileError: compile failed
```

This is a **codegen / store-lowering bug**, NOT a fused-attention or graph-GEMM bug, and NOT a
regression in the shipped route. It is a **BoltBeam (fork) bug, not an upstream tinygrad bug** — see
"Attribution" below. The fix lives entirely in this repo; there is no upstream dependency. The normal (with-argmax) prefill path compiles and runs fine, so
the benchmark numbers and shipped generation are unaffected. Only the `--logits-only` timing mode
is blocked.

## Exact reproduction

```bash
cd /home/ubuntu/tinygrad-arkey
bash extra/qk/gpu_wait_clear.sh 14 60 5          # wait for >=14GB free
timeout 240 env PYTHONPATH=. DEV=AMD DEBUG=5 \
  .venv/bin/python extra/qk/prefill/prefill_whole_synced.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --mode smoke --logits-only --no-artifact
# -> CompileError: compile failed  (the make_float32(...) = ... line is echoed by comgr)
```

**Minimal delta that isolates the trigger** (both on the same tree, same env):

| run | flag | result |
|-----|------|--------|
| `--mode smoke` | (no `--logits-only`) | **exit 0**, compiles + runs (254 tok/s cold) |
| `--mode smoke --logits-only` | `--logits-only` | **CompileError** |

So `--logits-only` is the sole trigger. `TINYGRAD_PREFILL_PACKED_WMMA=0` is **incidental** (8B is
graph-GEMM; the bug reproduces regardless). Off-profile `--start-positions/--whole-lengths` combos
hit the *same* error for the same reason — do not chase those; `--logits-only` is the root switch.

## The failing kernel (identified)

Kernel name: `r_16_2374_8_16_4_4_16_2_2_2_16_2_<hash>` (an `r_` = reduce kernel).
Buffer signature: `void ...(float* data0_77791232, float* data1_2097152, float* data2_512,
unsigned char* data3_16384, unsigned char* data4_510504960)`.

- `data0` has **77,791,232 elements = 512 tokens × 151936 vocab** → this is the **LM-head logits
  projection** (full sequence logits).
- `data4` (~510 MB) is the quantized `output`/lm_head weight; `data2_512` is a norm/scale vector.

This kernel **only exists in `--logits-only` mode**. With argmax, the final last-token slice /
argmax collapses the vocab reduce before this fusion is formed, so the offending store never
materializes. `--logits-only` keeps the full `512 × 151936` logits, and the **final RMSNorm-weight
multiply gets fused into the vocab reduce**, producing the bad store.

## The offending generated line (root symptom)

```c
make_float32(val43.x,val43.x,val43.y,val43.y, ... ,val42.w,val42.w)     // <-- LHS: a constructor CALL
  = make_float32(((*(buf0+0))*val44.x), ((*(buf0+1))*val44.x), ... , ((*(buf0+31))*val44.w));
```

Two tells:

1. **LHS is `make_float32(...)`** — a 32-wide vector *constructor* (defined at the top of the
   kernel as `static inline float32 make_float32(...){ return {...}; }`). Assigning to a returned
   struct is not an lvalue → `expression is not assignable`.
2. **LHS components are doubled**: `val43.x,val43.x,val43.y,val43.y,...`. That doubling is a
   **broadcast / stride-0 EXPAND** (each source lane written to two destination lanes). The store's
   *destination* was built out of a VECTORIZE/EXPAND of existing registers instead of an
   addressable location.

The RHS is `buf0[i] * val44.{x,y,z,w}` — a small constant buffer (`buf0`, the **RMSNorm weight**)
multiplied by a broadcast of a float4 (`val44`, the normed hidden). i.e. the fused
`norm_weight * normed_hidden` feeding the vocab projection.

## Root cause (mechanism)

The renderer's STORE rule is unconditional:

- `tinygrad/renderer/cstyle.py:84`
  `(UPat(Ops.STORE, src=(UPat.var('bidx'), UPat.var("var"))), lambda ctx,bidx,var: f"{ctx.render_access(bidx)} = {ctx[var]};")`

It emits `render_access(dest) = value;`. For a normal store, `dest` is an `Ops.INDEX` into a real
buffer and `render_access` yields `buf[idx]` (an lvalue). **Here `dest` resolved to a REG-addrspace
VECTORIZE of register components**, and `render_index`'s REG path
(`tinygrad/renderer/cstyle.py:264-269`, `if buf.addrspace == AddrSpace.REG and buf.op not in
{Ops.AFTER, Ops.BUFFER}: ...`) rendered it as the `make_float32(...)` constructor — an rvalue.

So the real bug is upstream, in **store lowering / devectorization**: a `STORE` whose destination
is a broadcast/expand of registers should have been **materialized into addressable
(scalar or GEP) stores** before render, and wasn't. The renderer is just the messenger.

## Where to fix (ranked pointers)

1. **`tinygrad/codegen/late/reg_store.py` → `pm_reg_store_devec`** (centralized repo-custom register-store devectorizer,
   wired in via `tinygrad/codegen/experimental.py:15`). This pass exists specifically to lower
   register-vector stores into distinct/scalar stores. It is the prime suspect: it does not handle
   the **broadcast/doubled-lane (stride-0 EXPAND) destination** case, leaving a VECTORIZE as the
   store target. Fix here first.
2. **`tinygrad/codegen/late/devectorizer.py`** — `pm_distinct_reg_store_devec`, `correct_load_store`,
   `devectorize_buf_and_index` (imported at `tinygrad/codegen/__init__.py:21-22`). The generic
   reg-store devectorization path; confirm whether the broadcast case is meant to be handled here vs
   the repo-custom pass, and that the two compose.
3. **`tinygrad/codegen/__init__.py:158-175`** — the STORE-with-`Ops.UNROLL`-in-value substitution
   (`store.src[-1]` unrolls → `indexed_unroll`). The doubled components look like an UNROLL/EXPAND
   with a **broadcast axis (`arg` present but stride 0)** that this substitution didn't fully index,
   so the destination kept a vector shape.

Correct behavior: a STORE destination must always be addressable — an `INDEX` into a real
(GLOBAL/LOCAL/REG-scalar) buffer. A VECTORIZE/EXPAND feeding the *destination* of a STORE is the
invariant violation to assert on and to lower.

## Suggested approach

1. Add a **codegen-level assertion** (cheap, pure-CPU): after the late devectorizer, walk the sink
   and fail if any `Ops.STORE`'s destination (`src[0]`) is an `Ops.VECTORIZE`/`Ops.EXPAND` (or a
   REG-addrspace INDEX whose index is non-scalar). That converts the opaque HIP CompileError into a
   precise, located tinygrad error and pins whether pass (1) or (3) is the miss.
2. Build a **tiny pure-python repro** (no model, fast, CPU-renderable): a matmul whose input is a
   broadcasted per-channel scale (`weight[j] * x_broadcast`) reduced over a large output dim, forced
   through the reg-store devectorizer, rendered to source. Assert the rendered source contains no
   `make_floatN(...) =` LHS. Put it under `extra/qk/` next to `reg_store_devec.py` as the regression
   test.
3. Fix the devectorizer so the broadcast/doubled-lane destination is expanded to addressable stores;
   re-run the assertion repro, then the GPU repro above (must reach exit 0), then a real
   `--logits-only` authority run for parity.

## Blast radius / severity

- **Blocks:** the `--logits-only` prefill timing path only (pure-logits measurement, no
  sampling/argmax). This is the *more correct* way to time prefill (you don't sample during
  prefill), so it is worth fixing — it currently prevents clean per-kernel attribution of the
  long-context "tax."
- **Does NOT block:** normal generation/decode, the shipped fused-attention route, graph-GEMM, or the
  published 8B benchmark table (all use the with-argmax path, which compiles). No correctness risk to
  shipped inference.
- **Class:** genuine tinygrad renderer/codegen invariant violation (unassignable store) that could
  bite any large reduce with a fused broadcast pre-scale, not just this kernel.

## Attribution: BoltBeam (fork), not upstream tinygrad

Confirmed by diffing the failing-path files against `upstream/master` (github.com/tinygrad/tinygrad):

- `tinygrad/codegen/__init__.py:158-175` — the STORE/UNROLL substitution block
  (`store_subs`/`indexed_unroll`) that forms the store whose destination becomes a VECTORIZE/EXPAND
  is **absent from upstream** (`git show upstream/master:tinygrad/codegen/__init__.py | grep
  indexed_unroll` → empty). Authored 100% by the repo owner under *"[codegen] bind physical attention
  lanes to output addresses"*.
- `tinygrad/codegen/late/devectorizer.py` — `+1022 / -0` vs upstream (fork-owned at this path).
- `tinygrad/renderer/cstyle.py` — heavily forked (236 ins / 254 del).
- `tinygrad/codegen/experimental.py` (loader for `extra.qk.*` passes) and
  `pm_reg_store_devec` is now centralized in `tinygrad/codegen/late/reg_store.py`; the former extra module was the
  BoltBeam-originated implementation.

So the pipeline that mints the unassignable store is BoltBeam-authored fork code, and the fix belongs
here — nothing to push upstream or wait on.

**Caveat (to fully close):** forked files prove the *pipeline* is BoltBeam's, not that stock upstream
tinygrad is immune to the same store-lowering gap. Running the pure-python repro (see "Suggested
approach") against `upstream/master` would settle whether the underlying devectorizer gap also exists
upstream. Either way BoltBeam owns the failing path as shipped.

## What is ruled out (with evidence)

1. **Not the fused-attention emitter / graph-GEMM.** The failing kernel is `r_...` (vocab reduce),
   not an attention or WMMA kernel. Removing `--logits-only` compiles the identical attention path.
2. **Not `TINYGRAD_PREFILL_PACKED_WMMA` / warmstart.** Reproduces with and without it; 8B doesn't use
   packed-WMMA.
3. **Not off-profile shapes.** The validated authority positions
   (`0,512,1024,2048,3584 × 512,1024,2048,4096`) also fail *with* `--logits-only` and pass *without*.
4. **Not a shipped-route regression.** The with-argmax path (benchmark table, 3623 tok/s pp512) is
   unaffected.

## Evidence artifacts (this session, on-box)

- Failing DEBUG=5 source (kernel name + offending line):
  `/home/ubuntu/.claude/jobs/6db6b205/tmp/tax_dbg5.log` (see line ~3734 and the kernel signature at
  ~3566). NOTE: this path is under a job tmp dir and is not durable — regenerate with the repro
  command above if it has been cleaned up.
- Passing (no-`--logits-only`) smoke run: exit 0, compiles + runs.
