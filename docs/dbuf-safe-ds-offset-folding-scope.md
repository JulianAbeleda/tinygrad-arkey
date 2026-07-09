# DBUF Safe DS Offset Folding Scope

## Big Picture

We are trying to make the generated AMD:ISA prefill GEMM follow the lean machine-code route:

```text
global_load_b128 -> ds_store_b128 -> barrier -> ds_load_b128 -> v_wmma
```

The current generated `DBUF both 4x2` route is now correctness-unblocked when LDS offsets stay materialized as VALU
address arithmetic. The wrong-result mode is specifically the legacy DS immediate fold:

```text
addr = dynamic_lds_base + static_const
ds_load/store_b128(addr, offset0=0)          # correct materialized form

addr = dynamic_lds_base
ds_load/store_b128(addr, offset0=static_const)  # currently unsafe when forced
```

So the next primitive is not a hand-asm retrace, not a B tile-key rewrite, and not scheduler tuning. The primitive is a
proof-based DS offset folder that only emits `offset0` when it is semantically identical to the materialized address.

## Current Facts

| Fact | Evidence | Status |
|---|---|---|
| Generated `DBUF both 4x2` can be correct. | `DEV=AMD:ISA`, `512x5120x5120`, `u0=4,u1=2,loc=2,unr=2`, materialized LDS offsets -> `status=ok`, about `6.6 TFLOPS`. | Proven. |
| The old failure is isolated. | Same route with `PREFILL_DBUF_LDS_CONST_IMM_UNSAFE=1` -> `WRONG rr=2.2e-01`. | Proven. |
| B tile-key/cadence is not the leading blocker. | Passing materialized-offset route uses the same B local cadence; `2x2` and `4x2` show the same B permutation pattern. | Demoted. |
| Hand/machine asm remains useful as a target shape. | It demonstrates the lean route can work, but it is not needed to identify the current failing generated primitive. | Reference only. |
| Final physical-register tracer is incomplete under remat. | Pre-isel symbolic proof passes; final analyzer reports false missing/alias windows because it keys by physical address registers. | Tooling blocker. |

## Parallel Wave Results

Current agent wave results:

| Lane | Result | Blocker |
|---|---|---|
| Lane A: proof helper | Added `LDSAddr` and `decompose_lds_index(ctx, idx, order=None)` in `tinygrad/renderer/isa/amd.py`; focused unit test passes. | None for Phase 1. |
| Lane B: tracer | `a_fragment_alias_probe.py` and `native_isa_l4_stream_probe.py` now normalize materialized const adds and DS immediates into byte windows, and separate `covered`/`alias`/`missing_store`/`out_of_bounds`/`unknown`. | Tracer-only; unknowns can recur for future unmodeled remat instructions. |
| Lane C: runtime matrix | `2x2 both`, `4x2 both`, `4x2 A-only`, and non-DBUF controls pass on the safe-current/materialized path; unsafe `4x2 both` is schedule-gate rejected. | Unsafe fold remains intentionally blocked. |
| Lane D: hand/machine spot-check | `extra/qk/prefill/wmma.py::build_gemm_lds2` uses VGPR dynamic bases and DS immediate offsets for static LDS offsets. | No fresh assembled disassembly was run; source-level builder evidence is sufficient for direction. |
| Phase 2 worker | Implemented safe DS offset folding using `LDSAddr`; `PREFILL_DBUF_LDS_CONST_IMM=1` now takes the proof path, while `PREFILL_DBUF_LDS_CONST_IMM_UNSAFE=1` remains the legacy repro path. | No correctness blocker; performance is mixed and needs Phase 5 review. |
| Phase 5 worker | Added `docs/dbuf-safe-ds-offset-folding-phase5-report.md`; repeated runs show `2x2 both` safe fold is slower despite lower instruction count, while `4x2 both` improves. | Promotion blocker: safe fold needs a selection guard or deeper DS scheduling/bank-behavior analysis. |

Phase 5 found the current blocker: safe folding is correct, but it is not an unconditional performance win. Promote it
only with a shape/route guard or after deeper DS scheduling/bank-behavior analysis.

Detailed Phase 5 report: `docs/dbuf-safe-ds-offset-folding-phase5-report.md`.

Latest post-Phase-2 runtime matrix on `DEV=AMD:ISA`, `M=512`, `out_f=in_f=5120`, `loc=2`, `unr=2`:

| Case | Status | TFLOPS | LDS bytes |
|---|---|---:|---:|
| `2x2 both`, materialized | PASS | 9.22 | 32768 |
| `2x2 both`, safe fold | PASS | 7.64 | 32768 |
| `4x2 both`, materialized | PASS | 6.59 | 49152 |
| `4x2 both`, safe fold | PASS | 6.90 | 49152 |
| `4x2 both`, unsafe fold | WRONG `rr=2.2e-01` | 0.00 | 49152 |
| `4x2 A`, materialized | PASS | 11.34 | 32768 |
| `4x2 A`, safe fold | PASS | 11.47 | 32768 |
| non-DBUF `4x2 both`, safe flag | PASS | 8.51 | 24576 |

## 100% Definition

100% for this scope means:

| Layer | Percent when complete | Required proof |
|---|---:|---|
| L0. Correctness baseline | 15% | Materialized-offset `DBUF both 4x2` passes on target shapes. |
| L1. Failure repro | 15% | Unsafe DS immediate fold still reproduces wrong result under an explicit diagnostic flag. |
| L2. Formal fold legality | 20% | Folder proves `addr + const` and `addr, offset0=const` target the same byte window for every DS op it rewrites. |
| L3. Compiler implementation | 20% | Safe folder emits `offset0` only for proven cases and falls back to materialized `V_IADD` otherwise. |
| L4. Tracer/proof tooling | 15% | Probe reports folded DS windows by byte range, not by physical address-register identity. |
| L5. Performance recovery | 10% | Correct folded route improves instruction count/register pressure/TFLOPS versus materialized baseline. |
| L6. Promotion policy | 5% | Unsafe flag remains diagnostic; safe fold becomes the only promotable path. |

Completion bar:

```text
safe-fold enabled:
  correctness PASS
  no unsafe override required
  final/proof tracer PASS or explicitly explains any inconclusive remat case
  performance >= materialized-offset baseline

unsafe-fold enabled:
  still allowed only as a repro/diagnostic path
  schedule gate rejects it for known-bad both-stage u0>2 candidates
```

## Blockers

| Blocker | What it means | Current state | Fix direction |
|---|---|---|---|
| B0. Unsafe `offset0` fold | Current legacy fold can change generated kernel semantics. | Fenced behind `PREFILL_DBUF_LDS_CONST_IMM_UNSAFE=1`. | Replace with proof-based fold. |
| B1. Offset field bounds | RDNA3 DS `offset0` is an 8-bit byte field in this encoder. | `_ds_addr_imm` bounds checks `0 <= imm <= 0xff`. | Keep as hard legality gate. |
| B2. Unit mismatch risk | Source indexes are in half elements; DS offsets are bytes. | Some proof rows report half-element constants; lowerer consumes bytes. | Make every fold proof report both element and byte units. |
| B3. Dynamic base equivalence | Need to prove folded base equals the materialized base with only the constant removed. | Not yet encoded as a reusable proof object. | Add a symbolic LDS address descriptor. |
| B4. Barrier/order preservation | Folding address math is pure, but memory op order must not move across barriers. | Current change does not reorder memory ops. | Folder must rewrite only the address operand/imm, never move DS ops. |
| B5. Remat/final tracer mismatch | Final probe sees different physical regs and reports false aliases. | Known tooling gap. | Track symbolic byte windows through `V_OFFSET`/`V_IADD`/remat, not physical reg names. |
| B6. Performance proof | Correct materialized route is slow. | About `6.6 TFLOPS` for target shape. | Fold only proven constants and measure deltas. |

## Safe Fold Contract

The fold may rewrite:

```text
base_addr = dynamic_expr
addr      = base_addr + const_bytes
ds_op(addr, offset0=0)
```

to:

```text
ds_op(base_addr, offset0=const_bytes)
```

only if all invariants hold:

| Invariant | Required check |
|---|---|
| Same LDS buffer | Both forms refer to the same `DEFINE_LOCAL` allocation. |
| Same dynamic expression | Removing the constant leaves exactly the same dynamic base expression. |
| Constant is byte-valued | The folded constant is in bytes, not half elements. |
| Field fits | `0 <= const_bytes <= 0xff`. |
| Alignment preserved | `const_bytes` is aligned for the DS op width and original access alignment. |
| No double folding | The address expression has not already had the same const removed into `offset0`. |
| Order unchanged | The DS op remains in the same dependency/order position. |
| Fallback exists | If any check fails, emit/materialize `V_IADD(addr, const)` with `offset0=0`. |

## Proposed Compiler Shape

Introduce an internal descriptor at AMD isel time:

```python
LDSAddr = {
  "buf": define_local_id,
  "dyn": symbolic_dynamic_expr,
  "const_bytes": int,
  "order": order_token,
}
```

Lowering pseudocode:

```python
def lower_lds_addr(idx):
  desc = decompose_lds_index(idx)
  addr = emit_dynamic_addr(desc.dyn)
  if safe_ds_offset_fold(desc, opcode):
    return addr, desc.const_bytes
  if desc.const_bytes != 0:
    addr = emit_v_iadd(addr, desc.const_bytes)
  return addr, 0

def safe_ds_offset_fold(desc, opcode):
  return (
    desc.buf is not None and
    desc.dyn is not None and
    is_byte_constant(desc.const_bytes) and
    0 <= desc.const_bytes <= 0xff and
    aligned(desc.const_bytes, opcode) and
    not desc.already_folded
  )
```

The current `PREFILL_DBUF_LDS_CONST_IMM` should eventually request this safe path. `PREFILL_DBUF_LDS_CONST_IMM_UNSAFE`
must remain the old repro-only path until deleted.

## Work Phases

### Phase 0: Baseline Lock

| Task | Parallel? | Done when |
|---|---|---|
| Record target materialized run. | Yes | `DBUF both 4x2`, `512x5120x5120`, `status=ok`, group segment `49152`. |
| Record unsafe repro. | Yes | Same route with `PREFILL_DBUF_LDS_CONST_IMM_UNSAFE=1` returns `WRONG rr=2.2e-01`. |
| Keep schedule gate fail-closed for unsafe mode. | Yes | Gate rejects only unsafe both `u0>2`, not safe materialized route. |

### Phase 1: Address Proof Object

| Task | Parallel? | Done when |
|---|---|---|
| Add symbolic LDS address decomposition helper. | No | It returns `(buf, dyn_expr, const_bytes)` for store and load indexes. |
| Unit-test byte/element conversion. | Yes after helper | Half-index constants are multiplied exactly once by itemsize. |
| Emit debug rows for folded candidates. | Yes after helper | Probe shows `dyn`, `const_bytes`, `opcode`, `folded`, `fallback_reason`. |

### Phase 2: Safe Folder

| Task | Parallel? | Done when |
|---|---|---|
| Implement guarded safe fold. | No | `PREFILL_DBUF_LDS_CONST_IMM=1` uses proof path, not unsafe path. |
| Keep fallback materialization. | No | All unproven or out-of-range cases compile with `offset0=0`. |
| Preserve unsafe repro flag. | Yes | `PREFILL_DBUF_LDS_CONST_IMM_UNSAFE=1` still uses legacy path for diagnostics. |

### Phase 3: Tracer Upgrade

| Task | Parallel? | Done when |
|---|---|---|
| Teach final probe DS byte windows. | Yes | `ds_load/store_b128` windows are reported as byte ranges. |
| Track remat chains. | Yes | `V_OFFSET`, `V_IADD`, constant remat, and DS imm are normalized into one address expression. |
| Split false-inconclusive from true violation. | Yes | Probe distinguishes `unknown`, `out_of_bounds`, `alias`, and `covered`. |

### Phase 4: Runtime Matrix

| Case | Required result |
|---|---|
| `2x2 both`, materialized | PASS. |
| `2x2 both`, safe fold | PASS. |
| `4x2 both`, materialized | PASS. |
| `4x2 both`, safe fold | PASS. |
| `4x2 both`, unsafe fold | FAIL or schedule-gate reject; diagnostic only. |
| `4x2 A-only`, safe fold | PASS. |
| non-DBUF `4x2 both`, safe fold if applicable | PASS or no-op. |

### Phase 5: Performance Gate

| Metric | Baseline | Target |
|---|---:|---:|
| Correct materialized target TFLOPS | about `6.6` | Safe fold must not regress. |
| Instruction count | materialized route | Lower VALU address math. |
| VGPR pressure | materialized route | Same or lower. |
| LDS bytes | `49152` for `4x2 both` | Same. |
| Correctness | PASS | PASS. |

## Parallelization Plan

These can run in parallel:

| Agent lane | Scope | Output |
|---|---|---|
| Lane A: proof helper | Implement/verify symbolic LDS address decomposition. | Patch + unit tests. |
| Lane B: tracer | Normalize final DS byte windows through remat. | Probe patch + before/after report. |
| Lane C: runtime matrix | Run materialized/safe/unsafe combinations across shapes. | Table of pass/fail/TFLOPS/LDS bytes. |
| Lane D: hand/machine comparison, optional | Compare DS address forms only, not full retrace. | Short note: which offsets hand asm folds and their byte ranges. |

These must be sequential:

```text
address proof helper -> safe folder -> runtime correctness matrix -> performance promotion
```

## Stop Conditions

Stop and re-scope if any of these happen:

| Stop condition | Meaning |
|---|---|
| Safe fold still gives `WRONG rr=2.2e-01`. | The proof missed a semantic difference; do not tune scheduler. |
| Materialized baseline regresses. | The correctness baseline was disturbed; revert/fix before continuing. |
| Final tracer cannot distinguish unknown from violation. | Do not use it as a correctness oracle; rely on runtime + pre-isel proof. |
| Performance does not improve after safe fold. | Offset folding is not the dominant remaining perf gap; move to scheduler/waitcnt/overlap. |

## Current Recommendation

Do not retrace hand asm now. The next implementation step is:

```text
Build the symbolic LDS address descriptor, then re-enable DS offset folding only through that proof path.
```

Hand asm should be used later as a spot-check for folded byte ranges after the compiler proof exists.
