# WMMA carrier shape-awareness — scope

Date: 2026-07-31

Status: scoped, not implemented. Branch boundary: tinygrad `exp`. Does not authorize promotion to
`dev`/`master`.

Companion to `docs/task_workflow/input/dtype-authority-decomposition-scope-20260731.md`, which censuses
dtype *authority* (D1 kernel ABI, D2 precision literals, D3 quant identity). **This scope covers the one
boundary that census does not reach**, and which its non-goals do not exclude: the WMMA carrier's shape.

---

## 1. The defect, stated by the codebase itself

`docs/dtype-orthogonality-amd-validation-20260729.md:20-24`:

> The final generic `Ops.WMMA` carrier is **intentionally** still represented as `float.vec(8)`. An EXP
> scalarization trial exposed a real dependency: **associative symbolic rewrites currently use dtype
> identity** to keep a complete `(8,)` WMMA value separate from one scalar projected lane. Removing that
> identity before those rewrites become shape-aware produced a `(8,)` versus `()` graph mismatch. **This
> is the next code migration boundary; it is not an AMD runtime failure and must not be papered over
> with broadcasting.**

Correctly diagnosed, and **that sentence is the only place it exists.** It has never been scoped.

---

## 2. Why it matters beyond AMD: the carrier is 8 on 2 of 13 families

`tc.elements_per_thread[2]` — the C-operand carrier width — across every registered tensor-core family:

| family | dims | `elements_per_thread` | C carrier | binary axes |
| --- | --- | --- | ---: | ---: |
| `amd_rdna3` | (16,16,16) | (16,16,8) | **8** | 3 |
| `amd_rdna4` | (16,16,16) | (8,8,8) | **8** | 3 |
| `amd_cdna3` | (16,16,32) | (8,8,4) | 4 | 2 |
| `amd_cdna4` | (16,16,128) | (32,32,4) | 4 | 2 |
| `amd_cdna_161616` | (16,16,16) | (4,4,4) | 4 | 2 |
| `amd_cdna_161632` | (16,16,32) | (8,8,4) | 4 | 2 |
| `amd_cdna_1616128` | (16,16,128) | (32,32,4) | 4 | 2 |
| `cuda_sm75` | (8,16,8) | (4,2,4) | 4 | 2 |
| `cuda_sm80` | (8,16,16) | (8,4,4) | 4 | 2 |
| `cuda_81616` | (8,16,16) | (8,4,4) | 4 | 2 |
| `cuda_81632_f8` | (8,16,32) | (16,8,4) | 4 | 2 |
| `cuda_8168_f16` | (8,16,8) | (4,2,4) | 4 | 2 |
| `cuda_8168_tf32` | (8,16,8) | (4,2,4) | 4 | 2 |
| `metal` | (8,8,8) | (2,2,2) | **2** | 1 |

**`float.vec(8)` is correct for RDNA3 and RDNA4 and wrong for the other eleven — including AMD's own
CDNA line.** This is not a Metal or CUDA accommodation; it is a representation that happens to match the
one GPU the code was written on.

---

## 3. What it currently blocks

- **72 `UOp verification failed` failures** at HEAD, zero at `0e41c260d^`. Signature:
  `Ops.MUL dtypes.float 2 [(Ops.STACK, dtypes.float.vec(8)), (Ops.STACK, dtypes.float.vec(8))]` —
  a scalar-declared consumer fed 8-wide sources, which is the `(8,)` versus `()` mismatch §1 names.
- **`test_online_softmax_tile.py`: 38 failed / 49 passed**, including both admitted attention grids.
  Plus 17 in `test_amd_attention_kv_tile_oob_guard.py`. Roughly half of the ~114 baseline failures the
  campaign diffs against.
- **The AMD fused prefill attention path does not render** — `FA-CTRL`
  (`scratchpad/fa_ctrl_amd_attention_rendered_source_equality.py`) records that failure as its current
  baseline.
- **Therefore the Metal fused attention port is blocked** at its first step
  (`docs/task_workflow/input/metal-fused-attention-port-scope-20260731.md`), because you cannot port from
  a reference that does not compile. That port is the largest measured lever on Metal: attention is
  **4.5% → 73.3%** of per-chunk prefill with depth (99% capture coverage), and prefill decays −51.1%
  against llama's −9.8%.

**Expected, not proven:** crossing this boundary should unblock the attention path, because the failure
signature is exactly the mismatch §1 describes. FA-CTRL is the falsifier — **it flips from recording a
failure to emitting hashes the moment the path renders.** If it does not, the attention breakage has a
second cause and this scope did not fix it.

---

## 4. The fact source already exists

`tinygrad/codegen/opt/kernel_lds.py:58`:

```python
def binary_axis_count(tc, operand_idx:int) -> int:
  """How many binary (size-2) upcast axes ``tc`` itself says operand ``operand_idx`` folds.
  ... Not a per-target constant: RDNA3's 16/16/8 gives 4/4/3, Metal's 2/2/2 gives 1/1/1."""
  return int(math.log2(tc.elements_per_thread[operand_idx]))
```

A shape-aware rewrite does not need a new fact source or a new registry. **It asks the descriptor.**
`postrange.py` already derives `tc_upcast_axes` the same way, so there is one existing derivation to
replay rather than a second to invent — the discipline `derive_wmma_operand_lane_layout` established.

---

## 5. Work packages

### WC0 — Exhaustive census. Compile-only. **Nothing moves until this is complete.**

Follow D2's method (`dtype-authority-decomposition-scope-20260731.md` §3.2 censused 103 precision-literal
references across 12 files before touching any).

Enumerate every site that:

1. **spells the carrier as a literal** — `.vec(8)`, `float.vec(8)`, `dtypes.float.vec(8)`, `8` used as a
   carrier width, and the `(8,)`/`()` pairing;
2. **uses dtype identity to distinguish a complete carrier from a projected lane** — the rewrites §1
   names. These are the load-bearing ones and they will not be found by grepping `vec(8)`;
3. **consumes the carrier assuming a width** — `.gep(i)` over a range, Horner folds, `STACK` construction.

Report per site: file:line, which of the three categories, whether the width is derivable from `tc` at
that point, and whether the site is on the AMD-only path or shared. **An `unclassified` bucket is
mandatory.**

Stop condition: if category 2 turns out to be a handful of rewrites rather than a diffuse pattern, say so
— that materially shrinks WC2 and should be reported before design work begins.

### WC1 — Establish the verification surface. Compile-only.

Three controls must exist and be recorded *before* any change:

- **Packed-WMMA six-row hashes** — `scratchpad/pg2_amd_all_routes_rendered_source_equality.py`:
  `0e4c2e9218a7 8e01063e3c8f ce03d94bb58a 5ced48b9fa7c b0df79b8bb58 349a2c8c521f`. These must not move.
- **FA-CTRL** — currently records a render failure; the acceptance signal is that it begins emitting
  hashes.
- **`test/unit` failing-test-id set** — 111 unique ids at HEAD. **Diff the set, never the count.** The
  regression this scope fixes was itself absorbed into that baseline for two days precisely because the
  count was watched instead of the contents. **Audit the set once, explicitly, and record what is in it.**

### WC2 — Make the rewrites shape-aware. Prerequisite: WC0, WC1.

Design follows WC0's census. Constraints, stated by §1 and by this campaign's method:

- **No broadcasting.** The validation doc forbids it by name.
- **Derive the width from `tc`**, never a literal — `binary_axis_count(tc, operand_idx)` is the existing
  derivation.
- **Replay, do not reimplement.** `postrange.py` already derives `tc_upcast_axes`; a second derivation
  will drift.
- **Fail closed** on any descriptor family the code cannot resolve, as
  `derive_wmma_operand_lane_layout` does for `amd_rdna4`/`amd_cdna3` and unexpressed dtype pairings.
- **No backend branches.** Thirteen AMD couplings were removed today with zero.

### WC3 — Verify across families. Prerequisite: WC2.

Compile-only render checks for at least one family from each carrier class — RDNA3 (8), CDNA (4),
CUDA (4), Metal (2) — proving the same source produces correct per-target carriers. This is the
deliverable that makes CUDA and Metal bring-up cheaper, and it is the reason to do this properly rather
than patch the AMD path.

---

## 6. Evidence contract

1. **AMD non-regression is structural** (no AMD hardware here). All six packed-WMMA hashes byte-identical
   after every commit; any movement stops work and is reported with the diff.
2. **Numeric behaviour must not change on AMD.** RDNA3's carrier is 8 before and after; if the rendered
   source moves, the change is not shape-awareness, it is a rewrite.
3. **Diff failing-test-id sets, never counts** — and audit the set's contents once (WC1).
4. **One defect per commit**, with a predicted signature stated in advance.
5. Every number from a command actually run. Three conclusions have been retracted in this campaign.
6. Compile-only throughout except where a packet says otherwise; GPU work serialised.

---

## 7. Non-goals

- **Not** re-doing `dtype-authority-decomposition-scope-20260731.md`'s D1/D2/D3. Complementary, not
  overlapping.
- **Not** the Metal fused attention port — that is a separate scope, unblocked by this one.
- **Not** changing any target's numeric behaviour, carrier width, or kernel output. RDNA3 stays at 8.
- **Not** forking `tinygrad/dtype.py`.
- **No** new env var, flag, registry, or module — the fact source already exists (§4).

---

## 8. Known limitations

- **No AMD hardware.** Non-regression is byte-identical rendered source only.
- **The causal claim in §3 is expected, not proven.** FA-CTRL is the falsifier.
- `0e41c260d` (2026-07-29, *"[uop] move AMD fragment lanes into descriptors"*) is where the 72 failures
  enter. Its parent renders cleanly. That commit is a legitimate stage of the dtype-orthogonality
  migration, **not a defect to revert** — the boundary it exposed is what this scope closes.
- The Stage 1 validation in `dtype-orthogonality-migration-20260729.md:72-74` classified what it saw as
  *"existing attention scheduling/barrier failures."* The bisect shows these specific verification
  failures were new with that commit. Worth reconciling with whoever owns the migration.
