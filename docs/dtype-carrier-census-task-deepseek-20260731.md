# TASK (deepseek): WC0 + WC1 — census the WMMA carrier surface, audit the failing baseline

Two scopes are queued behind this task and **both are blocked on it**:

- `docs/task_workflow/input/wmma-carrier-shape-awareness-scope-20260731.md` (WC0/WC1 are its first two packets)
- `docs/task_workflow/input/dtype-authority-decomposition-scope-20260731.md` (D1/D2/D3 — its NFC verification depends on WC1)

**This task is diagnostic-only. You produce two inventories. You change no behaviour.**

---

## §0 HARD BANS

1. ❌ **No fixes.** Not one. If you find the bug, write down where it is and stop. A "small obvious fix" here silently changes a carrier width and breaks a GPU nobody in this room owns.
2. ❌ **No conclusions about effort, difficulty, or timelines.** Report file:line and counts. Claude concludes.
3. ❌ **Do NOT grep `vec(8)` and call it a census.** The load-bearing sites are rewrites keyed on *dtype identity* — they will not contain the string `vec(8)`. If your census is a grep result, you have done the easy half and missed the point.
4. ❌ **No behaviour change, no NFC claim, no refactor.** Read-only except for the two output docs and any scratchpad probe you need.
5. ❌ **Do not revert `0e41c260d`.** It is a legitimate stage of an in-flight migration, not a defect. See §1.
6. ❌ **No new env var, flag, registry, or module.**

---

## §1 Established state — do NOT re-derive any of this

**The defect**, stated by the codebase in `docs/dtype-orthogonality-amd-validation-20260729.md:20-24`:

> The final generic `Ops.WMMA` carrier is **intentionally** still represented as `float.vec(8)`. An EXP scalarization trial exposed a real dependency: **associative symbolic rewrites currently use dtype identity** to keep a complete `(8,)` WMMA value separate from one scalar projected lane. Removing that identity before those rewrites become shape-aware produced a `(8,)` versus `()` graph mismatch. **This is the next code migration boundary; it is not an AMD runtime failure and must not be papered over with broadcasting.**

**Why it matters beyond AMD** — measured, `tc.elements_per_thread[2]` per family:

| carrier width | families |
| ---: | --- |
| **8** | `amd_rdna3`, `amd_rdna4` |
| **4** | 5 CDNA families, 6 CUDA families |
| **2** | `metal` |

`float.vec(8)` is correct for **2 of 13** registered tensor-core families.

**The failure it produces** — 72 occurrences at HEAD, **zero at `0e41c260d^`** (bisected, confirmed):

```
RuntimeError: UOp verification failed ... on Ops.MUL dtypes.float 2
[(Ops.STACK, dtypes.float.vec(8), None), (Ops.STACK, dtypes.float.vec(8), None)]
```

Reproduce in one line:
```
.venv/bin/python -m pytest test/unit/test_online_softmax_tile.py -q
```
Expect **38 failed / 49 passed** at HEAD. At `0e41c260d^` it is 26 failed / 61 passed, and none of those 26 are verification failures — they are "no AMD hardware on this Mac."

**The fact source already exists** — `tinygrad/codegen/opt/kernel_lds.py:58`:
```python
def binary_axis_count(tc, operand_idx:int) -> int:
  return int(math.log2(tc.elements_per_thread[operand_idx]))
```
`postrange.py` derives `tc_upcast_axes` the same way. **There is one existing derivation to replay, not a second to invent.** Do not propose a new registry.

**Known surfaces** (starting points, not the answer): `tinygrad/schedule/wmma/softmax.py`, `tinygrad/schedule/wmma/kernels.py`, `tinygrad/codegen/opt/postrange.py`, `tinygrad/uop/ops.py`, `tinygrad/renderer/isa/amd_attention_abi.py`.

---

## §2 WC0 — the census (your main task)

Enumerate **every** site in three categories. For each: `file:line`, category, whether the width is derivable from `tc` at that point, and whether the site is AMD-only or shared.

**Category A — the carrier spelled as a literal.** `.vec(8)`, `float.vec(8)`, `dtypes.float.vec(8)`, a bare `8` used as a carrier width, `(8,)`/`()` pairings. This is the grep-findable half.

**Category B — rewrites that use dtype identity to distinguish a complete carrier from a projected lane.** ⚠️ **This is the load-bearing category and the reason the task exists.** These are what §1's quote names. They will not contain `vec(8)`. Look for: associative/symbolic rewrite rules that compare or match on dtype; pattern matchers keyed on `dtype ==` or `dtype is`; anywhere a `(8,)` value and a `()` value are told apart by *type* rather than by *shape*. **If you find zero of these, say so explicitly and show what you searched** — that would materially shrink the follow-on work and is a real finding.

**Category C — consumers that assume a width.** `.gep(i)` over a range, Horner folds, `STACK` construction, anything iterating `range(8)`.

**Mandatory `unclassified` bucket.** Report it with its contents, never drop it.

Output: `docs/task_workflow/output/wc0-carrier-census-20260731.md`.

---

## §3 WC1 — audit the failing-test baseline

`test/unit` has ~114 failures that this campaign has diffed against all day. **Nobody has looked inside it.** That is how the 72-failure regression sat unnoticed for two days — the rule is "diff sets, never counts," and the set was inherited without audit.

Produce the inventory: every failing test id, grouped by root cause. At minimum separate:

- the 72 `UOp verification failed` carrier failures (§1)
- "no AMD hardware" / missing-tool failures — environmental, not defects
- everything else, **grouped by actual error signature, not by filename**

Output: `docs/task_workflow/output/wc1-failing-baseline-audit-20260731.md`.

This is the artifact both scopes need in order to make an NFC claim mean anything.

---

## §4 Guardrails

- **Compile-only. No GPU.** Another agent may hold the lane.
- **AMD non-regression** — run before you finish, prove unchanged:
  ```
  python3 scratchpad/pg2_amd_all_routes_rendered_source_equality.py
  ```
  Expect exactly: `0e4c2e9218a7 8e01063e3c8f ce03d94bb58a 5ced48b9fa7c b0df79b8bb58 349a2c8c521f`. You should not be able to move these — you are not changing code. If they move, **stop and report**, because something is wrong with your working tree.
- Search `__WMMA` / `simdgroup_multiply_accumulate`, **never `simdgroup_matrix`** — `MetalRenderer` never emits that string and it has caused two wrong conclusions here.
- Every number from a command you actually ran. **Three conclusions have been retracted in this campaign over unreconciled or fabricated figures.**
- Commit prefix from `[tensor] [uop] [engine] [codegen] [runtime] [nn] [docs] [test] [examples] [repo]`. Use `[docs]`. **Do not push.**

---

## §5 Deliverable + HARD STOP

Two docs, committed. Then **stop**.

Do not design the fix. Do not start WC2. Do not touch D1/D2/D3. The census determines what those become, and a census written by someone already implementing is a census bent toward the implementation.

**If Category B turns out to be a handful of rewrites rather than a diffuse pattern — stop and say so prominently.** That single finding changes the shape of everything downstream.

---

## §6 One-line job

**Find every place the code decides "this is a whole WMMA value" vs "this is one lane," write down where they are, and tell us what is actually in the 114.**
