# dtype authority decomposition: scope

Date: 2026-07-31 (rev 2 — re-graded against `structure/Development/coding-principles.md`)
Status: rev 3 — D1 and D3 landed on `nvidia-bringup-20260731` (D1: 91134defb/1341674af; D3: 19f033822,
53c8bdf8b, a0976d3ca, 90301f805, e37a94549); D2 executed to close the scope per explicit user direction.

Governing document: `structure/Development/coding-principles.md` + `structure/Development/tinygrad-coding-overrides.md`
(the authority for this repo). `knowledge_base/principles/codebase-organization-principles.md` is the general
backing those build rules are consistent with; where the two differ, the repo file wins.

## 1. End goal

**One-sentence reduction:** *Every dtype in the LLM layer is either read from whoever owns it — the model file,
the kernel artifact, the device — or is the single compute-precision decision; none of them is a literal restated
at a call site.*

The LLM layer contains ~103 `dtypes.*` references across 12 files. They look like one fact and are at least four,
with four different owners and four different lifetimes:

```text
storage format    -> the GGUF file        (fp16 d/dmin inside Q4_K superblocks)
kernel ABI        -> the kernel artifact  (what a searched kernel writes)
expressibility    -> the device           (renderer.supported_dtypes())
compute precision -> a decision           (the only one that is a choice)
```

Because they are written identically, the code *asserts* they are the same fact. That assertion is false, and a
reader cannot tell which occurrences they are allowed to change.

The metric is **one authoritative source per piece of knowledge**, not line count ("Reducing Code The Right Way":
*line count is not the metric, knowledge duplication is*). Three of the four categories should be **asked** of
their owner rather than restated; only the fourth needs a home. Fewer lines is a side effect, never the goal.

## 2. Why this is a design defect and not a preference

The load-bearing argument, which does not depend on any efficiency claim:

`execute_promoted_program(Tensor.empty(shape, dtype=dtypes.float32, device=...), *inputs, program=p)` requires the
**caller** to know the output layout and dtype of a kernel it did not author. The kernel is
`MACHINE_SEARCH_GENERATED`; its ABI is the artifact's property.

Two build rules name this directly:

- *"Do not rely on comments, naming, or caller discipline for rules the code can encode."* (Encode Invariants)
  The current seam is caller discipline.
- *"Move metadata to the owning value instead of modeling it as a separate graph node."* (Simplify
  Representations Before Adding Mechanisms — cited there as the upstream tinygrad pattern.) The ABI is metadata
  belonging to the program.

It currently works **by coincidence**: `describe_flash_decode_attention` (`flash_decode_attention.py:358-365`)
varies `staging`, `fused_combine`, `quant`, `rope`, `combine_stride`, `query_group_size`, `stage_width`, `Tc` —
**no dtype is in the search space**, so a re-search cannot invalidate the caller's guess. That is a property of
today's search space, stated nowhere near the call site and owned by a different module.

A design whose correctness rests on an unstated property of something else is worse designed even while it works.
**No bug is currently reachable through this seam; this scope is not justified by one and must not claim one.**

## 3. Exhaustive current-state census

### 3.1 D1 — kernel ABI restated by the caller (6 sites, 3 files)

| file:line | buffer | kernel source | spec object exists? |
| --- | --- | --- | --- |
| `decode_routes.py:71` | `Tensor.empty(binding.N, float32)` | `q4k_g3_lanemap_gemv_kernel(N, K)` (free function) | **no** |
| `decode_routes.py:114` | `Tensor.empty(binding.N, spec.partial_axis_extent, float32)` | `emit_q6k_gemv_kernel(spec)` | yes — `Q6KGEMVRouteSpec` (`decode_kernels.py:163`) |
| `decode_routes.py:119` | `Tensor.empty(binding.N, float32)` | `emit_q6k_vocab_scalar_reduce_kernel(spec)` | yes — same spec |
| `flash_decode_attention.py:499` | `Tensor.empty(Hq*S*(Hd+2), float32)` | `spec.emit_tile(Tc)` | yes — `FlashDecodeAttentionSpec` (`:341`) |
| `flash_decode_attention.py:503` | `Tensor.empty(Hq*Hd, float32)` | `spec.emit_combine()` | yes — same spec |
| `fused_attention.py:215` | `Tensor.empty(Hq*T*Hd, half)` | `fxn` + `identity` string | **no** |

Two of the six have no spec object to hang an allocator on. **This is the finding that shapes D1's design** — see
4.1. `execute_promoted_program` is declared at `kernel_program.py:55`; 10 call sites repo-wide, of which 6 are the
production ones above (the rest are `extra/llm_research/decode/`, `test/unit/test_llm_kernel_program.py`).

Note `flash_decode_attention.py:499`'s fp32 is **not** a free choice: the `+2` is the online-softmax running max
and sum, which require fp32 for stability. The defect is *who states it*, not *what it is*.

### 3.2 D2 — precision literals (103 references, 12 files)

| file | `dtypes.*` refs | expected category (UNVERIFIED — see 4.2) |
| --- | ---: | --- |
| `gguf.py` | 39 | storage format — **out of scope, must not move** |
| `decode_kernels.py` | 15 | kernel-internal UOp dtypes (ABI) |
| `model.py` | 14 | mixed: KV cache dtype, compute casts |
| `flash_decode_attention.py` | 11 | ABI + kernel-internal |
| `decode_routes.py` | 5 | compute casts + ABI |
| `fused_attention.py` | 5 | compute casts + ABI |
| `prefill_routes.py` | 3 | compute casts |
| `adapter.py` | 3 | LoRA param dtype |
| `model_facts.py`, `packed_wmma_prefill.py`, `qk_primitives.py`, `prefill_graph_gemm.py` | 2 each | mixed |

Two spellings of the same object are in use with no stated rule: `dtypes.half` (26) and `dtypes.float16` (30) are
the same interned instance (`dtype.py:200`). The split is *nearly* "C name in kernel-facing code, Python name at
tensor level" — `flash_decode_attention`/`fused_attention`/`packed_wmma_prefill` use `half` exclusively;
`gguf`/`prefill_routes`/`prefill_graph_gemm`/`decode_*` use `float16` exclusively — but `model.py` mixes both
(3 / 7), so it is a coincidence, not a convention. ("Consistency beats cleverness" — either state the rule or
collapse to one spelling.)

**The per-file category column is an expectation, not a measurement.** D2 therefore begins with a census slice
(4.2); do not act on the table.

### 3.3 D3 — quant identity is a stringly-typed status value (52 sites, 9 files)

This is a **named violation**, not a judgment call. `coding-principles.md`, Encode Invariants:
*"Prefer typed states over stringly-typed status values."*

- `packed_linear_quant` (`model_facts.py:54`) returns `""` / `"Q4_K"` / `"Q6_K"` — a stringly-typed status value —
  decided by attribute probing: `hasattr(linear, "q4k_storage")`.
- 36 bare `"Q4_K"` / `"Q6_K"` string literals across the repo.
- Re-matched through ad-hoc dicts downstream, e.g. `{"Q4_K": "q4k", "Q6_K": "q6k"}` (`prefill_routes.py:131`).
- Storage wrappers `Q4KPrimitiveStorage` / `Q6KPrimitiveStorage` (`qk_primitives.py:114-126`) carry byte
  accounting but no format identity.
- Files: `codegen/opt/packed_weight.py`, `llm/{decode_kernels, decode_routes, model, model_facts,
  model_route_plan, packed_wmma_prefill, prefill_routes, qk_primitives}.py`.

**D3 crosses out of `llm/` into `codegen/`.** It is the widest of the three and the only one touching the compiler.

Design note recorded here deliberately ("Explain Tradeoffs Close To The Code" — this belongs next to the code that
made the call, not in the knowledge base): Q4_K/Q6_K must **not** become `DType`s. They are block formats (Q4_K =
256 weights per 144-byte superblock; Q6_K = 256 per 210 bytes), so they are not element-addressable, which every
`DType` invariant assumes (`itemsize`, `nbytes()`, `.vec()`, pointer arithmetic). They are also not closed under
arithmetic — there is no `least_upper_dtype(Q4_K, float16)` — and are multi-tensor. The correct shape is a **peer**
of `DType`, not a member of `dtypes`. (llama.cpp made the opposite choice: `ggml_type` holds F16 and Q4_K in one
enum, and the type×op×backend matrix is the cited cause of its hand-kernel surface.)

## 4. Target design

### 4.1 D1 — the output descriptor belongs to `KernelProgram`

The census kills the obvious design. Per-spec `spec.alloc_output(device)` cannot cover `decode_routes.py:71` or
`fused_attention.py:215`, which have no spec object — adopting it would mean inventing two spec classes purely to
host an allocator, which is a new abstraction serving a refactor rather than the domain, and would violate the
No-new-file rule in spirit.

`KernelProgram` is the type that exists at **all six** sites and is constructed on the line adjacent to every
hand-allocated buffer. The output descriptor goes there, and `execute_*_program` allocates from it.

**Open decision — descriptor shape (two-sided; the reviewer picks):**

| option | argument | rule cited |
| --- | --- | --- |
| (a) two plain fields `output_shape`, `output_dtype` | the caller-facing surface stays maximally ordinary; no new type for a struct with no behavior | Keep Public Surfaces Boring |
| (b) an `OutputSpec` value | the pattern repeats 6× and is genuinely identical, so the abstraction is earned; one parameter instead of two, and a place for later fields | Rule of Three / AHA |

Both are defensible and the two governing rules split here. Recommendation: **(b)**, because Rule of Three is a
rule that can *fail* (you counted or you didn't), while "is this surface boring enough" cannot — and a criterion
that cannot fail is the weaker gate. Prior guidance in review said (a) on premature-abstraction grounds; that was
graded against the knowledge-base file, not this repo's, and Rule of Three supersedes it here.

`execute_promoted_program`'s `output: Tensor` positional parameter is **kept** as an accepted override for the
research/test call sites (`extra/llm_research/decode/`, `test/unit/test_llm_kernel_program.py`) so this slice does
not fan out into research code.

### 4.2 D2 — census first, then subtract

**D2.0 (census, no code change):** classify all 103 references into `{storage, abi, express, compute}`. The
row-by-row census is a **working artifact, not a doc section** — a 103-row table here would be a filing cabinet.
What lands in this document is (i) the classification rule and (ii) any reference that could not be classified,
which is a finding rather than a rounding error.

**D2.1 (subtract):** `storage` occurrences stay exactly where they are (`gguf.py` is the file format and must not
move). `abi` occurrences are absorbed by D1. `express` reads `DeviceCapabilities.supports_fp16` (published by
fp16-scope S4 — do not duplicate that work). Only `compute` remains.

**D2.2 (name the one decision):** compute precision gets a single owner and becomes a row in the fp16 scope's
authority table (section 6), which today has no row for precision at all. It is a derived value — the widest
precision the device expresses that the workload admits — not a constant. `sum_acc_dtype` (`dtype.py:274`) already
derives accumulate width from it and is currently **unused anywhere in `llm/`** ("never add what you can
compose").

**D2.3 (spelling):** pick `half` xor `float16` and apply it, or write the "C name in kernel-facing code" rule down.
Either is acceptable; the current state is neither.

### 4.3 D3 — `QuantFormat` as a peer value type

```python
@dataclass(frozen=True)     # interned, like DType
class QuantFormat:
  name: str                 # "Q4_K"
  block_elems: int          # 256
  block_bytes: int          # 144
  storage_roles: tuple[str, ...]   # ("words",) / ("halfs",)
```

`packed_linear_quant` returns `QuantFormat | None` instead of `str`; the `{"Q4_K": "q4k"}` dicts delete; the
storage wrappers carry their format. Later this is the natural home for the dequant program identity, which the
machine-search purity contract wants owned as data.

This also satisfies *"If similar things should be merged, fix the inputs first"* — unifying quant identity into a
typed canonical shape is the prerequisite for collapsing the downstream branches that fan out on the string.

## 5. Non-goals

- **Not** making Q4_K/Q6_K a `DType`. See 3.3.
- **Not** forking `tinygrad/dtype.py`. It is unmodified from upstream and there is no reason to diverge.
- **Not** changing any numeric behavior. Every slice is dtype-identical by construction; a changed output byte is
  a bug, and it also invalidates the NFC claim (section 6).
- **Not** touching `prefill_routes.py` dispatch, the fp16 admission slices, or `BOUNDED_PACKED_TILES`.
- **Not** re-deriving `supports_fp16`; fp16-scope S4 owns it.
- **No** new env var, flag, registry, control plane, or module.

## 6. Commit discipline

Every slice in this scope is behaviour-preserving, so every commit is **NFC**, per
`tinygrad-coding-overrides.md`:

- One owning-subsystem prefix per commit, from the allowed set. **`[refactor]` is not an allowed prefix.**
  - `kernel_program.py`, `llm/` routes, `llm/model*.py` → `[nn]`
  - `codegen/opt/packed_weight.py` (D3.2) → `[codegen]`
  - test-only changes → `[test]`
  - this document → `[docs]`
- Format: `[nn] NFC - move kernel output ABI onto KernelProgram`.
- **The NFC claim must be byte-proven, not asserted** — fixed-seed token parity or a golden hash, per the override.
  This is why section 7's gate is byte-identical output rather than "tests pass".
- Never mix an NFC refactor with a functional change. D3.2 touches `codegen/` and `llm/`: split by subsystem, not
  by convenience.

## 7. Slices and verification

| slice | change | verification |
| --- | --- | --- |
| D1.1 | output descriptor on `KernelProgram`; `_execute` allocates when no positional output is given | unit: program with descriptor allocates correctly; positional override still honored |
| D1.2 | migrate 6 production call sites | decode e2e **byte-identical** first token; decode tok/s unchanged |
| D3.1 | `QuantFormat` value type; `packed_linear_quant` return change | llm unit batch; prefill e2e byte-identical |
| D3.2 | delete downstream string dicts; migrate `codegen/opt/packed_weight.py` | codegen + llm unit batches; prefill e2e byte-identical |
| D2.0 | census; classification rule + unclassifiable findings recorded here | the rule and the findings list |
| D2.1–2.3 | subtract, name the one decision, settle spelling | llm unit batch; e2e byte-identical |

Every gate is byte-identical output, because none of these slices may change a number. That is a gate that can
fail, which is the point.

**Merge checklist:** the repo's own Practical Test (`coding-principles.md`, 14 questions) is the gate for each
slice — this scope does not invent a second one. Questions 1 (single source of truth), 5 (duplicate rule),
9 (encode the invariant rather than caller discipline), 14 (simplify the representation rather than add a
mechanism) are the load-bearing ones here.

## 8. Size

Honest estimate, flagged as an estimate. Measured in authorities collapsed, not lines removed.

| unit | files | sites | shape |
| --- | ---: | ---: | --- |
| **D1** | 4 (`kernel_program.py` + 3 route files) | 6 buffers | 1–2 commits. Small, self-contained, highest design value per site. |
| **D3** | 9, incl. `codegen/` | 52 (36 bare strings) | 2 commits. Named-violation fix; widest blast radius, only one leaving `llm/`. |
| **D2** | up to 12 | ≤103, most untouched | 1 census + 2–3 small commits. True size unknown until D2.0. Mostly subtraction. |

**Total: three independent units, roughly 5–8 commits, no new modules.**

Ordering: **D1 → D3 → D2.**

- D1 first: smallest, self-contained, and it absorbs D2's entire `abi` category — classifying those references
  before D1 deletes them would be wasted work.
- D3 second: *"If similar things should be merged, fix the inputs first."* Typed quant identity is the input
  unification that makes the downstream string branches collapsible.
- D2 last: its `express` category depends on fp16-scope S4 having landed.

## 9. Honest framing

This scope improves the design. It does not improve any number, does not unblock the NVIDIA campaign, and fixes no
reachable bug — the one safety story available (re-search invalidating a caller's ABI guess) was checked against
the search space and does not hold (section 2).

D3 is the exception to "this is taste": stringly-typed status values are a named anti-pattern in this repo's build
rules, so that unit is a rule violation being corrected, not a preference being applied.

Schedule it as design work on its own merits, or not at all. It must **not** be sold as correctness or performance
work, and it must not be interleaved with S1–S6: those slices gate on "decode unchanged," and D1/D2 touch the
decode path, so concurrent work would make that gate unattributable.

## 10. D2.0 census result (measured 2026-07-31)

Classification rule: every `dtypes.*` reference in the LLM layer belongs to exactly one category, decided by who
owns the fact:

- `storage` — the GGUF file format and its decode; the packed-weight storage carrier (uint32 words / uint16
  halfwords). Owned by the file. Cannot move.
- `abi` — the kernel artifact: UOp placeholders/consts/casts inside kernel builders, the packed-to-half operand
  carrier, and identity metadata naming the artifact's dtypes. Caller-restated output ABIs are absorbed by D1
  (6 sites, committed 91134defb/1341674af); artifact-internal references stay where the artifact lives.
- `express` — the device: one read, `device_facts.py:238` publishing `supports_fp16` (fp16-scope S4). Zero
  references in the census restate it.
- `compute` — the one decision: the widest precision the device expresses that the workload admits. Every
  reference in this column is a materialization of that decision on a path already gated by fp16 expressibility.

Measured counts (103 references, 12 files):

| category | count | files |
| --- | ---: | --- |
| storage | 41 | `gguf.py` 39, `qk_primitives.py` 2 |
| abi (artifact-internal) | 27 | `decode_kernels.py` 15, `flash_decode_attention.py` 9, `packed_wmma_prefill.py` 2, `model_facts.py` 1 (identity) |
| abi (output, absorbed by D1) | 6 | `flash_decode_attention.py` 2, `decode_routes.py` 3, `fused_attention.py` 1 |
| express | 0 | (the single owner read is `device_facts.py:238`) |
| compute | 27 | `model.py` 13, `fused_attention.py` 4, `decode_routes.py` 2, `prefill_routes.py` 3, `prefill_graph_gemm.py` 2, `adapter.py` 3 |
| unclassifiable (finding) | 2 | `model.py:1394` (index dtype), `model_facts.py:19` (dtype-name vocabulary) |

Findings (references that could not be classified into the four categories):

- `model.py:1394` `dtypes.int32` — an *index* dtype for the RoPE position map, not a precision. There is a fifth
  flavor (working/index dtypes) that is neither of the four; it is not a precision literal and needs no owner.
- `model_facts.py:19` `PROGRAM_DTYPES` — a dtype-*name* vocabulary for program identity metadata (`"float16"`,
  `str(dtypes.half)`, ...). It names dtypes as strings at the identity boundary, which is metadata, not a
  precision decision. Stays.

D2.1 result: subtraction is complete. Storage stays (`gguf.py` untouched by this scope's commits; the two
`qk_primitives.py` storage dtypes stay with the install specs). Output ABI was absorbed by D1 (6 sites). Express
is read once at `device_facts.py:238`; nothing in the census restates it.

D2.2 result: the one decision is named. Compute precision is the widest precision the device expresses that the
workload admits; its single owner is `DeviceCapabilities.supports_fp16` (the express read), which gates every
materialization in the compute column. `sum_acc_dtype` (`dtype.py:274`) composes accumulate width from it and is
not duplicated in `llm/`. The fp16-scope authority table gains the compute-precision row.

D2.3 result: spelling settled to a single spelling, `dtypes.float16`, across `tinygrad/llm` (19 `dtypes.half`
references collapsed; `half`/`float16` are the same interned instance per `dtype.py:200`, so this is
byte-identical by construction).
