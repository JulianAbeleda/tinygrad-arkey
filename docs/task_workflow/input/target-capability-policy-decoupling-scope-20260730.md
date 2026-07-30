# Target capability/policy decoupling scope

Date: 2026-07-30

Status: scoped, not implemented. Branch boundary: all work begins on tinygrad `exp`. This scope does not authorize
promotion to `dev`/`master`, hand-authored target kernels, or new selection-policy tables.

## 1. End goal

Production `tinygrad/llm/**` currently decides *where a fast path may run* with hardcoded target-string equality.
Five such gates exist. Each one silently disables a fast path on Metal and NVIDIA, with no error, no diagnostic, and
no census entry. One of them was measured today to be worth **3.2x on Metal**.

The end goal is that a fast path is admitted by exactly two questions, each answered by its existing owner:

```text
can this target EXPRESS the program?     -> renderer/device capability facts (tinygrad)
is this candidate PROMOTED here?         -> BoltBeam route manifest (measured evidence)
```

and that a target which cannot express a program fails loudly at lowering rather than falling back silently.

This is a decoupling, not a new subsystem. No new selection table, no new registry, no new control plane.

## 2. Pinned evidence

All numbers below were measured on 2026-07-30, Apple M4 10-core / Metal, Qwen3-8B-Q4_K_M
(sha256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`).

### 2.1 The five gates

| file:line | gate | domain |
| --- | --- | --- |
| `tinygrad/llm/qk_primitives.py:27` | `(backend, architecture, wave_size) == ("AMD","gfx1100",32)` | Q4_K/Q6_K decode primitives |
| `tinygrad/llm/prefill_candidate_runtime.py:162` | `target != {"backend":"AMD","arch":"gfx1100","wave_size":32}` -> raises | prefill candidates |
| `tinygrad/llm/model.py:72` | `backend == "AMD" and arch == "gfx1100"` | grid admission |
| `tinygrad/llm/admission.py:21` | `_TC_ATTN_TARGET_REQUIREMENTS = {"backend":"AMD","architecture":"gfx1100"}` | tensor-core attention |
| `tinygrad/llm/flash_decode_attention.py:384` | `device == "AMD" or device.startswith("AMD:")` | flash decode |

Only the first has a measured prize. The other four are structurally identical but **unmeasured** — treat them as
hypotheses, not as known wins.

### 2.2 Measured prize on the first gate

Diagnostic patch (reverted; repo clean) widening `eligible` for Metal, plus a Metal `simd_shuffle_xor` branch in
`warp_shfl_xor`, plus `METAL_HYBRID_REPLAY=1`:

| arm | tok/s | output tokens |
| --- | ---: | --- |
| generic Metal fallback | 5.37 | prelude 13876 / generated 38835 |
| ported quant primitives | **17.11** | prelude 13876 / generated 38835 (identical) |
| llama.cpp reference | 20.34 | identical |

253/253 installed primitives bound (216 Q4_K + 37 Q6_K). Isolated-kernel worst relative error `2.9e-7`.

### 2.3 Two blockers found beyond the gate

1. **`warp_shfl_xor` emits AMD ISA as a literal string.** `tinygrad/codegen/late/warp_reduce.py:22-27` renders
   `__builtin_amdgcn_ds_bpermute(...)` through `Ops.CUSTOMI`. `MetalRenderer` is a `CStyleLanguage` subclass sharing
   the same `CUSTOMI` template path, so it renders the AMD text verbatim and fails to compile. Widening the
   eligibility gate alone yields `use of undeclared identifier '__builtin_amdgcn_ds_bpermute'`.
2. **Metal ICB 4 GB offset limit.** `storage_mode="shared"` (`model.py:1247`) aliases packed weights into the
   4.68 GB GGUF buffer; `tinygrad/runtime/graph/metal.py:89` raises `Metal ICB offset exceeds 0xffffffff`.
   `METAL_HYBRID_REPLAY=1` routes around it. This makes MR3's hybrid replay a **prerequisite** for shared storage on
   Metal, despite MR4 classifying it as neutral in isolation.

### 2.4 Measurement basis established today

Unbatched per-kernel timing is a reliable basis and is far cheaper than xctrace:

```sh
JIT=0 DEBUG=2 PYTHONPATH=$PWD .venv/bin/python \
  extra/llm_research/decode/decode_runtime_overhead.py \
  --model <gguf> --ckpts 128 --nmeas 1 --reps 1 --warmup-decode 2 --out <json>
```

The last 803 kernel launches are exactly one decode token (matches the graph-admission census). Summing them
reproduced 5.31 tok/s against a batched 5.37 — unbatched and batched agree, so this basis is sound. Parser:
`scratchpad/diff_kernels.py`. The first reported bandwidth column is achieved traffic on packed bytes (validated:
lm_head 510,504,960 bytes / 63.64 ms = 8.02 GB/s, matching the reported 8.0). The second column includes cache hits
and must not be read as DRAM bandwidth.

### 2.5 Confounder that must be pinned

`90e93875c` (devectorizer/reg_store refactor) changed decode output from 83659/33235 to 13876/38835 — now matching
llama.cpp — and cost **2.03x** throughput (10.79 -> 5.31 tok/s unbatched, same kernels, same launch counts, bandwidth
collapse concentrated in two kernels). Any baseline in this scope must be captured on a **single pinned commit**;
comparing across that commit invalidates the comparison. Repairing that regression is **out of scope here** and
belongs in its own packet.

## 3. Architectural boundaries

### 3.1 One authority per concern

| Concern | Authority |
| --- | --- |
| can a target express an operation | `tinygrad/renderer/**` attributes and per-renderer templates |
| target hardware facts (lane width, offset limits) | `tinygrad/llm/device_facts.py` + renderer attributes |
| is a candidate promoted for a target | BoltBeam `route_manifest.v1.json` (via the hashed EXP snapshot) |
| shape admissibility | the existing `bind()` methods in `tinygrad/llm/decode_routes.py` |
| research/qualification gates | `extra/llm_research` only, never production defaults |

Do not add a second capability registry, target table, selection table, or promotion rule.

### 3.2 Required reuse

- Reuse the existing renderer attribute mechanism (`supports_float4`, `has_local`, `has_shared`, `shared_max`,
  `global_max`, `tensor_cores`) — `MetalRenderer` already conditions `tensor_cores` on `Apple7+`
  (`tinygrad/renderer/cstyle.py:475`). Add capabilities in that same declarative form.
- Reuse the existing per-device template pattern. `tinygrad/nn/__init__.py:361`
  (`elif device == "AMD": atomic_arg = "__hip_atomic_fetch_add(...)"`) is the in-repo precedent for a per-target
  intrinsic. A cross-lane shuffle belongs in the same shape as `barrier`, `float4`, `smem_prefix`.
- Reuse BoltBeam's route manifest and its exporter. Do not copy the asset; refresh only via
  `tools/export_route_manifest_snapshot.py`.
- Reuse `tinygrad/llm/device_facts.py` for target facts. Do not introduce a parallel facts object.
- Reuse the unbatched measurement basis in section 2.4 and `scratchpad/diff_kernels.py` for every before/after.

### 3.3 Separation of facts

These stay orthogonal and must never be collapsed into one boolean again:

- backend identity (AMD / METAL / NV / CPU);
- architecture string (gfx1100 / Apple9 / sm_80);
- lane width (AMD wave32 vs Metal simdgroup 32 — equal here, not equal in general; Metal reports `wave_size=None`);
- operation availability (cross-lane shuffle, tensor cores, atomics);
- resource limits (shared memory, indirect-command-buffer offset width);
- promotion status (measured and recorded by BoltBeam);
- shape admissibility.

A target string must not imply a capability. A capability must not imply promotion. An unreported `wave_size` must
never silently default to 32.

## 4. Evidence contract

Every package that changes admission or codegen must produce, on one pinned commit:

1. **Correctness**: prelude and generated token ids identical to the pre-change baseline at depth 128, plus the
   `prompt_evidence` sha256. A change in token output is a failure, not a result, unless the packet's explicit
   purpose is a correctness fix.
2. **Per-kernel before/after**: the section 2.4 run, both arms, diffed with `diff_kernels.py`, reporting total decode
   step ms, per-kernel ms, launch counts, and achieved GB/s. Identical kernel sets and launch counts across arms must
   be stated explicitly when true — that distinguishes a codegen change from a schedule change.
3. **Whole-model number**: median tok/s over at least 3 repetitions, with spread. This machine shows ~0.5-2.6%
   run-to-run variation; a delta inside that band must be reported as indistinguishable, never as a win.
4. **No silent fallback**: a target that cannot express a program must raise at lowering. Demonstrate the raise.

Do not hand-edit any emitted number, status, or classification.

## 5. Work packages

### TG0 — Freeze the baseline and promote the measurement tool

Prerequisite: none. Owns no behavior change.

- Pin one commit for the whole campaign; record it in every artifact.
- Capture the section 2.4 baseline on Metal: full per-kernel table, token identity, 3-repetition tok/s.
- Promote `scratchpad/diff_kernels.py` into the repo under `extra/llm_research/` (research surface, per the Boundary
  Rule — it must not become a production dependency), with a unit test over a captured fixture log so the parser
  cannot silently drift.

Stop condition: if the baseline does not reproduce 5.3-5.4 tok/s with tokens 13876/38835, stop and re-pin.

### TG1 — Cross-lane shuffle as a renderer-lowered operation

Prerequisite: TG0. This is the only genuinely new primitive in this scope.

- Replace the inline AMD string in `tinygrad/codegen/late/warp_reduce.py:22-27` with an operation lowered per
  renderer, in the same shape as `barrier`/`float4`.
- Provide: AMD `__builtin_amdgcn_ds_bpermute` (existing behavior, byte-identical output required), Metal
  `simd_shuffle_xor(value, mask)`, CUDA `__shfl_xor_sync` if it is a one-liner; otherwise leave CUDA unprovided.
- A renderer that does not provide it must raise at lowering with the operation name and the target — never fall back.
- Do **not** dispatch on `Device.DEFAULT`. That was a diagnostic shortcut; it breaks under multi-device and does not
  belong in codegen. Dispatch on the renderer.

Acceptance: AMD-rendered source for an unchanged kernel is byte-identical to the pinned baseline; Metal compiles;
an unprovided target raises.

### TG2 — Declare the missing target capabilities

Prerequisite: TG0.

- Add as declarative renderer/device facts: cross-lane shuffle availability, lane width (explicitly modelling
  "unreported"), and `max_indirect_buffer_offset` (Metal 0xffffffff; see section 2.3).
- Metal reports `backend=METAL, architecture=Apple9, wave_size=None`. Model the unreported case explicitly; do not
  default it.
- No admission logic changes in this package.

Acceptance: facts are readable for METAL, AMD, and CPU; a unit test pins Metal's ICB offset limit and unreported
lane width.

### TG3 — Split the quant gate into capability and policy

Prerequisites: TG1, TG2.

- `tinygrad/llm/qk_primitives.py:27` becomes two checks: (a) does the renderer provide every capability the primitive
  requires, (b) does the route manifest promote this candidate for the resolved target.
- The install gate at `tinygrad/llm/model.py:1035-1036` must consult the same two answers — today it duplicates the
  eligibility property, which is why a single `==` disabled installation entirely.
- Primitives declare the capabilities they require. Do not infer capability from a target string.
- Shape gates in `decode_routes.py` stay exactly where they are.

Acceptance: on AMD the admitted set is unchanged (byte-identical route census); on Metal admission is decided by
manifest content, and a manifest that does not promote Metal yields no binding **and a recorded census reason** — not
a silent fallback.

### TG4 — Metal promotion evidence for the quant route

Prerequisite: TG3. This is BoltBeam work, not tinygrad work.

- Record the Metal measurement as promotion evidence in the canonical manifest; refresh the EXP snapshot only via
  the single exporter.
- Until this lands, Metal binding stays off by manifest content, which is the correct fail-closed default.

Acceptance: hash-checked snapshot refresh; EXP fails closed on a stale or mismatched snapshot.

### TG5 — Prove the quant path end to end on Metal

Prerequisites: TG3, TG4.

- Full section 4 evidence contract, including the `METAL_HYBRID_REPLAY` interaction from section 2.3.
- Expected: ~5.4 -> ~17 tok/s with identical tokens. Report whatever is measured.

Stop condition: token divergence from baseline, or a delta inside the noise band. Either ends the packet.

### TG6 — Record the hybrid-replay prerequisite

Prerequisite: TG5.

- Metal + `storage_mode="shared"` requires hybrid replay because of the ICB offset limit. Express this as a
  capability requirement (TG2's `max_indirect_buffer_offset`), not as an environment flag users must know about.
- Amend the MR4 record: hybrid replay is neutral in isolation and a prerequisite in combination. Do not restate MR4's
  classification; add the interaction.

### TG7-TG10 — The remaining four gates, one package each

Prerequisite: TG5 must be complete, so the pattern is proven before it is replicated.

One package each for `prefill_candidate_runtime.py:162`, `model.py:72`, `admission.py:21`,
`flash_decode_attention.py:384`. Each package: identify the true capability requirement, split capability from
policy, produce the full section 4 evidence, and report the measured delta.

These are **unmeasured hypotheses**. A package that finds no win on Metal must still land the decoupling if it is
behaviour-neutral on AMD, and must record the null result. Do not assume they behave like TG3.

## 6. Dependency graph and safe parallelism

```text
TG0 ──┬─> TG1 ─┐
      └─> TG2 ─┴─> TG3 ─> TG4 ─> TG5 ─┬─> TG6
                                       ├─> TG7
                                       ├─> TG8
                                       ├─> TG9
                                       └─> TG10
```

TG1 and TG2 are independently parallelisable. TG7-TG10 are parallelisable **only after TG5**, and each needs the
exclusive GPU lane for its measurement — serialize all hardware runs.

## 7. Non-goals

- Repairing the `90e93875c` devectorizer regression (separate packet; see section 2.5).
- Promotion to `dev`/`master`.
- Hand-authored Metal kernels, or copying AMD ISA, lane maps, or route IDs into production as defaults.
- A general target-capability framework beyond the three facts TG2 names. Add the fourth when a fourth target needs
  it, not before.
- Changing the roofline, MR7 ranking method, or any BoltBeam promotion rule.

## 8. Known limitations

- **No AMD hardware is available on this machine.** Every package touching shared code paths (TG1 especially) can
  only verify AMD non-regression by rendered-source equality, not by execution. State this limitation in each
  packet; do not claim AMD non-regression from a passing Metal run.
- Absolute tok/s in this scope is measured against a tree carrying the 2.03x devectorizer regression. Deltas remain
  valid; absolute numbers will move when that packet lands.
- The trace-derived `selected_gpu_union` basis is unreliable (swings ±30% run to run, and the MR4 harness itself
  flagged 3 of 10 rows `:profiler`). Use the section 2.4 basis instead.
