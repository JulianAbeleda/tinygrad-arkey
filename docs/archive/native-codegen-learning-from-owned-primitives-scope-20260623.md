# Native tinygrad Codegen Learning — Charter (2026-06-23)

## Verdict: `NATIVE_CODEGEN_LEARNING_CHARTER_READY` + `NATIVE_CODEGEN_FIRST_EXPERIMENT_SCOPED`

A learning/codegen-capability charter (NOT a near-term W==D lane). It asks **which parts of the proven
escape-hatch attention primitive should become expressible in tinygrad's scheduler/renderer**, and defines one
bounded first micro-primitive experiment. Boundary: this is **not** a performance lane and must **not** displace
the runtime-KV core work or the default owned route.

## 1. Current closed primitives
- **Attention**: owned AMDGCN tile, default-on, near llama parity, ISA-confirmed. `ATTENTION_CLOSED_MAINTENANCE_ONLY`.
- **FFN GEMV**: Q4K GEMV warp, at/near llama parity. `GEMV_CLOSED_MAINTENANCE_ONLY`.

## 2. What tinygrad-native ALREADY learned (the existence proof)
**Q4K GEMV warp is tinygrad-native** — a UOp schedule (row/wave decomposition, K-block parallelism, warp
reduction), not a hand-kernel, and it **transferred to W==D at parity**. Lesson: *work decomposition can be
represented natively and win.* This is the template the attention primitive should follow.

## 3. What remains escape-hatch-only (the gap)
The owned attention tile is a hand-written HIP/AMDGPU **code object injected as an `Ops.PROGRAM` graph node** —
tinygrad-native codegen does **not** yet emit it. The escape-hatch-only pieces (ISA-confirmed):
- split-KV work decomposition + the split-KV combine;
- **LDS staging** (8KB group segment, `ds_store`/`ds_load`);
- **`v_dot2_f32_f16`** packed dot lowering;
- **cross-lane reduction** (`ds_bpermute`, warp reduce);
- the native fp16 cache dtype/ABI contract;
- precompiled binary graph-node injection.

## 4. Candidate compiler capabilities (the wishlist)
1. **First-class wave/cross-lane reductions** in the scheduler (so a reduce can lower to `ds_bpermute`/`shfl`-class
   ops instead of LDS/global round-trips).
2. **Explicit LDS tiling templates** (controlled `__shared__`-equivalent staging with static sizes).
3. **Vector-dot lowering controls** (force `v_dot2`/DP4A/WMMA-class ops where the dtype/shape allows).
4. A **split-KV decode-attention schedule** expressible natively (online softmax + PV + combine).
5. A **code-object/ISA feedback loop** — the renderer (or CI) consumes `extra/qk_isa_primitive_audit.py` output to
   detect a missing intended primitive (e.g. "expected v_dot2, got scalar fp16 ops").
6. **dtype/cache ABI propagation** (so a route can declare an fp16 cache contract and the scheduler honors it — the
   lesson from the fp32-as-fp16 bug).

## 5. Non-goals
- No immediate W==D target unless tied to a fresh residual gap (there is none open — attention is at parity).
- No replacing the working owned route prematurely.
- No broad renderer rewrite. No full native flash attention before the micro-primitives are proven.

## 6. First bounded learning experiment (the only thing to actually run, when funded)
**A tinygrad-native LDS + cross-lane reduction microkernel, ISA-audited.** Reproduce ONE owned-tile ISA property
(an LDS-staged, cross-lane-reduced dot/reduction over a small tile) **in a tinygrad-native UOp schedule**, then:
- local numeric correctness vs numpy;
- **ISA audit** via `extra/qk_isa_primitive_audit.py` → confirm `has_lds=true` and `has_cross_lane=true` were
  actually emitted by tinygrad's renderer (or document exactly what it emits instead);
- **no requirement to beat the default owned route** — the deliverable is "can tinygrad-native codegen express this
  primitive?", a YES/NO with ISA evidence.

### Why this experiment first
| experiment | goal | risk | pick |
|---|---|---|---|
| native `v_dot2` microkernel | controlled dot lowering | narrow | useful, but dot already works in GEMV warp |
| **native LDS + cross-lane reduction microkernel** | **prove the workgroup/wave primitive** | **medium** | **RECOMMENDED** — directly targets the owned-tile↔native gap, bounded, no default-route risk, uses the ISA guard |
| native split-KV toy attention | prove dataflow shape | higher | later, only after the micro-primitives pass |
| renderer↔ISA-audit feedback loop | automate missing-primitive detection | infra-heavy | later infrastructure |

## 7. Stop rules
- Do not replace the owned attention route.
- Do not pursue full native flash attention before micro-primitives are proven.
- Do not claim a performance win without W==D (this charter targets *expressibility*, not speed).
- Do not let this block runtime-KV core work if that is authorized (runtime-KV is the only parity-class speed prize).

## 8. Dependency note
This lane consumes Lane 3 (the ISA audit guard) as its YES/NO oracle. It is **lower priority than runtime-KV** (no
W==D need) and should run only when the owner wants longer-term tinygrad codegen capability, not decode speed.
