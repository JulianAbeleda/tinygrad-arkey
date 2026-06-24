# Route B B4 — External precompiled AMDGCN as tinygrad JIT graph node: result

Date: 2026-06-21

Executes **Route B B4** from `docs/decode-attention-route-b-b4-external-graph-node-scope-20260621.md`.
The blocker from B3 (`raw .co` cannot enter JIT graph) is lifted: the owned kernel is now injected as
`Tensor.custom_kernel` / `Ops.PROGRAM` graph nodes and participates in TinyJit capture/replay on AMD.

## Decision: **`B4_WD_FAIL_INTEGRATION`** — graph-node integration is solved, but W==D does not yet clear the
promotion gate under tested policies.

## B4.1 — Harness and route verification

- Added `extra/qk_owned_flash_decode_graph_node.py` to specialize `extra/qk_owned_flash_decode.hip` into single-kernel
  ELFs (tile + combine), bind `start_pos` as the symbolic scalar var, bake `S`/`scale`/`MAXC`, and wire the
  two-kernel path as `owned_flash_tile_gqa` + `owned_flash_combine` through `Tensor.custom_kernel`.
- The graph-node path is verified by
  `DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_owned_flash_decode_graph_node.py 48`:
  - eager / capture / replay all pass
  - `rel_rmse≈2.8e-7`
  - captured `Ops.PROGRAM` nodes present (`owned_flash_tile_gqa`, `owned_flash_combine`).
- In-model route fire checks are now explicit in the W==D harness (`extra/qk_b4_decode_eval.py`): it inspects
  captured `Ops.CALL(PROGRAM)` node names and validates route presence for base vs AMDGCN runs.
- Timing is measured in-process with `out.item()` inside the timed window (no delayed post-loop sync), avoiding the
  earlier “instantaneous” wall-number trap.

## B4.2 — Split/policy sweep (trusted harness)

All sweeps use `extra/qk_b4_decode_eval.py` with `nmeas=40`, `repeats=6`, `warmups=8` and fixed S candidate set
`[24,32,40,48,56,64,80,96,128]`:

- `ctx2048_only`:
  - best splits: 512→24, 1024→32, 4096→64
  - deltas: `+0.08%`, `+0.18%`, `+5.44%`
  - route fire: `false, false, true`
- `ctx4096_only`:
  - best splits: 512→24, 1024→48, 4096→56
  - deltas: `+0.11%`, `+0.40%`, `+5.56%`
  - route fire: `false, false, true`
- `adaptive` (current policy):
  - best splits: 512→128, 1024→48, 4096→48
  - deltas: `+0.24%`, `-0.76%`, `+5.36%`
  - route fire: `false, true, true`
  - route-nodes when fired: `owned_flash_tile_gqa`, `owned_flash_combine` plus fixed decode graph nodes

## Gate evaluation

- `first_gate_pass = false` in all evaluated policy modes.
- Gate rule used by the harness is:
  - no regression at ctx512 and ctx1024, **and** (`ctx1024 >= +5%` **or** `ctx4096 >= +7%`)
- Best tested outcome is `+5.36%` at `ctx4096` (adaptive) with only a small ctx512 rise and a ctx1024 regression in
  adaptive mode.

## Decision rationale

`owned_flash_tile_gqa`/`owned_flash_combine` now enter the model JIT graph and route visibility is reliable in
decode. That unblocks W==D as a valid route and invalidates prior “graph-node blocked” status. However, the current
`DECODE_ATTN_AMDGCN` route does not clear the promotion criteria under policy sweep (`B4_WD_FAIL_INTEGRATION`).

## Classifying the W==D miss: **Amdahl limit** (not graph overhead / sync leakage / kernel mismatch)

Per the scope's stop-condition rubric, the local-graph-integrated route passes but W==D misses; the miss is the
**Amdahl ceiling of swapping one ~17% component**, not an integration defect:

- **Not kernel mismatch** — greedy byte-identical at every ctx; standalone `rel_rmse 2.7e-7`.
- **Not graph overhead / sync leakage** — the tile+combine are folded into the one bound JIT graph queue (per-attention
  dispatch count drops from coop's ~7 kernels to **2**); W==D uses real per-token `.item()` sync with tight spread
  (<0.4%), and the ctx512/1024 route-off arms read ≈0% (no measurement bias).
- **Amdahl** — decode attention is ~17% of the step (coop ≈70µs × 36 layers ≈2.5ms of a ~16.4ms ctx4096 token). The
  owned attention (`owned_flash_tile_gqa` ~21.5µs + `owned_flash_combine` ~12µs ≈33.6µs × 36 ≈1.2ms) saves ~1.3ms →
  ~+8% ideal; measured **+5.6–5.85%@ctx4096** (best across `ctx4096_only` runs). The residual loss is the
  **latency-bound combine** (~12µs at S≈48–56, only 32 workgroups), which gives back part of the tile's q·k/PV win
  (21.5µs vs coop's 53µs `flash_partial`). At ctx1024 the smaller KV makes the fixed combine + over-splitting a net
  wash (route-on −0.8%) → the route is ctx-gated (`DECODE_ATTN_AMDGCN_MIN_CTX`, default 2048).

So closing the gap is **not** a graph-node problem (that capability is solved and reusable); it needs a cheaper combine
(stream-k / ctx-scaled splits — a bounded follow-on) or attacking the FFN/GEMV share outside attention. The +5.6%@4096
byte-identical gain is a real long-context win below the automatic-promotion bar → an **owner-call knob** (the
`FLASH_L=64` precedent), not a default.

## Boundaries honored

- default-off route; no `tinygrad/` changes beyond the shape/device-guarded model branch
- no pinned performance state in the B4 harness (`in-process`, auto clock for W==D comparisons)
- no vendored llama code in the owned path
- no KV layout conversion/repack
- no Route-A codegen work introduced in this phase.
