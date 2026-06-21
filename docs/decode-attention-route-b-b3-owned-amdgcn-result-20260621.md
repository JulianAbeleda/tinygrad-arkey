# Route B B3 — Owned hand-AMDGCN decode-attention tile (tinygrad KV layout): result

Date: 2026-06-21

Executes **Route B B3 only**. B1/B2 proved the *vendored* llama tile wins through tinygrad's HCQ (2.96× GPU) and that
the win survives launch integration (B2, 1.65× wall) — but the vendored kernel is non-promotable and W==D-blocked
because it reads llama's ggml KV layout. **B3 authors the first OWNED, promotable decode-attention tile for tinygrad's
NATIVE K/V layout** — llama's dataflow, our source.

## Decision: **`B3_LOCAL_PASS` — the owned kernel BEATS gqa_coop_vec locally (2.35× GPU-busy / 1.70× wall, near-exact). The project crosses "can we own the primitive?" → YES.** W==D blocked by graph-node integration (default stays off).

`extra/qk_owned_flash_decode.hip` (our source, hipcc→`.co`, launched via the B1 `NamedAMDProgram` + B2 one-bound-HCQ
queue) is value-correct (rel_max 3.5e-7 — far better than the vendored 1.3e-3) and beats the shipped winner
`gqa_coop_vec` on every local gate. It is promotable (owned, tinygrad-native layout, no repack) but `default_eligible
=false`: a production W==D needs the raw-HCQ kernel to enter the JIT-traced decode graph, which is blocked (below).

## B3.0 — Layout contract (`bench/qk-decode-attention-route-b-b3/tinygrad_kv_layout_contract.json`)
tinygrad's decode K/V layout (from the `gqa_coop_vec` winner's indexing, `qk_flash_decode.py`): **`K`/`V` = `[Hkv, MAXC,
Hd]` fp16** (kv-head major → position → head-dim; strides `[MAXC·Hd, Hd, 1]`), `Q` = `[Hq,Hd]`, GQA group 4
(`h = kvh·4 + g`), scale `1/√128`. **vs llama** (B2): position-major `[pos][8 kv-heads][128]` (`nb11=2048`). The owned
kernel reads tinygrad's layout natively — each kv-head's K/V is one contiguous `[MAXC,Hd]` block (naturally coalesced
per warp). **Describable without model surgery → stop condition NOT triggered.**

## B3.1 — Owned kernel (`extra/qk_owned_flash_decode.hip`)
Authored from scratch (llama dataflow, not vendored code). Two tiles + a combine:
- **v1** `owned_flash_tile`: one warp (32 lanes) per (q-head × split); each lane owns 4 head-dims as two `half2`;
  `__builtin_amdgcn_fdot2` → **`v_dot2_f32_f16`** for q·k; warp-shuffle reduce; register online softmax + PV.
- **v2 `owned_flash_tile_gqa` (shipped):** one 128-thread workgroup (4 warps = the 4 GQA q-heads) per (kv-head ×
  split); **K/V staged into LDS once and reused across all 4 q-heads** (the llama `ncols2=4` ingredient — removes v1's
  4× redundant K/V global reads). 8 KB LDS, 56 VGPR, **0 spill**.
- **`owned_flash_combine`:** one warp per q-head, log-sum-exp merge of the `S` split partials.
Launched via the **B2 one-bound-HCQ-queue** (1 doorbell, 2 dispatch packets, kernargs baked once). Compiled with hipcc
`-D__AMDGCN_WAVEFRONT_SIZE=32 --genco` → unbundle → bare gfx1100 ELF.

## B3.2 — Local A/B (clock-pinned, @ctx1024, S=48 — swept S∈{8..128}, optimum ~32–48)
| metric | owned tile | gqa_coop_vec | ratio | gate |
|---|---|---|---|---|
| **GPU-busy** | **18.9 µs** | 44.5 µs | **2.35×** | ≥1.5× ✅ |
| **wall (matched per-call sync)** | **41.6 µs** | 70.6 µs | **1.70×** | ≥1.5× ✅ |
| correctness vs numpy | rel_max 3.5e-7 | — | — | ≤1e-3 ✅ |
| ISA | v_dot2=2, 56 VGPR, 8 KB LDS, **0 spill** | — | — | ✅ |

**Measurement note (important):** the wall uses **both sides per-call-synced** (matched launch-overhead model) — the
*fair* comparison. An earlier coop-pipelined-vs-owned-synced reading gave a misleading 1.40× (coop's TinyJit pipelines,
the bound owned queue forced a sync per call). With matched sync, coop's per-call wall is 70.6 µs (its 44 µs GPU +
~26 µs sync) and owned is 41.6 µs → 1.70×. GPU-busy (the unambiguous kernel metric) is 2.35×; the equal additive sync
overhead makes the wall ratio conservative. Counts per replay: **2 dispatches, 1 doorbell, 1 graph replay.** Artifact
`bench/qk-decode-attention-route-b-b3/latest.json` (stamped; kernel/source hash, ISA, v_dot2, LDS, VGPR/spill, AQL/
doorbell/replay counts, layout-contract hash).

## B3.3 — W==D: BLOCKED by graph-node integration (default stays off)
Local passing *unlocks* W==D, and unlike the vendored kernel there is **no layout block** (the owned kernel reads
tinygrad's native layout). The remaining blocker is **graph injection**: the owned kernel is a raw hipcc `.co` launched
via HCQ, **not a tinygrad UOp/graph op**, so it cannot enter the JIT-traced decode graph that `model.generate` replays.
Injecting it needs either **Route-A native codegen** (forbidden this phase + the known UOp inexpressibility wall) or
**eager (un-jitted) decode** (non-production, Amdahl-limited). So a production W==D is gated on a *"schedule an external
precompiled kernel as a JIT graph node"* capability — a **bounded tinygrad feature, NOT Route-A codegen of the
attention**. W==D not run; **no `model.py` route, zero `tinygrad/` diff, default off**.

## B3.4 — Lifecycle
Registered `decode_attention_llama_flash_tile_owned_amdgcn` (candidate, family `north_star_flash_attn_tile` — this is
the first executable realization of that previously-`PRUNE_NEEDS_TEMPLATE` north-star row) + its binding, with
**`default_eligible=false`**, `wd_runner: none (blocked)`, and the full local-A/B result + ISA + kernel provenance.

## Classification & decision
**`B3_LOCAL_PASS` (task family: `LOCAL_PASS_WD_FAIL` — local passes, W==D not yet passable).** Not `FAIL_LOCAL_AB`
(it wins), not `BLOCKED_BY_KV_LAYOUT` (layout solved — the point of B3), not `BLOCKED_BY_ISA_QUALITY` (clean v_dot2,
0 spill, 2.35× GPU). **The owned, promotable primitive beats the winner locally — the project crosses from "can we
prove the idea?" to "can we own the primitive?": YES.** The single remaining step to promotion is the
external-kernel-as-graph-node capability (a bounded, scoped follow-on — explicitly *not* Route-A codegen).

## Boundaries honored
No vendored llama code in the owned artifact; no repack of tinygrad KV; **no defaults changed**; no Route-A native
codegen started; no generalization before the fixed-shape pass; **no `model.py` route; zero `tinygrad/` diff**.
`gqa_coop_vec` comparator SSOT; no closed-lane reopen; local A/B is GPU/wall diagnostic, not a benchmark headline.
