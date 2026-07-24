# HANDOFF — Fused Shared-Attention Enablement for 8B/14B (2026-07-23)

Self-contained entry point for a fresh session. Read this first; it links the supporting docs.

---

## 1. Mission (one paragraph)

Qwen3 **8B and 14B prefill** should run attention through the fused, compiler-native shared
flash-attention kernel (`amd_gfx1100_q16_grid_hd128_loop_attention`) and emit a BoltBeam artifact.
Today they silently fall back to ordinary SDPA. The kernel itself is **correct and fast in isolation**
(254 VGPR, 0 spills, ~3.7–4.4× faster than SDPA, numerically matching to ~6e-05). The blocker is
**not** perf and **not** the kernel — it is that the production *lowering pipeline* crashes before it
can build that kernel from a real model graph. Enabling this is a **compiler-robustness workstream**
(sized below), plus a plumbing tail.

## 2. What is DEAD — do NOT re-chase (each was disproven with measurement/evidence)

- **VGPR occupancy / the ≤192 cap.** The admission gate capped `vgpr ≤ 192`; production is 254. This
  cap was the target of a long "rotating-PV" effort. It is **unjustified**: occupancy on gfx1100 only
  changes bucket at ≤128 (not 192), the wave-count gain was never confirmed, and the one experiment
  that actually cut VGPR **regressed** perf 1.4–2.5% (`docs/ATTENTION_COMPACT_VGPR_LEASE_NEGATIVE_20260723.md`,
  `docs/SHARED_ATTENTION_LIVE_STATE_RESIDENCY_LEDGER_20260723.md`). Cap already raised to 256
  (commit `fa6d29633`).
- **The rotating-PV probe + its bespoke ABI.** A whole alternative accumulator-residency experiment
  (`RotatingPV*`, `SoftmaxBridge*`, `ROTATING_PV_SEQUENCE`). It never compiled to ISA, chased the dead
  ≤192 target, and was the sole source of a "re-lowering treadmill." **Retired** (~780 lines removed,
  commit `88b4e3ee1`). Do not resurrect it.
- **Register-pinning the accumulator.** Spiked; made VGPR pressure *worse* (145 vs 82 peak live vregs).
  LDS is the correct home for the accumulator.

## 3. Verified facts (hard evidence)

- Isolated fused kernel: **254 VGPR / 0 spills / 512 LDS**, all 4 routes (8b/14b × first/prefix).
  Source: `python3 -m extra.qk.generate_shared_attention_captures --output-dir <d>`.
- Isolated fused vs SDPA on real gfx1100 (synthetic Q/K/V): **8B 0.541ms vs 2.01ms (3.72×); 14B
  0.568ms vs 2.49ms (4.39×)**; correctness max-abs-err ~6.1e-05. (Fused number cross-checks the
  independent 0.5445ms baseline in the compact-lease doc.)
- 14B whole-model prefill on the **standard** `model.__call__` path runs fine (RESULT OK, 7.8s) — the
  "expression is not assignable" HIP crash seen earlier was **harness-only** (the logits_only shortcut).
- Hardware: AMD Radeon RX 7900 XTX, gfx1100, ROCm 7.2.4. Weights present in `/home/ubuntu/models/`
  (`Qwen3-8B-Q4_K_M.gguf`, `Qwen3-14B-Q4_K_M.gguf`).

## 4. Current tree state (as of this handoff)

- `fa6d29633` — VGPR admission cap 192→256 (+ tests). DONE.
- `88b4e3ee1` — rotating-PV retirement + triage docs. DONE.
- `86e6442c6` — **WIP partial fixes, pushed** (path still gated OFF, so inert in production):
  - `model.py`: Q/K/V cast to half at the attention boundary (Q was fp32 → silently failed
    `grid_shape` eligibility → SDPA every time). Correct, needed.
  - `rangeify.py`: build composite reduce from natural `(b,h,q_len,kv_len)` score shape (no fake-hd
    broadcast). Correct, non-regressing, but insufficient alone.
- Production is still on SDPA fallback (`prefill_tc_attn` is never enabled — see §6 plumbing).

## 5. THE REAL BLOCKER (where a new session starts)

Running a real 8B (or 14B) prefill through the fused path crashes in scheduling:
```
ValueError: bad reshape: () -> (1, 32, 1, 512, 128)
  tinygrad/schedule/rangeify.py (symbolic+reduce_collapse+debuf pass, ~line 938-951)
  -> _mop_index (rangeify.py:259) -> UOp.shape (ops.py:389)
```
Repro (self-contained, forces the path on): 
```
DEV=AMD PYTHONPATH=/home/ubuntu/tinygrad-arkey .venv/bin/python \
  /home/ubuntu/.claude/jobs/6db6b205/tmp/fused_prefill_force_8b.py
```
(If that scratch path is gone, it forces `config.prefill_tc_attn=True` + policy strategy
`FULL_RESIDENT_OVERLAY` after a normal `load_model_and_tokenizer`, then calls `model.logits(chunk,0)`
and inspects the schedule for a CALL carrying a non-None `candidate_context`.)

**Root cause (systemic):** the composite-reduce lowering (`tinygrad/codegen/late/composite_combines.py`
+ `lower_attention_semantic` in `rangeify.py:19-153`) uses **size-1 broadcast axes** to make the
score, V, m, l line up on one reduce axis. Under **full-model DAG sharing** (32 layers, hash-consed
nodes, one causal-mask node with 32 consumers, real RoPE/quant feeding Q/K/V), the generic collapse
passes (symbolic / reduce-collapse / const-fold) fold a degenerate broadcast axis down to a scalar
`PARAM` load **without updating the enclosing `RESHAPE` target shape** → shapeless node → panic. This
path has **never** successfully lowered at production graph scale (the 254-VGPR proof was only ever
run on a synthetic 61-node, single-consumer graph that calls the hand-built kernel directly, bypassing
composite-reduce). It recurs per input: the score-side instance is fixed (commit `86e6442c6`); the
**auxiliary V (`value_tile`) instance is the current crash**; more may follow.

**This is NOT a per-crash patch chain.** The correct fix is structural, and the FIRST task of the
workstream is to choose between two designs:
- **(1) Construct the composite-reduce inputs without generic-pass-fragile unit broadcast axes** —
  restructure `lower_attention_semantic` / `CompositeInputSpec` so score/V/m/l are expressed without
  size-1 axes that collapse passes mangle. (Cleaner, more invasive.)
- **(2) Make the collapse passes RESHAPE-preserving** — when a generic rewrite folds a broadcast
  source to a scalar, rebuild/remove the enclosing RESHAPE so no shapeless node survives. (More local,
  but touches generic passes used everywhere — higher blast radius.)

The already-built "collapse to one kernel" AST-swap (`postrange.py:328-361` → `wmma.py:545`) is
**correct and needs no work** — it's simply unreachable until scheduling survives. Fixing the crash
chain lets the real graph reach that swap.

## 6. Full work breakdown (dependency-ordered; see scope-A doc for detail)

| Phase | Task | Status / difficulty |
|---|---|---|
| A0 | Q/K/V half-cast at boundary | DONE (`86e6442c6`) |
| A1 | Score-broadcast fix | DONE (`86e6442c6`), insufficient alone |
| **A2** | **Structural design decision (1 vs 2 above) + fix V-side + iterate until real 8B prefill schedules clean** | **NOT STARTED — the core workstream; UNKNOWN # of fixes** |
| A3 | Confirm schedule reaches `postrange.py:328` native-swap and emits the 254-VGPR single kernel on the REAL graph | after A2 |
| A4 | End-to-end correctness: real 8B+14B fused vs SDPA (earns `model_*_prefill`) | after A3 |
| A5 | Whole-model prefill benchmarks KV 512..4096 (note: handoff doc defers these behind an in-flight multi-wave G2/G4/G5 experiment) | after A4 |
| A6 | **Build** decode-nonregression harness (earns `decode_nonregression_*`); none exists, nearest is `extra/qk/decode_runtime_overhead.py` | after A4 |
| A7 | Composite-proof collector: assemble `shared_attention_proof` (target+geometry+v2 artifact+8 flags) and wire memory-adaptive adapter activation into model load (zero production callers today) | after A4/A5/A6 |
| A8 | BoltBeam prefill-attention route entry in `BoltBeam/boltbeam/policy/route_manifest.py` (none exists; template from `decode_attention_*`) | after A5/A6/A7 |

8B and 14B share ONE geometry-driven lowering path (strategy string is metadata only); fixing A2
fixes both. 14B Hq=40 needs nothing extra (`AMDAttentionGridSpec.validate` requires `q_tokens%16==0`,
not `q_heads%16`).

## 7. Key files / commands

- Lowering crash site: `tinygrad/schedule/rangeify.py` (`lower_attention_semantic` 19-153; crash pass
  ~929-951; `_mop_index` 259), `tinygrad/codegen/late/composite_combines.py` (`_lower_composite_no_range_pm`,
  `_handle_no_range_generic` 205-272, `_pack_online_softmax_v_lanes` 173-192, `_combine_step_online_softmax_state`
  151-171).
- Native one-kernel swap (already correct): `tinygrad/codegen/opt/postrange.py:328-361`, kernel builder
  `tinygrad/schedule/wmma.py:545`.
- Model call site: `tinygrad/llm/model.py:600-616` (fused branch), mask at `model.py:589`.
- Eligibility/admission: `rangeify.py:48-49` (grid_shape), `tinygrad/uop/ops.py` `AMDAttentionGridSpec.validate`.
- Policy gate: `tinygrad/llm/prefill_policy.py:22` (`shared_attention_proven_eligible`, the 8 flags).
- Admission resource gate: `extra/qk/shared_attention_promotion.py:52` (VGPR cap, now 256).
- Isolated capture / proof: `extra/qk/generate_shared_attention_captures.py`, `extra/qk/shared_attention_evidence.py`,
  artifact `docs/artifacts/shared-attention-m10e1-20260723/shared_attention_proof.json`.
- Repro: `/home/ubuntu/.claude/jobs/6db6b205/tmp/fused_prefill_force_8b.py`; 14B: `.../test_14b_prefill.py`.
- Env: `DEV=AMD`, use repo `.venv/bin/python` (no `python3` on PATH); `AMD=1` is deprecated.

## 8. Related docs

- `docs/shared-attention-fused-enablement-scope-A-20260723.md` — the exhaustive scope + spike result.
- `docs/boltbeam-export-triage-8b-14b-20260723.md` — why the artifact was never produced (policy + cap).
- `docs/shared-flash-prefill-rewrite-handoff-20260723.md` — earlier rewrite framing (partly superseded).
- `docs/shared-flash-attention-rotating-pv-primitive-results-20260723.md` — the retired probe (history).
- `docs/ATTENTION_COMPACT_VGPR_LEASE_NEGATIVE_20260723.md`, `docs/SHARED_ATTENTION_LIVE_STATE_RESIDENCY_LEDGER_20260723.md`
  — the measured negatives that kill the VGPR-cutting lever.
- `docs/SHARED_ATTENTION_HANDOFF_20260723.md` — prior attention handoff (proof artifacts, multi-wave experiment).

## 9. First action for the new session

Do NOT continue crash-by-crash. Start A2 with the **design decision**: read `composite_combines.py`
and `lower_attention_semantic` in full, then decide (1) restructure inputs vs (2) harden collapse
passes — with the blast-radius trade-off explicit. Prototype the chosen approach against the §5 repro
and watch whether the crash family converges (each fix removes a whole class of degenerate-axis crash)
or persists (signal to switch design). Gate every change on: isolated capture still 254 VGPR/0 spills,
and `test_online_softmax_tile` + `test_shared_attention_compiler_capture` no new failures.
