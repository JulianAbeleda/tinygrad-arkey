# Claude Execution Prompt — Fused Shared-Attention Enablement (8B/14B)

You are resuming a compiler workstream. This is a directive prompt. Follow it in order. Be
skeptical of your own "it works" claims — this codebase has repeatedly produced false passes
(fallback-vs-fallback numerics, dict-returns misread as success). Verify every claim by running.

## Objective

Make Qwen3 **8B and 14B prefill** actually run attention through the fused compiler-native kernel
`amd_gfx1100_q16_grid_hd128_loop_attention` and emit a BoltBeam artifact. Today both silently fall
back to SDPA. The kernel is already correct + fast in isolation; the production *lowering pipeline*
crashes before it can build that kernel from a real model graph. Your job is the compiler-robustness
fix that lets it, then the enablement plumbing.

## Read first (do not skip)

1. `docs/HANDOFF_fused_attention_enablement_20260723.md` — the full map (mission, dead ends,
   verified facts, tree state, work breakdown, files, repros). THIS PROMPT ASSUMES YOU HAVE READ IT.
2. `docs/shared-attention-fused-enablement-scope-A-20260723.md` — exhaustive scope + the spike result.
3. Skim: `docs/boltbeam-export-triage-8b-14b-20260723.md` (why no artifact),
   `docs/ATTENTION_COMPACT_VGPR_LEASE_NEGATIVE_20260723.md` +
   `docs/SHARED_ATTENTION_LIVE_STATE_RESIDENCY_LEDGER_20260723.md` (why VGPR-cutting is dead).

## Environment (verify at start)

- Hardware: AMD RX 7900 XTX, gfx1100, ROCm 7.2.4. Weights: `/home/ubuntu/models/Qwen3-{8B,14B}-Q4_K_M.gguf`.
- Use `DEV=AMD` and the repo `.venv/bin/python` (NO `python3` on PATH). `AMD=1` is deprecated.
- Commit on master, never branch. Commit/push only when the user asks. Co-author line:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- `git log --oneline -5` should show `d3e0bd45f` (handoff) at or near HEAD. Partial fixes are in
  `86e6442c6` (already pushed, gated off, inert).
- Delegate concrete sub-work to sonnet subagents; keep the gating decisions in the main loop. Long
  GPU runs (>~5 min) should be YOUR background bash jobs, not a subagent's (subagents lose in-process
  runs at their turn boundary).

## Hard rules — do NOT do these (all disproven this project)

1. Do NOT try to reduce VGPR / raise occupancy. Occupancy needs ≤128 (not 192); cutting VGPR
   measured a perf REGRESSION. 254 VGPR is fine and ships.
2. Do NOT resurrect the rotating-PV probe / `RotatingPV*` / `SoftmaxBridge*` / `ROTATING_PV_SEQUENCE`
   (retired, ~780 lines, commit `88b4e3ee1`).
3. Do NOT whack-a-mole one crash at a time. The crash is a systemic family; fix it structurally (below).
4. Do NOT hand-build a per-route kernel. The single-kernel builder (`wmma.py:545`, swapped in at
   `postrange.py:328`) already exists and is correct — do not touch it.
5. Do NOT fabricate proof flags. The 8 `shared_attention_proof` flags (esp. `decode_nonregression_*`)
   must be earned from real runs.

## The blocker (what you are fixing)

Real 8B/14B prefill through the fused path crashes in scheduling:
```
ValueError: bad reshape: () -> (1, 32, 1, 512, 128)
  tinygrad/schedule/rangeify.py  (symbolic+reduce_collapse+debuf pass, ~938-951)
  -> _mop_index (rangeify.py:259) -> UOp.shape (ops.py:389)
```
Repro (forces the path on, ~minutes on GPU):
```
DEV=AMD PYTHONPATH=/home/ubuntu/tinygrad-arkey .venv/bin/python \
  /home/ubuntu/.claude/jobs/6db6b205/tmp/fused_prefill_force_8b.py
```
(If that scratch file is gone, recreate it: load 8B via `extra.llm.generate.load_model_and_tokenizer`,
`object.__setattr__(model.config,"prefill_tc_attn",True)`, force policy strategy `FULL_RESIDENT_OVERLAY`,
run `model.logits(chunk_of_512, 0)`, and inspect `schedule_linear()` for a CALL whose KernelInfo has a
non-None `candidate_context`.)

Root cause: composite-reduce lowering (`tinygrad/codegen/late/composite_combines.py` +
`lower_attention_semantic` `rangeify.py:19-153`) expresses score/V/m/l alignment with **size-1
broadcast axes**. Under full-model DAG sharing (32 layers, hash-consed nodes, one mask node with 32
consumers, real RoPE/quant feeding Q/K/V), generic collapse passes fold a degenerate broadcast axis
to a scalar PARAM load WITHOUT updating the enclosing RESHAPE → shapeless node → panic. Score-side is
already fixed (`86e6442c6`); the current crash is the **V (`value_tile`) side**; more inputs may follow.
This path has NEVER lowered at production graph scale.

## Work order

### Step 1 — DESIGN DECISION (do this before writing any fix)
Read `composite_combines.py` and `lower_attention_semantic` in FULL. Choose and justify:
- **(1) Restructure inputs**: build the composite-reduce inputs without generic-pass-fragile unit
  broadcast axes (extend `CompositeInputSpec`/construction so no size-1 axis exists to collapse).
  Cleaner; more invasive; touches the construction/lowering boundary.
- **(2) Harden collapse passes**: make the generic rewrite that folds a broadcast source to a scalar
  also rebuild/remove the enclosing RESHAPE, so no shapeless node survives. More local to the bug but
  touches passes used everywhere (blast radius — must not perturb non-attention graphs).
Write the decision + rationale into the scope-A doc before coding.

### Step 2 — Implement the chosen fix
Apply it. Then run the §blocker repro. Classify the outcome:
- **Converges**: the crash family is gone (score AND V AND siblings) — proceed.
- **Next-crash**: a new degenerate-axis crash on another input — if it's the SAME family your design
  should have covered, your design is wrong; reconsider (1)↔(2). If genuinely new class, log it.
- **Same crash**: fix ineffective; debug.

### Step 3 — Reach the native-swap and confirm the real kernel
Once `schedule_linear()` succeeds, confirm it reaches `postrange.py:328` and emits the single
`amd_gfx1100_q16_grid_hd128_loop_attention` kernel for the REAL graph (not just the harness). The
admission gate at `postrange.py:343-344` is strict (param slots [0,1,2,3], exact sizes, scale as a
plain CONST) — if it falls through to the slow generic scalar kernel, that's follow-up work.

### Step 4 — End-to-end correctness (8B and 14B)
Run real prefill through the fused path vs SDPA fallback; compare next-token logits (max abs err +
argmax agreement). MUST prove the fused kernel actually fired (kernel name / non-None candidate_context
in the schedule) — do not accept a 0.0 error that is fallback-vs-fallback. This earns
`model_8b_prefill` / `model_14b_prefill`.

### Step 5 — Enablement tail (plumbing; see scope-A A5-A8)
- Whole-model prefill benchmarks KV 512..4096 (note: prior handoff defers these behind an in-flight
  multi-wave G2/G4/G5 experiment — check status first).
- BUILD a decode-nonregression harness (none exists; nearest `extra/qk/decode_runtime_overhead.py`):
  run decode with shared-attention on vs off, assert token-parity + no perf regression. Earns
  `decode_nonregression_8b/14b`.
- Build the composite-proof collector: assemble `shared_attention_proof` (target+geometry+v2 artifact
  + 8 flags) from real evidence and wire memory-adaptive adapter activation into model load (zero
  production callers today: `tinygrad/llm/memory_adaptive_authority.py:49`, `model.py:144`). Verify a
  real `from_gguf` load now sets `prefill_tc_attn=True` and routes through the fused kernel.
- Add a BoltBeam prefill-attention route entry in `BoltBeam/boltbeam/policy/route_manifest.py`
  (template from `decode_attention_*`); cite the proof artifacts.

## Verification gates (run on EVERY change; a fix that breaks these is not a fix)

1. Isolated capture unchanged: `DEV=AMD .venv/bin/python -m extra.qk.generate_shared_attention_captures
   --output-dir <tmp>` → every route still `vgpr=254, vgpr_spills=0, sgpr_spills=0, scratch_bytes=0`.
2. `DEV=AMD .venv/bin/python -m pytest -q test/unit/test_online_softmax_tile.py
   test/unit/test_shared_attention_compiler_capture.py` → no NEW failures vs baseline (baseline has 6
   pre-existing fails; compare sets, don't just count). To get the true baseline, `git stash` your
   change and rerun.
3. The §blocker repro schedules further than before (or cleanly).

## Success criteria (the workstream is done when ALL hold)

1. Real 8B AND 14B prefill schedule + run through the fused kernel (proven by kernel name in schedule),
   numerically matching SDPA within tolerance (~<1e-2 rel).
2. The composite `shared_attention_proof` is assembled by production code at model load, with all 8
   flags earned from real runs, and a real `from_gguf` load enables `prefill_tc_attn` and produces the
   fused kernel + artifact.
3. BoltBeam has a prefill-attention route entry that consumes the artifact.
4. No regression: isolated capture still 254/0; the two unit-test files no new failures; SDPA fallback
   still correct where the fused path is not selected.
5. Perf: fused prefill ≥ SDPA on whole-model benchmarks for both models (the 3.7-4.4× isolated win
   should carry, but MEASURE — do not assume).

## If blocked

- If the crash family does NOT converge after the structural fix (step 2 keeps yielding new
  degenerate-axis classes), STOP and report: the composite-reduce lowering may need a deeper rework
  than either design (1) or (2); write the enumerated crash classes into the scope-A doc and escalate
  the design question to the user before spending more GPU time.
- If step 3's native-swap admission gate rejects the real AST, that is a separate, bounded task
  (make the real scalar AST match the `postrange.py:343-344` shape contract) — scope it, don't force it.
- Keep every partial fix committed as labeled WIP (path stays gated off, so it's inert) so progress
  is never lost.
