# Option A research measurement — A0/A1 PASS (JIT-dim gate cleared); A2+ status + handoff

Executed the research-measurement scope (`prefill-tensile-research-measurement-scope-20260619.md`) — Option A,
research-only, no default/ship/policy change. **A0 PASS, A1 PASS** — the single gating engineering step (injected
Tensile node JIT-capturable with correct dims) is cleared. A2 (standalone one-block harness) is blocked on a probe
plumbing quirk, not the mechanism; the clean path to the pp512 number is A3 in-model. Honest status + handoff below.

## A0 — preflight: PASS
- assets present: `kernarg_all.jsonl` (roles ffn_gate_up/ffn_down/attn_q_o/attn_k_v), `shape_matrix.json`, `inject.json`;
- eager injection smoke (`qk_tensile_inject.py`) still PASS, rel_err 3.7e-4.

## A1 — JIT-dim minimal proof: PASS [M] (`bench/qk-tensile-extraction/jit_dim_proof.json`, `extra/qk_tensile_jit_dim.py`)
The injected precompiled Tensile kernel runs **under TinyJit/HCQGraph with correct launch dims**, correct on replay.
4 TinyJit calls; cnt≥2 (HCQGraph replays) rel_err ~3.4–3.9e-4; eager regression still PASS. Two probe-local fixes (no
UOp surgery, no model/default change):
1. **rebindable kernargs** — `TensileRunner.fill_kernargs` binds the 4 pointer VAs via `bind_sints_to_buf` (offset 16:
   D,C,A,B consecutive) so symbolic JIT-rebound input VAs are graph-updatable (`struct.pack_into` broke on the
   symbolic `inp_0_0`);
2. **dim override** — monkeypatch `AMDComputeQueue.exec` to use `TensileRunner.tensile_global/local` (tinygrad cannot
   EMIT the `(4,96,1)/(128,1,1)` grid — it always reserves grid dim0 for the local threads; confirmed via range
   shaping + `OptOps.LOCAL`, both gave `(128,96,4)`).

**This was "the single remaining engineering gap for Option A" per the scope — it is cleared.** The mechanism works:
an external precompiled Tensile kernel executes as a JIT-captured tinygrad graph node, correct, no copies, no HIP
runtime.

## A2 — one-block graph route: BLOCKED on probe plumbing (not the mechanism)
`extra/qk_tensile_block_jit.py` routes a full FFN block (gate/up/down) in `[feature,T]` space under TinyJit. It is
blocked by a **standalone-harness quirk**: in this module's specific import/setup combination, the `R.get_runtime`
monkeypatch used to capture each role's kernel key during warmup does not intercept (the same pattern fires correctly
in `qk_tensile_inject.py` and in isolated repros incl. TensileRunner+exec-patch+TinyJit each separately). Time-boxed
bisection did not isolate the interaction. **This is a harness key-capture issue, not a limitation of the route** —
A1 already proved the node captures+runs correctly under JIT.

Recommendation: **do A2/A3 in-model, not in the standalone harness.** In the real PREFILL_V2 forward the model's own
realize/JIT compiles the trivial kernels natively, and the runtime swap can be installed after the first warm prefill
realize (or via a model-side primitive hook), sidestepping the standalone get_runtime-capture quirk entirely. The
`TensileRunner` + dim-override + rebindable-kernarg pieces (all A1-proven) are exactly what the in-model route needs.

## A3/A4 — in-model route + pp512/dNLL: NOT YET RUN
Requires routing PREFILL_V2's high-share linears (ffn_gate/up, ffn_down) through Tensile-injected `custom_kernel`
nodes behind `PREFILL_TENSILE_GEMM=1`, then warm pp512/pp1024 + dNLL. Expected ~1.40× (TPE-5 weighted) / ~1.74× FFN
matmul (TPE-6b). Not run — the standalone A2 de-risking stalled on the harness quirk; A3 is the next step.

## Verdict (this pass)
- **A1 PASS** — JIT-dim capture proven; the gating step is done.
- **A2/A3/A4 incomplete** — no in-model pp512 number yet; A2 standalone harness blocked on a key-capture plumbing
  quirk, A3 in-model not started. Per the scope's verdict set this is **not yet** PASS_RESEARCH (needs A4 pp512); it
  is a partial result: the mechanism is JIT-proven, the in-model measurement remains.

## Handoff to Codex (precise next step)
The hard part is done (A1). For the pp512 number, route in-model behind `PREFILL_TENSILE_GEMM=1`:
1. In the prefill Linear (model.py, `[out,in]` weight, `[feature,T]` activation), when the flag is set and shape ∈
   {ffn_gate/up, ffn_down}, replace the matmul with a `custom_kernel(out, A, B, fxn=trivial)` whose runtime is a
   `TensileRunner` for that role.
2. Install the runtime swap once after the first warm prefill realize (model realizes the trivial kernel natively →
   capture its program key from `runtime_cache` or a model-side hook → swap to `TensileRunner`); keep
   `AMDComputeQueue.exec` dim-override + rebindable `fill_kernargs` from A1.
3. Measure warm pp512/pp1024 + dNLL ≤0.01 vs PREFILL_V2; verify fallback (flag off == PREFILL_V2) and decode untouched.

Assets: `extra/qk_tensile_{runtime,jit_dim,inject,block_jit}.py`, `bench/qk-tensile-extraction/{jit_dim_proof,kernarg_all,shape_matrix}.json`.
All A1-proven and ready. No default/ship/policy change.

## Files
`extra/qk_tensile_jit_dim.py` (A1, PASS), `extra/qk_tensile_block_jit.py` (A2, in-progress; documented blocker),
`bench/qk-tensile-extraction/jit_dim_proof.json`, this doc. No kernel/model/default changes; runtime change is the
probe-local `AMDComputeQueue.exec` dim-override (research probes only, not imported by the model).
