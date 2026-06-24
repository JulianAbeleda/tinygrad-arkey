# Branch B: Flash/TC Attention on Concrete KV — Scope + Execution Plan

Date: 2026-06-20
Repo: `/home/ubuntu/tinygrad-arkey`, branch `qk-prefill-flag-leak-resolution`. GPU gfx1100. Model Qwen3-8B-Q4_K_M.
Follows: `docs/prefill-graph-route-attribution-result-20260620.md` (Step 1 → Branch B).

## Why Branch B (recap)

Step-1 audit: on the promoted graph route, **attention is ~47% of the prefill forward**, concentrated in one
naive non-TC reduce kernel `r_2_512_*start_pos*` at **30% of the whole forward**. PMC: attention is ALU-bound
with no tensor cores; the matmuls are already WMMA/cache-compute-bound (so Branch A int8-MMQ premise is
measured-false). llama spends ~4.4% on flash attention vs our ~47% → the gap lives here.

## Key finding that reopens this: the prior "REFUTED 0.79×" is INVALID

`extra/qk_prefill_tc_attention_measure.py` + `bench/qk-prefill-tc-attention/result.json` concluded "TC attention
REFUTED in-model 0.79×, symbolic KV blocks TC." That result is invalid on **two independent counts**:

1. **Env-var typo.** The harness sets `PREFILL_TC_ATTENTION`; the model reads `PREFILL_TC_ATTN` (model.py:40,876).
   So `PREFILL_TC_ATTN` was always 0 → **both arms ran SDPA**; the 0.79–0.92× was cross-subprocess clock noise
   (the documented clock confound).
2. **Symbolic start_pos.** The harness binds `vsp.bind(sp_val)` (a UOp), but the TC path requires
   `isinstance(start_pos, int)`. A bound UOp is not an int → the TC branch could never fire even with the right
   env name.

→ TC attention has **never been measured in its valid regime** (concrete int start_pos). The conclusion was
inferred from a harness that could not exercise the path.

## The path (already wired, flag-gated)

model.py:876–885 — `PREFILL_TC_ATTN` + `_prefill_v2` + `isinstance(start_pos,int)` + `T!=1` → explicit TC
attention: `Q@Kᵀ` (fp16 TC) → fp32 softmax → `P@V` (fp16 TC), GQA via broadcast. Fires on the concrete first
chunk (`start_pos==0`, always concrete per model.py:1285) or every chunk with `PREFILL_CONCRETE_KV=1`. The
attribution regime (start_pos=0, KV=512) already shows attention at 47% / the r_2_512 kernel at 30%, so the
default-on first-chunk regime is exactly where the lever applies.

## Execution plan

A correct A/B in the regime where the path can fire. **Iron law: synced measurement only; no cross-subprocess
clock comparison; gate correctness + quality.**

1. **Build a correct harness** `extra/qk_prefill_tc_attn_concrete_gate.py`:
   - **Concrete int start_pos** (0; KV=512) so `isinstance(int)` holds and TC fires. Graph route ON in both arms
     (isolate the attention delta on the promoted baseline).
   - **Same-process interleaved A/B** (the gold standard per the Tensile reconciliation), respecting the
     TinyJit flag-leak rule: capture the OFF jit fully with `PREFILL_TC_ATTN=False`, THEN set True and capture
     the ON jit fully. Build explicit `TinyJit(model.forward)` per arm (model.__call__ keys jits by start_pos →
     the two arms would collide on key 0). Replicate __call__ setup (`block._prefill_v2=True/_use_flash=False`,
     install `pr._WARMSTART_OPTS = model._pf16_warmstart` during capture).
   - **Kernel-identity assert (flag-leak guard):** dump per-kernel names from each captured graph; assert the ON
     graph contains a `wmma`/TC attention kernel the OFF graph lacks, and report the attention-bucket share for
     each. If identity fails, the A/B is void.
   - **Correctness:** `rel RMSE(off_out, on_out) < 1e-2`.
   - **Synced arbiter:** K=8 forwards / one `dev.synchronize()` / total/K, clock pinned `high`, best-of-N.
     Report ms512 OFF vs ON, speedup, tok/s, % of llama (3020).
2. **If it wins** (and identity + correctness pass): run the **quality gate** — sampled/chunked NLL dNLL ≤ 0.01
   + greedy-exact over a short generation with `PREFILL_TC_ATTN=1` (reuse the graph-gemm quality harness shape).
3. **Decision:** win + all gates → propose gated default-on (gfx1100, owner-approved, like the graph route).
   Else document the real in-regime number and rest. Write `docs/prefill-branch-b-tc-attention-result-<date>.md`.

## Risks / unknowns

- **Score materialization at long KV.** Explicit TC materializes the `Hq×T×KV` fp16 score tensor (117 MB at
  KV=3584); SDPA may fuse. At KV=512 (default-on regime) this is small (~16 MB) — start there. Long-context
  concrete-KV is a separate follow-up (needs `PREFILL_CONCRETE_KV`, K jits).
- **Does TC actually fire** for the (Hkv,G,T,Hd)×(Hkv,1,KV,Hd) shapes — the kernel-identity assert answers this
  empirically; if no wmma kernel appears, the cast-to-fp16 isn't triggering tensor cores and the path needs a
  reshape/contiguity fix before it's worth measuring.
- **Compile-fault risk** with multiple in-process 8B jits — keep to 2 jits (OFF/ON) at one context first.

## Gates (iron law)
rel RMSE < 1e-2 + sampled/chunked NLL dNLL ≤ 0.01 + greedy-exact + SYNCED arbiter vs llama + kernel-identity +
fallback/OOM. Default-off unless owner-approved; gfx1100-restricted. No BEAM.
