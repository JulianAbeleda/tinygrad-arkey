# Packed-WMMA 14B HW-fault: trace and localization (2026-07-24)

Supersedes the diagnosis in `docs/BOLTBEAM_GPU_HANG_DIAGNOSIS_HANDOFF_20260724.md`. That doc's *path* is
right; its framing ("debug the kernel/geometry") points at the wrong thing, and two of its file paths are
stale.

## THE HEADLINE: this is a REGRESSION, not a design flaw

Commit `c35b5ff53` (2026-07-21, "production packed-WMMA candidates") records, verbatim:

> Verified real Qwen3-**14B** pp512 via the NORMAL dispatcher (no override): opted-in **1854-1858 tok/s**
> median (past llama 1837), 100% coverage, **6/6 gates max_abs 0.0**, finite; opted-out byte-identical
> **354 tok/s**.

So three days ago all six canary gates passed on 14B shapes with zero error, and the route delivered
1854 tok/s. The "opted-out 354 tok/s" in that message matches **exactly** what we measured today
(352-364 tok/s with `TINYGRAD_PREFILL_PACKED_WMMA=0`). The documented ~5x 14B prize is `1854/354 = 5.2x`,
and it was once realized. **Something between `c35b5ff53` and HEAD broke it.**

## Corrections to the prior handoff
- `process_isolated.py` is at **`tinygrad/runtime/`**, not `extra/llm_research/`. The stale path caused a later agent
  to report the file missing.
- It conflates two distinct failures: (a) the GEMM HW-fault, and (b) a spawn deadlock where the child
  re-execs, dies on `budget 0.0GB, KV admits 0` because the parent holds ~19GB, and never drains the pipe
  so `run_isolated`'s timeout never fires. Both are real; only (a) is the performance blocker.
- "Never run 14B" (an instruction I gave agents) is **too strong and cost a day of evidence.** 14B runs
  fine *with* `TINYGRAD_PREFILL_PACKED_WMMA=0`; the thing never to do is run it *without*.

## Fault path (confirmed)
`tinygrad/llm/model.py:_build_packed_wmma_warmstart` -> `build_packed_wmma_warmstart_tables(self.
_prefill_v2_covered(), ubatch)` -> `gate_combo(quant, role, shape)` -> `_run_gate` -> `build_artifact` +
`run_canary` -> `run_isolated_guarded_execution` (spawned child).

Shapes are derived from the **live model's** covered linears, so 14B gates 14B shapes.

## Ruled out — all statically, no GPU, no fault risk

| layer | result |
|---|---|
| tile/shape divisibility | all four 14B shapes divide by `tm`/`tn`/`tk`; `K%256==0` for Q4_K superblocks |
| **admission** | **all 12 (quant,role) x {8B,14B} combos ADMIT** — not the discriminator |
| LDS overflow | ruled out by construction: `runtime_specs.py:448` checks `active_lds > capability.max_lds_bytes` and passes. Max demand is `Q4_K/ffn_down` at `(256+128)*80*2 = 61440 B` of 65536 |
| **compile** | **all 12 compile clean** and produce binary shas |

So the fault is at **execution** of an admitted, successfully compiled kernel.

## The codegen delta (the live lead)

Compiling all six 14B canary kernels at `c35b5ff53` (in a `git worktree`, compile-only) versus HEAD:
**all six differ in BOTH `binary_sha256` AND `source_sha256`.** The emitted kernel for this exact path
changed. Example (`Q4_K/ffn_down`, shape 512x5120x17408):

    GOOD c35b5ff53  binary e33eb2cfcdacfabf...  source c411513c65d79fcd...
    HEAD            binary ba3c9072fb47f493...  source 65319ac2156f17d6...

Reproduce with `scratchpad/canary_binary_sha.py` / `canary_res.py` (compile-only; `resource_summary`
carries only the two shas, not LDS/VGPR — worth extending).

## Geometry provenance — a real latent defect, but NOT this fault
`PACKED_WMMA_GEOM` (`extra/llm_research/prefill/packed_wmma_prefill_candidates.py:38`) is keyed by
**`(quant, role)` only, not shape**, and the candidate set contains only 8B payloads, so every role goes
through `rebind_full_kernel_workload`. That function's docstring delegates legality to admission
("Admission remains responsible for proving that the retained schedule is legal for the new shape"), but
`runtime_specs.py:307` **overwrites `applicability`** to assert the new profile, and admission
(`:392-395`) then validates the workload against the claim rebind just wrote. **The check is circular** —
which is why all 12 combos admit and admission can never fail.

The invariant slot (`applicability` with `exact_shape: true`) already exists and is being laundered. This
violates *encode-invariants* and the *"convenience APIs that bypass canonical validation"* anti-pattern,
and should be fixed regardless. But it is NOT the cause here: the geometries were gated at **14B** shapes
(per `c35b5ff53`), so 14B is the *proven* configuration, not the extrapolated one.

## Next step: bisect on the source sha (zero GPU risk)
495 commits separate `c35b5ff53` from HEAD; ~20 touch this path (`git log c35b5ff53..HEAD -- extra/qk/
prefill/ extra/llm_research/runtime_specs.py tinygrad/llm/model.py extra/llm_research/kernel_lds.py tinygrad/codegen/opt/
kernel_lds.py`). Because the kernel compiles at every commit, the transition points of `source_sha256`
enumerate a small candidate set **without ever executing the faulting kernel**. Each transition is then a
judgment call on whether the change could produce an out-of-bounds access.

Shortlist worth reading first (all 07-23/24, all on this path): `837889b30` scalarize packed wmma output
store (adds `.contiguous()`; a fix for the same make-floatN class, and its commit body is EMPTY — no
verification recorded), `200b36673` admit single-buffer attn_qo gate, `86d71d38c` bind admitted one-buffer
attention forward route, `3649217ae` dedup prefill packed-WMMA stack, `114277f36` LDS bank-conflict
row re-election (TODAY — touches `kernel_lds.py`, which this path uses; postdates the observed fault but
is a new independent risk to this geometry).

Note `one_buffer_payload` flips `buffer_count` 2->1 while leaving the LDS windows unchanged, so the
allocation becomes `windows.end x 1` instead of `x 2`. If the kernel still ping-pongs between two LDS
halves, the second half lands outside the allocation — a textbook OOB-LDS -> `memory_lost=1`. The
`_SINGLE_BUFFER_ATTN_QO` evidence block is shape-keyed to **8B** `(512,4096,4096)` and
`_single_buffer_attn_qo_admitted` compares `(quant, role, shape)` exactly, so it *declines* for 14B — but
this is the mechanism to check first if the bisect lands nearby.

## Why this matters
14B is currently 352-364 tok/s against a proven 1854. That is a **5.2x** regression on the larger model,
larger than every lever pursued this session combined, and it is also what makes 14B unusable as a
measuring instrument — which is why today's evidence had to be 8B-only.
