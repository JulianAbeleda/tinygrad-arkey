# Packed-WMMA 14B canary codegen: the transition commit (bisect result, 2026-07-24)

Continues `docs/packed-wmma-14b-fault-trace-20260724.md`. That doc established that all six 14B
canary kernels differ in both `binary_sha256` and `source_sha256` between `c35b5ff53` (last
provably-working state, 1854-1858 tok/s, 6/6 gates max_abs 0.0) and HEAD, and proposed a
compile-only bisect on `source_sha256` to localize the transition. This is that bisect.

## Method
Compile-only, zero GPU risk throughout. Used the committed probe
(`extra/llm_research/prefill/canary_codegen_identity_probe.py`), extended with a local (uncommitted)
`source_sha256` field for the predicate, run inside disposable `git worktree add --detach`
checkouts (never in the main tree), one per candidate commit, removed immediately after. Predicate:
`Q4_K/ffn_down` `source_sha256 == c411513c65d79fcd...` (good) vs anything else (bad). Binary search
over the 496 commits between `c35b5ff53` and HEAD, 9 compiles to converge, plus targeted checks of
the shortlist and a full six-kernel diff at the transition boundary.

## Result: single commit, exact, no further bisection needed

**`114277f36` "[codegen] THEORY 2 closed: LDS bank conflict eliminated to zero, throughput unmoved"**
(2026-07-24 21:46:45 -0400) is the **entire** transition. Confirmed by direct comparison of its
parent vs itself:

| kernel | parent (`23b8e05f`, good) `source_sha256` | `114277f36` (bad) `source_sha256` |
|---|---|---|
| Q4_K/attn_kv, Q6_K/attn_kv | `5946fe93d86a0095...` | `9470cc683d59df7f...` |
| Q4_K/attn_qo | `a9f90c61fe37894e...` | `f00d8eef26572...` |
| Q4_K/ffn_down | `c411513c65d79fcd...` | `65319ac2156f17d6...` |
| Q4_K/ffn_gate_up | `c753e876380bb1bd...` | `14bc717219...` |
| Q6_K/ffn_down | `ee9c68b9daa40c2b...` | `f82d988976...` |

All six values at `114277f36` are byte-identical to the values measured at HEAD in the prior
report — nothing after this commit changes anything on this path. All six values at its parent
(`23b8e05f`) are byte-identical to the `c35b5ff53` known-good values. **This one commit accounts
for 100% of the codegen delta for all 12 (quant,role)-x-{8B,14B} canary identities, not just the
14B ones checked here.**

It moves **all six kernels simultaneously** (both quants, all four roles) — a generic codegen
change, not something isolated to the `Q4_K/ffn_down` `bc=2`/512-thread/61440B path specifically.

## The shortlist is cleared, except the one item already flagged as new
Directly tested (compile-only) at the shortlisted commits themselves; `Q4_K/ffn_down`
`source_sha256` is unchanged (`c411513c65d79fcd...`, i.e. still good) at every one of:

- `837889b30` scalarize packed wmma output store
- `200b36673` admit single-buffer attn_qo gate
- `86d71d38c` bind admitted one-buffer attention forward route
- `3649217ae` dedup prefill packed-WMMA stack
- `c7da22a61` revert vocab route

None of these — including the `one_buffer_payload` `buffer_count` 2->1 mechanism the prior doc
flagged as the leading OOB-LDS theory — touch this codegen path at all for the 14B canary shapes.
That theory is **ruled out**, not just "declines to apply" as the prior doc reasoned from the
admission-gate shape key. The bisect makes it unambiguous: the source is byte-identical through
all of them.

## Correction to the prior trace doc
The prior doc's shortlist entry for `114277f36` read: *"today — touches `kernel_lds.py`, which
this path uses; **postdates the observed fault** but is a new independent risk to this
geometry."* That framing was backwards. `114277f36` does not postdate the fault as an unrelated
risk sitting next to the real cause — **it is the cause of the entire codegen delta**, full stop.
Everything else on the shortlist is now cleared.

## Mechanism, and why it's a real candidate for the HW fault despite looking correct on paper
`114277f36` adds `cooperative_store_row()` (`tinygrad/codegen/opt/kernel_lds.py`), applied inside
`instantiate_precontract_producer` and `build_precontract_lds_stage`, both used by every
packed-WMMA precontract LDS store. For `vectors_per_row == 4` and `rows % 8 == 0` it remaps the
Python-compile-time row index via `(raw_row//8)*8 + (raw_row%2)*4 + (raw_row%8)//2` — a bijection
within each aligned 8-row block (`0..7 -> 0,4,1,5,2,6,3,7`), intended to spread the cooperative
store's bank access across all 8 quads instead of colliding 2-per-quad.

Algebraically this is a closed, in-bounds permutation of `[0, rows)`; it doesn't touch
`window.base`, `stride_bytes`, or LDS allocation size, and the loop bound (`loads =
rows*vectors_per_row // threads`, checked to divide evenly) is derived independently of the
rotation, so by construction every row in `[0, rows)` is still written exactly once. I don't see
an OOB in the rotation arithmetic itself.

The gap is in **what was validated versus what changed**. The commit's own numerics evidence is
PMC/throughput on the **attention** row as an unchanged-path control, plus `max_abs_err` PASS at
`Hd=64/128` for **8B SDPA vs FUSED** — none of that exercises the 14B packed-WMMA GEMM canary path
this bisect shows the commit actually rewrites (all four roles, both quants, 14B shapes included).
The commit message itself says "candidate-set.json untouched; identities unchanged" — true for the
*canary identity* (a hash of the workload spec, unaffected by this), but that statement papers over
the fact that the *compiled source* for the exact kernels this session gates on did change, and
was never re-verified against the packed-WMMA numerics/canary gate after the change. Given the
canary path uses `bc=2` double-buffered LDS (two windows back to back) and the rotation function's
correctness was reasoned about for a single window in isolation, the two-buffer interaction (does
`rows` as passed in still equal the *tile* row count, or could a caller on the `bc=2` path be
passing a stale/mismatched `rows` derived from a window that no longer matches the operand's true
extent post-dedup from `3649217ae`?) is the one thing not directly checked here — compile-only
localization stops at "this commit changed the source"; whether the interaction with `bc=2` is
where the OOB actually lives is a fix-phase question, not answered by this bisect.

## Skipped commits
None. Every commit tested (bisect probes at 0-indexed 247, 371, 433, 464, 479, 487, 491, 492, 493,
plus direct tests of `837889b30`, `200b36673`, `86d71d38c`, `3649217ae`, `c7da22a61`, `c35b5ff53`,
and HEAD) imported and compiled cleanly. No `IMPORT_FAIL`/skip cases were hit on this path.

## Reproduction
Worktrees were created with `git worktree add --detach <dir> <sha>`, probed with `PYTHONPATH=<dir>
python3 <dir>/_probe_tmp.py` using a copy of the committed probe extended to also print
`ev["source_sha256"]` (that field already exists in the compile-evidence dict; the committed probe
just doesn't print it), and removed with `git worktree remove --force`. No dispatch, no GPU
execution, at any point.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>

> **Probe removed 2026-07-25.** `extra/llm_research/prefill/canary_codegen_identity_probe.py` is deleted. Its verdict --
> all six 14B canary code objects byte-identical to the 6/6-gated state -- is recorded above.
