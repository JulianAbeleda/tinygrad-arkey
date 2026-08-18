# NV three-front audit: schedule, reduce_output, vocab_aux (2026-08-18)

Date: 2026-08-18
Branch: `nvidia-bringup-20260731`, HEAD `daf591ad3`
Status: **audit findings, read-only. Exhaustive "why" for the three rows the
side-by-side comparison exposed (schedule / reduce_output / vocab_aux), with
arithmetic at HEAD and a per-row verdict. Companion docs: the three detailed
audits this summarizes (schedule, reduce_output, vocab_aux) plus
`nv-us-vs-llama-side-by-side-20260818.md` and the FUSE/HIDE/ELIMINATE ledger.**

The question being answered: the side-by-side shows we win 5 of 9 classes and
the total work ledger (node_sum -496.3 us below llama), yet lose 729.4 us of
wall. The three rows that are "ours to lose" are the schedule (overlap 0 vs
1125.1), reduce_output (+312.1), and vocab_aux (+59.5). This document answers
why, with arithmetic, and says whether anything can still move the needle.

## 1. Schedule: why we lose overlap (structural, closed)

Fresh exact wall ledger at HEAD this session (HCQGraph profiler, same harness
as the 08-17 account): `steady_tokens=6`, `node_sum=4514.75us`,
`union=4514.75us`, **`overlap=0.0us`** (wall 6.31ms in the profiling child,
which includes host-side item-sync overhead; the exact unprofiled authority
remains the 08-17 pair at 4.7883ms / 208.84 tok/s). The S4 PDL exec-path
wiring did not change the raw overlap: still zero, and the decode A/B this
session (205.99 vs 205.55 tok/s, tokens identical) is wall-neutral.

Why:

1. llama's overlap is produced by CUDA graph internal streams (865 overlapping
   kernel pairs/replay on one reported streamId, negative inter-kernel gap
   median -4961 ns) plus programmatic dependent launch (`ptxVersion >= 90`,
   `cudaTriggerProgrammaticLaunchCompletion` at kernel start).
2. The native NV runtime's QMD latch releases the dependent grid at the
   **last CTA trigger**, not at kernel start (probe rows: +64/+305/+905 us on
   synthetic spins; `nv-pdl-substrate-verdict-20260817.md`). Per-edge overlap
   on real decode kernels is therefore bounded to a final wave, and the
   decode A/B measures wall-neutral.
3. The mass llama hides is mostly its own non-fused cost structure:
   quantize_q8_1 + rope + kv_set_rows + rms_norm (~571-752 us of separable
   mass we already fused away; our node_sum is 496.3 us below llama's). There
   is no kernel mass left to hide on our side, and the exposed flash remainder
   (~140 us) is dependency-bound on the critical path with zero independent
   pairs at body parity.

Arithmetic: the transferable overlap ceiling is **~18-33 us** (launch-hiding
audit 08-13), i.e. ~0.7-1.3 tok/s, ceiling ~209.6-210.3. The 1125.1 us figure
is llama's own cost structure, not a lever we can pull.

**Verdict: structural. The schedule loss cannot be fixed by substrate work at
HEAD; the substrate is proven but economics-negative on the real route.**

## 2. reduce_output: why the +312.1 us row is mostly closed (bitwise-blocked)

The row is a taxonomy artifact plus a geometry gap that the exact-output
contract closes:

1. llama does not absorb the norms into its mmvq GEMV; it runs standalone
   `rms_norm_f32` kernels (303.5 us in llama's `norms` class). The wall
   account's "llama = 0.0" is a taxonomy fold, and the honest delta was
   +138.1 us (old census), not +312.1.
2. P1 per-row grid promotion is in the tree at HEAD and captured **+55.31 /
   +66.55 us of wall** with exact logits sha `6ec7227e...` identical.
3. The remaining q/k gap (~130 us at 1:1; ours 3.07-3.14 us/launch vs llama
   1.30 us/launch) is **bitwise-blocked**: llama's 1.30 us geometry requires a
   tree `block_reduce` that reorders the fp32 sumsq and flips the pinned
   full-logit token sha. P2 (the attempt) regressed -631.6/-672.5 us; M1 fold
   (+81.92 us) and phase6 (18.5 us slower) are NO-GO.

Fresh at-HEAD census confirms the row at 383.54 us / 91 launches with the
4096 side at parity. The ledger's "L1 open remainder ~0" is correct.

**Verdict: closed. No bitwise-preserving absorption exists in the record; the
remaining wall value is ~0.**

## 3. vocab_aux: why the +59.5 us row is hidden mass (closed)

llama never runs a GPU argmax: its mmvq writes the logits buffer, and top-1
happens on the host, hidden under llama's 1125.1 us of in-graph overlap. We
lower `logits.argmax(-1)` to a 4-kernel GPU chain (E_1187_32_4 x2 + r_32_4_1187
+ r_128_16_8_1187 + r_16_8 = 59.23 us, 5 launches), which hides behind our
319.87 us vocab GEMV anchor.

The 08-17 A/B removed ~16.5 us of tail node and moved the wall only 1.55 us
(~9.4% transfer). Scaling to the full fresh tail (59.23 us): honest remaining
wall value is **~5.6 us (~+0.25 tok/s)**, not 59.5 us. The fused lease route
is bit-exact but only renames the tail 4 -> 2 (scheduler lowers
`packed_argmax_from_tile_keys` to its own reduce pair); the harness
`tail_family` gate now catches the rename.

The only reopen path is codegen: a single-pass in-GEMV cross-tile argmax
(blocked: no atomics / grid sync in the PTX renderer) or a cheaper 32-bit
comparable key (the packed u64 route is 142.6 us vs Tensor.argmax's 71.9 us in
the microgate - the ordinary NV lowering of the 64-bit key build/reduce is the
cost, 2x slower).

**Verdict: closed as wall mass. Real ceiling ~5.6 us; below the +50 us
promotion bar. Reopens only via codegen, and even a topology-legal build is
capped there.**

## 4. The arithmetic summary at HEAD

| row | ledger node mass | honest wall value | verdict |
| --- | ---: | ---: | --- |
| schedule / overlap | 1125.1 (llama) / 0 (us) | ~18-33 us transferable | structural |
| reduce_output | 383.5 | ~0 (P1 landed, rest bitwise-blocked) | closed |
| vocab_aux | 59.2 | ~5.6 us (~+0.25 tok/s) | closed (hidden mass) |

The three rows are not "open levers" - the side-by-side's behind-rows mostly
collapse on inspection. The remaining clean wall lever is the host gap (L6,
~100.6 us ceiling, ~+4-5 tok/s), which the audits all point to as the last
non-codegen row. The two genuine upside rows remain the ELIMINATE/codegen
targets: searchable flash shape and cheaper packed-key reduce.

## 5. Evidence

- Fresh wall ledger at HEAD: `/tmp/nv_decode_wall_ledger_head2` (this session)
- Fresh census: `docs/task_workflow/evidence/nv-full-audit-census-head-20260818.json`
- Detailed audits: `/tmp/audit_schedule_why.md`, `/tmp/audit_reduce_output_why.md`,
  `/tmp/audit_vocab_aux_why.md`
- Authority: `nv-240-exact-wall-account-20260817.md` (residual 0.0),
  `nv-us-vs-llama-side-by-side-20260818.md`
