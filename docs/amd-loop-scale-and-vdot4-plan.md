# Scope: scale-the-substrate (S) + v_dot4 decode lowering (D)

Date: 2026-06-15. Two directions, grounded by two code-map agents. Execute S1 → S2 → D0 → D1.

## S1 — partial-schedule dataset → retrain → retry L2 (close the L2 boundary)
L2 failed because the model trained on COMPLETE 277-config schedules can't score native BEAM's PARTIAL
schedules. Fix: train on partial schedules harvested from real BEAM over its full action space.
- **Hook**: env-gated logging in `tinygrad/codegen/opt/search.py` right after `timed.append(...)` (~L164):
  when `BEAM_SCHEDULE_LOG` set, append `{shape(full_shape), opts(applied_opts), device_us}` per timed
  candidate. Default off = no behavior change.
- **Harvest** (`extra/qk_partial_schedule_log.py`): run real BEAM (amt≈4) on ~10 corpus shapes with the
  hook on → a JSONL of (shape, partial-opts, device_us) over BEAM's REAL action space (incl.
  SWAP/GROUP/THREAD the model was blind to).
- **Retrain + retry**: train the booster on the partial-schedule dataset; re-run the L2 warm-start filter
  on the fresh shape. Gate: SOME keep_k both saves wall-clock AND preserves quality ≥0.97 (the L2 gate).
  Honest: if it still fails, the OOD problem is deeper than the dataset (real result).

## S2 — does the loop generalize beyond matmul? (CONV; attention is just matmuls)
Map finding: attention's autotuning targets (Q@Kᵀ, Attn@V) ARE matmuls (already covered) — so the
genuinely new op is **conv**. Conv lowers to a final sum-reduce kernel (pad→im2col→reduce);
`helper_realized_ast(conv_result.realize())` yields that kernel's ast.
- **Dataset** (`extra/qk_conv_beam_log.py`): pick ~6 realistic conv shapes (ResNet/1x1/3x3); enumerate the
  same `gen_candidates()` opt space; live-time each → `conv_beam_log.jsonl`. Features: extend
  `_shape_feats` to conv dims (or map conv→(M,K,N)-equivalent of the reduce kernel).
- **Learnability + live** (`extra/qk_conv_loop.py`): N1-style leave-one-shape-out (does a cost model
  predict good conv configs?) + L0-style live guided search on a held-out conv shape. Gate mirrors N1/L0
  (top-1 high, beats lookup, guided@8≥0.95 live). Honest: conv may be flatter/less learnable — real result.

## D0 — v_dot4 lowering make-or-break (cheapest first)
AMD renders to HIP C++ → v_dot4 emits as the intrinsic `__builtin_amdgcn_sdot4(a,b,acc,0)` (signed) /
`__builtin_amdgcn_udot4` (unsigned). This fork already has the `Ops.QK_BLOCK_DOT`/`CUSTOMI` format-string
escape the existing Q4_K kernels use.
- **Build** (`extra/qk_vdot4_probe.py`): a minimal int8×4 dot kernel — `acc = sdot4(pack4(a), pack4(b),
  acc)` over an int8 reduction — via a custom op emitting the intrinsic. NO PatternMatcher needed
  (construct it explicitly like the existing custom kernels).
- **Verify (make-or-break)**: (1) compiles, (2) numerically correct vs a reference int dot, (3) the
  disassembly actually contains `v_dot4_i32_i8` (not scalar `v_mad`). If v_dot4 is emitted + correct → D0
  PASS, proceed to D1. If the compiler refuses / scalarizes → D0 documents that tinygrad's HIP path can't
  reach v_dot4 cleanly (real boundary), stop.

## D1 — v_dot4 Q4_K decode kernel + measure (gated on D0)
Route the Q4_K int-dot GEMV through the v_dot4 op (nibble-unpack → pack int8 → sdot4 accumulate →
per-group affine). Measure: (1) VALU/weight from disasm (target ~1.35 vs fp 4.06), (2) e2e decode tok/s
vs fp 58 / llama.cpp 104. Honest report: does DP4A codegen actually close the decode gap, or do other
costs (nibble unpack, occupancy) dominate? Either way it's the measured answer to the consolidated doc's
open lever.

## RESULTS (2026-06-15)

**S1 — GPU-BLOCKED (not a code failure).** Added the default-off `_BEAM_SCHEDULE_LOG` hook (search.py) +
`qk_partial_schedule_log.py`. But running real native BEAM over its FULL action space repeatedly HANGS
gfx1100 (`Wait timeout: signal not set`, `memory_lost=1` HW faults) — the action space contains
configs that hang this GPU, poisoning the process. Small ops and the curated 277-config substrate
(L0/L1) are fine; the full BEAM space is not. This is itself the infra reason the curated substrate
exists. S1 (harvest partial schedules over the real action space) is blocked on this hardware; the
hook stays for a future stable run.

**S2 — BLOCKED (opt-space mismatch + likely flat).** Conv ASTs build fine via
`helper_realized_ast(conv.realize())` (final reduce kernel), but the matmul candidate set
(`gen_candidates`, TC/UPCAST on axes 0/1) fails on conv's reduce kernel with `KernelOptError` — conv's
axis layout differs, needing a bespoke conv opt-candidate set. The conv reduce baseline is tiny (0.1 TF,
memory-bound), so it is likely "flat" (not a rich learnable substrate, like the GEMV). Deferred: needs a
conv-specific candidate set before learnability is even testable.

**D0 — PASS (major).** `qk_vdot4_builtin_d0.py`, `dp4a-d0/BUILTIN_VS_ASM_RESULT.md`. The schedulable
builtin `__builtin_amdgcn_udot4` (gfx1100, unsigned, `target("dot-insts")` attr) emits v_dot4 and at full
occupancy hits **169.6 Q4-GB/s ≈ fp's 173**, **2.54× over the asm-volatile v_dot4** (66.7), exact-correct.
Phase D's "DP4A is the wrong lever" was an **asm-volatile-barrier artifact**; the builtin realizes the
consolidated doc's predicted instruction-count floor (~1.58 VALU/weight vs fp 4.06). The decode
instruction-count lever is REAL and kernel-competitive — reopening the decode question.

**D1 — PARTIAL (kernel-competitive shown; e2e pending).** D0 already proved the builtin GEMV is
kernel-competitive with fp standalone. The e2e decode test needs the `target("dot-insts")` attr on
tinygrad's GENERATED kernel (a core render_kernel change — the inline CUSTOM-op body can't set it), and
faces the occupancy/pipelining wall that killed every prior standalone-fast kernel e2e (int-dot 242→136).
Open: whether the builtin's lower instruction/register count lets it pipeline e2e and beat fp's 58 tok/s.

## Honesty / pre-registration
- S1/S2/D each have a pre-registered gate; a null is a real, reported result (the program's pattern).
- The default-off search.py hooks keep tinygrad's normal behavior unchanged.
- D1's target is the decode gap; S1/S2 are about the loop's reach. They are independent.
