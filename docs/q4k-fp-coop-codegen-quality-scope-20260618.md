# Arc B — Q4_K ffn_gate fp-coop codegen quality (Phases 1–2) → **codegen lever REFUTED; one sub-gate candidate found**

Research primitive: `q4k_ffn_gate_fp_coop_codegen_quality`. Question: is the remaining Q4_K ffn_gate/up decode gap
to llama a *per-thread codegen* problem (register-alloc / scheduling / ILP) with a bounded renderer lever?
Scope: the **byte-identical fp path only** (q8/int-dot is separately pack-walled and excluded). Method: source/ISA
diff (Phase 1) → identify a bounded lever (Phase 2). gfx1100, Qwen3-8B-Q4_K_M. **No default changed.**

## Verdict

**REFUTED at Phase 1–2: there is no bounded fp codegen lever.** The premise rested on the ladder tinygrad 57% /
handwritten-HIP 65% / llama 70% — but **that ladder is the int-dot (sudot4 + q8) path, not the fp path.** On the
byte-identical **fp** path the metal says the opposite:

| kernel | %peak | VGPR | spills | inner-loop signature (per 32-weight block) |
|---|---:|---:|:--:|---|
| tinygrad fp-coop | **48** | — | — | `(float)(bitcast<half>(w))·(float)(act)` dequant+FMA (same arithmetic) |
| **handwritten HIP fp (W4A16, clang -O3)** | **49** | 93 | 0 | 34 `v_cvt_f32_ubyte` + 32 `v_fma_f32` + 32 `v_fma_mix_f32` + 32 `v_cndmask` + 59 `v_lshrrev`/40 `v_and` |
| handwritten HIP int-dot (sudot4) | 65 | **36** | 0 | 16 `v_dot4_i32_iu8` (4 MAC each) — **no convert, no per-weight FMA** |
| llama (int-dot) | ~70 | — | — | int-dot + QR4_K=2 unroll |

**tinygrad's fp path (48%) is already at clang-handwritten parity (49%)** — i.e. clang's *own best* hand-tuned fp
kernel only reaches 49%, so there is **no codegen ceiling above ~49–53% to chase on the fp path.** The fp inner
loop is **ALU-bound by the irreducible dequant**: per weight it must unpack the nibble (shift+mask), **convert
int→fp** (`v_cvt_f32_ubyte`), apply the `d·sc·q − dmin·mn` affine (`v_fma_mix`), and accumulate the dot
(`v_fma_f32`) — ~4 VALU/weight that register-allocation/scheduling cannot remove (confirmed: handwritten 93 VGPR,
**zero spills**, so it's not a register-pressure problem either). The **+8% codegen lever exists only on the
int-dot path** (36 VGPR, `v_dot4` collapses convert+FMA into one 4-MAC instruction), which is **lossy (rel 0.006) +
q8-pack-walled** and explicitly out of scope here. This matches Bank 3's standalone verdict (`bank3-w4a16-handwritten-20260618.md`:
"fp ALU-ceilinged ~49–53%, no codegen rescue") and the prior 4.06-VALU/weight disassembly count.

→ **Phase-2 kill gate met** ("source/ISA diff shows no small lowering lever"): clang's best fp kernel can't beat
49%, tinygrad is already there. No fp renderer/lowering micro-patch can clear the ≥10% isolated bar. *Do not* spend
on fp inner-loop codegen.

## Phase-1 byproduct — a work-decomposition candidate (not codegen): coop → ffn_gate routing

The diff surfaced that the **default-routed** ffn_gate uses the *base* GEMV (`q4k_gemv_partial`, LOCAL:0:64) at
**41%**, while the shipped `q4k_coop_partial_kernel` (routed only to attn_q/o today) reaches **47%** on the same
role — a **1.16× isolated** (DEBUG2 best-of-5, err 2.3e-6, byte-identical class). This is **work-decomposition**
(better coalescing/thread-mapping), *not* the codegen lever Arc B targeted. It was refuted in the coop arc at the
**1.3× isolated gate** (`bench/qk-mmvq-coop-q4k-ffn/role_inventory.json`) but **never measured in-model**, and its
+16% isolated clears a ≥10% bar with ~+6% Amdahl projection (ffn_gate/up = 44% of decode weight traffic).

**In-model W==D probe (this scope; `Q4K_FFN_COOP=1`, clean back-to-back A/B):**

| ctx | baseline (off) | coop-ffn (on) | Δ |
|---|---:|---:|---:|
| 128 | 71.8 | 72.5 | +1.0% |
| 512 | 68.2 | 69.2 | +1.5% |
| 1024 | 66.4 | 67.6 | +1.8% |
| 4096 | 60.7 | 62.1 | **+2.3%** |

Real, consistent, byte-identical-class, **growing with ctx** — but **+1.0–2.3% << the ≥5% in-model gate.** The
isolated 1.16× / Amdahl ~+6% **did not translate** (in-model the kernel is a smaller wall-share than its traffic
share, plus the coop reduce adds launches) — another instance of "isolated DEBUG2 misleads; only in-model W==D is
authoritative." **Verdict: CANDIDATE → sub-gate / deferred.** Not routed (the probe flag is reverted; the change is
3 lines, reproducible). Worth stacking only if bundled into a larger decode pass where +2% @ctx4096 is free.

## What this closes / leaves

- **CLOSED:** fp ffn_gate/up codegen-quality as a bounded lever (ALU-ceilinged ~49%, tinygrad already at handwritten
  parity, no spills). Do not reopen without changing the activation format (→ int-dot, which is the separate
  pack-walled / lossy arc).
- **DEFERRED (sub-gate):** coop→ffn_gate routing (+1–2.3% in-model, byte-identical) — a free stacking win, not a
  standalone ship.
- **Confirms the campaign conclusion:** the byte-identical decode surface is exhausted; the only path below the fp
  ALU ceiling is int-dot (lossy, q8-pack-walled — Arc-A-adjacent activation-lifecycle), and the only orthogonal
  beat-llama route is spec-decode/prefill's shared weight-reuse primitive (Arc A).

## Files
ISA diff: `extra/q4k_w4a16_handwritten.hip` (fp, 49%), `extra/q4k_mmvq_handwritten.hip` (int-dot, 65%), tinygrad
emitted source via `DEBUG=4 extra/q4_k_bench.py --tensor blk.0.ffn_gate.weight --primitive`. In-model probe:
`extra/qk_decode_runtime_overhead.py` with `Q4K_FFN_COOP` (model.py, reverted). No kernel/model/default changes shipped.
