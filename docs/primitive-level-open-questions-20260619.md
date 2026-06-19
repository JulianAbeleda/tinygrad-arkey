# Standing primitive-level open questions (audited) — what's actually left, against the principles (2026-06-19)

After the full decode + prefill + Tensile-extraction + codegen arcs, this is the audited map of what remains **at the
performance-primitive level**: for each phase/role, the *binding* primitive constraint (measured), the lever, its
state, and the principle-verdict on whether to pursue it. The point: stop treating "llama is faster" as one gap —
it's a small set of distinct, named primitive constraints, most already closed/deferred, with **two** genuinely open
difficult ones.

## The audit that reframed it (this pass) [M]
Hypothesis (principle: *reduce knowledge duplication / one authority*): is the prefill address-lowering inefficiency
the same root cause as the decode gap? **Refuted by disassembly:**

| phase/role | dominant kernel ALU (per the ISA) | binding primitive constraint |
|---|---|---|
| **prefill** fp16 GEMM | 120 `v_mov` + 40 `v_add` (vs 16 `v_wmma`) | **address-lowering**: per-load 64-bit global addrs, no `offset:` immediates, recomputed per k-tile |
| **decode** Q4_K MMVQ | 397 `v_cvt` + 160 `v_and` + 134 `v_lshl` + 384 `v_fma` (3 `v_mov`) | **dequant ALU**: fp unpack of 4-bit nibbles, then fp dot |

Different causes. The renderer addressing fix is **prefill-only**; decode is dequant-ALU-bound (the int-dot/q8 lever).
So they need separate fixes — the "one root cause" shortcut does not exist. (Principle satisfied: audited before
recommending a shared renderer arc.)

## The primitive-constraint map (binding constraint per role, with state)
| role / phase | binding constraint | lever | state | why (principle) |
|---|---|---|---|---|
| decode MMVQ (Q4_K/Q6_K linears) | fp **dequant ALU** + no weight reuse | q8 activation + native dot4 (int-dot), packed | **DEFERRED-behind-codegen** | int-dot in-kernel wins but whole-linear lost to the q8-pack reuse wall (reuse ceiling 2); producer needs LDS-reduction-fusion (Q8L-2 KILL). full-primitive-boundary: measured WITH pack cost → walled |
| decode attention (long ctx) | KV **locality / online-softmax** | flash-decode (shipped) + gqa_coop_vec (shipped) | **SHIPPED** (slope gap closed) | in-model W==D authority; byte-identical |
| prefill matmul | **address-lowering ALU** (this audit) | renderer: base+offset for strided global gathers + K-loop strength-reduce | **OPEN, project-level** | confirmed in ISA; not a kernel-UOp restructure (renderer decides addressing); outside BEAM's OptOp space |
| prefill attention (SDPA) | symbolic start_pos blocks concrete-TC | (left as SDPA; PREFILL_V2 only does matmuls) | **deferred** | measured ~0.8× when forced; not the bottleneck at pp512 |
| prefill matmul (external) | — | extract rocBLAS Tensile through HCQ | **PASS (1.41× llama in-model), policy-gated** | dependency (rocBLAS HSACO) — declined for default |

## The two genuinely-open DIFFICULT questions
1. **Prefill: can tinygrad's AMD renderer emit constant-stride global gathers as base + immediate offsets, and
   strength-reduce the K-loop pointer — and does it transfer in-model (pp512/dNLL)?**
   - Difficulty: core `renderer/cstyle.py` + symbolic-index simplification; broad test surface.
   - Value: closes ~half the prefill loop's ALU overhead → toward llama-class prefill, **no dependency, general**
     (every AMD matmul). The Tensile route already proved the ceiling is reachable in-model (1.41× llama), so the
     payoff is known; this is the dependency-free way to it.
   - Principle check: *in-model authority* (gate on warm pp512 + dNLL, not isolated TFLOPS); *test at the boundary*;
     *contain dangerous power* (renderer change → broad regression tests). **Outside BEAM's space** (addressing-mode is
     lowering, not an OptOp) → the BEAM-hang is irrelevant to it, so it must be hand-built.
2. **Decode: the q8-activation int-dot side-channel — can a fused RMSNorm→q8 producer feed packed dot4 without the
   pack cost dominating?**
   - Difficulty: needs an LDS-reduction-fusion (flash-style multi-granularity producer) — the Q8L-2 KILL capability.
   - Value: ~+3–4% e2e decode (the per-role delta audit ceiling), lossy (needs dNLL).
   - Principle check: *full primitive boundary* (the q8 pack/scale lifecycle, not just the dot4 intrinsic — already
     showed the pack wall); *label state* = deferred-behind-codegen; lower EV than #1.

## Verdict against the principles
- The frontier is **not diffuse** — it is exactly two named, audited primitive constraints (prefill address-lowering,
  decode dequant/q8), each requiring a specific tinygrad codegen capability, plus the shipped wins and the
  policy-gated external route.
- **Highest-leverage difficult question = prefill address-lowering** (#1): dependency-free, general, high payoff,
  ceiling proven reachable, and *outside* the BEAM-hang wall (it's a lowering fix, not a search). The principles point
  here: it has the cleanest in-model gate (pp512/dNLL), the broadest reuse (one renderer fix, all AMD matmuls), and an
  audited single cause.
- Decode q8 (#2) stays deferred (lower EV, deeper codegen, lossy) — consistent with *label-state* + *audit-before-build*.
- Both are project-level codegen, not session probes; the bounded kernel space is exhausted. The honest next move is a
  funded **prefill address-lowering renderer arc** (the dependency-free path to llama-class prefill), with decode q8 as
  a secondary deep arc.

## Files / provenance
This audit: decode kernel `/tmp/dec_r_32_32_4_48*.bin` (ISA mix above), prefill diagnosis
`prefill-codegen-wmma-issue-result-20260619.md`. Syntheses: `what-makes-a-performance-primitive-efficient-20260618.md`,
`performance-frontier-exhaustion-20260619.md`. No kernel/model/default changes (diagnosis + framing).
