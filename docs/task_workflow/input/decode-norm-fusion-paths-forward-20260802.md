# Fused decode RMSNorm - paths forward (review scope)

Status: review scope for the M3 decision recorded in
`m3-fused-norm-measurement-record-20260802.md`. M3 is landed closed-default; the question is
which path reopens it, and what the campaign does in the meantime. The paths do not all
compete; the complementarity matrix is section 5.

## 1. Current design (as landed, measured facts)

M3 is one opaque kernel per decode norm (`decode_rmsnorm_1_4096` for attn/ffn, `_32_128` and
`_8_128` for q/k) replacing the generic reduce + epilogue pair, selected under the closed
`decode_norm_fusion` promotion record. Measured on NV sm_120, Qwen3-8B Q4_K, d512:

| class | baseline (M2) | fused (M2+M3) | delta |
| --- | ---: | ---: | ---: |
| norm family kernels/token | 361 (876.5us) | 288 (810.0us) | -73 kernels, -66.5us |
| decode kernels/token | 1021 | 1093 | +72 |
| decode kernel us/token | 6256 | 6398 | +142 |
| decode tok/s | 173.45 | 168.42 | -3% |

The family alone is a small paper win; the decode-level regression comes from what the fused
state adds around it, all verified in the census trace:

- 144 input-boundary copies: 108 x 4096-elem (`E_32_32_4_86a2`, 1.47us) + 36 x 1024-elem
  (`E_8_32_4_dd98`, 1.57us). The custom-kernel transport contiguous()s every non-identity
  input; the norm inputs are lazy producers (residual, qkv, embedding) with no buffer identity
  at trace time. A rank-3 pass-through was tried; the copy is not elided.
- 72 output materializations (`E_32_32_4_3b0f`, 1.54us), one per attn/ffn norm: the flat
  `(numel,)` kernel output reshaped to `(1,1,4096)` does not satisfy the downstream consumer's
  contiguity, so the scheduler realizes it. Trace order per attn/ffn norm:
  `copy -> decode_rmsnorm -> copy` (3 launches replacing the legacy 2).
- The fused kernels are launch-bound at 3.2-5.0us (4.96us median for `decode_rmsnorm_1_4096`);
  the design doc's llama-shaped 2.12us end-state is not reachable until per-kernel host
  overhead drops.

Tokens are byte-identical (sha `9d6b3787...` 3/3, first token `151936` 3/3), so correctness
is not the blocker; economics is.

## 2. Path 1 - copy-free opaque boundary (transport substrate)

Change the opaque-kernel transport so a consumer can declare "index my input by logical shape";
the boundary preserves the non-identity view instead of contiguous()'ing it, and the emitter
indexes the producer's logical dims so the scheduler resolves strides. Emitter-by-emitter opt-in
(new boundary mode, default unchanged).

Expected outcome for the norm family: 144 input copies (215.3us) and 72 output
materializations (110.9us) disappear; the family becomes ~144 fused kernels at ~594.7us vs
876.5us legacy (~-280us/token, -216 launches). Same mechanism removes the copy tax for the
flash q/k/v and gemv x views. Does NOT change the launch floor; fused kernels stay 3.2-5.0us
until Path 2.

Risk: every opted-in emitter must be audited for flat-index assumptions (q6k coop is flat;
flash tile reads cache with identity - unaffected); the "opaque = simple buffer" contract
becomes two modes that must not drift; pg3 pins for opted-in emitters move deliberately.

## 3. Path 2 - per-kernel launch/host overhead (B3)

The P2/P3 finding: per-kernel host cost is the dominant term for sub-10us kernels (the
launch-side counterpart of the measured ~1.5-5us floor). B3 batches/deletes the host work per
launch. Already scoped in `decode-gap-per-target-lever-scope-20260802.md`.

Expected outcome: lowers the floor for ALL ~1000 kernels/token, not just norms - the largest
single decode lever (order +10-20% wall time at the current census, to be measured). It also
makes the design doc's 2.12us norm shape reachable. Note: Path 2 ALONE does not make M3 land -
with cheaper launches, fused (3 launches) still loses to legacy (2 launches) until the copies
go. Path 2 and Path 1/3 multiply: fewer kernels x cheaper launches.

## 4. Path 3 - scheduler-native norm fusion (no opaque boundary)

Give the generic RMSNorm lowering an in-kernel reduction so one norm lowers to ONE scheduler
kernel: no boundary, no copies, no transport change. The machinery exists - M2's q6k coop
in-kernel merge uses the same staged-shfl + smem-barrier building blocks. This is the design
doc section 9 Q1's shape (b), the narrow generic chain-fusion option.

Expected outcome for the norm family: 145 kernels (144 fused + final norm), ~594.7us of kernel
time, no copy/materialization kernels, same ~-280us/token as Path 1's norm result - without
touching the transport or any other opaque consumer. Because it is the generic path, prefill
norms benefit too, which is the right outcome but widens the blast radius; admission must stay
closed-default per shape/target until measured. Reduce-order parity gate (decode sha) applies
as today.

Risk: generic-path blast radius (any model's RMSNorm), two lowering shapes that must not
drift, occupancy on small norms.

## 5. Complementarity

| | Path 1 (transport) | Path 2 (launch) | Path 3 (generic) | Path 4 (M4/M5) |
| --- | --- | --- | --- | --- |
| Path 1 | - | multiplies (fewer kernels x cheaper launches) | partial substitute for the norm family; Path 1 also fixes flash/gemv views, Path 3 also fixes prefill norms | orthogonal |
| Path 2 | multiplies | - | multiplies | orthogonal |
| Path 3 | partial substitute | multiplies | - | orthogonal |
| Path 4 | orthogonal | orthogonal | orthogonal | - |

The two genuinely competing choices are Path 1 vs Path 3 for the norm copies: both remove the
same 216 kernels from the norm path, and doing both is redundant on that family. They differ
only in where the win lands (transport benefits every opaque consumer; generic lowering
benefits every norm on every target). Path 2 does not compete with anything - it is a
multiplier. M4/M5 (q4k epilogue absorption, flash combine normalization) proceed regardless of
the norm decision.

## 6. Expected outcomes by choice

| choice | norm family | decode (est.) | reachable end-state |
| --- | ---: | ---: | --- |
| keep closed (today) | 361 kernels, 876.5us | 173.45 tok/s | norm story parked; ~0.4ms epilogue claim via M4/M5 |
| Path 1 alone | ~144 kernels, ~595us | +4-6% (est.; ~180-184) | norm win lands; flash/gemv copy tax also removed; launch floor unchanged |
| Path 2 alone | unchanged (361) | +10-20% (est., to measure) | everything faster; M3 still loses (3 vs 2 launches) |
| Path 3 alone | ~145 kernels, ~595us | +4-5% (est.) | norm win lands on the generic path, all targets; prefill norms too (gated) |
| Path 1 or 3 + Path 2 | ~145 kernels at ~2-2.5us | doc's original 0.9-1.0ms claim becomes the measured question | llama-shaped plumbing end-state |

All est. numbers are arithmetic from the measured census medians, not measured; each choice
must be verified at the campaign's fixed-depth protocol (d512/d2048/d4096, sha pins) before the
record flips.

## 7. Open questions for review

1. Path 1 vs Path 3: is the generic in-kernel norm (Path 3) preferred over the transport change
   (Path 1), given Path 1's blast radius across flash/gemv emitters - or is Path 1 worth doing
   anyway for the other consumers' copy tax?
2. If Path 1: should view-preservation be a NEW boundary mode (opt-in per consumer) rather than
   a change to `custom_kernel`'s default, to keep the existing emitters' flat-index contract?
3. If Path 3: is an in-kernel reduce in the GENERIC lowering acceptable for decode shapes only
   behind a closed gate, or does that create two norm lowerings that can drift?
4. The 72 output materializations (`E_32_32_4_3b0f`): are they a norm-boundary artifact (fixed
   by Path 1) or a separate contiguity bug in the flash/attention consumer worth fixing
   independently?
5. Sequencing: run M4/M5 now and let the norm paths land when measured, or treat the norm
   story as non-optional per design doc section 6 and sequence one of Path 1/3 before M4?

HARD STOP after this section. No implementation on any path until this scope is reviewed.

---

## 8. Correction and llama-informed priority (2026-08-02)

Reconciling this scope against the llama trace (`decode-gap-per-target-lever-scope-20260802.md`
section 1) changes two claims and the sequencing. This section supersedes sections 3 and 6
where they disagree.

### 8.1 Path 2 is a prefill lever, not a decode lever (correction)

Decode is 95% GPU-busy (5.83ms busy of 6.12ms wall at d512) and the flash-decode rollout is
already graph-replayed into 6 batches (`batched 32/64/128/256/512/29` = 1021 programs/token) -
the B3 replay mechanism is in place for decode. The per-kernel host-cost ceiling for decode is
therefore ~5%, not the "order +10-20%" stated in section 3. The 840x per-kernel / 1.9x
wall-busy evidence is the PREFILL prime path (24.1ms busy / 44-46ms wall, 1.35M to_mv calls),
where B3 remains open. Path 2 stays a live lever for prefill; it is demoted for decode.

### 8.2 llama's shape argues for Path 3 (generic norm), not Path 1

llama's graph has no opaque boundary and no norm copy tax: RMSNorm is one generic kernel per
norm (`rms_norm_f32`, 145 nodes, 1.3-3.4us class) - exactly the shape Path 3 (scheduler-native
in-kernel norm) produces, and exactly the 145-kernel end-state the design doc already targets.
llama does NOT keep a two-kernel reduce+epilogue pair, and it does not pay a contiguous copy to
read its norm input. The M3 opaque emitter was the wrong shape for the norm family
specifically: it introduced a toll booth llama does not have. Path 1 remains useful only for
consumers whose inputs are non-identity views (flash/gemv), which is a separate question.

### 8.3 llama-informed priority

llama's advantage decomposes as: plumbing +1.05ms (no separate add/silu kernels; fused w1w3),
GEMV bandwidth +0.44ms (Q6_K 1.4 TB/s vs our 0.82/0.2; k/v 1.04 vs 0.2), vocab +0.24ms (single
mmq 303.75us vs our ~540us chain), flash +0.17ms (3.17+3.35us vs 7.6+3.6us per layer). Two
asymmetries cut the other way: llama pays q8_1 quantization (217 nodes, 0.482ms) that we do
not, and llama's per-kernel times (1.3-3.4us) are the same league as ours (1.6-3.9us) - the gap
is count and bandwidth, not launch economics.

Priority, largest measured mass first, each additive and closed-gate:

1. M4/M5 epilogue absorption (the +1.05ms plumbing class; llama's "no separate add/silu"
   shape). Independent of the norm decision; this is the biggest single next lever.
2. Path 3 - generic in-kernel norm (the norm half of that class; llama's `rms_norm_f32` shape,
   no copies, all targets). Path 1 only if a later measurement shows flash/gemv view consumers
   also pay a material copy tax.
3. L2/L5 GEMV bandwidth (+0.44ms; llama mmq blocks are 128 threads vs our lanes=32, Q6_K at
   1.4 TB/s) - diagnostic microbench first, per the substrate trichotomy.
4. L4 vocab substrate fusion (+0.24ms; scalar reduce + scatter into the coop kernel).
5. Flash tile (+0.17ms; 7.6 vs 3.17us) - values/occupancy first.

Path 2 moves to the prefill campaign (B3), where its evidence lives. The campaign's stated
endpoint stays 195-210 tok/s at d512 (llama 245.6; like-for-like closer once llama's q8_1
asymmetry is excluded).
