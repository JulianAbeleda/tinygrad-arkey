# NV GEMV mapping reopen -- static audit

Date: 2026-08-05. Scope: Qwen3-8B-Q4_K_M d512 native-NV decode. This is a
read-only code/artifact audit; it contains no timing and authorizes no route
change or GPU execution.

## Decision

Do **not** run the proposed native all-18 Q6 shared-Q8 token A/B as currently
described. The passing `q6k_shared_q8_reuse_microgate` proves a synthetic graph
of three same-shape Q6 consumers, but the model inventory says the Q6 attention
population is only `attn_v` on the 18 Q6 layers. Its Q/K siblings are Q4_K.
Multiplying the -90.028 us / three-Q6 synthetic result into a token recovery
would therefore test a graph that the model does not contain.

The cheapest decisive next step is CPU/render-only: make the existing semantic
manifest/capture report, for each actual attention-norm source, the ordered
`(role, quant, rows, K, source-buffer identity)` triplet and assert:

1. exactly one Q6 `attn_v` consumer is present on each Q6 layer;
2. Q/K are Q4_K consumers of the same norm source where the source aliases;
3. an actual cross-format shared-Q8 candidate cannot be built until Q4_K has a
   compatible Q8 consumer with the same numerical/output boundary;
4. no source has three Q6 consumers.

This can be implemented against the frozen CUDA semantic-call manifest and a
native render/capture-only construction; it needs neither device execution nor
default selection. It falsifies the synthetic-Q6 extrapolation before a costly
full-token run.

## Current comparison

The causal ledger partitions the 1646.170 us/token native/llama gap into
1108.082 us support work, 302.788 us quantized cores, and boundary bridges.
Thus a GEMV-only change cannot rationally target the whole gap. The measured
native quant-core deficit is concentrated in Q6 attention V/K labels and the
Q4/Q6 FFN-down substrate; Q4 gate/up is already fused on native NV and is not
a positive native deficit.

| population | installed native seam | llama artifact | status / implication |
| --- | --- | --- | --- |
| Q6 attention V, 1024x4096, 18 | `_emit_q6k_partial`, `parts=4`, then `partial.sum(axis=1)` | Q8_1 producer + Q6 MMVQ, 1024 blocks x 128 threads, 48 registers | strongest valid Q6 population; CUDA all-family replacement recovered 179--184 us but has no native credit |
| Q6 FFN down, 4096x12288, 18 | `_emit_q6k_coop`, NV row_tile=2 | Q8_1 + fused MMVQ, 4096 blocks x 128 threads, 46 registers | native same-shape gap is only about 1.19x; full-family correctness is not established |
| Q4 FFN down, 4096x12288, 18 | `q4k_g3_lanemap_gemv_kernel` | Q8_1 + fused MMVQ | CUDA family signal 65.8--66.1 us; residual fusion was neutral, so substrate rather than epilogue is implicated |
| Q4 gate/up, 12288x4096 | `q4k_g3_lanemap_gemv_w1w3_kernel(..., scalar)` | one fused Q8/MMVQ call | native already has one fused kernel; the Q4 quad load variant regressed in-loop |

## Ranked falsifiable hypotheses

### Direct Q4 G3 packed-fp16 audit (completed CPU/render-only)

The production `q4k_g3_lanemap_gemv_kernel(32, 1024)` was rendered for
`sm_120` with its real fp16 activation ABI and compiled to PTX using NVRTC,
without device execution. Its inner block has 32 fp16 activation loads, 8
uint32 weight/header loads, 32 `fma.rn.f32`, 34 `cvt.f32.f16`, and five warp
shuffles. There is no CUDA packed-fp16 dot instruction with the required fp32
accumulation contract: the renderer's `fdot2` provider deliberately expands to
two scalar fp32 products/adds. Using `half2` arithmetic would instead round the
products/partial sums in fp16 and is therefore a different numerical contract,
not an instruction-mapping substitution.

One closed-default source experiment reinterpreted each aligned group of four
fp16 activations as an 8-byte vector carrier while retaining scalar fp32
conversion and the exact recurrence. NVRTC scalarized it. Both control and
candidate PTX contained exactly 40 `ld.global`, 32 `fma.rn.f32`, and 34
`cvt.f32.f16`; the candidate's activation accesses were the same 32
`ld.global.b16` instructions. The experiment was reverted and needs no GPU
gate. Its exact falsifier was “fewer activation load instructions or fewer
fp32 arithmetic/conversion instructions in compiled PTX”; neither occurred.

This closes half2/fdot2/vector-load spelling as a route to the measured
~302.788 us/token quant-core deficit. A future direct-Q4 mapping reopen must
change the operand representation (for example, one shared Q8 producer feeding
packed integer dot consumers), lane ownership, or work/block schedule. Merely
spelling the same fp16 data as a vector cannot recover any ledger credit.

1. **Cross-format shared-Q8 is the only unclosed Q6 mapping hypothesis with
   mechanism-scale evidence, but the current three-Q6 benchmark has the wrong
   model topology.** Expected recovery: unknown; zero bookable until a real
   Q4+Q6 common-source construction passes full logits. Falsifier: static
   source/quant census above, followed only if admitted by a default-off
   capture whose actual pack count is one per Q/K/V norm source.

2. **The Q6 partial4 load/reduction mapping, not dp4a availability, is the
   remaining attention-V floor.** Expected native wall opportunity: at most
   the CUDA diagnostic's roughly 184 us/token directionally, not bookable.
   The direct-output and one-consumer Q8+DP4A variants were slower; that
   falsifies direct reduction and unshared packed-dot constructions. The next
   cheap test is render-only comparison of actual partial4 and a candidate:
   assert byte/halfword load widths, global-load count, register pressure, and
   output/reduction topology before GPU timing. Code seam:
   `tinygrad/llm/decode_kernels.py:_q6k_byte,_q6k_weight,_q6k_block_dot,_emit_q6k_partial`.

3. **Q4 FFN-down has a real substrate gap, but an ordinary-UOp Q8/MMVQ route
   must carry the entire activation and consumer contract.** Expected recovery:
   bounded directionally by 65.8--66.1 us/token in CUDA, zero native credit.
   Falsifier: CPU/render-only semantic comparison demonstrating that a Q8
   candidate preserves the existing fp16 activation and fp32 residual boundary
   without adapter/copy nodes; otherwise do not GPU-test a llama substitution.
   Code seam: `q4k_g3_lanemap_gemv_kernel` and
   `tinygrad/llm/decode_routes.py:_Q4KDecodeCandidate.execute`.

4. **Q6 FFN-down's achieved-bandwidth shortfall is smaller and correctness
   risk is higher than attention-V.** Expected recovery: low/moderate (same
   shape gap about 1.19x); do not prioritize before an exact local semantic
   gate. Falsifier: render/capture census that proves an in-core Q8 producer,
   Q6 consumer, and residual endpoint can replace the existing coop result
   without extra materializations. Code seam: `_emit_q6k_coop` and
   `decode_routes.py:_Q6KDecodeCandidate.execute`.

## Closed avenues

- Native Q6 direct-output reduction: +39.974 / +61.402 us in its complete
  microgate; no packed integer dot appeared in rendered source.
- Native one-consumer Q8+DP4A: +1.172 to +1.352 us; the CUDA provider does
  render `dp4a.s32.s32`, so merely adding the instruction is closed.
- Q4 gate/up load-style rewrite: the standalone quad pattern regressed in-loop;
  native scalar w1w3 fusion already eliminates the CUDA-only pair-count gap.
- Q4 attention Q/O llama substitutions and residual-only FFN-down epilogues:
  causal NO-GOs/wall-neutral in the ledger.

## Required GPU gate if static admission succeeds

Use a default-off native A/B/A on the *actual* mapped population, preserving
full-logit and generated-token checks. Report producer count, conversion/copy
nodes, changed semantic nodes, paired dispersion, and whether the source pack
is shared with Q4 as well as Q6 consumers. Never debit CUDA or synthetic
microgate deltas from the native 1646.170-us ledger.
