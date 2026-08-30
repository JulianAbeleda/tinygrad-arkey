# NVIDIA pp512 vocabulary many-row primitive (E1)

Date: 2026-08-29  
Packet: E1  
Status: **PASS: corrected full-logit numerical oracle**

The isolated primitive is `tinygrad/llm/q6k_vocab_manyrow.py`. It defines a
default-off explicit lease and a two-program lifecycle: the established Q8_1
producer writes a fresh 1,152-word packet, then a packed Q6_K cooperative MMVQ
consumer reads that packet and writes all 151,936 FP32 logits. There is no
top-1 shortcut, model integration, queue change, admission-policy change, or
production default change.

The E0 fixture remains the G0 authority:
`docs/task_workflow/evidence/nv-vocab-manyrow-e0-fixture-20260829/`.
The required replay must compare the complete producer + consumer lifecycle
against the installed vocabulary control over the full logits tensor, including
finite/nonzero and first/middle/last/adversarial sentinels. G0 replay was
attempted on the RTX 5090 using the immutable E0 packed weight and
final-hidden-row fixture. It did not reach the consumer: first, module import
failed because `Q8_GROUPS` referenced `K` before assignment; that narrow import
defect was fixed. The next replay reached `_unpack_q8_packet` and failed
because the imported FFN-down producer constants/emitter are fixed at
`K=12288` (`Q8_GROUPS=384`), while this ABI requires `K=4096`
(`Q8_GROUPS=128`); packet metadata therefore has shape 12288 rather than 4096
and cannot broadcast with the unpacked payload. No logits, sentinels, oracle
comparison, control comparison, or timing artifact was produced.

Exact replay environment: `DEV=NV QK_PRIMITIVE=1`, fixture
`docs/task_workflow/evidence/nv-vocab-manyrow-e0-fixture-20260829/`, RTX 5090.
The producer ABI blocker is resolved by parameterizing `emit_q8_provider` with
an explicit input width and using `k=4096` in the vocabulary primitive. The
resulting packet is 1,152 `uint32` words (1,024 payload + 128 metadata), with a
16-CTA grid and the distinct `q8_1_llama_provider_4096` kernel name. A focused
render/unit contract was added in
`test/unit/test_q4k_ffn_down_mmvq.py::test_provider_supports_vocab_k4096_packet_abi`.
Fresh G0 replay was run after the isolated producer ABI fix with the immutable
E0 hidden row and Q6_K weight. The producer emitted the expected 1,152-word
K=4096 packet and the consumer completed with shape `(1,1,151936)`; all
151,936 values were finite and nonzero. The full oracle comparison failed:
`max_abs=285.46103`, relative L2 `17.018793`, and argmax `88251` versus
reference `8503`. Sentinels also disagree substantially (first `-26.8555`
versus `2.04415`, middle `29.1638` versus `1.34719`, final `14.9560` versus
`2.86963`). This mismatch is not evidence against the Q6 consumer: the
fixture extractor captured the input to `model.output_norm`, while the
reference logits use its returned post-norm value. The extractor now records
that returned tensor, including loaded weight, epsilon, dtype cast, and order.
Regenerate a new immutable E0 fixture and rerun G0 before any consumer or
layout changes. No wall claim or promotion is authorized.
E2 remains prohibited.

## E1 corrected packet result (2026-08-29)

Fresh immutable fixture (the prior E0 directory was not overwritten):
`docs/task_workflow/evidence/nv-vocab-manyrow-e1-postnorm-fixture-20260829/`.
Its manifest records model SHA256
`d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`, Q6
weight SHA256 `05d7a76c29a32e067f41ad91df565341b21f4bedeeffbbf9e10957aaa9f78cc1`,
hidden-row SHA256 `b04adfc129794cf0d0c37328ee56d955dcd137137c4c7854bc2963dcbbb20c41`,
and reference-logits SHA256
`ba506b6ec83b639a0a22704eb07345626e96b0e6f02375b7cc78e66f22e84bae`.
The post-`output_norm` hidden-row sentinels are first `-3.2084198`, middle
`-0.8017600`, and last `-1.0661219`.

The isolated producer + consumer lifecycle completed on the RTX 5090 with
output shape `(1,1,151936)`; all 151,936 values were finite and nonzero. G0
full-logit comparison nevertheless failed: `max_abs=22.292755`, relative
L2=`1.295059`, argmax `98720` versus reference `8503`; sentinels were
`-2.813025/-0.101393/1.536214` versus `2.044147/1.347186/2.869625`.

**STOP.** The packet ABI and lifecycle are operational, but numerical
equivalence is not established. No consumer redesign, model integration, queue
change, performance claim, or E2 is authorized. A new packet must localize the
remaining post-`output_norm`/Q8/Q6 semantic mismatch.

## Corrected G0 closure

The remaining mismatch was in `_unpack_q8_packet`: `ds.repeat(32)` tiled the
entire 128-scale vector 32 times instead of repeating each group scale across
its 32 adjacent activation values. The E1-only fix broadcasts with
`ds.reshape(-1, 1).expand(-1, 32).reshape(-1)`.

The immutable fixture replay now passes the declared `rtol=0.02, atol=0.5`
full-logit oracle: `max_abs=0.093423`, relative L2=`0.005383`, and argmax
`8503` equals the reference. Sentinels are
`2.016314/2.044147`, `1.322650/1.347186`, and `2.867657/2.869625` for
first/middle/last. All 151,936 outputs remain finite and nonzero, and the
producer emitted the expected 1,152-word packet.

**PASS.** E1 is closed as an isolated primitive correctness result. The exact
root cause was compact-Q8 scale expansion semantics, not Q6 weight layout or
consumer row mapping. E2 composition remains separately gated and was not
performed.

## Scope boundary

## Matched lifecycle R9

Runner: `extra/llm_research/prefill/nv_vocab_e1_lifecycle_r9.py`  
Evidence: `docs/task_workflow/evidence/nv-vocab-manyrow-e1-r9-20260829.json`

The runner alternated the ordinary canonical Q6_K dequantized full-logit
matmul control and the E1 producer + Q6 consumer candidate for nine samples,
retaining full logits and checking finite/nonzero output and argmax on each arm.
Both arms produced `(1,1,151936)` and argmax `8503`. The candidate median was
`16449.309 us` (minimum `15871.895 us`), versus control median `3871.647 us`
(minimum `3838.416 us`), for a candidate-minus-control median of
`+12577.662 us`. The first control sample includes one-time allocation/cache
cost; all samples are retained in the JSON artifact.

This is a correctness PASS but a performance STOP for E1: the isolated
candidate is approximately 4.25x slower than the ordinary installed control.
No performance promotion or E2 composition is authorized.

E2 composition is not performed. The module is intentionally unreachable unless
a harness explicitly constructs `Q6KVocabManyRowAdmission` and calls it.
