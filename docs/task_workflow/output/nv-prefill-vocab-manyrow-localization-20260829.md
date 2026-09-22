# NVIDIA pp512 vocabulary many-row localization (E1 follow-up)

Date: 2026-08-29  
Packet: E1 diagnostic follow-up  
Status: **PASS: first mismatch localized**

## Authority and scope

This diagnostic uses the immutable E0 fixture under
`docs/task_workflow/evidence/nv-vocab-manyrow-e0-fixture-20260829/` and the
isolated primitive in `tinygrad/llm/q6k_vocab_manyrow.py`. It does not change
the Q6 consumer, model dispatch, admission, or performance claims.

## First divergent stage

The E0 extractor captures `captured[0]` from the call to `model.output_norm`:
it is the **pre-output-RMSNorm** final hidden row. The reference logits are
then produced by the normal `Transformer.logits` path, which applies
`output_norm` before the output projection. E1 passes the captured row directly
to `q8_provider`, so the consumer sees the wrong activation contract.

The owning expression is the output-projection boundary in
`tinygrad/llm/model.py`:

```python
x = self.output_norm(x)
return self.output(x)
```

The E1 call currently uses `source = x.reshape(K)...` directly. The required
correction target is therefore the fixture/replay boundary: provide the
post-`output_norm` row, or apply the exact output RMSNorm (including its loaded
weight, epsilon, dtype boundary, and ordering) before Q8 quantization.

## Evidence

The packed Q6 file has the expected canonical byte geometry: 151936 rows,
4096 input elements, 16 blocks per row, 210 bytes per Q6_K block, for
510504960 bytes. The consumer's row stride is consequently 16 * 105 uint16
halfwords. The Q6 block parser agrees with the installed consumer layout:
QL bytes [0,128), QH bytes [128,192), signed scales [192,208), and fp16 `d`
at bytes [208,210).

The compact-Q8 producer ABI is also structurally correct: K=4096 gives 128
groups, 1024 payload words plus 128 metadata words, hence 1152 uint32 words.
Metadata stores fp16 `d` in the low 16 bits and the fp16 raw sum in the high
16 bits, matching `emit_q8_provider` and `_unpack_q8_packet`.

With the pre-output-norm E0 row, independent Q6 dequantized one-row dots are
already inconsistent with the reference at the first row and remain so at
the middle and final rows:

| row | dequantized Q6 dot using E0 row | reference logit |
|---:|---:|---:|
| 0 | 9.87646 | 2.04415 |
| 1 | 3.58323 | 1.64719 |
| 75968 | -3.65670 | 1.34719 |
| 151935 | 12.19059 | 2.86963 |

This divergence exists before any many-row consumer output or argmax. The
E1 GPU result (`max_abs=285.46103`, argmax `88251` vs `8503`) is therefore a
downstream symptom, not evidence of a Q6 row-stride or output-weight
orientation defect.

## Verdict

**PASS.** Precise correction target: align the E0 fixture and E1 harness at
the post-`output_norm` activation boundary. Re-run packet-byte, one-row-dot,
and full-logit oracle checks after that fixture/parser correction. Do not
redesign the Q6 many-row kernel until those checks pass.
