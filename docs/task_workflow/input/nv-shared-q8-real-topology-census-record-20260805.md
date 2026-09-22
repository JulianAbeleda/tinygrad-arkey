# NV shared-Q8 real-topology census record (2026-08-05)

Evidence class: CPU-only GGUF metadata and static ABI construction. This record makes no GPU correctness or performance claim and authorizes no default or promotion.

## Question

Can one exact RMSNorm/Q8_1 provider feed all three attention projections in the real Qwen3-8B-Q4_K_M model, including its mixed Q4_K/Q6_K layers, without another provider or copy?

## Reproduction

```sh
PYTHONPATH=. python3 extra/llm_research/decode/shared_q8_real_topology_census.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf
PYTHONPATH=. python3 -m pytest -q test/unit/test_shared_q8_attention_boundary.py
```

## Findings

The 36-layer authority model has exactly two attention triples:

- `Q4_K/Q4_K/Q6_K`: 18 layers (0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 31, 32, 33, 34, 35).
- `Q4_K/Q4_K/Q4_K`: 18 layers (4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29).

The provider ABI is independent of the weight quantization: one `uint32[1152]` Q8_1 buffer contains 1,024 int8x4 packets and 128 `d|s` metadata words. Both `_emit_q4` and `_emit_q6` already consume that exact packet layout. The three consumers receive the identical provider Tensor/UOp, so the typed boundary inserts zero per-consumer providers and zero explicit copies.

The only missing seam was route admission/dispatch: `shared_q8_attention_call` required V to be `Q6KPrimitiveLinear` even though its Q4 consumer emitter already supported the real 1024-row V shape. The closed-lease route now accepts V as either Q4_K or Q6_K and selects the matching packed-weight storage/emitter. It remains impossible to enable through model loading or an environment default.

Focused result: 9/9 boundary tests pass, including common-provider identity and final Q4-V emitter identity `q4k_q8_dp4a_1024_4096`.

## Verdict

**STATIC PASS.** One common RMSNorm/Q8_1 source is structurally sufficient for all 36 real attention blocks and both consumer formats. There is no missing provider ABI. GPU numerical and wall qualification remain required; this record does not reopen the earlier full-model numeric/topology NO-GO by itself.
