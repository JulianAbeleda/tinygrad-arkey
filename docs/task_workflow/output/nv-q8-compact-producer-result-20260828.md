# NV pp512 compact Q8 producer gate

## Verdict

The llama-like compact topology passes the isolated research gate: every Q8
byte, FP32 scale, and FP16-rounded raw sum is bit-exact, while the complete
kernel remains below 4 us.

At `(M,K)=(512,4096)`, the final exact spelling measures 3.584 us minimum in
both hot R9 and a fresh-process R7, versus 104.156--105.609 us for the current
exact complete producer: about 29x faster, with zero mismatches across all
outputs.

No production code or routing changed.

## Geometry comparison

| producer | CTA geometry | ownership | service |
| --- | --- | --- | ---: |
| current exact tinygrad | 65,536 x 32-thread CTAs | one warp per 32-value group | 104.156 us |
| compact candidate | `(512,8)` x 128 threads = 4,096 CTAs | one thread loads float4; 8-lane subgroup owns 32 values | 3.584 us |
| llama MMQ Q8_1 | same float4/thread, 128-thread block, 8 K segments/row | DS4 packing | about 3.45 us/call from trace |

The candidate follows llama's `quantize_mmq_q8_1`: coalesced `float4` loads,
subgroup XOR reductions for max and raw sum, `char4` stores, and one metadata
writer per group.  The exact spelling uses 36 registers/thread, 1,024 bytes reported static
shared allocation, and 576 bytes reported local-memory reservation.  No
explicit shared memory appears in the source.

The logical packet traffic is approximately 11 MB: 8.39 MB FP32 input, 2.10 MB
Q8 values, and about 0.52 MB scales+sums.  The 3.58 us hot rate reflects strong
cache residency and cannot be labeled a DRAM bandwidth result without NCU.

## Correctness

Real legal fixture: Qwen3-8B block-0 normalized FFN boundary.

- finite: pass;
- scales: bit-exact, max abs zero;
- raw FP16-rounded sums: bit-exact, max abs zero;
- Q8 values: bit-exact, zero mismatches out of 2,097,152;
- result is stable across R9 and fresh R7 with identical hashes.

The differential sweep found 278 mismatches with CUDA `roundf`, then 115 with
`nearbyintf`.  Tinygrad's actual generated source uses neither direct spelling:
it constructs scale as `amax * float32(1/127)`, computes `value * (1/scale)`,
and emits an explicit truncation/parity ties-to-even sequence.  Reproducing
that exact operation order yields zero mismatches.  The adversarial mismatches
were all near half-integer boundaries, confirming the diagnosis.

## Consequence

The producer cost is now about 0.129 ms for 36 shared gate/up inputs instead of
3.75 ms.  Combined with a
future ~175 us packed main+fixup per projection, the gate/up population would
be about 12.73 ms and recover roughly 25 ms of the authority pp512 wall—the
original ~59 ms planning endpoint.

## Artifacts

- harness: `extra/llm_research/prefill/nv_q8_compact_producer_gate.py`;
- hot R9: `docs/task_workflow/evidence/nv-q8-compact-producer-20260828/hot-r9-exact.json`;
- fresh R7: `docs/task_workflow/evidence/nv-q8-compact-producer-20260828/cold-r7-exact.json`.
