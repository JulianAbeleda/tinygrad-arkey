# Native-NV Q6_K 1024x4096 direct-output microgate record

Date: 2026-08-05
Authority: `nv-q6k-1024x4096-native-nv-implementation-scope-20260804.md`
Verdict: **Gate 1 FAIL; exact ordinary-UOp cooperative direct-output construction refuted**

## Finding

The smallest native-owned construction was already expressible by tinygrad's generic Q6 lowering:

```text
packed Q6_K + fp16 activation
  -> q6k_gen_coop_1024_4096_inkernel
  -> contiguous fp32[1024]
```

It is numerically correct, but it is materially slower than the installed
`q6k_gen_partial_1024_4096_4 -> sum(axis=1).contiguous()` whole primitive.
Consequently, the CUDA oracle's 0.179--0.184 ms/token family recovery is not
explained by direct reduction or node-count removal alone.  Its missing
mechanism is the packed activation/instruction mapping, not merely a one-kernel
output topology.

No production selector remains, no model graph was changed, and the 18-role
wall arm was not run because the scope requires a positive whole-primitive
microgate before integration.

## Native measurements

Both arms used the same synthetic packed-Q6 bytes and fp16 activation, ran
under `flock -w 60 /tmp/nv_gpu.lock`, and used reverse-bracket native graph
replays.  The independent dequantized-Q6 matrix-vector reference set the
predeclared `atol=0.02`.

| candidate | correctness max abs | control midpoint | candidate | delta |
| --- | ---: | ---: | ---: | ---: |
| cooperative direct, row tile 2 | 1.669e-5 PASS | 113.338 us | 153.313 us | **+39.974 us** |
| cooperative direct, row tile 1 | 1.669e-5 PASS | 89.365 us | 150.766 us | **+61.402 us** |

The baseline's first bracket showed clock warm-up drift, so these timings are
not used as a global performance budget.  That caveat cannot reverse the
result: the candidate remained around 151--153 us across both independent
constructions while the later baseline brackets were 88--112 us.

## Mechanism classification

The sm_120 rendered source for row-tile 2 is 8,762 bytes and contains four
`__shfl_xor_sync` calls, zero `__dp4a` calls, and a scalar unpack/dequant/fp16
multiply recurrence.  Thus H3 (direct reduction topology) is refuted for this
construction and H5 is observed: the intended llama-like packed integer-dot
operation did not materialize through the current native-NV UOp grammar.

This does not refute Q8_1 plus a mapped Q6 integer dot.  It identifies the
next prerequisite precisely: a generic native-NV packed signed-int8 dot
primitive/instruction mapping, followed by an included-cost Q8_1 producer.
That requires a renderer/compiler-lowering amendment because this scope
explicitly allowed renderer audit but not renderer changes.  Substituting a
vendor source string, copied cubin, or inline assembly is prohibited.

## Accounting

The CUDA Q6_K family result remains a replicated 0.179--0.184 ms/token
diagnostic recovery signal.  This native experiment earns **0.000 ms/token**
residual credit.  The authoritative native/llama unexplained residual remains
**1.646170 ms/token**.  It does not establish decode parity.

Artifacts:

- compact evidence: `docs/task_workflow/output/nv-q6k-1024x4096-native-nv-direct-microgate-20260805.json`;
- reproducible harness: `extra/llm_research/decode/q6k_native_nv_direct_microgate.py`;
- raw lock-held rows: `/tmp/q6k_native_nv_direct_microgate.json` and
  `/tmp/q6k_native_nv_direct_rt1_microgate.json` (SHA256
  `23c5c8e405cfb07c7d93d70a320a55befc8b30bb6886f3a7276ffe5d9ab2a1c7`
  and `66acc3888e8d6183995eaf7d9ab31ae9fa916972748468b4205cafcfe5d035ea`).
