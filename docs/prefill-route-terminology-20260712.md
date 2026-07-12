# Prefill route terminology

Active work uses two route categories:

- **Pure**: every promoted prefill GEMM and supporting route is compiler/generated,
  with no handwritten `Ops.INS` or external backend atom.
- **Hybrid**: any part of the measured path uses a handwritten backend atom,
  even when another role uses a generated candidate.

The current gate/up-only policy is **hybrid role-selective**: `ffn_gate_up`
uses the generated buffer2 candidate, while `attn_qo`, `ffn_down`, and
`attn_kv` use the existing handwritten lean route. It must not be described as
pure.

Historical labels and filenames are retained only as immutable evidence keys.
New reports, benchmarks, and discussions should use `pure` or `hybrid`, plus the
role policy when needed.
