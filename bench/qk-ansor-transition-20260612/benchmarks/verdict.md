# QK Loop Benchmark Verdict

Candidates are compared against the current accepted generated policy, not
against explicit primitive flags. Raw accepts that fail confirmation are not
promoted.

## Summary

- models: `3`
- models with raw accept: `1`
- models with confirmed accept: `0`
- overall decision: `descriptor_knob_frontier_exhausted`

| model | matrix accepted | confirmed accepted | decision | confirmation |
|---|---|---|---|---|
| `8B` | `none` | `none` | `descriptor_knob_frontier_exhausted` | none |
| `14B` | `none` | `none` | `descriptor_knob_frontier_exhausted` | none |
| `32B` | `001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32` | `none` | `raw_accept_unconfirmed_or_rejected_by_confirmation` | 001-ffn-gate-blk-0-ffn-gate-weight-q4_k_packed_u32-p1-local32: tie (-2.29%) |
