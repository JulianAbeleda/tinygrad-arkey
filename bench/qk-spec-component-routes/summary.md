# Spec component routes SCR-0..4

Final verdict: `PROJECT_LEVEL_CLOSE`.

No TBF-3 implementation is earned. Attention generalization and grouped short-T linears both require new kernel families/project-level scheduler work.

## Gate state

- current T=5 verify: `4.652x` one pass
- required: `<=1.3-1.5x` one pass
- needed cut: `0.678`
- Q4_K: `2.916x`; Q6_K/lm_head: `5.831x`; attention/reduces: `3.061x`; linears group: `3.523x`

## Decision

`PROJECT_LEVEL_CLOSE`: reopen only with a measured component candidate, not by starting implementation from the current baseline.
