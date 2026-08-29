# Gate/up + K + Q/O combined evidence

Authoritative result: `../../output/nv-compiler-q4k-gkqo-combined-result.md`.

The bookable candidate is `candidate-safe-cut-v2-deep20-r9.json`; its matched
corrected gate/up+K control is `control-ready-r9.json`, and cross-arm logits are
in `compare-safe-cut-v2-vs-gatek.json`.

`combined-flash-direct-deps-cut-v2.json` is generated only from
`combined-default-v2.profile.jsonl` and `combined-default-v2.census.jsonl`.
The v1 policy is retained as negative evidence: it exposed an ambiguous exact
prefix and was rejected before execution. Files named `smoke` or generated
with profiling enabled are non-authoritative.
