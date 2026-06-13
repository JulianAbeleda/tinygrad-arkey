# QK Memory Access Audit

Date: 2026-06-13

This artifact gates the next packed-load optimization step after semantic
codegen v3 tied on 8B/14B.

Files:

- `vector-probe.json` / `vector-probe.md`: AMD capability probe for normal UOp
  `uint32.vec(4)` loads versus a raw custom-C `uint4` escape.
- `load-width/`: DEBUG=4 generated-source logs for both probe modes plus a
  parsed load-width report.
- `audit.json` / `audit.md`: combined decision using the vector probe, Family C
  v0 load-width report, model-scope roofline, and prior PMC smoke.

Verdict:

- normal UOp `uint32.vec(4)` global load support: `True`;
- raw custom `uint4` escape support: `True`;
- Family C v1 is unblocked as the next generated memory-access candidate;
- 32B should still stay skipped until 8B/14B show promise.
