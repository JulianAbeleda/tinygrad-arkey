# knowledge_center

Durable design knowledge for this hard fork — the "why" and the north-star principles behind the code, so decisions
don't get re-litigated. Not API docs (those live in `docs/`); this is the reasoning layer.

## Contents

- [minimization-principles.md](minimization-principles.md) — the reduced, cited principles for the smallest honest
  inference stack (authored/generated boundary, generate-don't-hand-write kernels, AOT compiler/runtime split,
  rules-as-data optimization, the hardware-submission floor). The north-star for any reduction/rewrite work.
- [100-percent-audit.md](100-percent-audit.md) — definition of done for a quant-grade fast + minimal AOT engine:
  the six-axis scorecard (have/partial/missing) and the ranked gap list with the critical path. The standing tracker.

## Conventions

- Every claim that can be, is **cited** (a project, paper, or in-repo file).
- Principles are stable; sizing numbers are timestamped snapshots (they drift as the tree changes).
- Related project docs: `docs/pure-machine-search.md` (the generated-vs-handwritten route contract),
  `docs/handwritten-kernel-exhaustive-lowering-scope-20260706.md` (the lowering L0->L5 plan).
