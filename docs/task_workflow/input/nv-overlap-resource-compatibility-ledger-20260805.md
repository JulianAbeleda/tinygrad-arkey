# NV overlap resource-compatibility ledger — 2026-08-05

## Result

**INCONCLUSIVE, fail closed.** Llama’s pinned d512 timeline shows 445.954 us of Q8_1 interval mass hidden behind MMVQ, but that is aggregate interval containment, not a named dependency-independent pair. The role manifest has llama grid/block/register/static/dynamic shared fields, but no local-memory field; the tinygrad logical-ready/census capture has call identities and durations but no compiled grid/block/register/shared/local tuple. Therefore no per-CTA complementarity calculation, occupancy bound, or highest-overlap-mass pair is defensible.

`extra/llm_research/decode/nv_overlap_resource_join.py` makes this absence executable and refuses partial rows. The accompanying unit test proves that missing local memory cannot be silently treated as zero.

The cheapest decisive next step is CPU-only: enrich one aligned tinygrad capture with per-call compiled resource metadata and join it to its logical-ready pairs. Only if that produces a dependency-independent pair with a positive complementary CTA residency bound should one native two-queue span A/B be authorized. Current evidence does not justify that probe.

## Capture-only collector gate

`route_b3_dag_attribution.attach_compiled_descriptors` is the no-behavior-change
join seam. A capture harness must pass the exact ordered `CallRecord` list and
one descriptor per occurrence: binary SHA-256, grid, block, registers/thread,
static/dynamic shared bytes, and local-memory bytes. Missing, extra, or partial
rows raise; names are never used as a surrogate for occurrence identity.

Hard gate for the deferred capture-only run: acquire `/tmp/gpu-bench.lock` only
after the active sweep releases it; emit all 875 rows with exact occurrence IDs
and hashes; then rerun `nv_overlap_resource_join.py`. A native overlap span
probe is authorized only if that output names an independent pair and proves a
positive resource-complementarity bound.
