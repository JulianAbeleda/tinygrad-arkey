# NV codegen scope: flash decode shape as searchable data (2026-08-18)

Date: 2026-08-18
Branch: `nvidia-bringup-20260731`
Status: **scope record, read-only. Exhaustive plan for the codegen/search work.**

## 0. Decision

The search/codegen substrate is already largely model- and target-neutral. The
work is not to rewrite the substrate or remove historical AMD-derived names;
it is to move the last hand-pinned flash geometry out of Python constants and
into the existing descriptor contract, then let BoltBeam/BubbleBeam/
FutureSight search it like any other candidate.

What stays untouched:

- tinygrad UOp lowering, renderers, and `KernelProgram` execution.
- the target-neutral provider protocol (`extra/llm_research/search_provider.py`).
- BubbleBeam legal-dimension and static-ranking primitives, which already
  consume caller-supplied target facts and never contain a vendor table in
  their shared path.
- `FlashDecodeCapability`, which already reads capability facts from the
  resolved renderer instead of inferring from a device string.

The concrete change is therefore narrow in blast radius and broad in search
coverage: expose the flash geometry knobs as descriptor fields, validate them
against the device facts, and measure the resulting candidates.

## 1. Why this is the next work

Same-session NV decode position (from `nv-us-vs-llama-side-by-side-20260818.md`
and `nv-full-audit-fuse-hide-eliminate-20260818.md`):

| quantity | tinygrad | llama |
| --- | ---: | ---: |
| wall | ~208.8 tok/s | ~246.4 tok/s |
| node sum | wins by ~496 us | - |
| overlap mass | 0 | ~1125 us |
| flash score body (production config) | 4.19 us | 3.16 us |

The remaining flash row is structural: the current template is fixed to
32 lanes, `QG` warps, `TK=16`, `S=48`, `stage_width=1`, and a 5-stage staged
shuffle reduce, while llama's traced shape is an 8-lane score reduce with a
3-stage shuffle ladder and much wider column parallelism. The isolated body
gap is ~+37 us over 36 nodes; the in-situ gap is ~+68 us. The honest ceiling
is ~+2.7 tok/s, and the hard gate for any searched shape is beating 4.19 us at
the production configuration with cold-L2 behavior, not a warm microbenchmark.

This is the codegen row, not a fusion row and not a scheduling row. The
substrate already knows how to emit a flash kernel; it just has no legal way
for search to vary the geometry.

## 2. What is already generic

An audit of the relevant surfaces confirms that the search pipeline is not a
per-vendor code path:

| layer | owner | current state |
| --- | --- | --- |
| provider protocol | `search_provider.py` | envelope, action set, fail-closed errors, target-neutral JSON lines; adapter seam exists |
| target facts | `tinygrad/llm/device_facts.py` | wave size, workgroup limits, shared memory, fp16, shuffle, tensor-core facts scanned from the live renderer |
| renderer capability | `flash_decode_attention.py:655-697` | flash admission is read from renderer capability objects, not from a device-name branch |
| generic legality | `bubblebeam_futuresight.py:118-242` | `propose_legal_dimensions` / `build_static_legality` consume supplied target facts |
| generic ranking | `bubblebeam_futuresight.py:146-207` | deterministic preference scoring, no backend policy |
| policy/evaluation | `BoltBeam` | owns candidate population, ranking, promotion, and evidence; not codegen |
| codegen | tinygrad | owns UOp lowering and emitted ISA |

This means no new search engine and no new compiler are required. The missing
piece is a schema/adapter boundary that carries flash geometry through that
existing loop.

## 3. The hand-pinned values in the flash template

The current production emitter is
`tinygrad/llm/flash_decode_attention.py:92-203`. The values below are the
only route-local values that need to become data; the rest of the emitter is
already a generic UOp builder.

| current location | value | controls | proposed descriptor field |
| --- | --- | --- | --- |
| `flash_decode_attention.py:102` | `LANES = 32` | score-reduce lane width, R, THREADS | `tile.lane_width` |
| `flash_decode_attention.py:102` | `WARPS = QG` | workgroup warps | `tile.warps` (default derived from `query_group_size`) |
| `flash_decode_attention.py:102` | `TK = 16` | token block / LDS tile | `tile.token_block` |
| `flash_decode_attention.py:103` | `RP = Hd // 64` | number of half2 dot pairs per lane | derive as `Hd // (lane_width * dot_pair_width)` |
| `flash_decode_attention.py:136-144` | `DECODE_STAGE_COALESCE` env fallback | staging coalesce width | `tile.stage_width` (already partially present) |
| `flash_decode_attention.py:171-172` | `DECODE_ATTN_BLOCK_TILE_INLINE_REDUCE` env | staged vs inline shuffle ladder | `tile.reduce_structure` |
| `flash_decode_attention.py:206-209` | combine `LANES = 32`, `R = Hd // 32` | combine lane geometry | `combine.lane_width` |
| `flash_decode_attention.py:259-261` | single-stage `S=4, LANES=32, TK=16` | closed research construction | `tile.*` if the construction is retained |
| `flash_decode_attention.py:363-388` | llama-vec `NKQ=8, LANES=32, WARPS=4` | llama trace transcription | `tile.score_group_width`, `tile.lane_width`, `tile.warps` |
| `flash_decode_attention.py:563` | spec `target = "amd_gfx1100"` | default target label | removed or set to `None`; resolved from device facts |
| `decode_routes.py:568` | `_FlashDecodeCandidate.target = "AMD"` | legacy label | derive from resolved renderer target |
| `flash_decode_attention.py:588-590` | kernel name omits geometry | JIT cache identity | include canonical geometry suffix when non-default |

Two notes keep the scope honest:

1. `WARPS = QG` and `S = 48` are shape-derived values, not vendor literals.
   They belong in descriptors because search should be able to vary them, not
   because they encode an AMD assumption.
2. Historical `wave32`/`amd_*` names are provenance labels. Renaming them is
   cosmetic and is only done in files already touched, never as a standalone
   refactor.

## 4. Descriptor contract

Extend the existing `FlashDecodeTileSpec` and `FlashCombineSpec` rather than
introducing a second owner.

### 4.1 Tile fields

```text
FlashDecodeTileSpec(
  Hq, Hd, Hkv, MAXC, split_count,
  staging="KV_BOTH",
  quant=False,
  rope=False,
  token_block=16,          # now searchable, not validator-locked
  lane_width=32,           # physical lane range width
  score_group_width=None,  # None means score_group_width == lane_width
  warps=None,              # None means warps == resolved query_group_size
  query_group_size=None,
  stage_width=1,
  reduce_structure="staged",
  dot_pair_width=2,
)
```

Defaults above reproduce the current production UOp byte-for-byte. Derived
values:

```text
G           = Hq // Hkv
QG          = G if query_group_size is None else query_group_size
warps       = QG if warps is None else warps
threads     = lane_width * warps
group_width = score_group_width or lane_width
R           = Hd // lane_width
RP          = Hd // (lane_width * dot_pair_width)
STAGES      = ceil(token_block * Hd / threads)
```

Legality is checked against target facts, not vendor names:

- `1 <= group_width <= lane_width`
- `lane_width` divides the target subgroup width when a shuffle ladder spans
  the physical lanes
- `Hd % (lane_width * dot_pair_width) == 0`
- `threads <= target.max_workgroup_threads`
- KV tile bytes plus combine scratch, when applicable, fit target shared-memory
  capacity
- `reduce_structure in {"staged", "inline"}`

### 4.2 Combine fields

```text
FlashCombineSpec(
  Hd, Hq, split_count,
  stride=None,
  output_fp16=False,
  lane_width=32,          # defaults to tile.lane_width
)
```

`NW` and `R` derive from `lane_width`; `Hd % lane_width == 0` becomes the
validator.

### 4.3 Identity and kernel naming

All geometry fields participate in canonical JSON and the candidate hash. The
production kernel name stays unchanged for the default geometry. Any non-default
geometry appends a short deterministic suffix so differently emitted programs
cannot collide in the JIT/program cache.

## 5. Search flow

The existing loop with one new payload shape:

```text
BoltBeam proposes flash candidate descriptors
  -> BubbleBeam filters against live target facts
  -> FutureSight statically orders the legal population
  -> tinygrad search provider admits / compiles / checks / measures
  -> BoltBeam ranks measured candidates and owns promotion
```

`search_provider.py` already owns the protocol shell; a small NV adapter (or a
generic CUDA adapter) is added beside the existing Metal adapter. Its
`describe` response supplies `subgroup_size`, workgroup limits, shared memory,
shuffle support, and fdot2 support. The provider must not import BoltBeam
types or make promotion decisions.

The flash geometry is represented as a `flash_decode_candidate.v1` payload
(or a provider-owned sidecar beside the existing `candidate_geometry` pattern
used for WMMA). The candidate hash covers the descriptor, so a change to any
field is a new identity, and the measured binary is bound to that identity.

The static cost model is the one place where NV-specific measured behavior is
allowed, and it must be a fact-derived estimate, not a hardcoded win table:
prefer larger column parallelism, shorter shuffle ladders, and enough LDS to
keep the score hot, with a penalty for cold-L2 launches. FutureSight orders
candidates; only BoltBeam's measured evaluation promotes.

## 6. Implementation phases and gates

### P0. Pin the baseline

1. Default suite green before edits: `DEV=CPU python3 -m pytest test/unit -q`.
2. Record current flash body (4.19 us production config), current kernel
   names, and token sha.
3. Add a regression fixture that asserts the production descriptor round-trip.

Gate: no implementation starts until the baseline is captured.

### P1. Descriptor-owned emitter, no behavior change

1. Extend `FlashDecodeTileSpec` / `FlashCombineSpec` with the fields above.
2. Thread them through the tile and combine builders.
3. Replace the two env-only switches with descriptor fields. Keep the env
   names only as explicit legacy aliases if needed by existing research
   harnesses, not as production defaults.
4. Fix the duplicate `FlashDecodeTileSpec` in
   `extra/llm_research/decode/flash_decode_attention_spec.py` so there is one
   canonical owner.
5. Remove the hardcoded `target` defaults and derive the label at bind time.
6. Add geometry fields to `kernel_name` identity and canonical JSON.
7. Add tests: default values reproduce the current G4/G5 UOp shape and kernel
   names, and invalid geometry fails at `validate()`.

Gate: AMD gfx1100 route byte-identical, production NV route bit-identical
token sha, default suite green. No search or promotion happens in this phase.

### P2. Make the shape searchable

1. Add `bench/qk-search-spaces/targets/nvidia_sm120.json` with measured facts;
   mark `backend_validated` true only after a live NV provider `describe`
   proves the facts.
2. Generate qwen3-8b q4/q5 sm120 profiles from the existing model profile
   derivation (`bench/qk-search-spaces/profiles/_schema.json`).
3. Add a `flash_decode_candidate.v1` schema and provider adapter support
   (admit/compile/check/measure).
4. Teach BubbleBeam legality for the tile fields and add a cold-L2-aware
   FutureSight ordering rule.
5. Add a generic target-facts adapter that does not branch on vendor names.

Gate: a CPU-only search can enumerate and classify the full legal geometry
space; the sm120 descriptor is loadable; no production route changes.

### P3. Measured search

1. Enumerate the legal NV population: `score_group_width`, `lane_width`,
   `token_block`, `stage_width`, `split_count`, `reduce_structure`.
2. Measure every candidate at the production configuration with device-side
   timing and a cold-L2 launch discipline, never warm microbenchmark numbers.
3. Correctness check against the oracle with bit-identical partial/combine
   semantics; token sha must remain the pinned value.
4. Rank and select candidates that beat the 4.19 us body.

Gate: at least one measured candidate beats 4.19 us under cold-L2 at the
production shape before any route A/B.

### P4. Promotion

1. Same-session wall A/B, reverse bracket, candidate vs control, exact token
   sha.
2. Census shows the expected kernel swap and no hidden fallback.
3. Apply the standing +50 us wall promotion bar (or package booking per the
   existing precedent).
4. Commit evidence JSON and book the ledger row.

Gate: promotion is only allowed with all P3/P4 evidence; a candidate that is
faster in isolation but flat on the route is recorded and not promoted.

### Track B: packed-key argmax

This is a separate ELIMINATE row. It reuses the same provider/measurement
substrate after P2, but has its own gate: a cheaper u64 key construction or
reduction that beats the ordinary `Tensor.argmax` chain without changing
token identity. It is lower priority than the flash body and is not a blocker
for P1/P2.

## 7. Non-goals and hard stops

- Do not rewrite BoltBeam/BubbleBeam/FutureSight into a new engine.
- Do not globally rename `wave`/`gfx`/`amd_*` identifiers; only update names
  in touched code.
- Do not regress the AMD gfx1100 flash route or any other promoted route.
- Do not promote a searched shape without P3/P4 gates.
- Do not use warm isolated microbenchmarks as the flash decision.
- Follow the standing working agreement: default suite green, one regression
  test per fix, GPU lock and timeout for every GPU command, no promotion
  without measured evidence.

## 8. Acceptance

The scope is complete when:

- the production flash G4/G5 geometry round-trips through a descriptor;
- `lane_width`, `score_group_width`, `token_block`, `split_count`,
  `stage_width`, and `reduce_structure` are legal dimensions in BubbleBeam
  without a vendor branch;
- a candidate can be compiled and checked through the provider on NV sm_120;
- the measured loop has a candidate that beats 4.19 us cold-L2 at the
  production configuration;
- any promoted shape passes the full wall A/B and token-identity gate and is
  booked in the ledger.

## 9. Evidence map

- baseline flash floor: `nv-flash-score-floor-test-head-20260816.md`
- device-side body truth: `nv-truth-audit-flash-body-20260813.md`
- exhaustive FUSE/HIDE/ELIMINATE audit:
  `nv-full-audit-fuse-hide-eliminate-20260818.md`
- FUSE/HIDE/ELIMINATE ledger:
  `nv-fuse-hide-eliminate-ledger-20260818.md`
- pure-machine-search ownership rules: `docs/pure-machine-search.md`
- working agreement: `nv-decode-working-agreement-20260808.md`

New evidence lands under `docs/task_workflow/evidence/` with dated names:
`nv-flash-geometry-search-<date>.json`,
`nv-flash-geometry-gate-<date>.json`,
`nv-flash-geometry-ab-<date>.json`.
