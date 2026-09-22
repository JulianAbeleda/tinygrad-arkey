# BubbleBeam/search substrate audit: remaining NV flash codegen gaps (2026-08-19)

Date: 2026-08-19
Repo: `/home/ubuntu/tinygrad-arkey`
HEAD: `d14e6964ee211f286b867616abe1daa9a137db5b`
Status: **read-only audit. No code or GPU work.**

This audit maps the current BubbleBeam/search substrate to the NV flash decode
gaps named in `nv-fuse-hide-eliminate-ledger-20260818.md` rows L3 and E1. It is
limited to the four search modules, the flash emitter, and the ledger/search
update section listed in the task.

## 1. Bottom line

- The schema and generic legality path are target-neutral. The remaining gaps
  are not in `flash_candidate_schema.py` or `bubblebeam_futuresight.py`; they
  are the CUDA-only measurement harness, the emitter's refusal to decouple
  `score_group_width` from `lane_width`, the hard-coded `S=48` promoted route
  for `Hq=32`, and the fact that `search_provider.CudaAdapter` is not connected
  to any real compiler/runtime adapter.
- Six axes are actually enumerated and searchable:
  `lane_width`, `token_block`, `stage_width`, `reduce_structure`,
  `dot_pair_width`, `split_count`.
- Two axes exist in the descriptor but are not searchable today:
  `score_group_width` is pinned to `None` and, when set, must equal
  `lane_width`; `warps` is pinned to `None` and only admits
  `warps >= query_group_size`.
- The llama structural target (128 threads, 8-lane score groups, 2 KV splits)
  cannot be represented by the current tile emitter without a structural
  emitter change, not just a new set of legal values.

## 2. Current searchable surface and its gates

`nv_flash_geometry_population.py` enumerates the Cartesian product in
`extra/llm_research/decode/nv_flash_geometry_population.py:35-48`:

```text
lane_width        {8, 16, 32}
token_block       {8, 16, 32}
stage_width       {1, 2, 4, 8}
reduce_structure  {staged, inline}
dot_pair_width    {2, 4}
split_count       {32, 48, 64}
```

`score_group_width` and `warps` are passed as `None` at
`nv_flash_geometry_population.py:91` and `nv_flash_geometry_search.py:93-94`.
The full product is `3 * 3 * 4 * 2 * 2 * 3 = 432`, matching the ledger's "432
legal candidates" at
`docs/task_workflow/input/nv-fuse-hide-eliminate-ledger-20260818.md:146-149`.

### 2.1 Descriptor-level target-agnostic gates

`extra/llm_research/flash_candidate_schema.py:147-200` (`validate`) owns the
pure geometry arithmetic:

| field | gate |
| --- | --- |
| `Hq`, `Hd`, `Hkv`, `MAXC`, `split_count` | positive; `Hq % Hkv == 0` |
| `staging` | `KV_BOTH` or `K_ONLY` |
| `quant`, `rope`, `output_fp16` | bool |
| `token_block`, `stage_width` | positive |
| `lane_width` | positive power of two |
| `score_group_width` | positive and must equal `lane_width`, else `None` |
| `query_group_size` | `1..Hq/Hkv` |
| `warps` | `>= query_group_size` (or `>= Hq/Hkv` when `query_group_size` is `None`) |
| `reduce_structure` | `staged` or `inline` |
| `dot_pair_width` | positive; `Hd % (lane_width * dot_pair_width) == 0` |
| `combine` (when present) | positive Hd/Hq/split_count; optional stride; combine lane width is pow2 and divides Hd |

The key structural lock is at
`flash_candidate_schema.py:163-167`: a non-null `score_group_width` must equal
`lane_width`. That encodes "sub-lane groups are invalid" at the schema layer,
before any target fact is consulted.

### 2.2 Fact-driven legality added by BubbleBeam

`extra/llm_research/bubblebeam_futuresight.py:192-222`
(`build_flash_legality`) adds only resource and physical-subgroup checks:

- `subgroup_size % lane_width == 0` when `derived_group_width > 1`
  (`bubblebeam_futuresight.py:217-218`).
- `derived_threads(tile) <= max_threads_per_threadgroup`
  (`bubblebeam_futuresight.py:219`).
- `local_memory_bytes(descriptor) <= max_threadgroup_memory_bytes`
  (`bubblebeam_futuresight.py:220`).

`derived_threads`, `derived_warps`, `derived_group_width`, and
`local_memory_bytes` live in `flash_candidate_schema.py:231-259`. With the
SM120 facts `{subgroup_size: 32, max_threads: 1024, max_local: 232448}`
(`nv_flash_geometry_population.py:29-33`), every one of the 432 combinations
passes this gate because the searched range never approaches the resource
limits.

### 2.3 Static priority

`bubblebeam_futuresight.py:225-260` (`build_flash_static_priority`) scores:

```text
column    = min(group_width, subgroup_size)
ladder    = reduce_stages(tile)
resident  = 2 if 2*local_memory_bytes fits else 1
hot       = 1 if staging == "KV_BOTH" else 0
relaunch  = bit_length(threads // stage_width - 1), clamped at 0
score     = column - ladder + resident + hot - relaunch
```

The tie break is candidate hash in both population builders
(`nv_flash_geometry_population.py:118` and
`nv_flash_geometry_search.py:115`). Since the six enumerated axes leave
`score_group_width = None`, `column` is always `min(lane_width, 32)`, so the
priority function never gets to rank a real sub-lane group.

### 2.4 Emitter-level legality (a separate pass)

Both population builders run the real emitter only after the schema/facts pass:

- `nv_flash_geometry_population.py:104-114` calls
  `describe_flash_decode_attention(...).validate()` and records
  `tile_validate:<error>` on failure.
- `nv_flash_geometry_search.py:103-113` does the same and labels the row
  `emitter_invalid`.

The emitter repeats and extends the schema gates in
`tinygrad/llm/flash_decode_attention.py:150-173` and
`tinygrad/llm/flash_decode_attention.py:652-674`:

- `stage_width` is restricted to `{1, 2, 4, 8}` in
  `flash_decode_attention.py:659`.
- `warps >= query_group_size` in `flash_decode_attention.py:666-667`.
- `score_group_width == lane_width` or `None` in
  `flash_decode_attention.py:662-664`.

There is one silent emitter fallback that the schema legality does not
surface. `CooperativeStageLaneMap.validate()`
(`flash_decode_attention.py:96-100`) requires
`token_block*Hd % (threads*stage_width) == 0`. In the tile builder this is
wrapped in `try/except` and falls back to `selected_width = 0`, which emits the
classic non-coalesced staging path (`flash_decode_attention.py:205-216`). A
candidate can therefore be "legal" in the schema while silently emitting a
different staging shape than its `stage_width` claims. The current axes happen
to avoid this because `warps` is pinned and
`token_block*Hd=2048` is divisible by `threads*{1,2,4,8}` for every searched
lane width.

## 3. What is hard-coded or out of reach

### 3.1 The measured search is CUDA-only, while production is DEV=NV

`nv_flash_geometry_search.py` documents its own CUDA discipline at
`nv_flash_geometry_search.py:2-8` and enforces it in two hard gates:

- `run_check` raises unless `Device.DEFAULT == "CUDA"` at
  `nv_flash_geometry_search.py:145-147`.
- `run_measure` raises the same guard at
  `nv_flash_geometry_search.py:183-185`.

Measurement also hard-codes CUDA literals: `_inputs(device="CUDA")` at line 51,
`dst` on `device="CUDA"` at line 212, and `Device["CUDA"].synchronize()` at
lines 215 and 218. The timing authority is `nsys profile --trace=cuda` plus
the CUPTI sqlite table `CUPTI_ACTIVITY_KIND_KERNEL`
(`nv_flash_geometry_search.py:233-237`), which has no DEV=NV equivalent.

The population `enumerate` mode is CPU-only and does not carry this gate, but
the actual correctness and timing evidence are both CUDA-only. The production
backend named in the task is `DEV=NV`, so as written the search cannot produce
NV evidence.

### 3.2 The emitter ties `score_group_width` to `lane_width`

The tile dot ownership formula is
`elem = pair_axis*(LANES*dot_pair_width) + lane*dot_pair_width`
(`flash_decode_attention.py:234`). Every lane must contribute to cover `Hd`.
The emitter states this explicitly and rejects a narrower group at
`flash_decode_attention.py:164-169`, and again at spec validation at
`flash_decode_attention.py:662-664`. `flash_candidate_schema.py:163-167` makes
the same choice at the descriptor boundary.

This means:

- `score_group_width < lane_width` is not merely unsearched; it is illegal.
- The physical-lane width (`lane_width`) is overloaded with the score-reduce
  width. llama needs them decoupled: physical lanes 32, score group 8, with
  four groups per warp scoring four different KV columns.

### 3.3 `S=48` is the frozen split count for `Hq=32`

The promoted G4 route carries `split_size=48` at
`flash_decode_attention.py:907-908`, and `_promoted_route_split_count` returns
48 for `(Hq=32, query_group_size=None)` at
`flash_decode_attention.py:50-56`. The control also hard-codes 48 in both
harnesses:

- `nv_flash_geometry_search.py:39` `CONTROL_NAME`, `:153` `_spec(48)`,
  `:199` `_spec(48, ...)`.
- `nv_flash_geometry_population.py:122`
  `describe_flash_decode_attention(32, 128, 8, 4608, 48, ...)`.

The production executor also rejects any non-promoted split at
`flash_decode_attention.py:927-932`, where
`(split_size, query_group_size, staging)` must equal the G4 or G5 promoted
triple. A direct `describe_flash_decode_attention` call can emit `S=2` for
research, but the production route guard would not run it without a route or
lease change.

### 3.4 The fixed llama template is not behind the descriptor

`flash_vec_llama_score_pv_kernel`
(`tinygrad/llm/flash_decode_attention.py:445-607`) is the existing transcription
of llama's shape. Its hard-coded facts are `NKQ=8, LANES=32, WARPS=4,
THREADS=128` at line 462, and it is fixed to `(Hd,Hq,Hkv) == (128,32,8)` at
line 460. It is called out as "closed-default, not routed" in its docstring
(line 446) and is not reachable from `FlashDecodeTileSpec`,
`describe_flash_decode_attention`, or the `flash_decode_candidate.v1` schema.
Its `S` is a positional parameter, so the split count is not a canonical
descriptor field there.

There is a documentation discrepancy to resolve before opening the axis: the
function docstring says llama uses `S=4` at context 512
(`flash_decode_attention.py:456-457`), while the ledger L3 target and the
pinned matched launch describe the structural target as 2 KV splits
(`grid.y=2`, 64 blocks) in
`docs/task_workflow/input/nv-fuse-hide-eliminate-ledger-20260818.md:28` and
`docs/task_workflow/input/nv-decode-path-pseudocode-fresh-ledger-20260816.md:40-51`.
Making `split_count` an explicit descriptor field also settles which of those
two values is the search target.

## 4. Is llama's shape representable in the current emitter?

The short answer is **no**, not as a legal `flash_decode_candidate.v1` tile
with only value changes. The target is:

```text
128 threads        (4 warps x 32 physical lanes)
8-lane score group (3 shuffle stages)
128 columns scored in parallel per block
2 KV splits
Q register-resident, K/V streamed, not per-tile LDS staged
```

The current tile emitter maps one score per physical lane and serially walks
the 16 tokens of a tile:

- `R = Hd // lane_width`, so each lane owns `R` dims.
- The dot reduce uses all `lane_width` lanes
  (`flash_decode_attention.py:229-246`).
- `token_in_tile` is a serial reduce over `token_block`
  (`flash_decode_attention.py:258-265`).

Setting `score_group_width=8` under the current ownership would sum only
`8/32` of the dot, which is why the emitter rejects it
(`flash_decode_attention.py:164-169`). Setting `lane_width=8` instead gives a
32-thread workgroup (`8 lanes * 4 warps`), not the 128-thread / 8-lane-group
shape. `split_count=2` is accepted by the schema but excluded from the search
axis (`nv_flash_geometry_population.py:40`) and rejected by the production
route guard (`flash_decode_attention.py:927-932`).

To represent the llama shape, one of two changes is required:

1. Generalize the tile emitter to a two-level ownership
   `(group, lane_in_group)` with `groups_per_warp = lane_width //
   score_group_width`, `R_group = Hd // score_group_width`, and one column
   scored per group. This also needs register-resident Q and parallel column
   scheduling, not the current serial `token_in_tile` loop.
2. Promote the existing `flash_vec_llama_score_pv_kernel` transcription behind
   `FlashDecodeTileSpec`/`describe_flash_decode_attention`, replacing its
   constants with descriptor fields (`score_group_width`, `lane_width`,
   `warps`, `split_count`) and its fixed shape gate with the schema gates.

Option 2 is less work and reuses the already-written transcription, but it is a
different emitter body from the tile kernel, so the descriptor/kernel-name
contract must be able to select between emitters rather than assuming one tile
shape.

## 5. Wiring plan

### 5.1 Ownership

| concern | current owner |
| --- | --- |
| canonical identity and hash | `extra/llm_research/flash_candidate_schema.py:203-205` (`candidate_hash` over canonical JSON) |
| short kernel-name identity | `tinygrad/llm/flash_decode_attention.py:60-85` `_tile_geometry_suffix` and `:688-698` `FlashDecodeTileSpec.kernel_name` |
| target-agnostic geometry legality | `flash_candidate_schema.py:147-200` `validate` |
| resource/subgroup legality | `bubblebeam_futuresight.py:192-222` `build_flash_legality` |
| emitter representability | `flash_decode_attention.py:150-173` and `:652-674` |
| static ranking | `bubblebeam_futuresight.py:225-260` `build_flash_static_priority` |
| finite expansion | `nv_flash_geometry_population.py:35-48` and `nv_flash_geometry_search.py:79-114` |
| admission | `search_provider.CudaAdapter.admit` (`search_provider.py:729-749`) |
| compile/check/measure | `nv_flash_geometry_search.py:133-247` directly, bypassing the provider |

The legality authority is split three ways and partially duplicated. The
recommended split is to keep schema arithmetic in
`flash_candidate_schema.py`, resource facts in `build_flash_legality`, and move
the emitter-specific checks into a single `FlashDecodeTileSpec.validate` that
does not disagree with the schema. In particular the schema's
`score_group_width == lane_width` rule must be relaxed in lockstep with the
emitter; relaxing only one creates an "admitted but unemittable" hole.

### 5.2 Minimal edits to run the search on DEV=NV

In `nv_flash_geometry_search.py`:

1. Add a `--device` argument (default `NV`, allow `CUDA` for the existing
   CUPTI path) and replace the two `Device.DEFAULT != "CUDA"` raises at lines
   145-147 and 183-185 with the resolved device check.
2. Thread the device through `_inputs` (line 51), the measure `dst` allocation
   (line 212), and both `Device[...].synchronize()` calls (lines 215, 218).
3. Replace the `nsys --trace=cuda` + `CUPTI_ACTIVITY_KIND_KERNEL` timing path
   (lines 233-247) with an NV-capable device timing source. The CUPTI sqlite
   parser is CUDA-specific and cannot simply be retargeted to DEV=NV.
4. Keep the CPU-only `enumerate` mode unchanged, and derive the hard-coded
   `SM120_FACTS` (lines 37-38) from the opened renderer/device facts rather
   than a module constant if the search is meant to generalize beyond sm120.

In `search_provider.py`:

1. Add an `NV` backend token (currently only `METAL` and `CUDA` at line 822)
   that constructs the same facts-driven adapter; the adapter itself is already
   target-neutral because `_nv_facts` never autodetects a device
   (`search_provider.py:627-643`).
2. Wire real `compile_fn`, `check_fn`, and `measure_fn` hooks, or point the
   adapter at the same `KernelProgram`/`execute_research_program` path the
   measured harness already uses.

### 5.3 Minimal edits to open the structural dimensions

1. Relax `score_group_width` in `flash_candidate_schema.py:163-167` from
   "equal lane_width or null" to a legal sub-lane group set, e.g. positive
   power of two dividing `lane_width`, plus `Hd % (score_group_width *
   dot_pair_width) == 0`.
2. Add matching emitter support. This is the non-trivial edit: re-express the
   dot ownership in `flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel`
   as `(group, lane_in_group)` so each group covers the full `Hd`, or select
   `flash_vec_llama_score_pv_kernel` for the sub-lane-group geometry.
3. Add `score_group_width` and `warps` to the enumerated axes in
   `nv_flash_geometry_population.py:42-49` and
   `nv_flash_geometry_search.py:85-90`, with `warps` constrained by
   `derived_threads <= max_threads`.
4. Add `split_count=2` to the searched split set and decide whether it is
   promoted-route geometry or a research-only emitter test. If promoted, update
   `FLASH_DECODE_G4`/`_promoted_route_split_count` and the route guard at
   `flash_decode_attention.py:907-932`; if research-only, the search harness
   can call `describe_flash_decode_attention` directly without the route guard.
5. Keep the descriptor and kernel-name suffix in sync: once
   `score_group_width` and `warps` become real dimensions, their suffixes
   already exist (`_sgw`, `_w`) at `flash_decode_attention.py:83-84`, but the
   canonical hash and the short name must both include the new legal values.

The ledger itself marks this as Case A (expose levers and search) versus
Case B (build the missing reduce/output primitive first) at
`docs/task_workflow/input/nv-buildable-lever-rows-exhaustive-scope-20260817.md:112-116`.
The representability analysis above says the llama target is Case B for the
current tile body: the 8-lane column-parallel mapping is a new primitive, not
a new value in the existing serial tile.

## 6. Provider shell gap

`search_provider.CudaAdapter` is a facts-validation shell, not a connected
compiler/runtime adapter:

- `live_backend` defaults to `False`, and `compile`, `check`, and `measure`
  all fail closed with `backend_unavailable` unless it is claimed
  (`extra/llm_research/search_provider.py:702-705`, `:751-776`).
- The CLI `--backend CUDA` constructs `CudaAdapter()` with no `live_backend`
  and no `*_fn` hooks (`search_provider.py:824`), so compile/check/measure can
  never succeed from the CLI.
- `_canned_cuda_compile`, `_canned_cuda_check`, and `_canned_cuda_measure`
  (`search_provider.py:646-688`) are deterministic fakes reachable only after
  `live_backend=True`; nothing in the repository sets that flag.
- The actual measured flash search does not call `process`, `process_line`, or
  `serve`; it imports `Device`, `Tensor`, `KernelProgram`, and
  `execute_research_program` directly
  (`nv_flash_geometry_search.py:26-30`). So the protocol shell is a parallel
  scaffold for flash geometry, not the path that produced
  `nv-flash-geometry-search-20260819.json`.
- `MetalAdapter` (`search_provider.py:303`) is a real Metal compiler/runtime
  adapter, but its shared `_candidate` admission path requires
  `boltbeam.full_kernel_candidate.v2` (`search_provider.py:169-174`), not
  `flash_decode_candidate.v1`, so it cannot be reused for flash geometry
  without a second adapter.

The minimal honest wiring is either:

- register the search harness's `_tile_program` / `_combine_program` /
  `_run_fused` path as the `CudaAdapter` (renamed NV adapter)
  `compile_fn`/`check_fn`/`measure_fn`; or
- state in the adapter's `describe` limitations that flash measurement is
  owned by the dedicated `nv_flash_geometry_search.py` harness until the
  adapter is connected.

Until one of those is done, the provider protocol's flash success path is
exercised only by unit tests against the canned fakes, not by the measured
search evidence.

## 7. Non-goals

This audit does not change code, run GPU commands, or reach over the network.
The promotion outcome is already closed by the ledger's clean-GPU verdict
(`nv-fuse-hide-eliminate-ledger-20260818.md:177-206`); this document only maps
what would have to change for a future DEV=NV search and for the unrepresented
llama structural shape.
