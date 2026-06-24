# QK Policy Storage Architecture

Date: 2026-06-12

Status: memory-aware policy, runtime storage accounting, and opt-in shared
primitive storage implemented. Shared storage is validated for full 8B, 14B,
and 32B generated-policy harness runs; sidecar remains the runtime default.

## Problem

The generated Q4_K/Q6_K policy can choose good primitive coverage for 8B and
14B, but the uncapped 32B sidecar policy does not fit in VRAM. The failure is
not search or correctness. The generated 32B policy passed semantic search and
parity, then failed during primitive storage install:

```text
MemoryError: Allocation of 70.31 MB failed on AMD. Used: 23.80 GB
```

The cause is duplicated storage. The model keeps the fallback GGUF-backed weight
graph, and each primitive wrapper also installs its own persistent packed-weight
Tensor. For 32B, the full generated sidecar would add about `17.8 GiB` of
primitive storage across `448` wrappers, which cannot coexist with the model on
the 24 GB card.

## Sidecar Policy Fix

The policy format now supports tensor-scoped entries in addition to the original
shape-scoped entries.

- Old format: one entry per `(ggml_type, rows, cols)` shape.
- New format: exact tensor entries keyed by tensor name, with fallback to shape
  entries for old artifacts.
- Runtime lookup checks tensor entry first, then shape entry.
- Each entry can carry storage metadata and a `policy_reason`.

`extra/qk_ansor.py --policy-max-storage-mb N` now emits a memory-capped policy:

1. Expand shape search results to real model tensors.
2. Estimate persistent primitive bytes per tensor.
3. Estimate benefit from fused baseline device time minus primitive device time.
4. Rank primitive candidates by `benefit_ms_per_mb`.
5. Select candidates until the byte cap is reached.
6. Emit fused-graph fallback entries for over-budget tensors.

The pipeline can compare this capped generated policy against a generic fused
baseline with `--reference-mode generic`, which was needed for sidecar 32B
experiments because the full explicit primitive baseline also OOMed.

## Runtime Controls

The runtime now reports actual primitive sidecar bytes under
`Q4K_PRIMITIVE_DEBUG`, `Q6K_PRIMITIVE_DEBUG`, or `QK_GENERATED_POLICY_DEBUG`.

Relevant environment variables:

- `QK_PRIMITIVE_MAX_STORAGE_MB=N`: cap total persistent primitive sidecar bytes
  across Q4_K and Q6_K installers. Tensors that would exceed the cap fall back
  and increment `runtime_storage_cap`.
- `QK_GENERATED_POLICY_STRICT=1`: turn storage-cap fallback into a loud error
  for generated-policy runs.
- `QK_PRIMITIVE_STORAGE=sidecar`: default fast path, preloads packed primitive
  storage on device.
- `QK_PRIMITIVE_STORAGE=q4_ondemand`: experimental Q4_K-only non-persistent
  path. It keeps the Q4 packed slice disk-backed and copies at decode time.
- `QK_PRIMITIVE_STORAGE=shared`: experimental shared-buffer path. It references
  the raw GGUF byte tensor already realized for the fallback graph through a
  typed buffer view, so selected Q4_K/Q6_K primitive tensors report
  `shared_bytes` instead of allocating a second persistent sidecar.

`q4_ondemand` is a negative prototype, not a recommended mode. It proves that
Q4 persistent sidecar bytes can be removed, but the per-token copy cost destroys
decode throughput.

`shared` is the first real dedup path. It has passed full 8B, 14B, and 32B
pipelines. The key property is now measured: selected Q4_K/Q6_K tensors report
`storage_bytes=0` and `shared_bytes` equal to the selected source ranges instead
of allocating duplicate primitive sidecars. It is recommended for generated
policy runs when memory behavior or cross-model consistency matters, but it is
not the code default because the older 8B sidecar artifact remains slightly
faster and sidecar is still useful as a performance control.

## 32B Capped Sidecar Result

Artifact: `bench/qk-policy-cap-20260612/32b-1536mb/`.

Command:

```sh
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_policy_pipeline.py \
  --model ~/models/Qwen3-32B-Q4_K_M.gguf \
  --out bench/qk-policy-cap-20260612/32b-1536mb \
  --device AMD --level 2 --iters 2 --benchmark 128 --repeats 3 \
  --max-extra-repeats 1 --profile auto --reference-mode generic \
  --policy-max-storage-mb 1536 --reuse
```

Decision:

| model | reference | capped policy | cap | selected primitive tensors | result |
|---|---|---|---:|---:|---|
| Qwen3-32B-Q4_K_M | generic fused graph | generated tensor policy | `1536 MB` | `144` | accept |

Decode:

| mode | avg tok/s | stable | note |
|---|---:|---|---|
| generic fused baseline | `3.44` | yes | pipeline labels this as `explicit` because it is the reference side |
| capped generated policy | `4.16` | yes | `20.98%` faster, greedy output A/B match |

Storage selection:

| selected role | tensors |
|---|---:|
| `attn_k` | `64` |
| `attn_v` | `64` |
| `ffn_down` | `16` |

Selected bytes: `1,600,389,120` under a cap of `1,610,612,736`.

The same policy re-run with runtime accounting reports exactly
`1,600,389,120` persistent bytes (`1526.25 MB`) and reaches about `4.30 tok/s`
after warmup in a short benchmark-4 smoke.

Important caveat: this is not an explicit-full-primitive comparison. The full
explicit Q4/Q6 primitive baseline does not fit for 32B. This result says a
memory-aware generated policy can recover some speed while fitting beside the
generic model graph.

This result is now historical/control evidence. The shared-storage run below is
the current 32B comparison because it fits the full explicit and generated
primitive policies without sidecar duplication.

## Shared Storage Validation

Artifact: `bench/qk-shared-storage-20260612/`.

8B validation:

| mode | Q4 wrappers | Q6 wrappers | storage bytes | shared bytes | result |
|---|---:|---:|---:|---:|---|
| generated shared smoke | `162` | `18` | `0` | `3,970,695,168` | warm decode about `57 tok/s`, greedy A/B match |
| full shared harness | `162` | `18` | `0` | `3,970,695,168` | accept, `52.07` vs `50.41 tok/s`, `3.31%` gain |

14B validation:

| mode | Q4 wrappers | Q6 wrappers | storage bytes | shared bytes | result |
|---|---:|---:|---:|---:|---|
| full shared harness | `240` | `40` | `0` | `7,918,387,200` | accept, `40.55` vs `21.77 tok/s`, `86.29%` gain |

32B smoke:

| mode | Q4 wrappers | Q6 wrappers | storage bytes | shared bytes | result |
|---|---:|---:|---:|---:|---|
| generated shared policy | `384` | `64` | `0` | `18,677,760,000` | full uncapped policy loads and decodes |

32B full harness:

```sh
DEV=AMD QK_PRIMITIVE_STORAGE=shared PYTHONPATH=. \
  .venv/bin/python extra/qk_policy_pipeline.py \
  --model ~/models/Qwen3-32B-Q4_K_M.gguf \
  --out bench/qk-shared-storage-20260612/32b \
  --device AMD --level 2 --iters 2 --benchmark 128 --repeats 3 \
  --max-extra-repeats 1 --profile auto --reference-mode explicit \
  --search-timeout 3600 --reuse
```

Decision:

| model | reference | generated policy | result | storage |
|---|---|---|---|---|
| Qwen3-32B-Q4_K_M | shared explicit Q4/Q6 primitives | shared generated policy | accept, `17.23` vs `11.15 tok/s`, `54.56%` gain | `storage_bytes=0`, `shared_bytes=18,677,760,000` |

The 8B, 14B, and 32B rows now pass parity, greedy output A/B, repeated decode,
and required profile gates. They are included in
`bench/qk-shared-storage-20260612/matrix-summary.md`, which is covered by the
matrix reproducibility test.

## Why This Is Architectural

Shape-level policy is the wrong abstraction for memory pressure. A single shape
decision can imply dozens of layer tensors. On 32B, choosing "primitive for this
shape" can silently allocate gigabytes of persistent sidecar storage.

Tensor-scoped policy is the minimum required control surface:

- it allows selecting high-benefit layers while leaving low-benefit layers on
  the fused graph;
- it records why each tensor was selected or fused;
- it makes memory a first-class search constraint rather than an after-the-fact
  OOM.

This moves the project in the Ansor direction without pretending the storage
problem is solved. The searcher now reasons about a model-level resource budget,
not only per-shape speed.

## Storage-Control Validation

Artifact: `bench/qk-storage-20260612/`.

8B smokes:

| mode | Q4 wrappers | Q6 wrappers | persistent storage MB | warm tok/s | verdict |
|---|---:|---:|---:|---:|---|
| generated sidecar | `162` | `18` | `3786.75` | `57.77` | fast path |
| generated runtime cap `512 MB` | `28` | `0` | `504.00` | `13.46` | cap works, install-order selection is slow |
| generated `q4_ondemand` | `162` | `18` | `708.75` | `0.55` | rejects per-token Q4 copy |

32B smokes:

| mode | Q4 wrappers | Q6 wrappers | persistent storage MB | warm tok/s | verdict |
|---|---:|---:|---:|---:|---|
| static generated cap `1536 MB` | `112` | `32` | `1526.25` | `4.30` | useful selected policy |
| full generated policy + runtime cap `1536 MB` | `43` | `0` | `1535.62` | `3.71` | guard works, but not an optimizer |

Interpretation:

- Runtime accounting matches the generated policy storage estimate.
- The runtime cap prevents OOM and explains skipped tensors.
- Install-order capping is slower than generated benefit-per-MB capping.
- Q4 on-demand removes persistent Q4 sidecar bytes but is too slow to use.

## Long-Term Design

The cap is a bridge. The long-term design should reduce or remove duplicate
storage:

1. Shared packed storage
   - A primitive wrapper can now reference the existing GGUF-backed raw byte
     tensor through a typed view under `QK_PRIMITIVE_STORAGE=shared`.
   - This avoids making a second persistent device copy when the source storage
     is already resident and can be viewed safely.
   - Current validation covers full 8B, 14B, and 32B harnesses. The promotion
     decision is recommendation-level, not runtime-default-level: use shared for
     generated-policy runs when memory behavior matters; keep sidecar available
     for the slightly faster 8B peak artifact.

2. Lazy materialization
   - Install primitive metadata at model load, not necessarily all sidecar
     buffers.
   - Materialize packed buffers only for tensors selected by a memory policy or
     first use.
   - Keep allocation failure local to one tensor and fall back loudly.

3. Budget-aware policy generation
   - Keep `persistent_bytes`, `benefit_ms`, and `benefit_ms_per_mb` in policy
     entries.
   - Support caps by absolute bytes and by remaining device memory.
   - Make policy decisions reproducible and artifact-pinned.

4. Runtime diagnostics
   - Report selected/skipped tensor counts and reasons.
   - Report estimated and actual persistent primitive bytes.
   - Fail loudly when a policy requests more storage than the runtime cap.

5. Search integration
   - Treat storage as part of the candidate objective:
     `speedup subject to persistent_bytes <= budget`.
   - Do not let BEAM or generated search choose a policy that cannot fit the
     target model/device pair.

## Next Work

Ordered by architectural value:

1. Keep shared storage explicit in the environment while it gets more soak.
2. Treat shared as the recommended generated-policy storage mode for memory
   behavior and cross-model consistency; keep sidecar as a control and 8B
   peak-speed fallback.
3. Use the shared-storage 8B/14B/32B matrix as the scaling point, not as a
   reason to restart model-specific tuning.
4. Do not resume kernel search from this track. The storage work is now
   infrastructure; use it to support higher-level loop/harness work.

Stop rule: do not spend more time on 32B candidate search until the storage
architecture can explain where every persistent packed-weight byte lives.
