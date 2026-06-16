# Upstream Surface Audit — 2026-06-16

Audit of the **upstream tinygrad** code this fork carries (everything inherited
from `tinygrad/tinygrad`, vs the fork-added `extra/qk_*`/`extra/llm_*` + the
`tinygrad/llm` decode/codegen diff). Question this answers: **what does this
AMD-decode fork actually use, what is dead weight, and what is the real cost of
pruning it.**

## Framing (why this is not a principles cleanup)

The coding principles target *knowledge-duplication in code you own and change*.
Upstream is none of that: it is inherited, well-factored for its own purpose, and
the fork does not modify it. Its LOC is **carrying cost** (repo size + cognitive
noise), not **maintenance cost** (you never touch it; it's free at runtime). So
the only real question is **keep-or-prune**, and that hinges on one thing:

> **Are you still merging from `upstream/tinygrad`?**
> - **Yes (tracking upstream):** prune *nothing*. Every deletion is a permanent
>   merge conflict on an upstream-owned path, and `autogen/` regenerates anyway.
> - **No (hard fork → an AMD-decode product):** pruning is possible but it is
>   one-time cascading surgery (dynamic backend loading, generated bindings,
>   cross-referencing subsystems), not a casual sweep.

## Footprint (code LOC, excl `.venv`/`.git`/`__pycache__`)

| area | LOC | notes |
|---|---:|---|
| `tinygrad/` total | 211,351 | of which **`autogen/` = 118,078** (generated) |
| `tinygrad/` core (non-autogen) | ~93,273 | the framework — needed |
| `test/` | 77,940 | upstream tests (+ fork's `test/external`) |
| `extra/` | 63,417 | ~21k fork-added, **~42k upstream** |
| `examples/` | 12,030 | upstream examples |

**Fork sanity check (measured):** the fork's decode + flywheel code imports
**zero** non-AMD backends or bindings (`ops_nv`/`ops_cuda`/`ops_metal`/`nv_pma`/
`qcom`/`dsp`/`torch_backend` → none). From the fork's view the entire non-AMD
surface is unused — but it is used by *other upstream code* (examples, `viz`,
backends, tests), so it is internally consistent and pruning cascades.

## Prune tiers (if hard-forking) — value vs cascade/divergence risk

### Tier 1 — `autogen/` non-AMD ≈ **91,021 LOC** (the mass)
By backend: `nv_580` 26,001 + `nv_570` 24,866 + `mesa` 10,532 + `mlx5` 10,523 +
`nv` 4,928 + `cuda`/`metal`/`webgpu`/`opencl`/`kgsl`… AMD+host-relevant autogen is
only ~27,057. **But:** it is *generated* (tinygrad's autogen tooling rebuilds it
from vendor headers) and dynamically referenced by backends. Deleting it is
pointless (regen restores it) and breaks non-AMD `Device`s. **Recommendation: do
not touch.** The real lever here is not deletion but *not generating* the non-AMD
targets — an upstream-tooling change, out of scope.

### Tier 2 — non-AMD `extra/` device bindings/tools ≈ **~22k LOC**
`nv_pma/` 14,773 (NVIDIA CUPTI) · `gemm/` 5,873 (keep `cdna_asm_gemm` ref) ·
`torch_backend/` 2,199 · `qcom_gpu_driver/` 1,429 · `dsp/` 1,343 · 13 upstream
example/tool orphans ~2,392. **Each has ≥1 upstream importer** (e.g. `nv_pma`←
`tinygrad/viz/serve.py`; `models`←37, `datasets`←10, `hcq`←12 are load-bearing) —
so a delete cascades into other upstream files. Standalone, truly-zero-importer
dirs are small: `amdpci` 1,541, `usbgpu` 400, `torch_hook` 364, `testsig` 294,
`mlx_driver` 167, `mmapeak`/`optimization`/`perfetto`/`viz`/`webgpu`/`mesa`/
`hcqfuzz` (~1.4k combined). **Recommendation: prune only the standalone zero-
importer set (~4k) if hard-forking; leave the rest (cascade > benefit).**

### Tier 3 — non-AMD runtime backends ≈ **~3k LOC**
`ops_nv` 845, `ops_qcom` 412, `ops_dsp` 313, `ops_webgpu` 221, `ops_metal` 192,
`ops_cuda` 133, `ops_cl` 132, `ops_hip` 68… **Dynamically loaded by `Device`
name.** Pruning breaks generality, diverges from core, and is tiny. **Recommendation:
do not touch** (keep `ops_amd` + host `ops_python`/`ops_cpu`/`ops_disk`/`ops_null`).

### Tier 4 — non-AMD tests in `test/` (large)
Hardware/backend tests for non-AMD targets. Pruning loses upstream coverage that
guards the core the fork depends on. **Recommendation: do not touch.**

## Verdict

- **If you still sync upstream → prune nothing.** The carrying cost is real but
  it is not the maintenance burden the principles exist to fight, and any prune
  buys permanent merge friction. This is the line-count-is-not-the-metric rule
  applied at the fork level.
- **If this is a hard fork → the only defensible in-place prune is Tier 2's
  standalone zero-importer dirs (~4k LOC).** The big numbers (autogen 91k,
  bindings 22k) are generated/cascading/dynamic — the right tool is a
  **distribution boundary** (ship/package only the AMD subset) or *not generating*
  non-AMD autogen, **not** deletion surgery on the tracked tree.
- **The fork's own reduction work stays where it was:** the `extra/qk_*`/`llm_*`
  scripts (repo-audit + script-map) — that is the code you own and the principles
  apply to. Upstream is a strategic keep/package decision, not an audit target.

## Decision needed (from the maintainer)

1. Do you still merge from `upstream/tinygrad`? (yes → stop here; no → continue)
2. If hard-forking: in-place prune Tier-2 standalone (~4k, low risk), or set up a
   packaging/distribution boundary for a slim AMD-only artifact (recommended over
   tree surgery)?
