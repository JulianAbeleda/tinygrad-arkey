# Decode large/small paths scope - 2026-06-19

Purpose: split the remaining decode work into the two paths that now match the evidence.

No model defaults are changed by this scope. The execution artifact is:

- `extra/qk_decode_path_split.py`
- `bench/qk-decode-path-split/result.json`
- `bench/qk-decode-path-split/large_artifact_inventory.json`
- `bench/qk-decode-path-split/small_q8_hardening.json`
- `bench/qk-decode-path-split/summary.md`

## Current Decode Authority

The decode mechanism is no longer ambiguous:

- tinygrad standalone GEMV is strong: about `76%` HBM peak;
- tinygrad in-model weight-GEMV falls to about `44%`;
- llama standalone is lower at about `57%`, but in-model holds about `54%`;
- B1 launch/env knob tuning failed;
- B2 runtime/cache identity is closed: direct role calls and in-model decode use the same program/launch identities.

So the large gap is not a hidden runtime wiring bug and not another standalone GEMV kernel. It is the in-model MMVQ
contract: activation-format reuse, low-VGPR/high-grid launch behavior, and scheduler/renderer behavior that keeps
memory-level parallelism high inside the full model.

## Path L - Large Decode Path

Goal: recover the parity-scale gap.

Target model:

- measured path: `44% -> 54%` HBM efficiency over the weight-GEMV bucket;
- expected decode movement: about `1.187x`;
- theoretical upper bound if tinygrad's standalone `76%` transferred perfectly: about `1.557x`;
- correctness target: byte-identical unless the path intentionally switches to lossy q8.

### L1 - Mature Artifact Inventory

Question: is there a Tensile-like decode MMVQ code-object family that tinygrad can import through HCQ?

Gate:

- find standalone Q4_K/Q6_K x q8_1 gfx1100 code objects;
- recover symbol, kernarg, launch contract;
- launch without in-process HIP runtime.

Executed result: `docs/decode-mmvq-artifact-import-discovery-result-20260619.md`.

Verdict: no ready HCQ artifact family found. llama.cpp has the mature source family and build objects, but not a
standalone Tensile-style code-object set. This closes TPE-style extraction as a bounded path.

### L2 - Source Import / Renderer-Scheduler Project

Question: should tinygrad learn or import the llama-style MMVQ contract from source?

This is the remaining large path:

- source-contract import from llama.cpp's HIP/CUDA MMVQ source and build objects; or
- native tinygrad AMD renderer/scheduler work to preserve the same low-VGPR/high-grid contract.

This is project-level work, not a small primitive edit. It crosses compilation, launch-contract extraction, scheduling,
and maintenance boundaries.

## Path S - Small Decode Path

Goal: keep the q8 FFN route as a clean research lever.

Measured authority:

- W==D decode movement: `1.051x` to `1.063x` across ctx `128/512/1024/4096`;
- dNLL: `+0.002887` over `160` tokens;
- isolated q8 lifecycle: `115.24us`;
- route is default-off and external-artifact policy-bound.

Executed result: `docs/decode-q8-research-route-hardening-result-20260619.md`.

Verdict: this path is done enough for research. It is not the parity path because it only covers dense FFN gate/up and
the reuse count is capped. It remains useful as an oracle and an optional research flag.

## Decision Table

| path | status | payoff | cost | next |
|---|---|---:|---|---|
| Large MMVQ contract preservation | live, project-level | `~1.187x` measured target; `~1.557x` theoretical | renderer/scheduler or source-import project | fund only as a project |
| Mature MMVQ artifact import | closed as bounded | same if artifact existed | no ready artifact found | reopen only with a real code-object family |
| Small q8 artifact route | pass research | `~1.05-1.06x` decode | lossy, external HSACO, default-off | keep as research flag |

## What Completion Looks Like

For decode, there are no more bounded map-first diagnostics left under the current target.

Completion now means one of:

1. accept the small q8 route as the research result and stop;
2. fund the large MMVQ contract project;
3. change the target regime, for example long-context KV, serving, CUDA, or alternative quantization.
