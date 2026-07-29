# `extra/` Organization Scope

Status: design and classification only. This document authorizes no path moves or renames.

## Goal

Organize the current `extra/` surface by purpose and lifecycle so a reader can tell whether a file is a production
authority, optional runtime support, debug tooling, research code, or an experiment. The existing names are historical
and must not be treated as ownership decisions.

The branch contract remains subtractive:

```text
exp    = production + debug + research/experimental
dev    = production + debug/qualification
master = production authorities + supported runtime surfaces
```

## Current top-level inventory

The current tree contains 198 tracked files:

| Current path | Count | Current role | Target organization | Branch default |
|---|---:|---|---|---|
| `extra/audit/**` | 8 | organization, lowering, fingerprint, compatibility, and boundary authorities | `extra/audit/**` | `master` |
| `extra/tools/**` | 2 | repository integrity and ISA-generation tools | `extra/tools/**` | `master` |
| `extra/llm/**` | 11 | mixed CLI, generation, evaluation, and benchmark tooling | `tinygrad/llm/{cli,generate,adapter}.py` plus `extra/llm/bench/**` | split: `master` CLI, `dev` benchmarks |
| `extra/llm_research/**` | 97 | mixed LLM runtime dependencies, codegen, scheduling, research, gates, and evidence | subdomains under `extra/llm_research/**` after runtime split | `exp`, with explicit migration exceptions |
| `extra/gpu_fault_analysis/**` | 2 | GPU fault and allocator reproducers | `extra/debug/gpu_fault_analysis/**` | `dev` |
| `extra/hardware/**` | 23 | PCI, PSP/GART, recovery, profiling, and trace tooling | `extra/debug/hardware/**` plus `extra/profiling/sqtt/**` | `dev`; SQTT boundary unresolved |
| `extra/remote/**` | 3 | remote power-cycle, repro, and transport qualification | `extra/debug/remote/**` | `dev` |
| `extra/nv_gpu_driver/**` | 11 | optional NV IOCTL backend and headers | `extra/drivers/nv/**` | unresolved: `master` only if supported |
| `extra/usbgpu/**` | 30 | TinyGPU protocol, installer, driver, qualification, and GPU lock | `extra/drivers/usbgpu/**` | `exp` unless formally supported |
| `extra/mesa/**` | 1 | optional Mesa setup/extraction helper | `extra/setup/mesa/**` | `master` while backend is supported |
| root `extra/setup_*.sh` | 8 | optional backend/toolchain setup scripts | `extra/setup/**` | `master` for supported backends; TinyGPU unresolved |
| root `extra/runtime_models.example.json` | 1 | CLI operating fixture | `extra/llm/runtime_models.example.json` | `master` with CLI |
| `extra/other/**` | 1 | temporary unclassified-surface policy | `extra/other/**` until classified | `dev`/`exp`; never inferred as production |

The count includes headers, fixtures, scripts, source files, protocol documents, and installer assets; it is not a count
of executable hot-path modules.

## Exhaustive file-level ledger

The organization map is not complete until every tracked path below `extra/` has one ledger row. The ledger must include
non-script assets because headers, fixtures, protocol documents, and installer sources can define runtime contracts even
when they are never imported by Python.

Each row must record:

- exact current path and file type;
- target path or target domain, without assuming that the current directory is authoritative;
- purpose in one sentence;
- branch owner: `master`, `dev`, `exp`, or `delete`;
- hot-path status: `default`, `conditional`, `operator-only`, `test/evidence-only`, or `none`;
- direct consumers in `tinygrad`, `extra`, `test`, `bench`, and docs;
- dependencies that must move with it;
- validation command and required hardware/toolchain;
- retention criterion, compact replacement, or deletion recovery commit;
- unresolved flag and the specific decision needed.

The initial ledger census must assert:

```sh
test "$(git ls-files 'extra/**' | wc -l | tr -d ' ')" = 198
```

No rename, directory move, or branch pruning is authorized until the ledger has 198 unique paths, its consumer/link
scan is clean, and every unresolved row names the human or follow-up task that will decide it.

## Target domains

### `extra/audit`

Canonical production evidence producers and repository audits. These are not runtime kernels, but their outputs and
tests are repository authorities. They remain on `master` and must not depend on research-only modules without an
explicit fixture boundary.

### `extra/tools`

Small repository-wide integrity and source-generation tools. They remain separate from audits because they operate on
the repository rather than asserting model or lowering behavior.

### `extra/llm`

The shipped LLM CLI belongs here. Benchmark, evaluation, scoring, and smoke-training scripts are qualification tools;
they should be grouped under `extra/llm/bench` and retained on `dev`, not mixed with the CLI entrypoint.

### `extra/llm_research`

This is the descriptive research surface formerly named `extra/qk`, but it is not a blanket production
classification. Its intended subdomains are:

```text
extra/llm_research/
  codegen/       kernel emitters, lowering probes, and compiler experiments
  scheduling/    lane maps, list/latency policies, and scheduling experiments
  hyperbeam/     BubbleBeam/HyperBeam candidate and route policy work
  futuresight/   Futuresight route prediction and candidate selection
  vocabulary/    kernel vocabulary, MMQ vocabulary, and route-shape metadata
  quantization/  Q4/Q6 packing, dequantization, and quantized route experiments
  harness/       benchmark, capture, timing, guarded execution, and worker harnesses
  evidence/      promotion gates, resource captures, manifests, and compact evidence producers
  vendor/        retained third-party research sources such as llama.cpp MMQ material
```

The root rename is complete; before any subdomain split, every `extra/llm_research` file must be assigned to one of these domains or marked unresolved. Files still
reached by `tinygrad/llm/route_ops.py` are migration debt: they must move into their `tinygrad/**` owner before the
research surface can be removed from `master`.

### `extra/debug`

Machine diagnostics, fault reproducers, PCI recovery, remote operations, and other tools that are useful for debugging
but are not part of normal model execution. This area belongs on `dev` and must use the GPU lock for hardware commands.

### `extra/profiling`

SQTT/rocprof trace decoding and example generation. The current `tinygrad/viz/profile.py -> extra/hardware/sqtt/roc.py
-> test.amd.disasm` dependency is an explicit boundary decision: either promote a minimal decoder into `tinygrad/viz`
or keep profiling strictly on `dev`.

### `extra/drivers`

Optional driver implementations and driver contracts. NV IOCTL and TinyGPU are separate products and must not share a
generic hardware bucket. TinyGPU remains `exp` until its supported-product status is decided.

### `extra/setup`

Installer and toolchain setup scripts. Setup scripts are not runtime code, but their supported backend contract must be
documented before they remain on `master`.

### `extra/other`

`extra/other` is an explicit quarantine for files that do not yet fit the named domains. It is not a fourth production
tier and must not become a dumping ground. Every file there requires a ledger row and a next disposition; new runtime
dependencies on it are forbidden.

## Organization rules

1. A directory name does not establish production ownership.
2. Production code must live under its `tinygrad/**` domain owner; `extra/llm_research` is not a permanent runtime API.
3. Research and qualification code may import production code, but production code must not import research helpers
   directly except through an explicitly tracked migration adapter.
4. Every retained `dev` or `exp` group needs a purpose, owner, and validation command.
5. Every removal needs a compact conclusion, consumer/link check, and recovery commit.
6. Whole-directory renames happen only after import, test, benchmark, artifact, and documentation references are
   inventoried.

## Completed root rename checkpoint

The opaque `extra/qk` root was renamed to `extra/llm_research` without changing its internal layout or runtime behavior.
Current imports, manifests, and operating commands use the descriptive namespace; historical recovery commands and frozen
evidence retain `extra/qk` where that name identifies the pre-rename source. The migration is propagated to `exp`, `dev`,
and `master`; subdomain moves remain gated by the ledger below.

## Required next classification pass

The next task is a file-level ledger for all 198 current `extra/**` files, starting with the 97 current `extra/llm_research`
files and the mixed files under `extra/llm`, `extra/hardware`, `extra/nv_gpu_driver`, and `extra/usbgpu`. The ledger
must record the target domain above, default-path reachability, branch owner, direct callers, tests, artifacts, and the
migration or retention criterion. No further subdomain move or branch pruning is authorized until that ledger is complete
and reviewed. The root rename itself was recorded in the rename checkpoint and does not move files between subdomains.
