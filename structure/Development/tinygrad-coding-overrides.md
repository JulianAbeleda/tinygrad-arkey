# tinygrad-arkey Coding Overrides

Project-specific layer on top of [coding-principles.md](coding-principles.md).
The principles file is the reusable, project-neutral scaffold; everything that is
specific to *this* fork (commit prefixes, portability rules, the AMD env
invariant, the anti-re-sprawl rule) lives here so the scaffold stays copyable.

## Commit Prefixes

Every commit carries exactly one owning-subsystem prefix. Allowed prefixes in
this repo (from history):

| prefix | owns |
|---|---|
| `[codegen]` | `tinygrad/` lowering, renderers, UOp/codegen, the linearizer |
| `[runtime]` | `tinygrad/` device/runtime/JIT/storage, ops execution |
| `[nn]` | model/`tinygrad/llm` changes (e.g. decode path in `model.py`) |
| `[test]` | `extra/` tooling, `test/`, and the `bench/` artifacts they produce |
| `[docs]` | docs and `structure/` content (unless it is a subsystem contract) |
| `[repo]` | repo plumbing (gitignore, CI, tooling config) |

**Hard rule (do not repeat the known gap):** `tinygrad/` core — renderer,
codegen, uop, runtime — is **`[codegen]`/`[runtime]` (or `[nn]` for the llm
model), never `[test]`**. `[test]` is for `extra/` + `test/` + their `bench/`
artifacts only.

One owning prefix per commit. Never bundle generated `bench/` artifacts with a
`tinygrad/` source change.

## NFC Discipline

Mark behaviour-preserving changes `NFC` (e.g. `[test] NFC - extract helper`).
Never tag a commit NFC if it changes any output byte, and never mix an NFC
refactor with a functional change — split them. An NFC claim on the flywheel
tooling must be **byte-proven** (golden/regeneration hash or fixed-seed token
parity), not asserted.

## Keep Artifacts And Fallbacks Portable

Committed `bench/` artifacts must regenerate **byte-identically on any checkout**.

- **No absolute checkout paths** in artifact contents — not in `source_files`,
  not in row `id`s. Derive ids/paths from repo-relative keys. (The
  `accepted_runtime` builder once slugged `str(path.parent)` — absolute — into
  row ids, baking `home-ubuntu-tinygrad-arkey-...` in; fixed to repo-relative.)
- **No machine-dependent float serialization** in golden-locked artifacts. Round
  transcendental/`libm` floats before serializing (e.g. cost-model
  `features.jsonl` rounds to 6 dp) so the hash is identical across macOS/Linux.
- The golden + reproduce-from-artifact tests are the machine-enforced version of
  this rule. A new locked artifact gets a portability assertion.

## Anti-Re-Sprawl Rule

The judging flywheel grew to ~7k LOC because, under speed, **every new experiment
got a new clone script or `build_*` function.** To prevent recurrence:

> A new experiment adds a **row to a source/spec table**, not a new file or a new
> `build_*` function. New scoring axes extend the scorer; new prediction backends
> extend the cost model; new staged batches are a row in the batch table. If you
> are copy-pasting a `main()` or a row-builder, stop — that is the re-sprawl this
> structure exists to kill.

One-off probes that have reached a verdict are deleted once their conclusion is
recorded in the session handoff (they stay in git history, reproducible from the
core). Do not leave dead probes wired into the CLI.

## AMD / Generation Invariants (irreducible — do not "simplify" away)

- **Env ordering is sacred:** `DEV`/`JIT`/`QK_PRIMITIVE_STORAGE` and the
  Q4K/Q6K primitive flags must be set **before** `from tinygrad import ...`.
  Shared generation code (`extra/llm/generate.py`) imports tinygrad lazily so the
  module can be imported without freezing the environment.
- **Subprocess isolation for generation is intentional:** the eval harness spawns
  a child per policy mode for clean per-run AMD/JIT device state + a JSON summary
  over stdout. Keep both entry points (in-process loop + isolated child).
- Do not run BEAM / risky schedule search on Mac/TinyGPU/remote paths. Do not
  make `QK_GENERATED_POLICY` a global default.
