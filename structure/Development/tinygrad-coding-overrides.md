# tinygrad-arkey Coding Overrides

Project-specific layer on top of [coding-principles.md](coding-principles.md).
The principles file is the reusable, project-neutral scaffold; everything that is
specific to *this* fork (commit prefixes, portability rules, the AMD env
invariant, the anti-re-sprawl rule) lives here so the scaffold stays copyable.

## Temporary Authored LOC Budget

The authored-source cap is temporarily 35,000 lines while the 14B autoscan route is implemented and proven end to end. After that proof,
run another consolidation/pruning pass and restore the 30,000-line cap. This is implementation headroom, not permission to retain dead probes,
duplicate route authority, or superseded compatibility paths.

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

## Naming Reflects Actuals (so deliberate look-alikes don't read as dups)

When two things are **deliberately** separate but look near-identical, the fix is
naming, not merging. A dedup pass will keep re-flagging them as accidental
duplicates — and someone will eventually "consolidate" them and destroy the
distinction — unless the *name* states why they differ.

Rule: if a function/constant intentionally mirrors another but must stay separate,
its name (or an adjacent one-line note) must encode the reason.

- **Validation-probe re-derivations of shipped code** get a `probe_` prefix. A
  probe must not import the thing it validates (or it regresses silently with it),
  so it re-derives the kernel — that is a feature, not a copy. Example:
  `extra/qk/decode_physical_tile.py` `probe_p1_crosslane_score_kernel` mirrors the
  shipped `flash_kernels.flash_p1_crosslane_score_whole_cache_kernel`; the `probe_`
  prefix marks the deliberate independence. (Keep the *emitted* kernel name stable
  when renaming the Python function, so gate artifacts don't shift.)
- **Protocol/wire identifiers** duplicated at both ends of a client↔server boundary
  (e.g. the `gfx1100_744c` discovery-profile value set by `amd_repro` and validated
  by `amdev`) are named for their role via the *env var / key* they live under
  (`AM_REMOTE_DISCOVERY_PROFILE`), not consolidated into one constant — the two ends
  are independent by design, like an HTTP header name written on both sides.
- **Per-arch parallel implementations** of one interface (e.g. `isa/amd.py` vs
  `isa/x86.py` `alloc_vregs`) share a name on purpose — that IS the actual (same
  role, different arch); leave them.

If a reviewer or a dedup audit has to ask "is this a duplicate?", the name failed.

## One IR, One Engine (upstream smallness rule)

Upstream stays ~24k counted lines by architecture, not terseness: a single UOp
graph is the only representation from tensor graph down to register allocation,
and nearly every transformation — codegen passes, symbolic math, even autodiff
(`mixin/gradient.py`) — is a `PatternMatcher` table run by the one generic
`graph_rewrite` engine. Backends are string-emitting pattern tables
(`renderer/cstyle.py`); knobs are `ContextVar`s; binding boilerplate is
generated (`autogen/`), never handwritten.

Fork rules (the tinygrad-specific form of "Prefer data over code" and
"Simplify Representations Before Adding Mechanisms"):

- **No second representation.** Work that transforms kernels or graphs is
  written as a `PatternMatcher` table run through `graph_rewrite` — not a
  hand-rolled traversal, visitor class, or bespoke pass framework.
- **A new knob is a `ContextVar`**, not a parameter threaded through call
  signatures or an ad-hoc config object.
- **Prefer adding a rewrite rule** to adding a class, phase, or file
  (composes with the anti-re-sprawl rule above).
- **Watch the second-system threshold.** `extra/qk` is ~16k counted lines
  against a ~24k core. When qk machinery starts re-implementing something the
  engine already does (traversal, matching, scheduling, config), that is the
  "second hidden system" anti-pattern from coding-principles — collapse it
  into rules over the existing engine instead of growing it.

## AMD / Generation Invariants (irreducible — do not "simplify" away)

- **Env ordering is sacred:** `DEV`/`JIT`/`QK_PRIMITIVE_STORAGE` and the
  Q4K/Q6K primitive flags must be set **before** `from tinygrad import ...`.
  Shared generation code (`extra/llm/generate.py`) imports tinygrad lazily so the
  module can be imported without freezing the environment.
- **Subprocess isolation for generation is intentional:** the eval harness spawns
  a child per policy mode for clean per-run AMD/JIT device state + a JSON summary
  over stdout. Keep both entry points (in-process loop + isolated child).
- Do not run risky schedule search on Mac/TinyGPU/remote paths. Do not
  make `QK_GENERATED_POLICY` a global default.
