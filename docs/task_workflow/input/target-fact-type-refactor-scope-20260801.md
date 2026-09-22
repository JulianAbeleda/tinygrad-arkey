# Target facts: make the declaration discipline mechanical instead of documentary

Date: 2026-08-01

Status: scoped, not implemented. Sequenced **after** the qualification-canonical-path pass
(`prefill-qualification-canonical-path-scope-20260801.md`, C1-C3). Written for review before code.

## 1. The problem: the rule is enforced by prose

The campaign's central rule is that a target fact must carry its authority and that absence must stay
distinguishable from a value. The repo states that rule well and follows it -- but it is enforced entirely by
comments and review, not by anything a machine checks.

The clearest evidence is `tinygrad/renderer/cuda.py:19-25`. Setting `wave_size = 32` takes **six lines of
comment** arguing that this particular 32 is a verified width and not the banned silent default, citing
`target-capability-policy-decoupling-scope-20260730.md` section 3.3 by section number. That comment is
load-bearing: delete it and the next reader sees a bare `32` indistinguishable from the defect. The same
justification is restated at `renderer/__init__.py:72-89`, `cstyle.py:496-497` and `cstyle.py:605-624` --
four independent prose copies of one invariant.

And it has already failed to replicate. `extra/llm_research/runtime_specs.py:33-45` declares:

```python
@dataclass(frozen=True)
class FullKernelCapability:
  backend: str = "AMD"
  arch: str = "gfx1100"
  wave_size: int = 32
```

`FullKernelCapability()` with no arguments yields a valid AMD row, so "forgot to specify" and "meant AMD" are
indistinguishable -- the identical defect class that `wave_size: int|None = None` exists to prevent one
directory over. The principle held where it was written down and did not survive the trip to the next module.
That is what a documentary invariant does.

## 2. The actual type-level defect: three absences, one spelling

`None` currently means three different things in `Renderer`, and the difference is carried only in prose:

| field | what `None` means | what the consumer must do |
| --- | --- | --- |
| `wave_size` (CPU) | **unreported** -- the width exists in hardware, this renderer does not verify it | must not guess; must not treat as 32 |
| `max_indirect_buffer_offset` (AMD, CPU) | **not applicable** -- no such constraint exists on this backend | nothing to check; absence is correct and final |
| `lds_read_before_next_write_ordered` | **undeclared** -- target has not asserted the guarantee | take the *safe* path (emit the barrier) |

The third has **inverted polarity** from the other two, which `renderer/__init__.py:96-101` calls out
explicitly: for `lds_bank_dwords` an unknown target forgoes an optimization, for MB2 an unknown target gains a
barrier. Same spelling, opposite consequence, distinguished only by whoever reads the comment.

So the refactor is not "wrap values in a box." It is: **give the three absences three names, and attach the
citation to the value rather than to the comment above it.**

## 3. Blast radius (surveyed 2026-08-01)

Smaller than it looks. Production readers of the declarative facts:

- `codegen/opt/postrange.py:286` -- `self.ren.shared_max`
- `codegen/opt/postrange.py:601-608` -- threads `lds_bank_dwords`, `lds_bank_cycle_lanes`,
  `lds_read_before_next_write_ordered` into `kernel_lds.build_precontract_lds_stage`
- `codegen/opt/kernel_lds.py:608-610, 613-640, 793` -- consumes those three
- `llm/device_facts.py:274` -- reads `shared_max` when present

And two facts with **no production reader at all**:

- `Renderer.wave_size` -- read only by `test/unit/test_target_capability_facts.py`. Every other `.wave_size`
  in the tree belongs to a different object (`geometry.wave_size`, `mapping.wave_size`, `grid.wave_size`).
- `Renderer.max_indirect_buffer_offset` -- set by `MetalRenderer`, read only by tests.

That is worth stating plainly: the two fields the discipline is most often justified on are, today, pure
declaration. It makes them a zero-risk pilot, and it is also a fair question whether they earn their place
(see §7).

Existing harness to reuse, not replace: `test/unit/test_target_capability_facts.py` already constructs real
renderers headlessly (`_amd(arch)`, `_metal()`, `_cpu()`) and pins unreported-vs-known-32 as its Fact 2. The
refactor extends that file; it does not start a new one.

## 4. Where the type lives: the fork question

`tinygrad/renderer/__init__.py` is a divergence surface -- `upstream` is `tinygrad/tinygrad`, and this file
already carries four local commits (TG2, TG7, PG1, MB2). Every field whose *type* changes there is merge pain
on every future rebase, forever.

Three options:

- **A. `Fact[T]` in `tinygrad/renderer/__init__.py`.** Widest reach, changes the declared type of core
  fields, maximal fork cost. Also the least justified by §3: the core fields have almost no readers.
- **B. `Fact[T]` in `extra/llm_research/` only.** Zero fork cost (`extra/` is even excluded from ruff lint per
  `pyproject.toml`), but cannot type a core `Renderer` field.
- **C. Hybrid (recommended).** Core fields keep their upstream-shaped `int|None` / `bool|None` declarations
  unchanged. A new adapter reads a real `Renderer` and *lifts* those declarations into typed, cited facts.
  The capability rows are then built from the adapter.

**C is the recommendation, and the pilot is the capability table, not the renderer.** The leverage is exactly
where the discipline already broke: the amended qualification scope requires every capability row to be
"declared from measured facts with citations", and today those citations live in a markdown table in a scope
doc. Moving them into the row is the whole point, it fixes the module that actually regressed, and it costs
nothing at the fork boundary.

## 5. Shape

```
extra/llm_research/target_facts.py

  Provenance  = Measured | Declared | Derived
  Absence     = Unreported | NotApplicable | Undeclared

  Fact[T]     -- value: T, provenance: Provenance, citation: str (required, non-empty)
  Absent      -- absence: Absence, citation: str (why it is absent, also required)

  TargetFact[T] = Fact[T] | Absent
```

Non-negotiables:

1. **A `Fact` cannot be constructed without a non-empty citation.** That is the entire mechanism; a default or
   an optional citation reproduces the defect at one remove.
2. **`Absent` also carries a citation.** "Apple does not publish threadgroup bank structure" is a fact about
   the world and is exactly as load-bearing as a number.
3. **Consuming a `TargetFact` requires handling the absence explicitly** -- no `.value or 32`, no
   `getattr(..., 32)`. The safe-path polarity (§2) is chosen at the *consumer*, per fact, and named there.
4. **`Fact` never enters a serialized payload.** See §6.

## 6. Hard constraint: canonical identity must not move

`_canonical_full_kernel_identity` hashes the normalized payload, and pinned hashes exist in tree
(`ANCHOR_SINGLE_BUFFER_CANDIDATE_HASH`, the promoted route table, every pinned candidate identity). The
capability is passed *alongside* the payload rather than serialized into it, so today it does not contribute
to the hash.

That must stay true. `Fact` lives at the row/capability level and never inside `normalized`. Verification is
cheap and must be in the first slice that touches `runtime_specs.py`: assert the anchor hash is unchanged.

## 7. Slices with verification

| slice | change | verification |
| --- | --- | --- |
| F0 | No code. Name the three absences in `test/unit/test_target_capability_facts.py` as explicit pins: CPU `wave_size` is *unreported*, AMD `max_indirect_buffer_offset` is *not applicable*, Metal `lds_read_before_next_write_ordered` is *undeclared and therefore barrier-emitting* | the three tests pass against today's tree unchanged; the taxonomy is forced to be real before any type exists |
| F1 | `extra/llm_research/target_facts.py`: `Fact`/`Absent`/`TargetFact` per §5 | unit tests: citation-less construction raises; each absence kind round-trips; `Fact` is frozen |
| F2 | `facts_for_renderer(renderer) -> TargetFacts` adapter lifting the declared `Renderer` attributes with citations | reuses `_amd()/_metal()/_cpu()` from the existing harness; AMD gfx1100 -> wave 32 Declared, gfx942 -> 64, CPU -> Unreported, Metal LDS banks -> Absent(NotApplicable) |
| F3 | `FullKernelCapability` rows constructed from `TargetFacts`; the all-AMD-defaults constructor is removed so a row cannot be built without stating its target | existing `test_runtime_specs.py` AMD identity pins unchanged; `ANCHOR_SINGLE_BUFFER_CANDIDATE_HASH` unchanged (§6); AMD admission outcomes identical before/after; NV e2e decode ratchet unchanged (~156 tok/s, digits `50994`) |
| F4 | *Deferred, not authorized by this scope.* Lifting `Fact` into `tinygrad/renderer/__init__.py` itself | would require the fork-cost decision in §4 to be made deliberately, with upstream contribution considered first |

F3 is the slice that pays for the whole scope. F1 and F2 are cheap and inert on their own -- if F3 does not
land, they are dead weight and should be reverted rather than left as an unused layer.

## 8. Ordering, and the one thing that could make this not worth doing

**Must follow C1-C3** of the qualification scope. F3 rewrites how capability rows are constructed; C1 rewrites
what is in them. Doing both at once means neither has a stable before/after to diff against.

**Honest risk:** these two scopes overlap with the centralization question. If the capability table later
dissolves into a *derived view* over `Renderer` + `tc` (the natural follow-on to C1, discussed but not yet
scoped), then `Renderer` becomes the single source and part of `Fact`'s value evaporates -- the citation would
just be "read from the renderer." The residue that survives either way is the **absence taxonomy** (§2),
because three different `None`s with opposite consequences is a defect no amount of centralization fixes.

So: if the derived-view decision is imminent, do **F0 only** and wait. F0 is useful under every outcome and
costs one test file. F1-F3 should not start until the table's future shape is settled.

## 9. Review questions

1. Is the three-way absence taxonomy complete, or is there a fourth case in tree (e.g. "declared but not yet
   measured on this hardware" -- the AMD lane gate in the qualification scope looks like one)?
2. Should `Provenance` distinguish `Measured` from `Declared` at all, given that BoltBeam owns measurement and
   promotion status is explicitly a separate axis (decoupling scope §3.3)? Or does provenance collapse to
   "citation string" and the tri-state is over-modelling?
3. F3 removes the all-defaults `FullKernelCapability()` constructor. How many call sites rely on it today, and
   is the churn worth it in the same pass that C1 is already rewriting those rows?
4. Does `Fact` belong in `extra/` at all, or -- since `extra/` is excluded from ruff lint -- does putting a
   type-discipline mechanism in an unlinted directory undercut the point?
