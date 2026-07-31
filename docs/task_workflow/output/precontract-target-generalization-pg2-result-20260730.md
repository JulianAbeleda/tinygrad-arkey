# PG2 result — M1 feasibility re-probe after generalization

Date: 2026-07-30

Status: **PG2 complete. Verdict GREEN, with one qualification stated in section 3.** Compile-only; no GPU workload.
Closes `docs/task_workflow/input/precontract-target-generalization-scope-20260730.md`. Does not authorize
promotion to `dev`/`master`.

## 1. Verdict

The packed-weight WMMA precontract path -- the only route to Metal's matrix units that fits the 12.7 GB budget --
**lowers on Metal**.

| question (from the original M1 probe) | before PG0 | after PG0/PG1a/PG1 |
| --- | --- | --- |
| does it lower without error? | `KernelOptError: RDNA3 WMMA descriptor dims drifted` | **yes** |
| does it emit tensor-core ops? | 0 | **129 `__WMMA`, 1 `simdgroup_multiply_accumulate`** |
| does `lds_bytes` fit 32768? | never reached | **25600** |

`simdgroup_matrix` occurrences: 0 -- expected, and not evidence of absence. `MetalRenderer` never emits that
string; it renders a `__WMMA_*` helper wrapping `simdgroup_multiply_accumulate`
(`tinygrad/renderer/cstyle.py:534-546`). Searching for it produced one wrong conclusion earlier in this campaign.

## 2. What changed, and what did not

Three commits, all pure generalization. **No Metal-specific code was added at any point.**

| commit | layer | nature |
| --- | --- | --- |
| `49ca7fb23` | `validate_rdna3_wmma_descriptor` | frozen `_RDNA3_DIMS/_ELEMENTS/_OPTS/_SWIZZLE/_REMAPS` -> derived from `tc` |
| `9be5a9c12` | `assemble` binary-axis fold | literal `4` -> `log2(tc.elements_per_thread[i])`; unrolled Horner -> reduction |
| `f446c4cb4` | cooperative-store rotation | literals `32`/`8` -> declared `lds_bank_dwords`/`lds_bank_cycle_lanes` |

The first two were **derivable snapshots**: values the general machinery already computed, frozen as constants.
The third was **genuine hardware structure**, so it was declared as a target fact rather than derived.

AMD is byte-identical throughout: rendered source sha256 `ce03d94bb58a706fc567d30e385beebb4724a8ac9af32f05600e51fd13599251`,
17 `__WMMA`, verified at each step by re-rendering the stashed tree rather than by assertion. **Render-only** --
this machine has no ROCm compiler, so no step of this is an AMD execution result.

Failing-test-id sets are unchanged at every commit (114 failing, identical IDs); passing rose 1793 -> 1804 from
added coverage.

## 3. The qualification on GREEN -- read before starting M1

**The probe used AMD's tuned geometry tuple**, `(256, 64, 32, 8, 1, 1)`, reused verbatim from
`PACKED_WMMA_ROUTES`'s only `ffn_gate_up` row. So GREEN proves the *mechanism* lowers on Metal. It does **not**
prove that geometry is right for Metal, and it is not a performance result -- nothing was executed.

Two consequences for M1:

1. **Metal will qualify without the cooperative-store rotation.** `MetalRenderer` leaves `lds_bank_dwords` and
   `lds_bank_cycle_lanes` unreported, because Apple does not publish threadgroup memory's bank count or
   per-cycle lane width the way AMD's ISA manuals do (searched; no authoritative source found). The rotation is
   always *correct* to skip -- it is an exact one-writer cover of the tile regardless of banking -- so an unknown
   target forgoes the optimization rather than inheriting AMD's bank model. Any geometry M1 promotes is therefore
   measured on an **unrotated store**, and if Apple's banking is ever established there may be further headroom
   on top of whatever M1 measures.
2. **Geometry is M1's to search, not to inherit.** The AMD tuple is tuned to gfx1100's LDS budget and register
   file. Metal's constraints are 32768 B threadgroup memory and 1024 threads, both already declared and already
   enforced by `propose_legal_dimensions`.

## 4. Known limitations

- No AMD hardware. All AMD non-regression is rendered-source equality, never execution.
- Compile-only. No throughput number is claimed or implied by this document.
- Two further instances of the same frozen-constant pattern remain outside the exercised path, recorded by PG1a:
  `validate_precontract_wmma_abi`'s `(A,4),(B,4),(C,3)` loop (reached only via `WMMAConsumerAdapter`, which
  nothing but its own test calls) and `postrange.py:497-499`'s C-axis fold (reached only when
  `buffer_count > 1`). Whoever qualifies such a route will hit them.
- `cooperative_store_octet_rows` is dead code -- nothing calls it repo-wide. It was parameterized anyway because
  its docstring named RDNA3, and its dead status is recorded rather than disguised.
