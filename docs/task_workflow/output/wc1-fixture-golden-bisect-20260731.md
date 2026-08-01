# WC1 HIGH-2 — fixture golden movement: bisect, review, regenerate

Answers the triage item "fixture SHA mismatch" (`wc1-baseline-triage-solution-scope-20260731.md`
§4): which commits moved the emitted-code fixtures, whether the movement was intentional, and
the reviewed regeneration that closes the silent-movement hole.

## What the fixture locks

`test/unit/test_amd_isa_extraction_fixtures.py` emits three plain WMMA matmuls on
`AMD:ISA:gfx1100` (`tc_16x16x16_unrolled`, `tc_16x16x64_unrolled`, `tc_16x16x64_rolled`) and
compares binary SHA, mnemonic SHA, byte count, and instruction count against goldens. The
goldens were introduced in `15c31063d` (2026-07-13) and last matched the emitter there.

## Movement attribution (measured by emitting at each commit)

The goldens match the emitter **only** at `15c31063d`. Every commit from `81dd858f5` onward
emits different bytes; the movement is confined to 2026-07-16..17 and is stable from
`470a4e3e` through HEAD (including the migration `0e41c260d` and WC2 `3ec8557f1`, which do
**not** change these fixtures).

| commit | date | subject | rolled fixture delta |
| --- | --- | --- | ---: |
| `81dd858f5` | Jul 16 07:50 | generalize memory-adaptive execution and decode authority | 121 -> 172 inst |
| `e6dba6f4` | Jul 16 13:21 | localize private address recipes | 172 -> 172 (mnemonic SHA changes) |
| `5b68cb8a` | Jul 16 19:07 | localize complete memory address roots | 172 -> 214 inst |
| `d395d706` | Jul 17 16:50 | preserve scarce register candidates | mnemonic SHA -> final `834da5...` |
| `470a4e3e` | Jul 17 17:14 | drain vector stores through VSCNT | binary SHA -> final `2edf8aae...` |

All five are reviewed `[amd]`/`[codegen]` commits by the repo owner, each with unit tests in
the same commit. None are accidental. Between `5b68cb8a` and `d395d706^` a regalloc cluster
(`0d91841d3` prioritize pressure release, `a293391b0` localize constrained leases,
`c05188028` rematerialize renderer-safe fills, `272d73af8` reduce wide pipeline register
pressure, et al.) flips intermediate hashes (`8afa1695df33`, `238dff44cae7`) before settling
at `4393b863`; the cluster is same-series intentional codegen and the endpoints are unique.

## Corrections to prior records

- The triage's "goldens last written at `3e152b218`" is off by the manifest-move commit: that
  commit only moved proof-manifest imports, and the fixture still fails there. The goldens'
  last matching commit is `15c31063d`.
- The triage's "the emitter moved after `3e152b218`" is off by one window: the movement starts
  at `81dd858f5` (Jul 16 07:50), which is an ancestor of `3e152b218` (Jul 16 08:40). The
  unreviewed-delta conclusion stands; the window does not.

## Regeneration

Golden values replaced with HEAD-emitted values, measured twice for determinism (identical
runs). Full fixture file passes: `6 passed, 3 subtests passed`.

| fixture | binary_sha256 | mnemonic_sha256 | bytes | inst |
| --- | --- | --- | ---: | ---: |
| `tc_16x16x16_unrolled` | `4a558d21...` | `f415079c...` | 972 | 149 |
| `tc_16x16x64_unrolled` | `65215110...` | `4d3e8fec...` | 2952 | 452 |
| `tc_16x16x64_rolled` | `2edf8aae...` | `834da500...` | 1380 | 214 |

The fixture now detects *future* emitter movement again; the Jul 16-17 delta is on record
instead of silently blessed.
