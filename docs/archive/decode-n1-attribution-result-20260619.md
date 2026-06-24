# Decode N1 native scheduler attribution result - 2026-06-19

Artifacts:

- `extra/qk_decode_n1_attribution.py`
- `bench/q8-ffn-amd-scheduler-project/n1_attribution.json`

## Verdict

**N1_COMPLETE_NO_N2_START**.

Do not begin a bounded native q8 scheduler/codegen N2 patch.

## Gate Result

| item | value |
|---|---:|
| full tinygrad ASM -> hipcc/LLD oracle gap | `73.109us` |
| N2 candidates `>=30us` | `0` |
| largest bounded attribution | `14.087us` |
| PMC profile runnable | yes |
| SQTT capture runnable | yes |
| SQTT decode usable | no |

SQTT captured non-empty RDNA3 HCQ instruction-trace blobs, but the local decoder fails on every trace with
`unknown cdna format word=0xf4080100`. So the final scheduler/resource bucket remains unattributed at timeline level.

## Attribution

| feature | attributed movement | decision |
|---|---:|---|
| dot4 instruction selection | `0.000us` | already matched; closed |
| global-load shape/coalescing | `14.087us` | below gate; no standalone N2 |
| waitcnt grouping | `0.837us` | closed |
| reduction topology | `13.305us` | below gate; no standalone N2 |
| `s_clause` / `s_delay_alu` scheduler markers | unknown | static diff only; needs hardware attribution |
| register/live-range/resource scheduler | unknown | project-level backend work only |
| local-y descriptor / launch contract | unknown/low EV | do not reopen for decode speed |

## Meaning

The q8 native gap is real, but it is not explained by a bounded primitive we can responsibly implement now. The observable
small features do not clear the `30us` bar, and the larger suspected movement is the general AMD scheduler/resource
model.

Next native work, if funded, is tooling first: make RDNA3 HCQ SQTT decode usable or build an equivalent PMU/timeline
attribution path. Otherwise the decode kernel/lifecycle work rests on the accepted default-off q8 artifact route.
