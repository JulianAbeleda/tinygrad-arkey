# Decode Oracle Gate/Up Extraction Result - 2026-06-20

Verdict: `PASS_DECODE_ORACLE_GATEUP_HSACO_METADATA_DISASM_EXTRACTED`

The first executable decode oracle step is complete. The exact `q8_mmvq_gateup` HIP/LLD oracle artifact is reproducibly rebuilt from the existing source, written to the ignored bench artifact directory, and checked against the existing manifest/loader/oracle contract.

## Artifact Identity

| field | value |
| --- | --- |
| artifact | `bench/qk-decode-primitive-transfer/oracle/q8_mmvq_gateup.hsaco` |
| symbol | `q8_mmvq_gateup` / `q8_mmvq_gateup.kd` |
| sha256 | `9d00b0723a6aa92d54f18e152678352d6b19d04ace9cbf605637c6abcf0287a5` |
| bytes | `4720` |
| source sha256 | `08255027d138def2e346e4503428cd1c461b90fee4c74522a541a4fc566aa83b` |

All identity fields match `bench/q8-ffn-amd-scheduler-project/artifact_build_manifest.json`.

## Resource Envelope

| field | value |
| --- | ---: |
| kernarg bytes | 40 |
| LDS/group bytes | 16 |
| private bytes | 0 |
| VGPR count | 26 |
| VGPR spills | 0 |
| SGPR count | 18 |
| SGPR spills | 0 |
| max workgroup size | 128 |
| wavefront size | 32 |

This closes the first DNR-3C9 missing-info item for the gate/up oracle resource envelope at the artifact level.

## ISA Contract

The disassembly gate matches the existing oracle instruction contract:

| group | count |
| --- | ---: |
| `dot4` | 16 |
| `fma` | 5 |
| `convert` | 6 |
| `VALU` | 120 |
| `SALU` | 197 |
| `DS` | 7 |
| `barrier` | 1 |
| `global_load` | 11 |
| `global_store` | 1 |
| `shuffle` | 5 |
| `branch` | 5 |
| `waitcnt` | 20 |

The parser was aligned with the existing local classifier convention: `fma` family includes `v_fma*`, `v_mad*`, and `mad_mix`.

## What This Changes

Decode now has a concrete oracle artifact to map against. The next path is no longer "try another native q8 schedule"; it is OES-4 semantic ISA mapping:

1. Stage-label `q8_mmvq_gateup.disasm.txt` into q8 semantic blocks.
2. Compare those blocks to native/C7C PC and resource ledgers.
3. Name the first missing native mechanism, or prove the gap is not in the kernel body.
4. Only then build a native schedule or search objective.

Probe: `extra/qk_decode_oracle_gateup_extract.py`

