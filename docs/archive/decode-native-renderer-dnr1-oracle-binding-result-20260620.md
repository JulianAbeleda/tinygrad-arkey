# Decode Native Renderer DNR-1 Oracle Binding Result - 2026-06-20

## Verdict

`PASS_DNR1_DECODE_Q8_ORACLE_BINDING_STRUCTURAL`

DNR-1 is complete as a structural gate. The q8 artifact oracle is now representable through the native AMD schedule
contract layer without changing defaults, lowering to ISA, launching a native candidate, or making a performance claim.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_native_renderer_dnr1_oracle_binding.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_native_renderer_dnr1_oracle_binding_result.json
```

## Bound Contract

| item | bound value |
|---|---|
| producer runtime | `q8_artifact_import_producer` |
| producer launch | `global=[1,1,1]`, `local=[1024,1,1]`, kernarg `32`, LDS `4096`, private `0` |
| gate/up runtime | `q8_artifact_import_gateup` |
| gate/up launch | `global=[12288,2,1]`, `local=[32,4,1]`, kernarg `40`, LDS `16`, private `0` |
| work decomposition | 128 threads per row; y selects gate/up; 16 Q4_K blocks; `sub=tid&7`; `kb=tid/8` |
| grouped oracle ISA | dot4 `16`, global load `11`, LDS/ds `7`, barrier `1`, shuffle `5`, store `1` |

## Gate Meaning

This does not mean decode has a native fast kernel. It means the oracle contract has been turned into an executable
structural object, so later native lowering work has a precise target and cannot silently drift from the measured q8
artifact.

## Next

DNR-2 is the first real implementation wall: native address/data-format lowering for block_q8_1 activation loads,
packed Q4_K weights, min/scale correction, y-role gate/up selection, reduction, and output store. DNR-2 must prove
numeric correctness before DNR-3 scheduler/resource timing work or BEAM/search starts.
