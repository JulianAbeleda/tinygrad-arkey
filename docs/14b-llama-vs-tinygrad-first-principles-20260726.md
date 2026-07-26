# 14B llama.cpp versus tinygrad: first-principles synthesis

Status: `INCONCLUSIVE / TOOL_FAILURE`. This LUNA-044 synthesis deliberately does not infer llama dispatch, resource use, or causal superiority. LUNA-001 found no installed ROCm trace/counter executable; LUNA-021 consequently retained only device/lock controls, and dependent llama/tinygrad capture cards did not run.

## Normalized comparison dimensions

| Dimension | Source-proven | Measured / retained | Unresolved evidence |
|---|---|---|---|
| Algorithm | Both map Qwen 14B GQA as Hq=40/Hkv=8 (G=5); both have prompt and one-token decode lifecycles. | Historical full-model tok/s rows and focused G4/G5 timing exist. | Selected algorithms, fusion boundaries, output parity. |
| Shape policy | llama batch/ubatch and flash are admission-dependent; tinygrad prefill and flash routes are shape/policy dependent. | None for this task. | Actual ctx128/512/4096 selections. |
| Dispatch | llama join keys are ggml op/template/HIP launch; tinygrad has named route seams. | No dispatches. | Counts, order, grids, synchronization. |
| Traffic | F16 logical KV write is 192 KiB/token; payload is 24/96/768 MiB at ctx128/512/4096. | Geometry formula only. | Weight, KV-read, transient, duplicate and measured bytes. |
| Compute/resources/locality | Candidate source families and G5 coupling are mapped. | Focused deep G5 normalized cost is worse than G4. | Instructions, VGPR/SGPR/LDS/scratch, occupancy, cache/coalescing. |
| Correctness | tinygrad route semantics and cache-store locations are source-mapped. | Historical compile failure is retained. | Failing UOp ancestry, route attribution, llama output and token parity. |

## Hypothesis table

| Question / mechanism | Verdict | Basis | Exact next evidence |
|---|---|---|---|
| llama completes ctx128 while tinygrad emits illegal STORE because of a different selected runtime path | `INCONCLUSIVE` | llama completion was not run in LUNA-021; tinygrad semantic capture is absent. | LUNA-021 smoke/trace plus LUNA-030/031 and route logs. |
| tinygrad ctx128 implicitly reaches unsafe direct-packed fallback | `INCONCLUSIVE` | Source permits it after a packed candidate declines, but no decision log exists. | LUNA-034 with selected-packed ctx512 and allowed direct-packed controls. |
| ctx128 is a generic lowering defect | `INCONCLUSIVE` | Historical HIP symptom is insufficient to identify semantic owner. | Full STORE ancestry across 14B-128, 14B-512, 8B-128. |
| G5 deep flash path contributes to depth decay | `SUPPORTED, bounded` | Retained focused timing: G5/G4 rises from 1.0217x to 1.3223x. | Per-family full-model wall share to quantify Amdahl contribution. |
| register spilling, occupancy loss, or a long-loop lifetime interaction causes G5 cliff | `INCONCLUSIVE` | Scope identifies this as remaining mechanism class, not proof. | Matching code-object resource/disassembly rows, then discriminating counters. |
| KV reread per query group, partial fifth wave, split=48, dispatch growth, VRAM pressure | `REFUTED for current route` | Retained scope exclusions. | New contradictory route/resource evidence only. |

## Consequences

Track A has no approved repair candidate. Track B may retain only the G5 lifetime/allocation/resource candidate family, gated on resource evidence and full-model Amdahl contribution. No static source map supports copying llama HIP, changing G5 ownership, padding away ctx128, or promoting profiler time.
