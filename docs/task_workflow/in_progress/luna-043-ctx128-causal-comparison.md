# LUNA-043: context-128 causal comparison

Verdict: `TOOL_FAILURE`; no causal owner can honestly be named.

## Established facts

| Case | Fact | Evidence class |
|---|---|---|
| llama 14B ctx128 | The intended reference lifecycle is source-mapped, but no ordinary smoke or dispatch trace ran. | source-proven + LUNA-021 tool-failure manifest |
| tinygrad 8B ctx128 | Historical scope says it passes at the cited commits. No LUNA-030 retained capture exists here. | retained historical assertion, not this task's runtime evidence |
| tinygrad 14B ctx128 | Historical scope records deterministic pre-timed-decode COMGR rejection of an invalid constructed-vector STORE. | retained historical assertion |
| tinygrad route boundary | Packed candidate decline may fall through to direct-packed; the env guard alone does not prove per-shape safety. | source-proven LUNA-014 |

## Causal decision

The earliest *known* divergence is not established. The failing semantic STORE ancestry, ctx128 route decision log, ctx512 selected-packed positive control, allowed direct-packed control, and 8B captured program are all missing. Therefore neither ``missing route coverage`` nor ``generic lowering failure`` is selected.

Required evidence: one LUNA-031 capture containing the semantic UOp STORE ancestry and HIP excerpt; LUNA-034 route decisions with both positive controls; LUNA-030 8B capture; and a llama ctx128 ordinary completion/trace once a collector exists. A constructed vector must not be named/materialized as a repair merely to make HIP compile.
