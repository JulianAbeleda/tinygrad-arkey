# NV P0 ordinary route profile ledger (2026-09-05)

Fresh RTX 5090 runs used the normal `model(...)` entrypoint at `max_context=4608`, R9, three warmups, and `PROFILE=1` with `HCQ_GRAPH_PROFILE_JSON`.

Candidate environment: `DEV=NV QK_PRIMITIVE=1 NV_COMPILER_Q4_IMMA_PP512=1 NV_COMPILER_Q4_IMMA_K_PP512=1 NV_COMPILER_Q4_IMMA_QO_PP512=1 NV_COMPILER_Q4_IMMA_UNROLL=4 NV_COMPILER_Q6_IMMA_PP512=1 NV_COMPILER_Q6_IMMA_PP512_ROLES=ffn_down NV_LLAMA_FULL_PACKED_PP512=0`.

Control environment: `DEV=NV NV_COMPILER_Q4_IMMA_K_PP512=0 NV_LLAMA_FULL_PACKED_PP512=1`.

The candidate profile contained 78 graph submissions and 15,067 profile entries; the control contained 78 submissions and 20,943 entries. Candidate PROGRAM census included 108 Q8 compact records, 18 Q6 Stream-K mains, and 18 destination-major fixups. Control included 2,808 llama Q4 main and 2,808 llama fixup entries across the captured submissions. Unknown entries remain explicitly unclassified in the raw JSONL.

Wall results from the profiled runs (read from their JSON summaries, distinct from the earlier unprofiled runs) were candidate median 69.840668 ms (minimum 67.727103 ms) and control median 39.45399 ms (minimum 39.167262 ms). The generated candidate is materially slower and remains opt-in. Profile raw files and exact stdout/stderr logs are retained under `/tmp/nv-p0/profile-{candidate,control}.{jsonl,log}`; the corresponding JSON summaries are `/tmp/nv-p0/profile-{candidate,control}.json`.

For the final six profiled submissions, active entry sums were candidate gate/up 21.004 ms, Q6-down main 3.878 ms, Flash 3.313 ms, and vocab 2.916 ms; control Q4 main 22.707 ms, Q6 main 4.702 ms, Flash 1.901 ms, and vocab 0.308 ms. These are overlapping active sums, not elapsed wall. PROGRAM families remain shape-identified in raw metadata; the largest unclassified candidate families were 72-call and 18-call E families and require exact metadata decoding before assigning Q/O versus down roles.
