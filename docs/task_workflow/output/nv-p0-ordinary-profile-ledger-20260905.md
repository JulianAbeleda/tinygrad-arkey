# NV P0 ordinary route profile ledger (2026-09-05)

Fresh RTX 5090 runs used the normal `model(...)` entrypoint at `max_context=4608`, R9, three warmups, and `PROFILE=1` with `HCQ_GRAPH_PROFILE_JSON`.

Candidate environment: `DEV=NV QK_PRIMITIVE=1 NV_COMPILER_Q4_IMMA_PP512=1 NV_COMPILER_Q4_IMMA_K_PP512=1 NV_COMPILER_Q4_IMMA_QO_PP512=1 NV_COMPILER_Q4_IMMA_UNROLL=4 NV_COMPILER_Q6_IMMA_PP512=1 NV_COMPILER_Q6_IMMA_PP512_ROLES=ffn_down NV_LLAMA_FULL_PACKED_PP512=0`.

Control environment: `DEV=NV NV_COMPILER_Q4_IMMA_K_PP512=0 NV_LLAMA_FULL_PACKED_PP512=1`.

The candidate profile contained 78 graph submissions and 15,067 profile entries; the control contained 78 submissions and 20,943 entries. Candidate PROGRAM census included 108 Q8 compact records, 18 Q6 Stream-K mains, and 18 destination-major fixups. Control included 2,808 llama Q4 main and 2,808 llama fixup entries across the captured submissions. Unknown entries remain explicitly unclassified in the raw JSONL.

Wall results from the same runs were candidate median 66.64 ms (minimum 65.27 ms) and control median 39.45 ms (minimum 39.17 ms). The generated candidate is materially slower and remains opt-in. Profile raw files and exact stdout/stderr logs are retained under `/tmp/nv-p0/profile-{candidate,control}.{jsonl,log}`; the corresponding JSON summaries are `/tmp/nv-p0/profile-{candidate,control}.json`.

This ledger reports observed graph populations and elapsed wall; overlapping profile entry sums are not treated as wall time.
