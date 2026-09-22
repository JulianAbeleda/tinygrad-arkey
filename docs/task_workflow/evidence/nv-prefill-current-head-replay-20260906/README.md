# Current-HEAD historical pp512 replay — 2026-09-06

This checkpoint repairs and replays the retained 198-role generated composition
(gate/up, K, Q/O, and Q4-V) at current HEAD. It explicitly sets
`NV_COMPILER_Q6_IMMA_PP512=0`; it is a historical-composition check and does
not represent the current promoted Q6 FFN-down route.

The first run found that `/tmp/q4v-asset` is ephemeral. It was rebuilt with
`nv_compiler_q4v_asset_build.py`. A second stopped run found that the old
harness accepted an unset Q6 lease even though unset now enables promoted Q6
down. The repaired harness requires an explicit zero for this legacy arm. A
third stopped run identified that the composed Q/O wrapper selected the
four-buffer fused-residual PROGRAM while supplying its three-buffer projection
ABI. The wrapper now uses the plain Q/O PROGRAM and records that exact context
identity; residual addition remains in the model graph.

`historical-r9.json` passes token 198, finite output, exact replay, 198 main and
producer calls, and 198 canonical weight bases. Samples are 71.770503,
71.202654, 71.021383, 68.260605, 68.253061, 68.100774, 67.168619, 67.151808,
and 67.145605 ms. The strong within-run descent means this is functionality
and raw timing evidence only, not a stable matched performance qualification.
