# Q6_K compiler model-lifecycle evidence

Authoritative interpretation: structural/correctness PASS, combined wall
FAIL. The 87 ms preliminary files are retained as rejected evidence showing
the rank-contaminated control and must not be used for a performance claim.

Final authority:

- `repro-k-authority-fixed-r9.json`: historical gate/up+K graph reproduced;
- `final-control-r9.json`: matched 70 ms-class control;
- `final-candidate-a-r9.json` and `final-candidate-b-r9.json`: corrected A/B/A
  candidate confirmations;
- `final-compare.json`: correctness pass plus explicit performance failure;
- `role-v-r9.json`, `role-down-r9.json`: population attribution;
- `current-tree-structural.json`: post-review fresh-process combined census and
  A/B/A replay requalification. It embeds SHA-256, byte count, and nanosecond
  mtime for every Q6 compiler/model source used by the run; all embedded hashes
  were rechecked against the workspace after artifact creation;
- `q6-regression-r9.json` and `q6-regression-artifacts/`: real V/down oracle,
  sentinel, readonly, SASS, and resource regression after the fallback fix.

Pinned runtime policy: `HCQ_NUM_COMPUTE=2`,
`HCQ_NV_READY_PLACEMENT=1`, and empty
`HCQ_NV_MULTI_QUEUE_PROGRAMS`, `HCQ_NV_MULTI_QUEUE_INDICES`, and
`HCQ_NV_MULTI_QUEUE_CUT_POLICY`.
