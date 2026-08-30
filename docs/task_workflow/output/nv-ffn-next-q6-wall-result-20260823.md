# NV FFN Q6 down four-warp wall result (2026-08-23)

## Findings

* **[MEASURED] WALL_PASS.** The candidate four-warp Q6 FFN-down admission
  measured `4.728057 ms/token`; reverse control midpoint measured
  `4.780930 ms/token`. Delta: **-52.873 us/token**.
* **[MEASURED]** Control A/C were `4.786741` and `4.775119 ms/token`.
  Candidate was between both controls, so the result is not a one-sided
  control drift.
* **[MEASURED]** All three arms produced the same token stream SHA:
  `f25083e5d0a754131283b40c03f52e688fee9f175bea7ae106805e7d628d7905`.
* **[MEASURED]** The bracket used fresh child processes, settled continuous
  windows, depth 512, 32 timed tokens, five samples per arm, and 18 Q6 FFN
  down blocks. GPU state was reset after the run.
* **[INFERRED]** This is a small but wall-relevant FFN improvement and clears
  the existing +50 us/token promotion bar by approximately 2.9 us/token.
* **[UNMEASURED]** This does not establish the full 163.5 us FFN ceiling or
  explain the remaining device gap. It also does not prove that the route is
  non-regressive across other depths or targets.

## Evidence

Raw arms and hashes are retained under:

`docs/task_workflow/evidence/nv-ffn-next-q6-wall-20260823/result/`

The measurement script was the existing research-only
`extra/llm_research/decode/q6k_ffn_down_four_warp_wall_bracket.py`; no
production file was changed.

## Next gate

Run the corresponding exact-live triage for gate/up and the Q4 down row. Do
not count this result toward a parity claim until a second fresh reverse
bracket confirms the same SHA-gated delta and the candidate is checked for
non-regression at the supported decode depths.
