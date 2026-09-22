# HCQ runtime overlay probe

Date: 2026-08-29. Temporary source overlay only; the shared source tree was not modified.

## Overlay

- Source: `/home/ubuntu/tinygrad-arkey`
- Temporary `PYTHONPATH`: `/tmp/tinygrad-hcq-overlay-20260829-If9viW`
- Only replaced file: `tinygrad/runtime/graph/hcq.py`
- Replacement: `git show 131b22a8b:tinygrad/runtime/graph/hcq.py`
- Hashes: `overlay-hashes.txt`

## Run

Exact command is in `command.txt`. It used `PROFILE=0`, Q4+Q4V candidate mode,
3 warmups, 9 measured rounds, deep replay, the current absolute cut policy, and
`flock -w 600 /tmp/gpu-bench.lock`.

## Result

- Process return code: `0`
- Harness status: `PASS`
- Token: `198`
- Deep-20 replay: exact (`all_cycles_exact=true`)
- Candidate census: 198 weight args; gate/up 72, K 36, QO 72, V 18; 198 Q8 producers; 0 copies; 0 fp16 overlays
- Wall samples (ms): 71.977467, 72.270115, 72.133649, 72.164417, 72.152444, 72.131885, 72.166020, 72.157655, 72.135123
- Median: **72.152444 ms**

Authority median was `72.121122 ms`; overlay delta is `+0.031322 ms` (+0.0435%),
within measurement noise and slower. Verdict: **STOP**. No performance attribution
or promotion is supported. Raw stdout, stderr, JSON, and logits are retained here.
