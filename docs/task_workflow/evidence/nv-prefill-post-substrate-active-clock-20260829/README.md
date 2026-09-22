# Active-clock diagnostic

Date: 2026-08-29. One GPU session, serialized by `flock -w 600 /tmp/gpu-bench.lock`.

Command:

```sh
NV_COMPILER_Q4_IMMA_PP512=1 NV_COMPILER_Q4_IMMA_K_PP512=1 NV_COMPILER_Q4_IMMA_QO_PP512=1 HCQ_NUM_COMPUTE=2 HCQ_NV_MULTI_QUEUE_CUT_POLICY=/home/ubuntu/tinygrad-arkey/docs/task_workflow/evidence/nv-compiler-q4k-gkqo-20260828/combined-flash-direct-deps-cut-v2.json HCQ_NV_READY_PLACEMENT=0 PROFILE=0 DEV=NV PYTHONPATH=. python3 extra/llm_research/prefill/nv_compiler_q4k_gkqo_model_arm.py --arm candidate --q4-v --rounds 9 --warmups 3 --deep-replay --out docs/task_workflow/evidence/nv-prefill-post-substrate-active-clock-20260829/tinygrad-candidate.json --logits-npz docs/task_workflow/evidence/nv-prefill-post-substrate-active-clock-20260829/tinygrad-candidate.npz --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf
```

Telemetry is in `telemetry.csv`, sampled with `nvidia-smi` every 100 ms (875 samples), including graphics/SM/memory clocks, power, temperature, P-state, utilization, and active throttle reasons. The runner returned `0`; deep-20 replay and token correctness passed, with token `198`. R9 wall samples were 71.955971 to 72.179321 ms, median `72.121122 ms`.

Observed active clock: among P0 samples, graphics-clock median `2760 MHz`, maximum `2955 MHz` (steady tail was `2955 MHz`; memory `14001 MHz`). Temperature reached `43 C`, power draw `152.41 W` against a `600 W` limit, and active throttle reasons remained `0x0000000000000000`.

Verdict: **hardware/session throttling does not explain the 72.121122 ms result versus the 67.235719 ms authority**. The GPU entered P0 and reached its 2955 MHz ceiling without an active throttle reason; the 4.885403 ms gap is therefore a software/measurement-path difference, not observed clock or power throttling.
