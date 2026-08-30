# Phase 1 result

Status: **FAIL**

```json
{
  "schema": "tinygrad.nv-packed-qk-q8-streamk-gate-phase1.v2",
  "status": "FAIL",
  "threshold": "PASS iff mandatory gates pass, candidate min+median beat both controls, and median <=250 us",
  "commands": [
    "flock -w 1200 /tmp/gpu-bench.lock env PYTHONPATH=. DEV=NV .venv/bin/python /home/ubuntu/tinygrad-arkey/extra/llm_research/prefill/nv_q4_imma_geometry_sweep.py --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --out docs/task_workflow/evidence/nv-all-native-stack-20260830/q4-opt/owners-160.json --rounds 3"
  ],
  "files_changed": [
    "extra/llm_research/prefill/nv_packed_qk_q8_streamk.py",
    "extra/llm_research/prefill/nv_packed_qk_q8_gate_qualify.py",
    "docs/task_workflow/output/nv-packed-qk-q8-streamk-gate-phase1-result-20260829.md",
    "docs/task_workflow/output/nv-packed-qk-q8-streamk-gate-phase1-result-20260829.json"
  ],
  "model": "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf",
  "tensor": "blk.0.ffn_gate.weight",
  "shape": {
    "M": 512,
    "N": 12288,
    "K": 4096
  },
  "gpu_arch": "sm_120",
  "dtype": "fp16->Q8/int8->fp32",
  "metadata": {
    "producer_grid": [
      512,
      8,
      1
    ],
    "producer_block": [
      128,
      1,
      1
    ],
    "main_grid": [
      170,
      1,
      1
    ],
    "main_block": [
      256,
      1,
      1
    ],
    "fixup_grid": [
      384,
      1,
      1
    ],
    "fixup_block": [
      256,
      1,
      1
    ],
    "workspace_bytes": 22286672,
    "schedule": "stream-k split-K with deterministic slot-map fixup"
  },
  "inputs": [],
  "observed_failure": "stage=case-0-candidate RuntimeError: MMU fault: 0x201F083000 | NV_PFAULT_FAULT_TYPE_PTE | READ"
}
```
