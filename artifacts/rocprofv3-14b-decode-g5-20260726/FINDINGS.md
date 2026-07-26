# 14B tinygrad decode rocprofv3 trace finding

Verdict: **FAIL_CLOSED_NO_KERNEL_TRACE**.

The ctx512 authority workload completed successfully on `f54f06e27e3db50f287480705d8c668106712fce` with the real Qwen3-14B Q4_K_M model. It used flash decode, reported 1,133 captured programs per token, and measured 14.55496 ms/token (68.7051 tok/s) for W.

The profiler command exited zero but created no requested absolute `ctx512/rocprof` output directory and no JSON trace. The raw rocprof stderr contains initialization/finalization records for the parent and multiprocessing children, but no emitted kernel records. Therefore rocprof did not provide evidence of tinygrad GPU dispatches. No page-fault, GPU-reset, or other fault strings were present in the raw session logs.

Ctx4096 was intentionally not launched: the required first successful *kernel trace* was absent, even though the workload authority JSON is nonempty and valid.

Exact ctx512 command (run under `flock -n /tmp/gpu-bench.lock`):

```bash
/opt/rocm/bin/rocprofv3 --kernel-trace --output-format json --output-directory /home/ubuntu/worktrees/luna-tinygrad-g5-trace/artifacts/rocprofv3-14b-decode-g5-20260726/ctx512/rocprof -- python3 extra/qk/decode/decode_runtime_overhead.py --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf --ckpts 512 --max-context 4608 --nmeas 40 --reps 5 --out /home/ubuntu/worktrees/luna-tinygrad-g5-trace/artifacts/rocprofv3-14b-decode-g5-20260726/ctx512/authority.json
```
