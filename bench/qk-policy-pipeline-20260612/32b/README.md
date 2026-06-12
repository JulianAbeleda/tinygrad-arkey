# QK Policy Pipeline: Qwen3-32B-Q4_K_M.gguf

Date: 2026-06-12

- commit: `a96534155`
- device: `AMD`
- model size: `32B`
- generated policy: `policy.json`

## Decision

- status: `blocked`

Reasons:

- decode blocked by GPU memory during primitive install: MemoryError: Allocation of 70.31 MB failed on AMD. Used: 23.80 GB

## Artifacts

- `search.json`, `policy.json`, `semantic-report.md`
- `policy-parity.json`, `policy-parity.md`
- `explicit-run1.log`

## Policy Parity Summary

```json
{
  "by_format": {
    "Q4_K": 385,
    "Q6_K": 65
  },
  "effective_mismatches": 320,
  "explicit_installed": 320,
  "explicit_reasons": {
    "policy_fallback": 130,
    "policy_primitive": 320
  },
  "generated_installed": 448,
  "generated_reasons": {
    "policy_fused": 1,
    "policy_missing": 1,
    "policy_primitive": 448
  },
  "generated_unsupported": 0,
  "raw_differences": 450,
  "same_effective": 130,
  "same_raw": 0,
  "total": 450
}
```


## Failure Tail

```
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/tinygrad-arkey/tinygrad/tensor.py", line 1417, in _wrapper
    ret = fn(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/contextlib.py", line 81, in inner
    return func(*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/tinygrad-arkey/tinygrad/tensor.py", line 224, in realize
    run_linear(*Tensor.linear_with_vars(*to_realize), update_stats=do_update_stats)
  File "/home/ubuntu/tinygrad-arkey/tinygrad/engine/realize.py", line 256, in run_linear
    for call in linear.src: pm_exec.rewrite(call, ctx)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/tinygrad-arkey/tinygrad/uop/ops.py", line 1385, in rewrite
    if (ret:=match(uop, ctx)) is not None and ret is not uop: return ret
             ^^^^^^^^^^^^^^^
  File "<string>", line 3, in compiled_match
  File "/home/ubuntu/tinygrad-arkey/tinygrad/engine/realize.py", line 160, in exec_copy
    dest, src = bufs[0].ensure_allocated(), bufs[1].ensure_allocated()
                ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/tinygrad-arkey/tinygrad/device.py", line 141, in ensure_allocated
    def ensure_allocated(self) -> Buffer: return self.allocate() if not self.is_initialized() else self
                                                 ^^^^^^^^^^^^^^^
  File "/home/ubuntu/tinygrad-arkey/tinygrad/device.py", line 156, in allocate
    self._bufs[self.device] = opaque if opaque is not None else self.allocator.alloc(self.nbytes, self.options)
                                                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/tinygrad-arkey/tinygrad/device.py", line 263, in alloc
    return super().alloc(size, options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/tinygrad-arkey/tinygrad/device.py", line 233, in alloc
    except (RuntimeError, MemoryError) as e: raise MemoryError(f"Allocation of {size_to_str(size)} failed on {self.dev.device}. "
                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
MemoryError: Allocation of 70.31 MB failed on AMD. Used: 23.80 GB
```