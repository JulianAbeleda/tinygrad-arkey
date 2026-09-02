# NV direct-copy 2x2 synthetic microgate (2026-09-01)

## Verdict: `SYNTHETIC_PASS_RELEASE_ONE_REAL_Q6_STATIC_RERUN`

`candidate_const_restrict` passes the hard static gate and improves on candidate_writable span 82 -> 71. It releases one matching real-Q6 static compile only, not correctness or timing.

## Arms

| Arm | Arch | const restrict | Span | REG | Stack | LDG opcode | Stable | Hard pass |
|---|---|---:|---:|---:|---:|---|---:|---:|
| `candidate_writable` | `sm_120` | False | 82 | 63 | 0 | `{'LDG.E': 18}` | True | True |
| `candidate_const_restrict` | `sm_120` | True | 71 | 60 | 0 | `{'LDG.E.CONSTANT': 18}` | True | True |
| `llama_writable` | `sm_120a` | False | 105 | 56 | 0 | `{'LDG.E': 18}` | True | True |
| `llama_const_restrict` | `sm_120a` | True | 107 | 56 | 0 | `{'LDG.E.CONSTANT': 18}` | True | True |

## Fixed contract

- 64 independent FP32 accumulators remain live across an existing barrier, 18 paired global-u32-to-shared copies, and a publication barrier.
- Exact memory census is 18 LDG, 18 STS, 1 LDS, 1 STG, and 2 BAR.
- Inline PTX, volatile, noinline, function splitting, MEMBAR, and atomics are forbidden.
- A hard arm requires span <=160, stable repeated cubin, and zero stack/LDL/STL/local spill traffic.

## Contrasts

```json
{
  "Q_at_candidate": {
    "left": "candidate_writable",
    "right": "candidate_const_restrict",
    "span_improvement_left_minus_right": 11,
    "register_delta_left_minus_right": 3,
    "stack_delta_left_minus_right": 0
  },
  "Q_at_llama": {
    "left": "llama_writable",
    "right": "llama_const_restrict",
    "span_improvement_left_minus_right": -2,
    "register_delta_left_minus_right": 0,
    "stack_delta_left_minus_right": 0
  },
  "F_at_writable": {
    "left": "candidate_writable",
    "right": "llama_writable",
    "span_improvement_left_minus_right": -23,
    "register_delta_left_minus_right": 7,
    "stack_delta_left_minus_right": 0
  },
  "F_at_const_restrict": {
    "left": "candidate_const_restrict",
    "right": "llama_const_restrict",
    "span_improvement_left_minus_right": -36,
    "register_delta_left_minus_right": 4,
    "stack_delta_left_minus_right": 0
  }
}
```

## Toolchain

- NVRTC: `13.2`
- nvdisasm: `Cuda compilation tools, release 12.8, V12.8.55`
- cuobjdump: `Cuda compilation tools, release 13.2, V13.2.86`

No kernel was launched and no Q6 production or research builder was modified.
