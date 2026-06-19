# ATT in-model role join summary

```json
{
  "activation": {
    "decode_enabled": true,
    "in_features": 4096,
    "kernel_mode": "partial",
    "linear_name": "blk.0.attn_output.weight",
    "linear_type": "Q4KPrimitiveLinear",
    "out_features": 4096,
    "parts": 1,
    "shape": [
      1,
      1,
      4096
    ],
    "warm_output_shape": [
      1,
      1,
      4096
    ]
  },
  "gates": {
    "att_body_packets": "PASS",
    "att_start_stop_sync": "PASS",
    "decode_primitives_enabled": "PASS",
    "programs_captured": "PASS",
    "q4k_native_coop_present": "PASS"
  },
  "program_call_count": 3,
  "trace_body_packets": 16137,
  "variants": [
    {
      "calls": 1,
      "global_size": [
        32,
        1,
        1
      ],
      "kernargs_alloc_size": 16,
      "lib_bytes": 4208,
      "lib_sha16": "5cbd23799115269e",
      "local_size": [
        32,
        1,
        1
      ],
      "prof_prg_counter": 13,
      "program_name": "E_32_32_4n1",
      "role": null,
      "runtime_class": "AMDProgram",
      "vals_len": 0
    },
    {
      "calls": 1,
      "global_size": [
        256,
        1,
        1
      ],
      "kernargs_alloc_size": 24,
      "lib_bytes": 6104,
      "lib_sha16": "c5d614003c16e974",
      "local_size": [
        16,
        8,
        1
      ],
      "prof_prg_counter": 6,
      "program_name": "q4k_coop_partial_4096_4096",
      "role": "attn_q/o_native_q4k_coop",
      "runtime_class": "AMDProgram",
      "vals_len": 0
    },
    {
      "calls": 1,
      "global_size": [
        32,
        1,
        1
      ],
      "kernargs_alloc_size": 16,
      "lib_bytes": 4336,
      "lib_sha16": "143886410d9ab3d2",
      "local_size": [
        32,
        1,
        1
      ],
      "prof_prg_counter": 12,
      "program_name": "r_32_32_4_8",
      "role": "role_reduce_or_glue",
      "runtime_class": "AMDProgram",
      "vals_len": 0
    }
  ],
  "verdict": "PASS_INMODEL_ROLE_JOIN_NATIVE_Q4K_COOP"
}
```
