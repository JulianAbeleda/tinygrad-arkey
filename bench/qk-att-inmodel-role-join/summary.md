# ATT in-model role join summary

```json
{
  "blocked_roles": {
    "ffn_gateup_pair": "MemoryError('Allocation of 4.68 GB failed on AMD. Used: 0 B') on rerun; individual ffn_gate/ffn_up artifacts remain valid"
  },
  "roles": {
    "ffn_gate": {
      "activation": {
        "capture_mode": "inmodel_activation",
        "decode_enabled": true,
        "in_features": 4096,
        "kernel_mode": "partial",
        "linear_name": "blk.0.ffn_gate.weight",
        "linear_type": "Q4KPrimitiveLinear",
        "out_features": 12288,
        "parts": 1,
        "shape": [
          1,
          1,
          4096
        ],
        "warm_output_shape": [
          1,
          1,
          12288
        ]
      },
      "gates": {
        "att_body_packets": "PASS",
        "att_start_stop_sync": "PASS",
        "decode_primitives_enabled": "PASS",
        "native_coop_present": "PASS",
        "programs_captured": "PASS"
      },
      "program_call_count": 2,
      "trace_body_packets": 47143,
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
          "lib_sha16": "7bdc80cbb3533005",
          "local_size": [
            32,
            1,
            1
          ],
          "prof_prg_counter": 13,
          "program_name": "E_32_32_4n2",
          "role": null,
          "runtime_class": "AMDProgram",
          "vals_len": 0
        },
        {
          "calls": 1,
          "global_size": [
            192,
            1,
            1
          ],
          "kernargs_alloc_size": 24,
          "lib_bytes": 6128,
          "lib_sha16": "236fd9e8841b577f",
          "local_size": [
            64,
            1,
            1
          ],
          "prof_prg_counter": 14,
          "program_name": "q4k_gemv_partial_12288_4096_1",
          "role": "ffn_gate/up_native_q4k_gemv",
          "runtime_class": "AMDProgram",
          "vals_len": 0
        }
      ],
      "verdict": "PASS_INMODEL_ROLE_JOIN_NATIVE_COOP"
    },
    "ffn_up": {
      "activation": {
        "capture_mode": "inmodel_activation",
        "decode_enabled": true,
        "in_features": 4096,
        "kernel_mode": "partial",
        "linear_name": "blk.0.ffn_up.weight",
        "linear_type": "Q4KPrimitiveLinear",
        "out_features": 12288,
        "parts": 1,
        "shape": [
          1,
          1,
          4096
        ],
        "warm_output_shape": [
          1,
          1,
          12288
        ]
      },
      "gates": {
        "att_body_packets": "PASS",
        "att_start_stop_sync": "PASS",
        "decode_primitives_enabled": "PASS",
        "native_coop_present": "PASS",
        "programs_captured": "PASS"
      },
      "program_call_count": 2,
      "trace_body_packets": 47132,
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
          "lib_sha16": "7bdc80cbb3533005",
          "local_size": [
            32,
            1,
            1
          ],
          "prof_prg_counter": 13,
          "program_name": "E_32_32_4n2",
          "role": null,
          "runtime_class": "AMDProgram",
          "vals_len": 0
        },
        {
          "calls": 1,
          "global_size": [
            192,
            1,
            1
          ],
          "kernargs_alloc_size": 24,
          "lib_bytes": 6128,
          "lib_sha16": "236fd9e8841b577f",
          "local_size": [
            64,
            1,
            1
          ],
          "prof_prg_counter": 14,
          "program_name": "q4k_gemv_partial_12288_4096_1",
          "role": "ffn_gate/up_native_q4k_gemv",
          "runtime_class": "AMDProgram",
          "vals_len": 0
        }
      ],
      "verdict": "PASS_INMODEL_ROLE_JOIN_NATIVE_COOP"
    }
  },
  "verdict": "PASS_PARTIAL_ROLE_JOINS_WITH_PAIR_BLOCKED"
}
```
