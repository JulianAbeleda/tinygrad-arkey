# ATT in-model role join summary

```json
{
  "roles": {
    "ffn_down": {
      "activation": {
        "capture_mode": "q6_surface_fallback",
        "decode_enabled": true,
        "in_features": 12288,
        "kernel_mode": null,
        "linear_name": "blk.0.ffn_down.weight",
        "linear_type": "Q6KPrimitiveLinear",
        "out_features": 4096,
        "parts": 1,
        "shape": [
          1,
          1,
          12288
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
        "native_coop_present": "PASS",
        "programs_captured": "PASS"
      },
      "program_call_count": 3,
      "trace_body_packets": 148598,
      "variants": [
        {
          "calls": 1,
          "global_size": [
            128,
            1,
            1
          ],
          "kernargs_alloc_size": 16,
          "lib_bytes": 4200,
          "lib_sha16": "a4e0ed6571fb4f89",
          "local_size": [
            32,
            1,
            1
          ],
          "prof_prg_counter": 0,
          "program_name": "E_128_32_3",
          "role": null,
          "runtime_class": "AMDProgram",
          "vals_len": 0
        },
        {
          "calls": 1,
          "global_size": [
            1024,
            1,
            1
          ],
          "kernargs_alloc_size": 24,
          "lib_bytes": 6112,
          "lib_sha16": "9b15e7a6f48723cc",
          "local_size": [
            4,
            16,
            1
          ],
          "prof_prg_counter": 1,
          "program_name": "q6k_coop_partial_4096_12288",
          "role": "ffn_down_native_q6k_coop",
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
          "lib_bytes": 4600,
          "lib_sha16": "e056b1c738dc6aff",
          "local_size": [
            32,
            1,
            1
          ],
          "prof_prg_counter": 2,
          "program_name": "r_32_32_4_16",
          "role": "role_reduce_or_glue",
          "runtime_class": "AMDProgram",
          "vals_len": 0
        }
      ],
      "verdict": "PASS_INMODEL_ROLE_JOIN_NATIVE_COOP"
    },
    "lm_head": {
      "activation": {
        "capture_mode": "q6_surface_fallback",
        "decode_enabled": true,
        "in_features": 4096,
        "kernel_mode": null,
        "linear_name": "output.weight",
        "linear_type": "Q6KPrimitiveLinear",
        "out_features": 151936,
        "parts": 1,
        "shape": [
          1,
          1,
          4096
        ],
        "warm_output_shape": [
          1,
          1,
          151936
        ]
      },
      "gates": {
        "att_body_packets": "PASS",
        "att_start_stop_sync": "PASS",
        "decode_primitives_enabled": "PASS",
        "native_coop_present": "PASS",
        "programs_captured": "PASS"
      },
      "program_call_count": 3,
      "trace_body_packets": 264117,
      "variants": [
        {
          "calls": 1,
          "global_size": [
            32,
            1,
            1
          ],
          "kernargs_alloc_size": 16,
          "lib_bytes": 4192,
          "lib_sha16": "07d7605a770353bc",
          "local_size": [
            32,
            1,
            1
          ],
          "prof_prg_counter": 3,
          "program_name": "E_32_32_4",
          "role": null,
          "runtime_class": "AMDProgram",
          "vals_len": 0
        },
        {
          "calls": 1,
          "global_size": [
            37984,
            1,
            1
          ],
          "kernargs_alloc_size": 24,
          "lib_bytes": 6120,
          "lib_sha16": "9300ebc8996a7ab8",
          "local_size": [
            4,
            16,
            1
          ],
          "prof_prg_counter": 4,
          "program_name": "q6k_coop_partial_151936_4096",
          "role": "lm_head_native_q6k_coop",
          "runtime_class": "AMDProgram",
          "vals_len": 0
        },
        {
          "calls": 1,
          "global_size": [
            1187,
            1,
            1
          ],
          "kernargs_alloc_size": 16,
          "lib_bytes": 4616,
          "lib_sha16": "2a8531661d24865f",
          "local_size": [
            32,
            1,
            1
          ],
          "prof_prg_counter": 5,
          "program_name": "r_1187_32_4_16",
          "role": "role_reduce_or_glue",
          "runtime_class": "AMDProgram",
          "vals_len": 0
        }
      ],
      "verdict": "PASS_INMODEL_ROLE_JOIN_NATIVE_COOP"
    }
  },
  "verdict": "PASS_ALL_ROLE_JOINS"
}
```
