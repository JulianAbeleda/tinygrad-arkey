# Phase 1 result

Status: **FAIL**

```json
{
  "schema": "tinygrad.nv-packed-qk-q8-streamk-gate-phase1.v2",
  "status": "FAIL",
  "threshold": "PASS iff mandatory gates pass, candidate min+median beat both controls, and median <=250 us",
  "commands": [
    "flock -w 1200 /tmp/gpu-bench.lock env PYTHONPATH=. DEV=NV .venv/bin/python /home/ubuntu/tinygrad-arkey/extra/llm_research/prefill/nv_packed_qk_q8_gate_qualify.py --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --out docs/task_workflow/evidence/nv-all-native-stack-20260830/q4-opt/resource-audit-v2.json --rounds 3"
  ],
  "files_changed": [
    "extra/llm_research/prefill/nv_packed_qk_q8_streamk.py",
    "extra/llm_research/prefill/nv_packed_qk_q8_gate_qualify.py",
    "docs/task_workflow/output/nv-packed-qk-q8-streamk-gate-phase1-result-20260829.md",
    "docs/task_workflow/output/nv-packed-qk-q8-streamk-gate-phase1-result-20260829.json"
  ],
  "resource_audit": {
    "compiler": {
      "main": {
        "name": "q4k_imma_stream",
        "regs_usage": 255,
        "shared_size_bytes": 58880,
        "local_size_bytes": 600,
        "max_threads": 256,
        "lib_bytes": 141232
      },
      "fixup": {
        "name": "q4k_imma_fixup",
        "regs_usage": 28,
        "shared_size_bytes": 1024,
        "local_size_bytes": 576,
        "max_threads": 2048,
        "lib_bytes": 8552
      }
    },
    "control_binding_type": "CompilerPP512Binding",
    "control_runtime_audit": "unavailable (binding has no NVProgram main/fixup)"
  },
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
  "inputs": [
    {
      "case": 0,
      "input_checksum": "a071ccaeba04326105a8c41e4d8cd3aff530d1e29a2d5780972d09dc4924dd8d",
      "input_pointer": 137929723904,
      "weight_pointer": 137657057280,
      "output_pointer": 137870966784,
      "workspace_pointer": 137935978496,
      "finite": true,
      "reference_finite": true,
      "overwrite": true,
      "deterministic_20": true,
      "repeat_checksums": [
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
        "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c"
      ],
      "output_checksum": "880bf9920d2fcf2e5012a9a30de69c3a55e6589a4534c709f399d6151edb3f5c",
      "reference_checksum": "893103155e9e58d6030ea7608e434b9bd758fb34056f59928517f37120301f10",
      "max_abs": 5.7220458984375e-06,
      "max_rel": 0.13513512909412384,
      "mismatch_count": 0,
      "first_mismatch": null,
      "correctness_pass": true
    },
    {
      "case": 1,
      "input_checksum": "ffd4ee709737cbad455e88531c9409468fa8ac2a6c67f1863a4fd38daba3a8c4",
      "input_pointer": 138015678464,
      "weight_pointer": 137657057280,
      "output_pointer": 137990504448,
      "workspace_pointer": 138021961728,
      "finite": true,
      "reference_finite": true,
      "overwrite": true,
      "deterministic_20": true,
      "repeat_checksums": [
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
        "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814"
      ],
      "output_checksum": "8d47cb24f9513a3cfaa7bf4fdaae6ee7ebe7003b99ed568735d664f8e4ad6814",
      "reference_checksum": "c5ebff1eccf66a9ca4797764bff8a864fec5b5cb25f77a534f04affa9e3ef633",
      "max_abs": 4.76837158203125e-06,
      "max_rel": 0.2059859186410904,
      "mismatch_count": 0,
      "first_mismatch": null,
      "correctness_pass": true
    }
  ],
  "activation_changes_output": true,
  "timing": {
    "control_a": {
      "min_us": 408.708,
      "median_us": 416.212,
      "p95_us": 431.6626,
      "mean_us": 421.7894074074074,
      "samples_us": [
        427.573,
        414.809,
        413.016,
        499.097,
        432.913,
        416.212,
        428.264,
        420.701,
        413.156,
        426.591,
        413.818,
        412.054,
        428.475,
        414.81,
        412.305,
        428.745,
        415.63,
        409.74,
        425.93,
        419.288,
        409.419,
        427.093,
        415.069,
        416.663,
        425.971,
        412.264,
        408.708
      ]
    },
    "candidate": {
      "min_us": 653.528,
      "median_us": 661.603,
      "p95_us": 688.8173999999999,
      "mean_us": 672.9707037037037,
      "samples_us": [
        677.022,
        653.798,
        656.122,
        832.033,
        664.128,
        661.392,
        686.079,
        661.603,
        657.666,
        683.895,
        658.518,
        659.93,
        684.687,
        664.087,
        656.073,
        684.867,
        657.244,
        654.019,
        688.543,
        657.786,
        658.297,
        685.137,
        661.924,
        660.982,
        688.935,
        661.914,
        653.528
      ]
    },
    "control_b": {
      "min_us": 408.507,
      "median_us": 413.928,
      "p95_us": 424.666,
      "mean_us": 416.32414814814814,
      "samples_us": [
        420.6,
        419.869,
        409.72,
        429.728,
        413.137,
        411.523,
        424.287,
        415.01,
        410.431,
        420.92,
        419.148,
        410.612,
        422.964,
        414.168,
        410.802,
        424.568,
        411.924,
        408.638,
        424.708,
        412.766,
        410.812,
        423.085,
        412.686,
        408.507,
        423.436,
        412.775,
        413.928
      ]
    },
    "r9": [
      {
        "repetition": 0,
        "control_a": {
          "min_us": 413.016,
          "median_us": 414.809,
          "p95_us": 426.29659999999996,
          "mean_us": 418.466,
          "samples_us": [
            427.573,
            414.809,
            413.016
          ]
        },
        "candidate": {
          "min_us": 653.798,
          "median_us": 656.122,
          "p95_us": 674.932,
          "mean_us": 662.314,
          "samples_us": [
            677.022,
            653.798,
            656.122
          ]
        },
        "control_b": {
          "min_us": 409.72,
          "median_us": 419.869,
          "p95_us": 420.5269,
          "mean_us": 416.7296666666667,
          "samples_us": [
            420.6,
            419.869,
            409.72
          ]
        }
      },
      {
        "repetition": 1,
        "control_a": {
          "min_us": 416.212,
          "median_us": 432.913,
          "p95_us": 492.4786,
          "mean_us": 449.4073333333333,
          "samples_us": [
            499.097,
            432.913,
            416.212
          ]
        },
        "candidate": {
          "min_us": 661.392,
          "median_us": 664.128,
          "p95_us": 815.2425,
          "mean_us": 719.1843333333334,
          "samples_us": [
            832.033,
            664.128,
            661.392
          ]
        },
        "control_b": {
          "min_us": 411.523,
          "median_us": 413.137,
          "p95_us": 428.0689,
          "mean_us": 418.12933333333336,
          "samples_us": [
            429.728,
            413.137,
            411.523
          ]
        }
      },
      {
        "repetition": 2,
        "control_a": {
          "min_us": 413.156,
          "median_us": 420.701,
          "p95_us": 427.5077,
          "mean_us": 420.707,
          "samples_us": [
            428.264,
            420.701,
            413.156
          ]
        },
        "candidate": {
          "min_us": 657.666,
          "median_us": 661.603,
          "p95_us": 683.6314,
          "mean_us": 668.4493333333334,
          "samples_us": [
            686.079,
            661.603,
            657.666
          ]
        },
        "control_b": {
          "min_us": 410.431,
          "median_us": 415.01,
          "p95_us": 423.35929999999996,
          "mean_us": 416.57599999999996,
          "samples_us": [
            424.287,
            415.01,
            410.431
          ]
        }
      },
      {
        "repetition": 3,
        "control_a": {
          "min_us": 412.054,
          "median_us": 413.818,
          "p95_us": 425.3137,
          "mean_us": 417.48766666666666,
          "samples_us": [
            426.591,
            413.818,
            412.054
          ]
        },
        "candidate": {
          "min_us": 658.518,
          "median_us": 659.93,
          "p95_us": 681.4984999999999,
          "mean_us": 667.4476666666667,
          "samples_us": [
            683.895,
            658.518,
            659.93
          ]
        },
        "control_b": {
          "min_us": 410.612,
          "median_us": 419.148,
          "p95_us": 420.7428,
          "mean_us": 416.8933333333334,
          "samples_us": [
            420.92,
            419.148,
            410.612
          ]
        }
      },
      {
        "repetition": 4,
        "control_a": {
          "min_us": 412.305,
          "median_us": 414.81,
          "p95_us": 427.1085,
          "mean_us": 418.53000000000003,
          "samples_us": [
            428.475,
            414.81,
            412.305
          ]
        },
        "candidate": {
          "min_us": 656.073,
          "median_us": 664.087,
          "p95_us": 682.627,
          "mean_us": 668.2823333333333,
          "samples_us": [
            684.687,
            664.087,
            656.073
          ]
        },
        "control_b": {
          "min_us": 410.802,
          "median_us": 414.168,
          "p95_us": 422.0844,
          "mean_us": 415.978,
          "samples_us": [
            422.964,
            414.168,
            410.802
          ]
        }
      },
      {
        "repetition": 5,
        "control_a": {
          "min_us": 409.74,
          "median_us": 415.63,
          "p95_us": 427.4335,
          "mean_us": 418.03833333333336,
          "samples_us": [
            428.745,
            415.63,
            409.74
          ]
        },
        "candidate": {
          "min_us": 654.019,
          "median_us": 657.244,
          "p95_us": 682.1047,
          "mean_us": 665.3766666666667,
          "samples_us": [
            684.867,
            657.244,
            654.019
          ]
        },
        "control_b": {
          "min_us": 408.638,
          "median_us": 411.924,
          "p95_us": 423.30359999999996,
          "mean_us": 415.0433333333333,
          "samples_us": [
            424.568,
            411.924,
            408.638
          ]
        }
      },
      {
        "repetition": 6,
        "control_a": {
          "min_us": 409.419,
          "median_us": 419.288,
          "p95_us": 425.2658,
          "mean_us": 418.21233333333333,
          "samples_us": [
            425.93,
            419.288,
            409.419
          ]
        },
        "candidate": {
          "min_us": 657.786,
          "median_us": 658.297,
          "p95_us": 685.5184,
          "mean_us": 668.2086666666667,
          "samples_us": [
            688.543,
            657.786,
            658.297
          ]
        },
        "control_b": {
          "min_us": 410.812,
          "median_us": 412.766,
          "p95_us": 423.5138,
          "mean_us": 416.0953333333334,
          "samples_us": [
            424.708,
            412.766,
            410.812
          ]
        }
      },
      {
        "repetition": 7,
        "control_a": {
          "min_us": 415.069,
          "median_us": 416.663,
          "p95_us": 426.05,
          "mean_us": 419.60833333333335,
          "samples_us": [
            427.093,
            415.069,
            416.663
          ]
        },
        "candidate": {
          "min_us": 660.982,
          "median_us": 661.924,
          "p95_us": 682.8157,
          "mean_us": 669.3476666666667,
          "samples_us": [
            685.137,
            661.924,
            660.982
          ]
        },
        "control_b": {
          "min_us": 408.507,
          "median_us": 412.686,
          "p95_us": 422.0451,
          "mean_us": 414.7593333333333,
          "samples_us": [
            423.085,
            412.686,
            408.507
          ]
        }
      },
      {
        "repetition": 8,
        "control_a": {
          "min_us": 408.708,
          "median_us": 412.264,
          "p95_us": 424.6003,
          "mean_us": 415.6476666666667,
          "samples_us": [
            425.971,
            412.264,
            408.708
          ]
        },
        "candidate": {
          "min_us": 653.528,
          "median_us": 661.914,
          "p95_us": 686.2329,
          "mean_us": 668.1256666666667,
          "samples_us": [
            688.935,
            661.914,
            653.528
          ]
        },
        "control_b": {
          "min_us": 412.775,
          "median_us": 413.928,
          "p95_us": 422.48519999999996,
          "mean_us": 416.71299999999997,
          "samples_us": [
            423.436,
            412.775,
            413.928
          ]
        }
      }
    ]
  },
  "structural": {
    "pass": true,
    "canonical_packed_weight": true,
    "weight_expansion_or_hot_copy": false,
    "candidate_launches": [
      "q8_compact_record_fp16",
      "q4k_imma_stream",
      "q4k_imma_fixup"
    ],
    "allocation_inside_timed_candidate": false,
    "synchronization_inside_timed_candidate": false,
    "stable_output_pointer": true,
    "stable_workspace_pointer": true
  },
  "observed_failure": "candidate did not beat both controls on minimum and median"
}
```
