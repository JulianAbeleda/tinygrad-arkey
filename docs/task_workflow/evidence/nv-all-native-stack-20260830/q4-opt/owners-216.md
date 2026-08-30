# Phase 1 result

Status: **FAIL**

```json
{
  "schema": "tinygrad.nv-packed-qk-q8-streamk-gate-phase1.v2",
  "status": "FAIL",
  "threshold": "PASS iff mandatory gates pass, candidate min+median beat both controls, and median <=250 us",
  "commands": [
    "flock -w 1200 /tmp/gpu-bench.lock env PYTHONPATH=. DEV=NV .venv/bin/python /home/ubuntu/tinygrad-arkey/extra/llm_research/prefill/nv_q4_imma_geometry_sweep.py --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --out docs/task_workflow/evidence/nv-all-native-stack-20260830/q4-opt/owners-216.json --rounds 3"
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
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
        "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296"
      ],
      "output_checksum": "83a64dfe5950bb5fa148d1d5ba4920b36eab7388187632add0c4f23e5cc83296",
      "reference_checksum": "893103155e9e58d6030ea7608e434b9bd758fb34056f59928517f37120301f10",
      "max_abs": 7.62939453125e-06,
      "max_rel": 0.1458333283662796,
      "mismatch_count": 0,
      "first_mismatch": null,
      "correctness_pass": true
    },
    {
      "case": 1,
      "input_checksum": "ffd4ee709737cbad455e88531c9409468fa8ac2a6c67f1863a4fd38daba3a8c4",
      "input_pointer": 138013581312,
      "weight_pointer": 137657057280,
      "output_pointer": 137988407296,
      "workspace_pointer": 138019864576,
      "finite": true,
      "reference_finite": true,
      "overwrite": true,
      "deterministic_20": true,
      "repeat_checksums": [
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
        "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb"
      ],
      "output_checksum": "4c03f6b2ceb32f13c5cf551bdd2e9d3182c4ff1af44f3e116a0edb09bfce34bb",
      "reference_checksum": "c5ebff1eccf66a9ca4797764bff8a864fec5b5cb25f77a534f04affa9e3ef633",
      "max_abs": 4.76837158203125e-06,
      "max_rel": 0.22535210847854614,
      "mismatch_count": 0,
      "first_mismatch": null,
      "correctness_pass": true
    }
  ],
  "activation_changes_output": true,
  "timing": {
    "control_a": {
      "min_us": 409.82,
      "median_us": 414.819,
      "p95_us": 428.03700000000003,
      "mean_us": 419.9975925925926,
      "samples_us": [
        426.572,
        414.67,
        413.387,
        488.759,
        416.353,
        412.535,
        426.261,
        412.946,
        411.704,
        425.981,
        412.996,
        415.32,
        427.694,
        417.826,
        410.792,
        426.101,
        414.819,
        409.82,
        427.363,
        417.936,
        410.121,
        423.196,
        412.705,
        410.382,
        428.184,
        414.339,
        411.173
      ]
    },
    "candidate": {
      "min_us": 673.315,
      "median_us": 683.304,
      "p95_us": 712.2795,
      "mean_us": 693.5131111111111,
      "samples_us": [
        699.335,
        678.666,
        680.078,
        827.094,
        689.646,
        686.049,
        709.945,
        683.304,
        682.062,
        706.889,
        684.076,
        673.436,
        708.04,
        686.681,
        677.373,
        704.634,
        682.924,
        676.041,
        708.743,
        678.255,
        675.71,
        703.402,
        679.938,
        674.638,
        713.28,
        681.3,
        673.315
      ]
    },
    "control_b": {
      "min_us": 408.819,
      "median_us": 411.994,
      "p95_us": 423.461,
      "mean_us": 415.015962962963,
      "samples_us": [
        419.519,
        411.934,
        409.34,
        431.21,
        415.301,
        410.491,
        423.536,
        412.515,
        409.069,
        423.286,
        411.894,
        411.022,
        422.013,
        411.062,
        410.922,
        422.243,
        411.804,
        409.42,
        420.721,
        412.205,
        408.819,
        419.729,
        412.756,
        409.53,
        421.943,
        411.994,
        411.153
      ]
    },
    "r9": [
      {
        "repetition": 0,
        "control_a": {
          "min_us": 413.387,
          "median_us": 414.67,
          "p95_us": 425.3818,
          "mean_us": 418.2096666666667,
          "samples_us": [
            426.572,
            414.67,
            413.387
          ]
        },
        "candidate": {
          "min_us": 678.666,
          "median_us": 680.078,
          "p95_us": 697.4093,
          "mean_us": 686.0263333333334,
          "samples_us": [
            699.335,
            678.666,
            680.078
          ]
        },
        "control_b": {
          "min_us": 409.34,
          "median_us": 411.934,
          "p95_us": 418.7605,
          "mean_us": 413.59766666666667,
          "samples_us": [
            419.519,
            411.934,
            409.34
          ]
        }
      },
      {
        "repetition": 1,
        "control_a": {
          "min_us": 412.535,
          "median_us": 416.353,
          "p95_us": 481.5184,
          "mean_us": 439.21566666666666,
          "samples_us": [
            488.759,
            416.353,
            412.535
          ]
        },
        "candidate": {
          "min_us": 686.049,
          "median_us": 689.646,
          "p95_us": 813.3492,
          "mean_us": 734.263,
          "samples_us": [
            827.094,
            689.646,
            686.049
          ]
        },
        "control_b": {
          "min_us": 410.491,
          "median_us": 415.301,
          "p95_us": 429.6191,
          "mean_us": 419.00066666666663,
          "samples_us": [
            431.21,
            415.301,
            410.491
          ]
        }
      },
      {
        "repetition": 2,
        "control_a": {
          "min_us": 411.704,
          "median_us": 412.946,
          "p95_us": 424.9295,
          "mean_us": 416.9703333333334,
          "samples_us": [
            426.261,
            412.946,
            411.704
          ]
        },
        "candidate": {
          "min_us": 682.062,
          "median_us": 683.304,
          "p95_us": 707.2809000000001,
          "mean_us": 691.7703333333334,
          "samples_us": [
            709.945,
            683.304,
            682.062
          ]
        },
        "control_b": {
          "min_us": 409.069,
          "median_us": 412.515,
          "p95_us": 422.4339,
          "mean_us": 415.04,
          "samples_us": [
            423.536,
            412.515,
            409.069
          ]
        }
      },
      {
        "repetition": 3,
        "control_a": {
          "min_us": 412.996,
          "median_us": 415.32,
          "p95_us": 424.9149,
          "mean_us": 418.099,
          "samples_us": [
            425.981,
            412.996,
            415.32
          ]
        },
        "candidate": {
          "min_us": 673.436,
          "median_us": 684.076,
          "p95_us": 704.6077,
          "mean_us": 688.1336666666667,
          "samples_us": [
            706.889,
            684.076,
            673.436
          ]
        },
        "control_b": {
          "min_us": 411.022,
          "median_us": 411.894,
          "p95_us": 422.1468,
          "mean_us": 415.40066666666667,
          "samples_us": [
            423.286,
            411.894,
            411.022
          ]
        }
      },
      {
        "repetition": 4,
        "control_a": {
          "min_us": 410.792,
          "median_us": 417.826,
          "p95_us": 426.7072,
          "mean_us": 418.77066666666667,
          "samples_us": [
            427.694,
            417.826,
            410.792
          ]
        },
        "candidate": {
          "min_us": 677.373,
          "median_us": 686.681,
          "p95_us": 705.9041,
          "mean_us": 690.698,
          "samples_us": [
            708.04,
            686.681,
            677.373
          ]
        },
        "control_b": {
          "min_us": 410.922,
          "median_us": 411.062,
          "p95_us": 420.9179,
          "mean_us": 414.66566666666665,
          "samples_us": [
            422.013,
            411.062,
            410.922
          ]
        }
      },
      {
        "repetition": 5,
        "control_a": {
          "min_us": 409.82,
          "median_us": 414.819,
          "p95_us": 424.9728,
          "mean_us": 416.91333333333336,
          "samples_us": [
            426.101,
            414.819,
            409.82
          ]
        },
        "candidate": {
          "min_us": 676.041,
          "median_us": 682.924,
          "p95_us": 702.463,
          "mean_us": 687.8663333333334,
          "samples_us": [
            704.634,
            682.924,
            676.041
          ]
        },
        "control_b": {
          "min_us": 409.42,
          "median_us": 411.804,
          "p95_us": 421.1991,
          "mean_us": 414.489,
          "samples_us": [
            422.243,
            411.804,
            409.42
          ]
        }
      },
      {
        "repetition": 6,
        "control_a": {
          "min_us": 410.121,
          "median_us": 417.936,
          "p95_us": 426.4203,
          "mean_us": 418.4733333333333,
          "samples_us": [
            427.363,
            417.936,
            410.121
          ]
        },
        "candidate": {
          "min_us": 675.71,
          "median_us": 678.255,
          "p95_us": 705.6942,
          "mean_us": 687.5693333333334,
          "samples_us": [
            708.743,
            678.255,
            675.71
          ]
        },
        "control_b": {
          "min_us": 408.819,
          "median_us": 412.205,
          "p95_us": 419.8694,
          "mean_us": 413.915,
          "samples_us": [
            420.721,
            412.205,
            408.819
          ]
        }
      },
      {
        "repetition": 7,
        "control_a": {
          "min_us": 410.382,
          "median_us": 412.705,
          "p95_us": 422.1469,
          "mean_us": 415.42766666666665,
          "samples_us": [
            423.196,
            412.705,
            410.382
          ]
        },
        "candidate": {
          "min_us": 674.638,
          "median_us": 679.938,
          "p95_us": 701.0556,
          "mean_us": 685.9926666666667,
          "samples_us": [
            703.402,
            679.938,
            674.638
          ]
        },
        "control_b": {
          "min_us": 409.53,
          "median_us": 412.756,
          "p95_us": 419.0317,
          "mean_us": 414.005,
          "samples_us": [
            419.729,
            412.756,
            409.53
          ]
        }
      },
      {
        "repetition": 8,
        "control_a": {
          "min_us": 411.173,
          "median_us": 414.339,
          "p95_us": 426.7995,
          "mean_us": 417.89866666666666,
          "samples_us": [
            428.184,
            414.339,
            411.173
          ]
        },
        "candidate": {
          "min_us": 673.315,
          "median_us": 681.3,
          "p95_us": 710.082,
          "mean_us": 689.2983333333333,
          "samples_us": [
            713.28,
            681.3,
            673.315
          ]
        },
        "control_b": {
          "min_us": 411.153,
          "median_us": 411.994,
          "p95_us": 420.9481,
          "mean_us": 415.03000000000003,
          "samples_us": [
            421.943,
            411.994,
            411.153
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
