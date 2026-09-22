# Phase 1 result

Status: **FAIL**

```json
{
  "schema": "tinygrad.nv-packed-qk-q8-streamk-gate-phase1.v2",
  "status": "FAIL",
  "threshold": "PASS iff mandatory gates pass, candidate min+median beat both controls, and median <=250 us",
  "commands": [
    "flock -w 1200 /tmp/gpu-bench.lock env PYTHONPATH=. DEV=NV .venv/bin/python /home/ubuntu/tinygrad-arkey/extra/llm_research/prefill/nv_q4_imma_geometry_sweep.py --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --out docs/task_workflow/evidence/nv-all-native-stack-20260830/q4-opt/owners-144.json --rounds 3"
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
      "min_us": 408.859,
      "median_us": 416.994,
      "p95_us": 433.4235,
      "mean_us": 421.3412222222222,
      "samples_us": [
        434.256,
        415.11,
        416.994,
        494.77,
        422.294,
        417.475,
        426.201,
        412.324,
        409.179,
        426.091,
        414.549,
        408.859,
        427.173,
        414.469,
        409.91,
        426.592,
        420.13,
        409.921,
        426.522,
        414.308,
        411.774,
        425.841,
        413.388,
        416.843,
        431.481,
        418.336,
        411.423
      ]
    },
    "candidate": {
      "min_us": 670.39,
      "median_us": 684.406,
      "p95_us": 716.634,
      "mean_us": 694.8482222222223,
      "samples_us": [
        698.022,
        676.161,
        670.39,
        849.035,
        695.117,
        686.3,
        708.011,
        684.406,
        675.489,
        712.77,
        686.22,
        675.018,
        718.29,
        679.196,
        675.29,
        711.016,
        679.798,
        676.111,
        703.913,
        680.86,
        679.577,
        709.514,
        683.264,
        679.878,
        704.765,
        687.372,
        675.119
      ]
    },
    "control_b": {
      "min_us": 408.107,
      "median_us": 412.766,
      "p95_us": 424.4204,
      "mean_us": 415.3888148148148,
      "samples_us": [
        420.891,
        410.982,
        410.852,
        425.249,
        415.802,
        411.824,
        423.086,
        411.294,
        410.261,
        424.598,
        412.686,
        412.264,
        420.22,
        412.035,
        408.107,
        423.085,
        413.767,
        408.457,
        420.34,
        413.637,
        409.109,
        424.006,
        412.816,
        412.195,
        422.604,
        412.565,
        412.766
      ]
    },
    "r9": [
      {
        "repetition": 0,
        "control_a": {
          "min_us": 415.11,
          "median_us": 416.994,
          "p95_us": 432.52979999999997,
          "mean_us": 422.12,
          "samples_us": [
            434.256,
            415.11,
            416.994
          ]
        },
        "candidate": {
          "min_us": 670.39,
          "median_us": 676.161,
          "p95_us": 695.8359,
          "mean_us": 681.5243333333333,
          "samples_us": [
            698.022,
            676.161,
            670.39
          ]
        },
        "control_b": {
          "min_us": 410.852,
          "median_us": 410.982,
          "p95_us": 419.9001,
          "mean_us": 414.2416666666667,
          "samples_us": [
            420.891,
            410.982,
            410.852
          ]
        }
      },
      {
        "repetition": 1,
        "control_a": {
          "min_us": 417.475,
          "median_us": 422.294,
          "p95_us": 487.52239999999995,
          "mean_us": 444.84633333333335,
          "samples_us": [
            494.77,
            422.294,
            417.475
          ]
        },
        "candidate": {
          "min_us": 686.3,
          "median_us": 695.117,
          "p95_us": 833.6432,
          "mean_us": 743.4839999999999,
          "samples_us": [
            849.035,
            695.117,
            686.3
          ]
        },
        "control_b": {
          "min_us": 411.824,
          "median_us": 415.802,
          "p95_us": 424.3043,
          "mean_us": 417.625,
          "samples_us": [
            425.249,
            415.802,
            411.824
          ]
        }
      },
      {
        "repetition": 2,
        "control_a": {
          "min_us": 409.179,
          "median_us": 412.324,
          "p95_us": 424.8133,
          "mean_us": 415.90133333333335,
          "samples_us": [
            426.201,
            412.324,
            409.179
          ]
        },
        "candidate": {
          "min_us": 675.489,
          "median_us": 684.406,
          "p95_us": 705.6505,
          "mean_us": 689.302,
          "samples_us": [
            708.011,
            684.406,
            675.489
          ]
        },
        "control_b": {
          "min_us": 410.261,
          "median_us": 411.294,
          "p95_us": 421.90680000000003,
          "mean_us": 414.88033333333334,
          "samples_us": [
            423.086,
            411.294,
            410.261
          ]
        }
      },
      {
        "repetition": 3,
        "control_a": {
          "min_us": 408.859,
          "median_us": 414.549,
          "p95_us": 424.9368,
          "mean_us": 416.49966666666666,
          "samples_us": [
            426.091,
            414.549,
            408.859
          ]
        },
        "candidate": {
          "min_us": 675.018,
          "median_us": 686.22,
          "p95_us": 710.115,
          "mean_us": 691.336,
          "samples_us": [
            712.77,
            686.22,
            675.018
          ]
        },
        "control_b": {
          "min_us": 412.264,
          "median_us": 412.686,
          "p95_us": 423.40680000000003,
          "mean_us": 416.516,
          "samples_us": [
            424.598,
            412.686,
            412.264
          ]
        }
      },
      {
        "repetition": 4,
        "control_a": {
          "min_us": 409.91,
          "median_us": 414.469,
          "p95_us": 425.9026,
          "mean_us": 417.184,
          "samples_us": [
            427.173,
            414.469,
            409.91
          ]
        },
        "candidate": {
          "min_us": 675.29,
          "median_us": 679.196,
          "p95_us": 714.3806,
          "mean_us": 690.9253333333334,
          "samples_us": [
            718.29,
            679.196,
            675.29
          ]
        },
        "control_b": {
          "min_us": 408.107,
          "median_us": 412.035,
          "p95_us": 419.4015,
          "mean_us": 413.454,
          "samples_us": [
            420.22,
            412.035,
            408.107
          ]
        }
      },
      {
        "repetition": 5,
        "control_a": {
          "min_us": 409.921,
          "median_us": 420.13,
          "p95_us": 425.94579999999996,
          "mean_us": 418.881,
          "samples_us": [
            426.592,
            420.13,
            409.921
          ]
        },
        "candidate": {
          "min_us": 676.111,
          "median_us": 679.798,
          "p95_us": 707.8942,
          "mean_us": 688.975,
          "samples_us": [
            711.016,
            679.798,
            676.111
          ]
        },
        "control_b": {
          "min_us": 408.457,
          "median_us": 413.767,
          "p95_us": 422.15319999999997,
          "mean_us": 415.103,
          "samples_us": [
            423.085,
            413.767,
            408.457
          ]
        }
      },
      {
        "repetition": 6,
        "control_a": {
          "min_us": 411.774,
          "median_us": 414.308,
          "p95_us": 425.3006,
          "mean_us": 417.5346666666667,
          "samples_us": [
            426.522,
            414.308,
            411.774
          ]
        },
        "candidate": {
          "min_us": 679.577,
          "median_us": 680.86,
          "p95_us": 701.6077,
          "mean_us": 688.1166666666667,
          "samples_us": [
            703.913,
            680.86,
            679.577
          ]
        },
        "control_b": {
          "min_us": 409.109,
          "median_us": 413.637,
          "p95_us": 419.6697,
          "mean_us": 414.36199999999997,
          "samples_us": [
            420.34,
            413.637,
            409.109
          ]
        }
      },
      {
        "repetition": 7,
        "control_a": {
          "min_us": 413.388,
          "median_us": 416.843,
          "p95_us": 424.9412,
          "mean_us": 418.6906666666667,
          "samples_us": [
            425.841,
            413.388,
            416.843
          ]
        },
        "candidate": {
          "min_us": 679.878,
          "median_us": 683.264,
          "p95_us": 706.889,
          "mean_us": 690.8853333333334,
          "samples_us": [
            709.514,
            683.264,
            679.878
          ]
        },
        "control_b": {
          "min_us": 412.195,
          "median_us": 412.816,
          "p95_us": 422.88699999999994,
          "mean_us": 416.339,
          "samples_us": [
            424.006,
            412.816,
            412.195
          ]
        }
      },
      {
        "repetition": 8,
        "control_a": {
          "min_us": 411.423,
          "median_us": 418.336,
          "p95_us": 430.1665,
          "mean_us": 420.41333333333336,
          "samples_us": [
            431.481,
            418.336,
            411.423
          ]
        },
        "candidate": {
          "min_us": 675.119,
          "median_us": 687.372,
          "p95_us": 703.0257,
          "mean_us": 689.0853333333333,
          "samples_us": [
            704.765,
            687.372,
            675.119
          ]
        },
        "control_b": {
          "min_us": 412.565,
          "median_us": 412.766,
          "p95_us": 421.6202,
          "mean_us": 415.97833333333335,
          "samples_us": [
            422.604,
            412.565,
            412.766
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
