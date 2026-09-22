# Phase 1 result

Status: **FAIL**

```json
{
  "schema": "tinygrad.nv-packed-qk-q8-streamk-gate-phase1.v2",
  "status": "FAIL",
  "threshold": "PASS iff mandatory gates pass, candidate min+median beat both controls, and median <=250 us",
  "commands": [
    "flock -w 1200 /tmp/gpu-bench.lock env PYTHONPATH=. DEV=NV .venv/bin/python /home/ubuntu/tinygrad-arkey/extra/llm_research/prefill/nv_q4_imma_geometry_sweep.py --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --out docs/task_workflow/evidence/nv-all-native-stack-20260830/q4-opt/owners-192.json --rounds 3"
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
      "min_us": 408.598,
      "median_us": 414.098,
      "p95_us": 427.2924,
      "mean_us": 418.88307407407405,
      "samples_us": [
        426.542,
        414.96,
        411.564,
        483.92,
        420.08,
        414.098,
        424.478,
        414.038,
        408.759,
        423.606,
        415.742,
        409.35,
        422.143,
        412.045,
        412.015,
        424.859,
        412.125,
        415.621,
        427.614,
        413.828,
        408.688,
        424.669,
        412.756,
        409.6,
        425.199,
        412.946,
        408.598
      ]
    },
    "candidate": {
      "min_us": 673.796,
      "median_us": 680.58,
      "p95_us": 712.3114,
      "mean_us": 693.6145185185185,
      "samples_us": [
        696.249,
        676.822,
        673.797,
        834.889,
        690.739,
        680.58,
        712.45,
        681.461,
        680.629,
        711.988,
        677.964,
        675.7,
        707.54,
        680.258,
        679.508,
        711.237,
        679.407,
        679.778,
        705.796,
        684.948,
        677.082,
        707.52,
        677.884,
        673.796,
        711.087,
        680.569,
        677.914
      ]
    },
    "control_b": {
      "min_us": 408.528,
      "median_us": 413.307,
      "p95_us": 426.04139999999995,
      "mean_us": 415.5899259259259,
      "samples_us": [
        419.718,
        410.913,
        413.307,
        426.501,
        415.171,
        412.575,
        424.969,
        411.553,
        409.901,
        420.27,
        411.393,
        410.481,
        419.679,
        412.385,
        411.424,
        418.477,
        415.39,
        410.391,
        421.051,
        412.335,
        409.7,
        422.564,
        413.417,
        419.067,
        427.012,
        412.756,
        408.528
      ]
    },
    "r9": [
      {
        "repetition": 0,
        "control_a": {
          "min_us": 411.564,
          "median_us": 414.96,
          "p95_us": 425.38379999999995,
          "mean_us": 417.6886666666667,
          "samples_us": [
            426.542,
            414.96,
            411.564
          ]
        },
        "candidate": {
          "min_us": 673.797,
          "median_us": 676.822,
          "p95_us": 694.3063,
          "mean_us": 682.2893333333334,
          "samples_us": [
            696.249,
            676.822,
            673.797
          ]
        },
        "control_b": {
          "min_us": 410.913,
          "median_us": 413.307,
          "p95_us": 419.0769,
          "mean_us": 414.646,
          "samples_us": [
            419.718,
            410.913,
            413.307
          ]
        }
      },
      {
        "repetition": 1,
        "control_a": {
          "min_us": 414.098,
          "median_us": 420.08,
          "p95_us": 477.536,
          "mean_us": 439.366,
          "samples_us": [
            483.92,
            420.08,
            414.098
          ]
        },
        "candidate": {
          "min_us": 680.58,
          "median_us": 690.739,
          "p95_us": 820.474,
          "mean_us": 735.4026666666667,
          "samples_us": [
            834.889,
            690.739,
            680.58
          ]
        },
        "control_b": {
          "min_us": 412.575,
          "median_us": 415.171,
          "p95_us": 425.368,
          "mean_us": 418.08233333333334,
          "samples_us": [
            426.501,
            415.171,
            412.575
          ]
        }
      },
      {
        "repetition": 2,
        "control_a": {
          "min_us": 408.759,
          "median_us": 414.038,
          "p95_us": 423.434,
          "mean_us": 415.7583333333333,
          "samples_us": [
            424.478,
            414.038,
            408.759
          ]
        },
        "candidate": {
          "min_us": 680.629,
          "median_us": 681.461,
          "p95_us": 709.3511000000001,
          "mean_us": 691.5133333333333,
          "samples_us": [
            712.45,
            681.461,
            680.629
          ]
        },
        "control_b": {
          "min_us": 409.901,
          "median_us": 411.553,
          "p95_us": 423.62739999999997,
          "mean_us": 415.47433333333333,
          "samples_us": [
            424.969,
            411.553,
            409.901
          ]
        }
      },
      {
        "repetition": 3,
        "control_a": {
          "min_us": 409.35,
          "median_us": 415.742,
          "p95_us": 422.8196,
          "mean_us": 416.23266666666666,
          "samples_us": [
            423.606,
            415.742,
            409.35
          ]
        },
        "candidate": {
          "min_us": 675.7,
          "median_us": 677.964,
          "p95_us": 708.5856,
          "mean_us": 688.5506666666668,
          "samples_us": [
            711.988,
            677.964,
            675.7
          ]
        },
        "control_b": {
          "min_us": 410.481,
          "median_us": 411.393,
          "p95_us": 419.3823,
          "mean_us": 414.048,
          "samples_us": [
            420.27,
            411.393,
            410.481
          ]
        }
      },
      {
        "repetition": 4,
        "control_a": {
          "min_us": 412.015,
          "median_us": 412.045,
          "p95_us": 421.1332,
          "mean_us": 415.401,
          "samples_us": [
            422.143,
            412.045,
            412.015
          ]
        },
        "candidate": {
          "min_us": 679.508,
          "median_us": 680.258,
          "p95_us": 704.8118,
          "mean_us": 689.102,
          "samples_us": [
            707.54,
            680.258,
            679.508
          ]
        },
        "control_b": {
          "min_us": 411.424,
          "median_us": 412.385,
          "p95_us": 418.9496,
          "mean_us": 414.496,
          "samples_us": [
            419.679,
            412.385,
            411.424
          ]
        }
      },
      {
        "repetition": 5,
        "control_a": {
          "min_us": 412.125,
          "median_us": 415.621,
          "p95_us": 423.93519999999995,
          "mean_us": 417.53499999999997,
          "samples_us": [
            424.859,
            412.125,
            415.621
          ]
        },
        "candidate": {
          "min_us": 679.407,
          "median_us": 679.778,
          "p95_us": 708.0911,
          "mean_us": 690.1406666666667,
          "samples_us": [
            711.237,
            679.407,
            679.778
          ]
        },
        "control_b": {
          "min_us": 410.391,
          "median_us": 415.39,
          "p95_us": 418.1683,
          "mean_us": 414.75266666666664,
          "samples_us": [
            418.477,
            415.39,
            410.391
          ]
        }
      },
      {
        "repetition": 6,
        "control_a": {
          "min_us": 408.688,
          "median_us": 413.828,
          "p95_us": 426.23539999999997,
          "mean_us": 416.71,
          "samples_us": [
            427.614,
            413.828,
            408.688
          ]
        },
        "candidate": {
          "min_us": 677.082,
          "median_us": 684.948,
          "p95_us": 703.7112000000001,
          "mean_us": 689.2753333333334,
          "samples_us": [
            705.796,
            684.948,
            677.082
          ]
        },
        "control_b": {
          "min_us": 409.7,
          "median_us": 412.335,
          "p95_us": 420.1794,
          "mean_us": 414.36199999999997,
          "samples_us": [
            421.051,
            412.335,
            409.7
          ]
        }
      },
      {
        "repetition": 7,
        "control_a": {
          "min_us": 409.6,
          "median_us": 412.756,
          "p95_us": 423.47769999999997,
          "mean_us": 415.675,
          "samples_us": [
            424.669,
            412.756,
            409.6
          ]
        },
        "candidate": {
          "min_us": 673.796,
          "median_us": 677.884,
          "p95_us": 704.5563999999999,
          "mean_us": 686.4,
          "samples_us": [
            707.52,
            677.884,
            673.796
          ]
        },
        "control_b": {
          "min_us": 413.417,
          "median_us": 419.067,
          "p95_us": 422.21430000000004,
          "mean_us": 418.34933333333333,
          "samples_us": [
            422.564,
            413.417,
            419.067
          ]
        }
      },
      {
        "repetition": 8,
        "control_a": {
          "min_us": 408.598,
          "median_us": 412.946,
          "p95_us": 423.9737,
          "mean_us": 415.581,
          "samples_us": [
            425.199,
            412.946,
            408.598
          ]
        },
        "candidate": {
          "min_us": 677.914,
          "median_us": 680.569,
          "p95_us": 708.0352,
          "mean_us": 689.8566666666667,
          "samples_us": [
            711.087,
            680.569,
            677.914
          ]
        },
        "control_b": {
          "min_us": 408.528,
          "median_us": 412.756,
          "p95_us": 425.58639999999997,
          "mean_us": 416.09866666666665,
          "samples_us": [
            427.012,
            412.756,
            408.528
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
