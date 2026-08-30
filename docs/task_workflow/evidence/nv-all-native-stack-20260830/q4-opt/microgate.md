# Phase 1 result

Status: **FAIL**

```json
{
  "schema": "tinygrad.nv-packed-qk-q8-streamk-gate-phase1.v2",
  "status": "FAIL",
  "threshold": "PASS iff mandatory gates pass, candidate min+median beat both controls, and median <=250 us",
  "commands": [
    "flock -w 1200 /tmp/gpu-bench.lock env PYTHONPATH=. DEV=NV .venv/bin/python /home/ubuntu/tinygrad-arkey/extra/llm_research/prefill/nv_packed_qk_q8_gate_qualify.py --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --out docs/task_workflow/evidence/nv-all-native-stack-20260830/q4-opt/microgate.json --rounds 9"
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
      "input_pointer": 137919238144,
      "weight_pointer": 137657057280,
      "output_pointer": 137860481024,
      "workspace_pointer": 137925492736,
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
      "input_pointer": 138005192704,
      "weight_pointer": 137657057280,
      "output_pointer": 137980018688,
      "workspace_pointer": 138011475968,
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
      "min_us": 406.785,
      "median_us": 411.894,
      "p95_us": 428.776,
      "mean_us": 414.5642962962963,
      "samples_us": [
        429.778,
        417.475,
        411.894,
        414.108,
        410.892,
        413.147,
        413.146,
        410.261,
        408.578,
        491.524,
        421.773,
        414.349,
        412.235,
        411.113,
        414.62,
        414.639,
        411.834,
        411.995,
        427.854,
        417.365,
        411.053,
        413.548,
        411.744,
        410.182,
        410.051,
        409.44,
        410.562,
        429.508,
        417.034,
        414.71,
        410.752,
        416.564,
        410.321,
        409.069,
        410.342,
        409.61,
        428.776,
        420.861,
        411.954,
        412.496,
        409.129,
        409.821,
        408.448,
        406.845,
        410.08,
        432.934,
        418.106,
        413.428,
        408.859,
        410.582,
        411.324,
        412.726,
        407.646,
        408.307,
        426.141,
        415.961,
        412.576,
        408.258,
        415.582,
        409.83,
        410.341,
        408.979,
        406.785,
        427.544,
        418.236,
        413.828,
        412.094,
        409.68,
        413.638,
        408.688,
        408.247,
        409.91,
        428.005,
        415.511,
        412.866,
        409.65,
        410.201,
        409.57,
        414.108,
        409.509,
        408.578
      ]
    },
    "candidate": {
      "min_us": 637.518,
      "median_us": 644.652,
      "p95_us": 684.066,
      "mean_us": 651.1331358024692,
      "samples_us": [
        678.135,
        649.08,
        648.919,
        640.695,
        640.364,
        643.299,
        641.706,
        643.56,
        641.756,
        830.682,
        657.746,
        654.48,
        644.421,
        649.511,
        645.534,
        640.414,
        643.851,
        643.87,
        678.896,
        649.371,
        642.178,
        647.658,
        639.843,
        641.596,
        641.035,
        645.574,
        637.518,
        683.044,
        644.542,
        646.586,
        647.046,
        641.195,
        642.137,
        648.008,
        639.773,
        641.976,
        696.279,
        654.44,
        648.85,
        643.861,
        643.7,
        645.564,
        643.329,
        645.133,
        644.622,
        709.674,
        651.906,
        644.652,
        645.293,
        645.974,
        641.276,
        645.924,
        646.946,
        639.663,
        689.296,
        646.094,
        646.977,
        649.661,
        639.672,
        642.128,
        644.271,
        639.683,
        644.391,
        681.171,
        651.625,
        651.495,
        637.559,
        641.265,
        646.566,
        638.34,
        641.125,
        639.934,
        684.066,
        653.229,
        644.703,
        646.415,
        647.637,
        641.306,
        640.885,
        639.412,
        639.793
      ]
    },
    "control_b": {
      "min_us": 407.636,
      "median_us": 411.453,
      "p95_us": 427.744,
      "mean_us": 413.43967901234566,
      "samples_us": [
        426.03,
        417.806,
        413.167,
        414.188,
        409.84,
        409.45,
        411.423,
        408.738,
        409.58,
        427.744,
        415.461,
        411.874,
        412.305,
        415.931,
        409.159,
        411.133,
        409.941,
        415.892,
        433.194,
        414.97,
        411.413,
        408.258,
        412.766,
        412.886,
        409.91,
        411.333,
        408.898,
        426.482,
        414.028,
        410.762,
        412.886,
        410.272,
        414.98,
        411.504,
        409.079,
        411.594,
        424.218,
        418.998,
        412.576,
        408.218,
        410.421,
        407.796,
        411.093,
        408.708,
        418.026,
        427.784,
        415.121,
        412.946,
        411.613,
        410.923,
        412.585,
        409.47,
        410.733,
        408.278,
        431.542,
        414.289,
        411.814,
        410.702,
        410.262,
        412.786,
        410.232,
        410.842,
        408.488,
        430.399,
        415.321,
        411.513,
        407.636,
        409.98,
        409.881,
        416.283,
        410.803,
        408.518,
        427.414,
        413.437,
        410.101,
        410.732,
        408.478,
        411.453,
        408.858,
        413.668,
        408.798
      ]
    },
    "r9": [
      {
        "repetition": 0,
        "control_a": {
          "min_us": 408.578,
          "median_us": 413.146,
          "p95_us": 424.8568,
          "mean_us": 414.3643333333333,
          "samples_us": [
            429.778,
            417.475,
            411.894,
            414.108,
            410.892,
            413.147,
            413.146,
            410.261,
            408.578
          ]
        },
        "candidate": {
          "min_us": 640.364,
          "median_us": 643.299,
          "p95_us": 666.513,
          "mean_us": 647.5015555555556,
          "samples_us": [
            678.135,
            649.08,
            648.919,
            640.695,
            640.364,
            643.299,
            641.706,
            643.56,
            641.756
          ]
        },
        "control_b": {
          "min_us": 408.738,
          "median_us": 411.423,
          "p95_us": 422.74039999999997,
          "mean_us": 413.358,
          "samples_us": [
            426.03,
            417.806,
            413.167,
            414.188,
            409.84,
            409.45,
            411.423,
            408.738,
            409.58
          ]
        }
      },
      {
        "repetition": 1,
        "control_a": {
          "min_us": 411.113,
          "median_us": 414.349,
          "p95_us": 463.6236,
          "mean_us": 422.6757777777778,
          "samples_us": [
            491.524,
            421.773,
            414.349,
            412.235,
            411.113,
            414.62,
            414.639,
            411.834,
            411.995
          ]
        },
        "candidate": {
          "min_us": 640.414,
          "median_us": 645.534,
          "p95_us": 761.5075999999999,
          "mean_us": 667.8343333333333,
          "samples_us": [
            830.682,
            657.746,
            654.48,
            644.421,
            649.511,
            645.534,
            640.414,
            643.851,
            643.87
          ]
        },
        "control_b": {
          "min_us": 409.159,
          "median_us": 412.305,
          "p95_us": 423.0188,
          "mean_us": 414.3822222222222,
          "samples_us": [
            427.744,
            415.461,
            411.874,
            412.305,
            415.931,
            409.159,
            411.133,
            409.941,
            415.892
          ]
        }
      },
      {
        "repetition": 2,
        "control_a": {
          "min_us": 409.44,
          "median_us": 411.053,
          "p95_us": 423.6584,
          "mean_us": 413.5332222222222,
          "samples_us": [
            427.854,
            417.365,
            411.053,
            413.548,
            411.744,
            410.182,
            410.051,
            409.44,
            410.562
          ]
        },
        "candidate": {
          "min_us": 637.518,
          "median_us": 642.178,
          "p95_us": 667.086,
          "mean_us": 647.0743333333334,
          "samples_us": [
            678.896,
            649.371,
            642.178,
            647.658,
            639.843,
            641.596,
            641.035,
            645.574,
            637.518
          ]
        },
        "control_b": {
          "min_us": 408.258,
          "median_us": 411.413,
          "p95_us": 425.9044,
          "mean_us": 413.7364444444445,
          "samples_us": [
            433.194,
            414.97,
            411.413,
            408.258,
            412.766,
            412.886,
            409.91,
            411.333,
            408.898
          ]
        }
      },
      {
        "repetition": 3,
        "control_a": {
          "min_us": 409.069,
          "median_us": 410.752,
          "p95_us": 424.5184,
          "mean_us": 414.21222222222224,
          "samples_us": [
            429.508,
            417.034,
            414.71,
            410.752,
            416.564,
            410.321,
            409.069,
            410.342,
            409.61
          ]
        },
        "candidate": {
          "min_us": 639.773,
          "median_us": 644.542,
          "p95_us": 669.0296,
          "mean_us": 648.2563333333334,
          "samples_us": [
            683.044,
            644.542,
            646.586,
            647.046,
            641.195,
            642.137,
            648.008,
            639.773,
            641.976
          ]
        },
        "control_b": {
          "min_us": 409.079,
          "median_us": 411.594,
          "p95_us": 421.88120000000004,
          "mean_us": 413.5096666666667,
          "samples_us": [
            426.482,
            414.028,
            410.762,
            412.886,
            410.272,
            414.98,
            411.504,
            409.079,
            411.594
          ]
        }
      },
      {
        "repetition": 4,
        "control_a": {
          "min_us": 406.845,
          "median_us": 410.08,
          "p95_us": 425.61,
          "mean_us": 413.1566666666667,
          "samples_us": [
            428.776,
            420.861,
            411.954,
            412.496,
            409.129,
            409.821,
            408.448,
            406.845,
            410.08
          ]
        },
        "candidate": {
          "min_us": 643.329,
          "median_us": 645.133,
          "p95_us": 679.5434,
          "mean_us": 651.7531111111111,
          "samples_us": [
            696.279,
            654.44,
            648.85,
            643.861,
            643.7,
            645.564,
            643.329,
            645.133,
            644.622
          ]
        },
        "control_b": {
          "min_us": 407.796,
          "median_us": 411.093,
          "p95_us": 422.13,
          "mean_us": 413.33933333333334,
          "samples_us": [
            424.218,
            418.998,
            412.576,
            408.218,
            410.421,
            407.796,
            411.093,
            408.708,
            418.026
          ]
        }
      },
      {
        "repetition": 5,
        "control_a": {
          "min_us": 407.646,
          "median_us": 411.324,
          "p95_us": 427.0028,
          "mean_us": 413.76800000000003,
          "samples_us": [
            432.934,
            418.106,
            413.428,
            408.859,
            410.582,
            411.324,
            412.726,
            407.646,
            408.307
          ]
        },
        "candidate": {
          "min_us": 639.663,
          "median_us": 645.924,
          "p95_us": 686.5668,
          "mean_us": 652.3675555555556,
          "samples_us": [
            709.674,
            651.906,
            644.652,
            645.293,
            645.974,
            641.276,
            645.924,
            646.946,
            639.663
          ]
        },
        "control_b": {
          "min_us": 408.278,
          "median_us": 411.613,
          "p95_us": 422.7188,
          "mean_us": 413.2725555555556,
          "samples_us": [
            427.784,
            415.121,
            412.946,
            411.613,
            410.923,
            412.585,
            409.47,
            410.733,
            408.278
          ]
        }
      },
      {
        "repetition": 6,
        "control_a": {
          "min_us": 406.785,
          "median_us": 410.341,
          "p95_us": 422.069,
          "mean_us": 412.717,
          "samples_us": [
            426.141,
            415.961,
            412.576,
            408.258,
            415.582,
            409.83,
            410.341,
            408.979,
            406.785
          ]
        },
        "candidate": {
          "min_us": 639.672,
          "median_us": 644.391,
          "p95_us": 673.442,
          "mean_us": 649.1303333333333,
          "samples_us": [
            689.296,
            646.094,
            646.977,
            649.661,
            639.672,
            642.128,
            644.271,
            639.683,
            644.391
          ]
        },
        "control_b": {
          "min_us": 408.488,
          "median_us": 410.842,
          "p95_us": 424.64079999999996,
          "mean_us": 413.43966666666665,
          "samples_us": [
            431.542,
            414.289,
            411.814,
            410.702,
            410.262,
            412.786,
            410.232,
            410.842,
            408.488
          ]
        }
      },
      {
        "repetition": 7,
        "control_a": {
          "min_us": 408.247,
          "median_us": 412.094,
          "p95_us": 423.82079999999996,
          "mean_us": 413.54055555555556,
          "samples_us": [
            427.544,
            418.236,
            413.828,
            412.094,
            409.68,
            413.638,
            408.688,
            408.247,
            409.91
          ]
        },
        "candidate": {
          "min_us": 637.559,
          "median_us": 641.265,
          "p95_us": 669.3526,
          "mean_us": 647.6755555555555,
          "samples_us": [
            681.171,
            651.625,
            651.495,
            637.559,
            641.265,
            646.566,
            638.34,
            641.125,
            639.934
          ]
        },
        "control_b": {
          "min_us": 407.636,
          "median_us": 410.803,
          "p95_us": 424.75260000000003,
          "mean_us": 413.37044444444444,
          "samples_us": [
            430.399,
            415.321,
            411.513,
            407.636,
            409.98,
            409.881,
            416.283,
            410.803,
            408.518
          ]
        }
      },
      {
        "repetition": 8,
        "control_a": {
          "min_us": 408.578,
          "median_us": 410.201,
          "p95_us": 423.0074,
          "mean_us": 413.1108888888889,
          "samples_us": [
            428.005,
            415.511,
            412.866,
            409.65,
            410.201,
            409.57,
            414.108,
            409.509,
            408.578
          ]
        },
        "candidate": {
          "min_us": 639.412,
          "median_us": 644.703,
          "p95_us": 671.7312000000001,
          "mean_us": 648.6051111111111,
          "samples_us": [
            684.066,
            653.229,
            644.703,
            646.415,
            647.637,
            641.306,
            640.885,
            639.412,
            639.793
          ]
        },
        "control_b": {
          "min_us": 408.478,
          "median_us": 410.732,
          "p95_us": 421.9156,
          "mean_us": 412.5487777777778,
          "samples_us": [
            427.414,
            413.437,
            410.101,
            410.732,
            408.478,
            411.453,
            408.858,
            413.668,
            408.798
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
