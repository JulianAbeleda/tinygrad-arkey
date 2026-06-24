# qk-prefill-clock-dpm-authority

Artifacts for the prefill clock/DPM authority project. Scope:
`docs/archive/prefill-clock-dpm-authority-scope-20260619.md`. Driver: `extra/qk_prefill_clock_dpm_authority.py`.

- `supported_controls.json` — P0 control inventory on this RX 7900 XTX (amd-smi absent; rocm-smi+sysfs;
  perf-levels auto/high/manual/profile_peak/profile_standard/profile_min_sclk; determinism via
  `rocm-smi --setperfdeterminism`; no OverDrive ranges; DPM sclk{500,~1498,2304}/mclk{96,456,772,1249}).
- `telemetry.jsonl`, `dpm_probe_matrix.json`, `prefill_clock_matrix.json` — produced by running P1-P3.

Done = (A) a telemetry-verified clock lane holds for both WMMA and Tensile, OR (B) proof this card can't be
pinned -> all prefill claims become telemetry-binned by measured sclk.
