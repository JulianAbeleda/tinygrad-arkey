# NV installed-island campaign evidence index (2026-08-22)

Branch: `nvidia-bringup-20260731`
HEAD: `6570abc025514273faa100c66b979e531585a1e1`

Reconstruct every retained file with:

```text
cd docs/task_workflow/evidence/nv-installed-islands-20260822
find phase0 phase1 phase2 phase3 phase4 phase5 phase6 phase7 phase8 phase9 phase10 -name sha256.txt -print -exec sha256sum -c {} \;
```

| phase | evidence | report |
| --- | --- | --- |
| 0 | `phase0/provenance.json` | `../output/nv-installed-islands-phase0-provenance-result-20260822.md` |
| 1 | `phase1/tg-control-a.json`, `tg-control-b.json`, `tg-profile.json`, `tg-profile.jsonl`, `tg-profile-ledger.json`, `llama-endpoint.json`, `llama-pdl0.nsys-rep`, `llama-pdl0.sqlite`, `llama-pdl0-dag.json`, `llama-graph-dump.txt`, `wall-summary.json` | `../output/nv-installed-islands-phase1-endpoint-result-20260822.md` |
| 2 | `phase2/island-manifest.json` | `../output/nv-installed-islands-phase2-manifest-result-20260822.md` |
| 3 | `phase3/qk-p-decomposition.json` | `../output/nv-installed-islands-phase3-qk-result-20260822.md` |
| 4 | `phase4/o-decomposition.json`, `o-exact.*`, `o-ncu.*` | `../output/nv-installed-islands-phase4-o-result-20260822.md` |
| 5 | `phase5/ffn-decomposition.json`, `gateup-exact.*`, `downq4-exact.*`, `downq6-exact.*`, `q4k-down-capture.json` | `../output/nv-installed-islands-phase5-ffn-result-20260822.md` |
| 6 | `phase6/flash-decomposition.json`, `flashscore-exact.*`, `combine-exact.*`, `flashscore-ncu.*`, `combine-ncu.*` | `../output/nv-installed-islands-phase6-flash-result-20260822.md` |
| 7 | `phase7/kv-projection-capture.json`, `kv-projection-partition.json`, `qproj-capture.json`, `qproj-decomposition.json`, `q4g3kv-exact.*`, `q4coopkv-exact.*`, `q6q8k-exact.*`, `q6v4-exact.*`, `r8completion-exact.*`, `*-hcq.json` | `../output/nv-installed-islands-phase7-kv-projection-result-20260822.md` |
| 8 | `phase8/vocab-exact.*`, `vocab-hcq.json`, `tail-capture.json`, `tail-decomposition.json`, `tail-p-map.json`, `*.exact.*`, `*.hcq.json` | `../output/nv-installed-islands-phase8-vocab-result-20260822.md` |
| 9 | `phase9/norm-capture.json`, `norm-decomposition.json`, `norm-p-map.json`, `*.exact.*`, `*.hcq.json` | `../output/nv-installed-islands-phase9-norm-result-20260822.md` |
| 10 | `phase10/island-ledger.json` | `../output/nv-installed-islands-phase10-ledger-result-20260822.md` |

Analysis tools added under `extra/llm_research/decode/`:

```text
nv_installed_island_phase1_ledger.py
nv_installed_island_manifest.py
nv_qk_installed_p_decomposition.py
nv_hcq_dispatch_slope_general.py
nv_cubin_ncu_launcher.py
nv_installed_island_phase8_tail.py
```

No production, renderer, scheduler, runtime, or model file was changed by this
campaign. The seven tracked modifications (`tinygrad/engine/jit.py`,
`tinygrad/llm/decode_routes.py`, `tinygrad/renderer/cuda.py`,
`tinygrad/runtime/graph/hcq.py`, `tinygrad/runtime/ops_nv.py`,
`extra/llm_research/decode/full_token_dag_capture.py`,
`extra/llm_research/decode/nv_norm_native_wall_ab.py`) are pre-existing user
work and were left untouched.
