# Whole-Repo Principles Cleanup — Inventory

HEAD `3772b0eed` · 2436 tracked · 1923 project rows · 466 vendor files in 19 dirs

## By recommendation

- **ARCHIVE_PROVENANCE**: 1469
- **KEEP_DOC_AUTHORITY**: 182
- **KEEP_TEST**: 153
- **KEEP_LIVE_TOOLING**: 58
- **KEEP_CORE**: 29
- **KEEP_LIBRARY_HELPER**: 27
- **IGNORE_EXTERNAL_VENDOR**: 19
- **DELETE**: 5

## By subsystem

| subsystem | ARCHIVE_PROVENANCE | DELETE | IGNORE_EXTERNAL_VENDOR | KEEP_CORE | KEEP_DOC_AUTHORITY | KEEP_LIBRARY_HELPER | KEEP_LIVE_TOOLING | KEEP_TEST |
|---|---|---|---|---|---|---|---|---|
| audit_tooling |  |  |  |  |  |  | 5 |  |
| bench_artifact | 618 |  |  |  |  |  |  |  |
| core_runtime |  |  |  | 5 |  |  |  |  |
| docs | 671 |  |  |  | 149 |  |  |  |
| evaluator_search_ledger |  |  |  |  |  |  | 19 |  |
| extra_qk_tooling | 170 | 5 |  |  |  | 27 | 34 |  |
| root_config |  |  |  | 24 |  |  |  |  |
| structure | 10 |  |  |  | 33 |  |  |  |
| test |  |  |  |  |  |  |  | 153 |
| vendor |  |  | 19 |  |  |  |  |  |

## DELETE candidates (5) — proof: no importer/doc/test/ledger ref

- `extra/qk_amd_bb5a2_pipelined_dataflow_probe.py`
- `extra/qk_amd_bb5a2_real_lowering_integration_probe.py`
- `extra/qk_amd_bb5a8_authority_kernel_capture_probe.py`
- `extra/qk_amd_bb5a8_tensile_mapping_probe.py`
- `extra/qk_decode_unknown_bucket_source_map.py`
