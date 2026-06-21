# Whole-Repo Principles Cleanup — Inventory

HEAD `e33a7975d` · 2468 tracked · 1613 project rows · 853 vendor files in 26 dirs

## By recommendation

- **ARCHIVE_PROVENANCE**: 923
- **KEEP_DOC_AUTHORITY**: 334
- **KEEP_TEST**: 260
- **KEEP_LIVE_TOOLING**: 48
- **KEEP_CORE**: 27
- **IGNORE_EXTERNAL_VENDOR**: 26
- **KEEP_LIBRARY_HELPER**: 21

## By subsystem

| subsystem | ARCHIVE_PROVENANCE | IGNORE_EXTERNAL_VENDOR | KEEP_CORE | KEEP_DOC_AUTHORITY | KEEP_LIBRARY_HELPER | KEEP_LIVE_TOOLING | KEEP_TEST |
|---|---|---|---|---|---|---|---|
| audit_tooling |  |  |  |  |  | 3 |  |
| bench_artifact | 427 |  |  |  |  |  |  |
| core_runtime |  |  | 5 |  |  |  |  |
| docs | 358 |  |  | 301 |  |  |  |
| evaluator_search_ledger |  |  |  |  |  | 18 |  |
| extra_qk_tooling | 128 |  |  |  | 21 | 27 |  |
| root_config |  |  | 22 |  |  |  |  |
| structure | 10 |  |  | 33 |  |  |  |
| test |  |  |  |  |  |  | 260 |
| vendor |  | 26 |  |  |  |  |  |

## DELETE candidates (0) — proof: no importer/doc/test/ledger ref

