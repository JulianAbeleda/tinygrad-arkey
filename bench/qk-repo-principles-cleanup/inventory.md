# Whole-Repo Principles Cleanup — Inventory

HEAD `460785c73` · 2253 tracked · 1708 project rows · 466 vendor files in 19 dirs

## By recommendation

- **ARCHIVE_PROVENANCE**: 1311
- **KEEP_DOC_AUTHORITY**: 182
- **KEEP_TEST**: 115
- **KEEP_LIVE_TOOLING**: 64
- **KEEP_CORE**: 29
- **IGNORE_EXTERNAL_VENDOR**: 19
- **KEEP_LIBRARY_HELPER**: 7

## By subsystem

| subsystem | ARCHIVE_PROVENANCE | IGNORE_EXTERNAL_VENDOR | KEEP_CORE | KEEP_DOC_AUTHORITY | KEEP_LIBRARY_HELPER | KEEP_LIVE_TOOLING | KEEP_TEST |
|---|---|---|---|---|---|---|---|
| audit_tooling |  |  |  |  |  | 5 |  |
| bench_artifact | 618 |  |  |  |  |  |  |
| core_runtime |  |  | 5 |  |  |  |  |
| docs | 671 |  |  | 149 |  |  |  |
| evaluator_search_ledger |  |  |  |  |  | 19 |  |
| extra_qk_tooling | 12 |  |  |  | 7 | 40 |  |
| root_config |  |  | 24 |  |  |  |  |
| structure | 10 |  |  | 33 |  |  |  |
| test |  |  |  |  |  |  | 115 |
| vendor |  | 19 |  |  |  |  |  |

## DELETE candidates (0) — proof: no importer/doc/test/ledger ref

