# JSON RS Coverage Gate

- status: `fail`
- min selected train rows per category: `20`

| category | status | selected | accepted attempts | attempts | near miss | reason |
|---|---|---:|---:|---:|---:|---|
| `code` | `pass` | 20 | 36 | 544 | 100 |  |
| `compiler` | `fail` | 0 | 0 | 544 | 158 | selected_train_rows 0 < 20 |
| `string` | `pass` | 23 | 46 | 544 | 44 |  |

## Failures

- compiler: selected_train_rows 0 < 20
