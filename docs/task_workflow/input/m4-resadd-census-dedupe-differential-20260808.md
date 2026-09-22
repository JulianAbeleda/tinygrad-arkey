# M4 resadd census dedupe: open-arm token differential (normalized seen-key vs pre-dedupe)

Date: 2026-08-08
Branch: `nvidia-bringup-20260731`. Context: census over-emission fix for the open arm
(production residual fold active). Baseline open-arm census was 32,906 kernels/token
(18.5x bloat vs the 948-kernel closed arm); the candidate fix
(`2df55de10`, normalized seen-set key dropping per-parent arg positions 0 and n-1)
collapsed the census to 2,246 kernels/token but changed the measured token stream.

## Protocol

Same-session, lock-held (`flock -w 600 /tmp/gpu-bench.lock`), 20G cgroup cap,
Qwen3-8B-Q4_K_M, d512, temperature 0, chunk_size 32. Open arm = module override
`mrp._DECODE_Q4K_EPILOGUE_RESADD_PROMOTED_TARGETS = frozenset({("NV","sm_120")})`.
Pin (closed/record arms, `m4-resadd-s4-gate-run-record-20260806.md`): sha
`227ad3ce9621f2c382cc722a3c2f1677637d3e3f2bfbf37d6ca652f98880eb4e`, first token `271`,
census 948 kernels / epi 0 / legacy 72 / copy class 1 / resadd 72.

## Results

| arm / tree | census kernels/token | first token | verdict |
| --- | --- | --- | --- |
| closed, HEAD `2df55de10` (with fix) | 948 | `271` 3/3 | pin matches, unchanged |
| open, PRE-dedupe (`6b0003880`, via `/tmp/m4_base_wt`) | 32,906 | `[271, 474, 330]` | first token equals pin 271 |
| open, HEAD `2df55de10` (with fix) | 2,246 (epi 36 / legacy 36 / resadd 36) | `60231` 3/3 | diverges from pin |

Open-arm fold was token-exact at the pre-dedupe commit (`271` first token), so the
normalized seen-key (dropping slots 0/n-1 from the emission seen-set) is what breaks the
stream: it over-collapses LIVE per-parent bindings, not only dead re-emissions. The
candidate fix is therefore NOT promotable as-is; the seen-key must be refined (e.g., key
by the buffers the nested body's kernels actually reference, or normalize only positions
whose kernels never reference them) so every live parent binding survives while dead
duplicate chain emissions still collapse.

## Evidence

`/tmp/m4_base_open_tok.py`, `/tmp/m4_base_open_tok.log` (pre-dedupe open first3),
`/tmp/m4_fix_probe.log` (post-fix both arms, census + pins), `/tmp/m4_dedupe_head.log`
(differential item dump), `/tmp/m4_key_probe.log` (arg variance across parents).
