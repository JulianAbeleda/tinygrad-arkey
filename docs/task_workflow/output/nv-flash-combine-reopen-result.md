# NV Flash combine reopen result

## Outcome

The current installed endpoint authority is 4.060523 ms/token / 246.274 tok/s.
The older 4.094502-ms / 244.230-tok/s checkpoint is pre-promotion history and
must not be used for current projections.

The apparent 11.391-us/token Flash-combine debt is not an isolated combine-body
debt. A single native CUDA binary compiled and timed llama's retained
`flash_attn_combine_results<128>` and tinygrad's two S6 combine spellings under
the same launch and event protocol:

| six-part combine | median native launch | registers | barriers |
|---|---:|---:|---:|
| tinygrad shared weights | 1.796896 us | 29 | 1 |
| tinygrad register weights | **1.787891 us** | 32 | 0 |
| llama retained combine | 1.818675 us | 40 | 1 |

Tinygrad's register-weight combine is about 0.031 us/launch faster in this
matched primitive. The separate Python/TinyJit microgate is not a cross-runtime
authority because host synchronization dominates its roughly 116-us result.

## Consequence

Do not port llama's combine arithmetic or charge 11.391 us/token as an
available kernel-body recovery. The production-row ordering must arise after
the primitive: score-to-combine handoff state, graph placement, cache/input
conditioning, profiler attribution, or a difference in the charged output
contract. The next causal test must replay each runtime's actual predecessor
and time the combine kernel itself, then reconcile kernel duration with the
score-start-to-combine-end island span.

No wall recovery or tok/s movement is booked by this diagnostic.

## Production conversion and NCU closure

The installed graph was then captured with the promoted S6 score name and its
real eleven-kernel predecessor interval. Twenty settled replay samples give:

| combine state | median |
|---|---:|
| hot repeat | 1.312 us |
| immediately after real score | 1.312 us |
| after complete production interval | 1.344 us |
| complete interval, then combine reheat | 1.312 us |
| score start through combine end | 5.824 us |

Production conditioning therefore contributes only 0.032 us/layer, or about
1.15 us/token. It does not explain the cross-runtime production-row ordering.

The same replay refreshes tinygrad's complete Flash conversion ratio. Its hot
score-to-combine-end island is 5.824 us/layer; the installed production rows
are 194.048 + 48.448 = 242.496 us/token, or 6.736 us/layer. The resulting
installed hot-to-production drop is approximately **15.7%**. The historical
36.35% value is superseded. The retained llama 17--19% band suggests likely
conversion parity, but remains protocol-unmatched and therefore provisional.

Matched single-launch NCU removes CUDA-event launch overhead and closes the
body comparison even more strongly:

| combine | NCU duration | dynamic instructions | global-load sectors |
|---|---:|---:|---:|
| tinygrad register weights | **1.888 us** | **11,648** | 5,952 |
| tinygrad shared weights | 2.080 us | 8,960 | 4,800 |
| llama | 2.528 us | 32,672 | 3,136 |

Tinygrad trades more load sectors for far fewer instructions and finishes
faster. The old 48.448-versus-37.057-us/token row difference is an unmatched
profile/accounting observation, not an optimization pool. Flash-combine work
is closed as the path to 248 tok/s.

The honest remaining Flash lever is the installed score row. Its observed
31.100-us/token difference is large enough in accounting to move the current
4.060523-ms endpoint to about 4.029423 ms/token / 248.174 tok/s, but it still
requires a positive production-conditioned score construction before booking.

## Evidence

- `docs/task_workflow/evidence/nv-flash-combine-reopen-20260827/native-ab-r9.json`
- `docs/task_workflow/evidence/nv-flash-combine-reopen-20260827/conversion-r20.json`
- `docs/task_workflow/evidence/nv-flash-combine-reopen-20260827/ncu-tiny_register.csv`
- `docs/task_workflow/evidence/nv-flash-combine-reopen-20260827/ncu-tiny_shared.csv`
- `docs/task_workflow/evidence/nv-flash-combine-reopen-20260827/ncu-llama.csv`
- `docs/task_workflow/evidence/nv-flash-combine-reopen-20260827/llama-timing-r9.txt`
- `docs/task_workflow/evidence/nv-flash-combine-reopen-20260827/tiny-s6-register-microgate-r9.json`
- `extra/llm_research/decode/nv_flash_combine_native_ab.py`
- `extra/llm_research/microbench/llama_fattn_combine_iso.cu`
