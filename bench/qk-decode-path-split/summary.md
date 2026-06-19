# Decode large/small path split

- commit: `c4fa1c145`
- large path: `NO_READY_HCQ_ARTIFACT__SOURCE_IMPORT_OR_RENDERER_PROJECT_LEVEL`
- small path: `DONE_AS_RESEARCH_FLAG__NOT_PARITY_PATH`

## Large Path

- llama source family found: `True`
- llama build objects found: `True`
- ready HCQ code-object family found: `False`
- decision: no Tensile-like decode artifact was found; parity-scale decode is source-import or renderer/scheduler project work.

## Small Path

- W==D speedup range: `1.051x` to `1.063x`
- dNLL: `0.002887` over `160` tokens
- default changed: `False`
- decision: q8 route is done enough as a research flag; it is not the parity path.
