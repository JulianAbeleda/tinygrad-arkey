# Fresh pp512 tinygrad/llama endpoint bracket

The final endpoint authority is a tinygrad R9 / llama R9 / tinygrad R9 bracket
on the current machine state.

| arm | settled median |
|---|---:|
| tinygrad A | 35.152125 ms |
| llama | 35.334424 ms |
| tinygrad C | 35.221056 ms |

The mean tinygrad median is 35.186591 ms, 0.147834 ms (0.42%) faster than the
intervening llama median. Both tinygrad arms select token 198 and their full
logits are bit-exact. Llama sample zero is retained but excluded under the
predeclared first-use graph-setup convention.

`cross-runtime-bracket.json` is the machine-readable comparison and
`llama-unprofiled-r9.json` is the raw llama output.
