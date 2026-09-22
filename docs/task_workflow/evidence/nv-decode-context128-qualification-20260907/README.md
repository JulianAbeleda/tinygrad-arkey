# Decode context 128 qualification

Ordinary generated decode selects SDPA at context128 and measures5.383666 ms/token median (min observed5.255469;185.747 tok/s) over two continuous three-token windows after excluded capture/warmup. Its rollout graph contains526 generated programs with526 admitted source/binary records; token streams are retained and GPU stayed P0 at2572MHz.

Matched llama (`llama-bench`, depth128, six generated tokens, five repetitions) reports241.090 tok/s, or4.181244ms/token. Run from the llama repository cwd because this instrumented build dumps a graph relative to cwd.

A default-off forced generated Flash diagnostic measures4.366347ms/token median (min4.248354;229.024 tok/s) with418 generated programs. It recovers1.017319ms/token versus ordinary SDPA but remains about0.185103ms/token behind llama. Therefore context128 currently fails parity: route policy is the dominant deficit, with a smaller Flash/kernel endpoint gap. No policy is promoted from this cell.
