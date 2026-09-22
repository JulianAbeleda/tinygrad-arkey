# Decode context 2048 qualification

Ordinary generated Flash measures4.456750ms/token median,4.343116ms observed minimum and224.379 tok/s over two continuous three-token windows after excluded capture/warmup. Matched llama reports4.500439ms/token average (223.710 tok/s). Generated median is0.043690ms (0.97%) faster than llama's average, and its minimum is0.157323ms faster. Metrics remain explicitly non-identical aggregates (generated median/min versus llama average).

The generated run uses the ordinary Flash route with captured renderer source/binary census and token evidence. GPU stayed P0 at2572MHz with no hardware, thermal, or software throttle reason during both windows. The harness's GPU-state schema does not record memory-used bytes, so bounded-memory remains an open cell rather than inferred from this run.
