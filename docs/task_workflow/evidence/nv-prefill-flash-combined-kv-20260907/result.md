# NV pp512 combined-KV Flash handoff closure (2026-09-07)

A typed experimental Flash binding consumed the original `assigned_kv` AFTER owner with the exact persistent-cache layout: K plane offset 0, V plane offset `Hkv*capacity*Hd`, and per-head stride `capacity*Hd`. The generated fragment address contract was exact and default packed behavior remained offset zero.

The full selected Qwen3-8B smoke passed: token 198, canonical 252/252 generated weight arguments, zero overlays, exact logits and every deep stage across three replay cycles. The candidate removed the two split E4096 K/V pack families (36 each), but scheduling materialized one combined E8192 family (36), so it did not establish the required zero-copy handoff.

Matched fresh-process A/B/C, warmups 9, rounds 9, replay 5:

- control A median/min: 47.707827 / 47.620403 ms
- combined candidate median/min: 48.243861 / 48.171274 ms
- control C median/min: 48.033532 / 47.922753 ms
- control midpoint median: 47.8706795 ms
- candidate regression: 0.3731815 ms median
- control midpoint min: 47.771578 ms
- candidate min regression: 0.399696 ms

The route is rejected and all production changes are reverted. The result localizes the issue: merely replacing two compacting cache-view copies with one combined materialization loses, despite one fewer launch. A useful direct cache handoff requires an opaque PARAM boundary that accepts the original strided AFTER owner without materializing it. The next higher-value action is folding the 9.984 us/layer Q RoPE+fp16 E2048 producer into generated Flash Q loading.
