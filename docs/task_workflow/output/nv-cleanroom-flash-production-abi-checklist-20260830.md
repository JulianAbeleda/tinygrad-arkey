# Clean-room Flash production ABI checklist

The standalone oracle is an independent numerical contract, not a drop-in
production tensor ABI.

## Production contract observed in `TransformerBlock._attention`

- Projection inputs are `(B,T,dim)`; Q is reshaped to `(B,Hq,T,Hd)`, K/V to
  `(B,Hkv,T,Hd)`.
- Qwen3-8B pp512 uses `B=1`, `T=512`, `Hq=32`, `Hkv=8`, `Hd=128`.
- The model stores Q/K/V in fp16/fp32 according to the active route; the Flash
  result is reshaped `(B,Hq,T,Hd)` and cast to `q.dtype` before O projection.
- K/V are written to `cache_kv` with leading KV selector, giving the logical
  cache shape `(2,B,Hkv,max_context,Hd)`; Flash reads the assigned cache-after
  store, not the transient projection buffer.
- Non-ring reads use KV extent `start_pos + T`; full-ring reads use the whole
  `max_context` extent. `start_pos` is an absolute position in fill/decode and
  a wrapped write slot in ring mode.
- Q RoPE uses frequencies at `start_pos:start_pos+T`. K RoPE uses the same
  absolute positions when stored roped; rope-at-read stores unroped K and
  rotates from the gathered ring frequency table at read time.
- The mask is causal-lower-right, materialized over concrete max-context rows
  and sliced to `(T,start_pos+T)`. It is not the upper-left `is_causal` mask.

## Mismatch warnings for the standalone fixture

- Fixture arrays are head-major `[H,S,D]` fp32, while production tensors are
  `[B,H,T,D]` and commonly fp16 at the Flash boundary.
- Fixture K/V contain all 512 positions directly; production K/V come from
  cache-after state and may have `start_pos` offset, ring wrapping, or a
  shorter live extent.
- Fixture uses `mask[q,k] = (k <= q)` for equal-length lower-right semantics;
  production must use `k <= start_pos + q` over the cache extent.
- Fixture frequencies are per-position `[S,D]`; production frequencies may be
  absolute-position slices or ring-gathered slot-relative rows.
- Fixture output is `[Hq,S,D]`; production Flash output must be
  `[B,Hq,T,D]`, then `(B,T,Hq*D)` for the downstream O projection.

## Required adapter checks

1. Assert B/T/Hq/Hkv/Hd and contiguous strides at the candidate boundary.
2. Assert cache offset and ring mode explicitly; never infer them from buffer size.
3. Compare full output and finite values, preserving canaries around logical
   output and cache writes.
4. Run both `start_pos=0` and a nonzero offset before any promotion.
