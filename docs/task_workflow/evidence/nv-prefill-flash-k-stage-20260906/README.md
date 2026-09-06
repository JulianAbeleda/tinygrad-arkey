# NV pp512 generated Flash K-stage promotion

The generated NV prefill Flash route now groups four physical warps in one CTA and cooperatively stages each 16x128 K tile once. The conversion repair keeps Q, V, online softmax, FP32 accumulation, direct FP16 output, launch count, and graph ABI unchanged.

Two lowering defects had prevented the retained staging substrate from being valid: same-shaped K/V stages collapsed to one shared allocation, and shared fragment lowering omitted the second NV MMA call's eight-column offset. Allocation identity now includes `stage_generation`, and shared K/V fragment addresses include `call_off`.

The full 2,097,152-element fixture is finite and agrees with both SDPA and the llama arithmetic oracle (`max_abs=0.001953125`). Direct device timing over the final 100 settled samples improves from 118.02 us to 101.79 us median (13.75%). V-only staging is rejected (126.78 us), and K+V staging is weaker (106.18 us), so production uses K-only staging.

A fresh current252 control/candidate bracket preserves token 198, exact five-cycle recurrent logits replay, 252 generated mains, canonical packed weights, no weight copies, and zero V/down overlays. Median wall improves from 50.152759 ms to 49.490628 ms; minimum improves from 50.102013 ms to 49.398375 ms. `PREFILL_NV_K_STAGE=0` is the explicit rollback.

A follow-up CTA remap that grouped the four GQA query heads over one `(KV head, query tile)` was rejected before timing. The typed `PackedFragmentLoopSpec` verifier does not currently admit a Q fragment whose canonical group expression depends on the physical warp lane. The passing promoted arm retains four adjacent query tiles per CTA; it still shares each K tile across all four consumers. Reopen the GQA-head mapping only with an explicit multiwave grid descriptor rather than weakening the existing verifier.
