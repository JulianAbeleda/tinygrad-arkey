"""
M1: Fused prefill attention kernel — scalar-scored (fdot2) bridge.
M2 will swap the dot-reduce to WMMA. This isolates fusion correctness from WMMA.

Reuses flash_kernels.py: online-softmax, LDS K/V staging, WAR barrier.
New: M-query tile per workgroup (M=4 for M1, later scaled in full build).
"""
from __future__ import annotations
from extra.qk.flash_common import _F32, _fexp, _fc, _fki, _ceildiv, Tensor, dtypes, getenv, AddrSpace, AxisType, KernelInfo, Ops, UOp
from extra.qk.kv_load import make_kv_element_loader

def flash_prefill_scalar_kernel(Hd: int, Hq: int, Hkv: int, M: int, T: int, KV: int, TK: int = 16):
    """
    Fused prefill attention kernel with scalar (fdot2) scoring.

    One workgroup processes M query rows, one GQA head group.
    K/V staged in LDS (TK rows per block). Online softmax over KV blocks.
    Scores computed per (M-row, TK-token) via warp-reduced fdot2.

    Args:
        Hd: head dimension (must be 128)
        Hq: total query heads (40 for 14B)
        Hkv: KV heads (8 for 14B)
        M: query rows per workgroup (4 for M1)
        T: total query tokens
        KV: total KV tokens
        TK: K/V block size (tokens per LDS block)
    """
    if Hd != 128: raise ValueError(f"This kernel requires Hd=128, got {Hd}")
    G = Hq // Hkv
    LANES = 32
    WARPS = G  # one warp per GQA query head
    THREADS = LANES * WARPS
    R = Hd // LANES  # 4 — each thread handles 4 head_dim elements
    RP = Hd // 64  # 2  — two fdot2 ops per full dot product
    NB = _ceildiv(KV, TK)  # number of KV blocks
    scale = 1.0 / (Hd ** 0.5)

    def kernel(pout: UOp, q: UOp, cache: UOp) -> UOp:
        # Simplified: fp16 K/V directly (no quant, no rope for MVP)
        kv_load = make_kv_element_loader(cache, Hd, kvscale=None, freqs=None)

        # Grid: kvh (KV head) × s (split — unused for MVP, keep for structure)
        kvh = UOp.range(Hkv, 0, AxisType.GLOBAL)
        s = UOp.range(1, 1, AxisType.GLOBAL)  # single split for MVP

        # Thread indexing
        lane = UOp.range(LANES, 10, AxisType.LOCAL)
        warp = UOp.range(WARPS, 11, AxisType.LOCAL)
        h = kvh * G + warp  # global query head index
        tid = warp * LANES + lane

        # LDS for K/V staging (TK * Hd halfs)
        ksh = UOp.placeholder((TK * Hd,), dtypes.half, 230, addrspace=AddrSpace.LOCAL)
        vsh = UOp.placeholder((TK * Hd,), dtypes.half, 231, addrspace=AddrSpace.LOCAL)

        # M accumulators per thread: acc[M][R], den[M], mx[M]
        acc = [UOp.placeholder((R,), _F32, 232 + 3 * m, addrspace=AddrSpace.REG) for m in range(M)]
        den = [UOp.placeholder((1,), _F32, 232 + 3 * m + 1, addrspace=AddrSpace.REG) for m in range(M)]
        mx  = [UOp.placeholder((1,), _F32, 232 + 3 * m + 2, addrspace=AddrSpace.REG) for m in range(M)]

        # Initialize accumulators
        chain = None
        za = UOp.range(R, 2)
        for m in range(M):
            init_acc = acc[m].after(kvh if chain is None else chain, s)[za].store(0.0).end(za)
            init_den = den[m].after(init_acc)[0].store(0.0)
            init_mx  = mx[m].after(init_den)[0].store(-float("inf"))
            chain = init_mx

        # KV block loop
        b = UOp.range(NB, 3, axis_type=AxisType.REDUCE)

        # K/V staging into LDS (cooperative, one element per thread)
        STAGES = _ceildiv(TK * Hd, THREADS)
        st = UOp.range(STAGES, 4, axis_type=AxisType.REDUCE)
        i_stage = st * THREADS + tid
        tt_stage = i_stage // Hd
        e_stage = i_stage % Hd
        t_stage = b * TK + tt_stage
        in_bounds = (tt_stage < TK) & (t_stage < KV)
        t_safe = in_bounds.where(t_stage, t_stage.const_like(0))

        kstore = ksh[i_stage].store(kv_load(0, kvh, t_safe, e_stage), i_stage < (TK * Hd))
        vstore = vsh.after(kstore)[i_stage].store(kv_load(1, kvh, t_safe, e_stage), i_stage < (TK * Hd))
        bar = UOp.barrier(UOp.group(vstore.end(st)))

        # Per-K-token: compute scores for all M query rows, merge online-softmax
        for tt in range(TK):
            chain_before = bar if tt == 0 else chain
            tt_idx = UOp.const(dtypes.weakint, tt)

            for m in range(M):
                # Query row offset for this workgroup: workgroup_id * M + m
                # We use idx0 for the M dimension (split across workgroups)
                q_row = UOp.range(_ceildiv(T, M), 8, AxisType.GLOBAL) * M + m  # simplified — actual indexing via gidx
                # FIXME: proper indexing — for MVP we hardcode single-workgroup M rows
                # Each thread computes dot product between Q[h, m, :] and K[tt, :]
                # using fdot2 over RP chunks, then warp-reduce.

                # Dot product accumulator (per thread)
                dotp = UOp.placeholder((1,), _F32, 240 + m, addrspace=AddrSpace.REG)
                di = dotp.after(chain_before)[0].store(0.0)
                dotp = dotp.after(di)

                # Inner reduce over RP fdot2 chunks
                rp = UOp.range(RP, 6 + m, axis_type=AxisType.REDUCE)
                e2 = rp * 64 + lane * 2
                qp = UOp(Ops.STACK, dtypes.half.vec(2), (
                    q[h * Hd + m * Hd * Hq + e2].cast(dtypes.half),
                    q[h * Hd + m * Hd * Hq + e2 + 1].cast(dtypes.half)))
                kp = UOp(Ops.STACK, dtypes.half.vec(2), (
                    ksh.after(bar)[tt * Hd + e2],
                    ksh.after(bar)[tt * Hd + e2 + 1]))
                d2 = UOp(Ops.CUSTOMI, _F32, (dotp.after(rp)[0], qp, kp),
                         arg="__builtin_amdgcn_fdot2({1}, {2}, {0}, false)")
                du = dotp[0].store(d2).end(rp)

                # Warp reduce sum
                from extra.qk.amd_warp_reduce import warp_reduce_sum
                sc = warp_reduce_sum(dotp.after(du)[0], lane, LANES) * scale

                # Online softmax merge
                old_m = mx[m].after(tt_idx if m == 0 else den[m-1].after(chain_before))[0]
                new_m = old_m.maximum(sc)
                corr = _fexp(old_m - new_m)
                p = _fexp(sc - new_m)

                dd = UOp.range(R, 7 + m)
                d = lane * R + dd
                vd = vsh.after(bar)[tt * Hd + d].cast(_F32)

                accu = acc[m][dd].store(acc[m].after(
                    tt_idx if m == 0 else acc[m-1].after(chain_before)
                )[dd] * corr + p * vd).end(dd)
                denu = den[m].after(accu)[0].store(den[m].after(tt_idx)[0] * corr + p)
                mxu0 = mx[m].after(denu)[0].store(new_m)

                chain = mxu0
                # Update the references for next iteration
                acc[m], den[m], mx[m] = acc[m].after(chain), den[m].after(chain), mx[m].after(chain)

        # WAR barrier
        bar2 = UOp.barrier(UOp.group(chain))
        mxu = bar2.end(b)

        # Write output: M query rows × Hd elements per workgroup
        af = [acc[m].after(mxu) for m in range(M)]
        lf = [den[m].after(mxu) for m in range(M)]
        mf = [mx[m].after(mxu) for m in range(M)]

        # FIXME: proper writeback — for M1 correctness check only
        base = h * T * Hd  # base offset for this head
        out_chain = None
        for m in range(M):
            dd2 = UOp.range(R, 8)
            d2 = lane * R + dd2
            prev = out_chain if out_chain is not None else mxu
            pv = pout[base + m * Hd + d2].store(af[m][dd2]).end(dd2)
            out_chain = pv

        return out_chain.end(kvh, s, lane, warp).sink(
            arg=_fki(f"flash_prefill_scalar_{Hq}_{Hd}_M{M}"))

    return kernel

# M1 test harness
if __name__ == "__main__":
    import os, time
    os.environ.setdefault('DEV', 'AMD')
    from tinygrad import Tensor, dtypes, Device
    from tinygrad.helpers import getenv
    import numpy as np

    print("M1: Fused prefill kernel — scalar scoring bridge")
    print("=" * 60)

    Hd, Hq, Hkv = 128, 40, 8
    M, T, KV = 4, 2048, 2048
    TK = 16

    # Build the kernel
    kfn = flash_prefill_scalar_kernel(Hd, Hq, Hkv, M, T, KV, TK)
    print(f"Kernel built: M={M}, T={T}, KV={KV}, TK={TK}")

    # Test with random data
    q = Tensor.randn(1, Hkv, Hq//Hkv, T, Hd, dtype=dtypes.float16).contiguous().realize()
    cache = Tensor.randn(2, Hkv, 1, KV, Hd, dtype=dtypes.float16).contiguous().realize()  # [K/V, heads, 1, tokens, Hd]

    # SDPA reference
    def sdpa():
        qr = q.reshape(1, Hkv, Hq//Hkv, T, Hd)
        kr = cache[0:1].reshape(1, Hkv, 1, KV, Hd)
        vr = cache[1:2].reshape(1, Hkv, 1, KV, Hd)
        scale = Hd ** -0.5
        # causal mask
        import numpy as np
        mask_np = np.triu(np.ones((T, KV)), k=1) * float('-inf')
        mask_t = Tensor(mask_np, dtype=dtypes.float16, device=q.device).reshape(1, 1, 1, T, KV).contiguous().realize()
        scores = (qr @ kr.transpose(-1, -2)).float() * scale
        scores = scores + mask_t
        s = scores.softmax(-1)
        return (s.cast(dtypes.float16) @ vr)

    out_ref = sdpa()
    out_ref.realize()
    Device[Device.DEFAULT].synchronize()
    ref_np = out_ref.numpy()

    print(f"Reference output shape: {ref_np.shape}")
    print(f"Reference output stats: mean={ref_np.mean():.6f}, std={ref_np.std():.6f}")
    print("\nM1 kernel defined — next: compile + verify via the gate script.")
