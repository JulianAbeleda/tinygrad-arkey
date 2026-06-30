# R0 residual lever triage

**Verdict:** AMD_ISA_RESIDUAL_TRIAGE_INCONCLUSIVE_NO_LIVE_LEVER  
**Selected R1 lever:** None  
**Why:** no SMALL live lever with a credible W==D path: R1A (lds_accum, the real structural lever) is regalloc-blocked with no design proof; R1B-hoist is a verified false positive (tinygrad already hoists) and R1B-strength-reduce needs unbounded induction-variable codegen for negligible wall; R1C FMA cluster <5; R1D waitcnt is non-primary/capped. The residual is STRUCTURAL (register accumulators, N5A) -> needs a regalloc feature, not a bounded lever.

## Lever feasibility

- **R1A_REGALLOC_FEATURE**: lds_accum inner-loop ds=25 (structural, PMC 48x). The real lever, but BLOCKED: removing needs loop-carried physical accumulators; N5A proved the single-def regalloc can't represent it and NO regalloc-model design proof is ready -> per scope, stop, do not select.
- **R1B_ADDRESS_STRENGTH_REDUCE**: address_index live hot-loop ops=44. RIGOROUS linearized check: loop-invariant-hoistable-INSIDE-loop=0 (==0 => tinygrad ALREADY hoists invariants -> hoist is a FALSE POSITIVE, the earlier hoistable count was invariant+consumed-in-loop = already-hoisted ops). strength-reducible(loop_var*const) in-loop=5 -> real but needs INDUCTION-VARIABLE codegen (renderer feature, not a bounded peephole) for negligible wall (tile ~10% of wall, 5/197 VALU). NOT a bounded credible lever (hoist false-positive; strength-reduce = unbounded IV-codegen feature, negligible W==D).
- **R1C_LOCAL_CODEGEN_CLEANUP**: FMA-fusable pairs=3, removable vgpr_copy movs=0 -> live removable cluster=3. too small (<5 live hot-loop VALU).
- **R1D_WAITCNT_THRESHOLDS**: waitcnt total=35 (hot=31). N2B did NOT name wait as primary category -> capped/<2% expected W==D; last resort only.

## UOp-graph LICM/strength-reduce check

```
{
 "n_inner_loops": 7,
 "int_alu_total": 43,
 "invariant_int_alu": 22,
 "loop_invariant_hoistable_REAL_inside_loop": 0,
 "loop_var_times_const_strength_reducible_inside_loop": 5,
 "interpretation": "hoist_real==0 => tinygrad's linearizer ALREADY hoists invariant address math out of the inner loops by construction -> R1B-hoist is a FALSE POSITIVE (the earlier invariant+consumed-in-loop count double-counted already-hoisted ops). The only real in-loop address lever is strength-reduction of loop_var*const, which needs induction-variable codegen (a renderer feature, NOT a bounded peephole)."
}
```

## Triage rows

- **other_classifier_split**: {"total": 49, "labels": 14, "branches": 0, "real_op_families": {"s_cbranch_scc0": 7, "s_branch": 7, "s_load_b64": 3, "s_load_b32": 1, "v_xor_b32_e32": 5, "v_cndmask_b32_e32": 8, "s_barrier": 1, "v_pack_b32_f16": 2, "s_endpgm": 1}, "note": "
- **address_index_live**: {"total": 114, "hot_loop_depth>=1": 44, "v_mul_lo": 43, "v_add_nc": 51, "verdict": "LIVE (vector address ops feed global/LDS load addresses). N1B's 'dead' was the SCALARIZED prefix (s64/s65 unconsumed because the live path stayed vector); t
- **waitcnt_binding**: {"total": 35, "hot_loop_depth>=1": 31, "depth>=2": 27, "note": "PMC (N2B) did not name wait as the primary category; finer thresholds reduce inst count but unlikely to close 35-40%."}
- **lds_accum_stage_roundtrips**: {"ds_load_store_total": 31, "ds_bpermute_cross_lane": 5, "ds_at_depth>=2_inner_per_token": 25, "ds_at_depth<=1_staging": 6, "verdict": "inner-loop (depth>=2) ds load/store = DEFINE_REG accumulator RMW per token (the structural owned-vs-nati
- **pv_softmax_fusion**: {"pv_softmax_static": 16, "hot_loop": 16, "fma_fusable_mul_then_add_pairs": 3, "note": "only 3 adjacent v_mul_f32->v_add_f32 fusable -> tiny static win"}
- **mov_copy_source**: {"total": 42, "by_cause": {"immediate": 23, "S2V": 19}, "hot_loop": 18, "note": "S2V = loop-counter/SGPR->VGPR for address math (needed); immediate = const materialization (some inline-foldable); vgpr_copy = candidate removable carrier move