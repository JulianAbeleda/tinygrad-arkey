# TG-P7 Final Default Flip

Verdict: **TG_P7_BLOCKED_PURITY_DEBT_REMAINING**

Strict default purity: **TINYGRAD_DEFAULT_PURITY_FAIL** (strict census exit 2).

## Hot-route final state

| route | provenance | default? | rollback |
|---|---|---|---|
| decode_q4k_g3_generated | machine_authored_generated | yes | {'BUBBLEBEAM_FUTURESIGHT': '0'} |
| decode_q6k_coop_generated | machine_authored_generated | yes | {'DECODE_Q6K_GENERATED': '0'} |
| decode_attention_owned_two_kernel | external_handwritten_kernel | yes | {'DECODE_ATTN_AMDGCN_TILE': '0'} |
| decode_flash_block_tile_g5_konly | machine_authored_generated | yes | {'DECODE_FLASH_BLOCK_TILE_G5': '0'} |
| prefill_pipe_role_selective_generated | machine_authored_generated | yes | {'PREFILL_GENERATED_SCHEDULE': '0'} |

## Prerequisites

- TG_P2: PASS (Q4_K G3 policy-driven)
- TG_P3: PASS (Q6_K generated coop)
- TG_P4: PASS (prefill generated schedule)
- TG_P5: REFUTE (generated 8B attention correct but slower; owned kept)
- TG_P6: PASS (pure-search diagnostic mode)

## Outcome

4 of 5 hot default routes are now machine_authored_generated and BoltBeam-selectable, with handwritten oracles one rollback flag away. The 5th (8B decode attention) is honestly blocked on speed. This is the correct terminal state given TG-P5; the north-star PASS would require a faster generated attention route (a future capability, not a packaging step).

The 8B owned attention (`decode_attention_owned_two_kernel`) is the sole remaining `external_handwritten_kernel`
default. The generated G5 replacement is correct and route-bound but slower (TG-P5), so it stays default-off and
owned HIP remains the default per the stop rule. Full `TINYGRAD_DEFAULT_PURITY_PASS` is therefore not reachable
without a faster generated attention route.
