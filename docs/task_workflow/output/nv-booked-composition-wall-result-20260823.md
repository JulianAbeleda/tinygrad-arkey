# Booked-route composition reverse bracket (2026-08-23)

## Findings first

1. **[MEASURED] The two accepted routes compose successfully.** With both
   routes disabled, the reverse-bracket control midpoint is
   `4817.770031 us/token`. With the production policies untouched, the
   candidate is `4697.288625 us/token`. Recovery is `120.481406 us/token`
   (`2.564914%`), verdict `WALL_PASS`.
2. **[MEASURED] Correctness closes.** Every arm reproduces token-stream SHA
   `f25083e5d0a754131283b40c03f52e688fee9f175bea7ae106805e7d628d7905`.
3. **[MEASURED] Route state is exact.** Control A and Control C have zero Q/K
   norm+RoPE blocks and zero Q6 FFN-down four-warp blocks. The production
   candidate has all 36 Q/K blocks and the model's 18 Q6 FFN-down blocks.
4. **[MEASURED] Clocks are matched.** All arms are P0 with memory at 14001 MHz;
   observed SM clocks are 2797, 2805, and 2790 MHz for Control A, candidate,
   and Control C respectively.
5. **[MEASURED] Composition exceeds the conservative component floor.** The
   separate booked floors sum to `93.429891 us/token`; the same-session
   composed bracket recovers `120.481406 us/token`, an excess of
   `27.051516 us/token`. There is no negative composition interaction.
6. **[MEASURED] The current production endpoint is `4697.288625 us/token`, or
   `212.888770 tok/s`.** It is `530.621958 us/token` from the 240 tok/s target.
7. **[MEASURED] The old frozen-control projection and fresh endpoint reconcile
   with zero residual.** The fresh disabled control is `46.347031 us/token`
   slower than the old `4771.423 us/token` control, while composed recovery is
   `27.051516 us/token` larger than the component floor. Therefore:

   ```text
   measured_current - old_projected
     = control_session_drift - recovery_excess
     = 46.347031 - 27.051516
     = 19.295516 us/token

   identity residual = 0.000000 us/token
   ```

8. **[UNMEASURED] 240 tok/s remains unmeasured.** This bracket validates the
   installed composition; it does not identify or recover the remaining
   `530.621958 us/token`.

## Protocol

The order was control/candidate/control. Every arm used a fresh process, depth
512, five settled 32-token windows, the shared GPU lock, and locked RTX 5090
clocks. Controls explicitly removed both accepted routes after model load. The
candidate did not use a research admission override: it asserted and retained
the shipped production policies.

## Verdict

`COMPOSED_WALL_PASS`

The current measured baseline for subsequent work is `4697.288625 us/token`.
The previous `4677.993109 us/token` value is retained only as a superseded
cross-session ledger projection.
