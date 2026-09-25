# GameTerm overnight self-training controller proof — 2026-09-07

This is a bounded one-round lifecycle proof of the resumable overnight controller. It began from the qualified Qwen3-8B adapter, trained every completion token plus `<|im_end|>`, saved a new checkpoint, started the tinygrad server, waited for readiness, performed a logged generation warm-up, evaluated four unseen-wording cases through the real GameTerm harness, stopped the server, and atomically finalized its ledger.

## Lifecycle result

- Controller wall time: **134.82 seconds**.
- Training steps: **1** (the minimum lifecycle proof setting).
- Full-token held-out loss: **16.242996 -> 15.052478**.
- Checkpoint digest: `c79631bf6124f0da951c3a5bf42781bda8b104097333e90bc531e38b14f4e0ed`.
- GameTerm turns completed: **4/4** with **zero infrastructure failures**.
- Exact behavioral score: **1/4** after one update; the score was not the lifecycle acceptance criterion.
- Final controller status: `max_rounds`.
- Resume check: rerunning the identical configuration retained exactly one completed round and launched no duplicate training or server process.
- Post-proof process audit: no trainer, controller, tinygrad server, or GameTerm harness remained running.

The one-update score is not presented as a converged result. The earlier MVP evidence separately records the real **0/4 -> 1/4** behavior change. This proof establishes that the controller can repeat the complete autonomous lifecycle and stop cleanly.

## Overnight target

The overnight fixture uses four meaningful labels that are each exactly one Qwen token: `READ`, `WATCH`, `OPEN`, and `BLOCK`. This keeps strict exact-completion evaluation inside GameTerm's provider timeout. The controller trains each label and EOS, continues from the prior round's adapter, decays the learning rate, and stops immediately when all four held-out exact checks pass. It otherwise stops at the round or wall-time bound. A finite four-case suite demonstrates that defined behavior; it does not establish complete understanding of every possible GameTerm interaction.

## Overnight command

Run in a persistent shell or under `nohup`; use a durable output directory. Defaults are 96 steps per round, 12 rounds, and an eight-hour limit.

```bash
DEV=NV python3 -m extra.llm.bench.gameterm_self_training_loop \
  --repo /home/ubuntu/tinygrad-self-training \
  --model /home/ubuntu/storage/models/Qwen3-8B-Q4_K_M.gguf \
  --input /home/ubuntu/tinygrad-self-training/test/fixtures/llm/qwen_output_lora_overnight.jsonl \
  --harness /home/ubuntu/gameterm_beta/target/debug/gameterm-harness \
  --out /home/ubuntu/storage/self-training/overnight \
  --system-prompt 'Reply with exactly one uppercase action label: READ, WATCH, OPEN, or BLOCK. No punctuation. /no_think' \
  --server-max-tokens 1 \
  --init-adapter /home/ubuntu/tinygrad-self-training/docs/task_workflow/evidence/gameterm-qwen-self-training-20260907/adapter
```

The controller only resumes when its full training and serving configuration matches the existing ledger. Each round retains commands, training stdout/stderr, model summary, adapter, warm-up request/response, server logs, raw normalized GameTerm events, and the exact score ledger.

For detached operation, `extra/llm/bench/start_gameterm_self_training_overnight.sh` launches this configuration and writes `launcher.pid`; the matching `stop_gameterm_self_training_overnight.sh` requests a clean process-group stop.
