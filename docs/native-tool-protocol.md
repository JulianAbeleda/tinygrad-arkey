# Native tool protocol

Tool-enabled `/v1/chat/completions` requests use the loaded GGUF's Jinja chat
template and the shared `tinygrad.llm.chat.NativeChat` owner. Install the
optional dependency with `pip install -e '.[llm]'`. Requests without tool
fields/history retain the existing text serializer. Sending `tools: []`
explicitly selects native formatting for no-tool control probes.

The currently qualified parser is the Qwen XML-call dialect, selected by GGUF
architecture (`qwen2` or `qwen3`) and template protocol markers, never filenames.
Other dialects, missing templates/dependencies, non-string message content,
assistant prefill, forced/required/none tool choice, and parallel tool generation
fail explicitly. `tool_choice: "auto"` and `parallel_tool_calls: false` are
supported. Native formatting fixes `enable_thinking=false`; this is recorded
alongside the template SHA-256 by native training reports.

One complete `<tool_call>` JSON object becomes one standard `tool_calls` entry.
Arguments must be a JSON object and the function must be in the request catalog.
Malformed/truncated, mixed prose-plus-call, parallel, duplicate-key, and unknown
function output returns `invalid_model_output` (HTTP 502); nothing is repaired.
Plain text remains a valid answer. JSON Schema argument semantics and permission
checks remain the executing harness's authority. Call IDs are transport IDs,
not model-generated evidence. Histories require unique IDs and exactly one
linked result for every call before further conversation.

Tool-enabled output is buffered until generation completes and validation passes,
including when `stream: true`; this adds first-byte latency. Valid SSE responses
contain indexed tool deltas and `finish_reason: "tool_calls"`. Nonstream responses
contain the same call shape. Cancellation cannot release an executable call.
The server never executes tools.

The existing Qwen smoke trainer accepts native JSONL rows with `id`, `source_id`,
`tools`, and `messages`, whose final message is the verified assistant target.
Use `--completion-scope all` and an empty CLI system prompt. Prior tool calls and
observations belong in `messages`; wire arguments remain JSON strings. The loader
validates protocol structure, **not whether a demonstration is correct**: the
experience collector must independently verify effects before approving targets.
The shared template renders both the generation prefix and complete conversation;
training refuses a prefix mismatch and includes the actual first EOS/EOT token.
Legacy `prompt`/`completion` rows retain their previous byte/token behavior.
The current smoke trainer still adapts only the output projection and is not a
qualified general tool-learning trainer merely because serialization is supported.

CPU tests use the Qwen3-0.6B GGUF template fixture, real `SimpleTokenizer`, actual
HTTP requests, and scripted model tokens. They establish protocol behavior, not
model competence. GameTerm's current real terminal/workspace services are
macOS-only; the Linux localhost harness fails to build because `real_registry`
is unavailable. No OS-effect or real learning claim follows from these tests.

## Qualification checkpoint

The focused CPU command is:

```sh
DEV=CPU python3 -m pytest -q test/unit/test_llm_runtime_state.py \
  test/unit/test_tinygrad_llm_cli_boundary.py test/unit/test_llm_cli_server_generation.py \
  test/unit/test_qwen_lora_smoke_train.py test/unit/test_sft_smoke_train.py
python3 sz.py
```

On 2026-09-08 it passed 30 tests. A separate CPU metadata-only check using the
complete local Qwen3-0.6B Q8_0 vocabulary verified the same serving/training
prefixes: 144 prompt + 22 target tokens for the first Unicode file call;
180 prompt + 8 target tokens for the answer after a Unicode tool observation.
Both ended at token 151645, the model's actual EOS, without training the
no-think prefix. Template SHA-256:
`57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361`.
These checks loaded tokenizer metadata only, not model weights onto a GPU.
