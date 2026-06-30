# Tinygrad Client Context Envelope v1

Date: 2026-06-30

Status: This is a forward-looking specification, not yet-implemented behavior. It is part of the runtime / client separation effort described in `tinygrad-runtime-client-separation-roadmap-20260630.md`, and it fills in the "Phase R6: Client-Side Context Contract" deliverable. It describes how a client (the proprietary app, an OpenCode-style tool, or any other caller) should pack prompt context before sending a request to the tinygrad runtime. Nothing here changes the runtime itself.

## The one rule that makes this work

The line between the client and the runtime is a normal JSON HTTP API. That JSON stays JSON. We do not invent a custom wire format, and we do not ask the runtime to understand anything new.

The trick is this: all of the rich structure — which repo we are looking at, which files, which lines, the session summary, the actual task — lives **inside** the `message.content` string as plain text. We dress that text up with XML-style tags so a human (and the model) can read its structure, but those tags are just characters in a string. The runtime never parses them.

So there are really two layers:

```text
HTTP payload:     JSON   (the runtime cares about this)
message.content:  free text, optionally XML-tagged   (only the model "reads" this)
```

The runtime takes the whole `content` string, runs it through the tokenizer, and feeds the tokens to the model. It does not know or care that there was a `<repo>` tag in there. To the runtime it is all just text. That is the entire point: it keeps the runtime a dumb, fast, portable inference server, and it keeps every smart decision about *what context to include* on the client side, where it belongs.

This also means the envelope can evolve freely. You can add new sections, rename tags, or change the layout, and the runtime needs no update — only the client and the model's understanding of the convention matter.

## Why XML-style tags inside the text

We could pack context as a blob of markdown or as nested JSON-in-a-string. XML-style tags win for a few plain reasons:

- They are easy for the model to recognize as structure (open tag, content, close tag).
- They carry attributes cleanly (`path`, `start`, `end`, `commit`), which is exactly what file citations need.
- They are forgiving: a stray angle bracket does not break a JSON parser, because the runtime is not parsing it at all.
- They read well to a human debugging a prompt.

To be clear, this is a *convention*, not a parser contract. There is no schema validation on the runtime side. If the client emits malformed tags, the only consequence is that the model may read the context less cleanly.

## The recommended sections

A fully-formed envelope has up to four sections, in this order. Only the parts you actually have need to appear.

### `<runtime>` — what model and how much room

A small header telling the model (and any human reading the prompt) which model this was packed for and how big the context window is. This is informational; the runtime already knows its own model and `max_context`. It is useful mostly so the prompt is self-describing and so the client's own budgeting math is visible.

```xml
<runtime>
  <model id="qwen3-8b-q4k" max_context="4096"/>
</runtime>
```

### `<session_summary>` — the compressed past

A short, client-produced summary of the conversation or working session so far. This is how the client keeps long sessions inside the context budget: instead of replaying every old turn verbatim, it summarizes the older history into a few sentences and puts it here. The runtime has no memory of past turns, so if you want the model to "remember," it goes here (or in the message history) — packed by the client.

```xml
<session_summary>
User is refactoring the GGUF loader. Earlier we confirmed max_context is
capped by the GGUF metadata. Open question: how prefix-cache reuse interacts
with a changed system prompt.
</session_summary>
```

### `<repo>` — the relevant code

The slice of the repository the model should look at. The repo is identified by its local path and the git commit it was read at (so the context is reproducible and cacheable — see the index adapter spec). Inside it, each included file is a `<file>` element carrying its path and the exact line range that was pulled.

```xml
<repo path="/home/ubuntu/tinygrad-arkey" commit="a1b2c3d">
  <file path="tinygrad/llm/model.py" start="1551" end="1600">
... the actual lines of code ...
  </file>
</repo>
```

### `<task>` — what to actually do

The user's real request for this turn. This is the thing the model must answer or act on. Keep it last so it is the freshest thing in the model's view.

```xml
<task>
Explain how oversized prompts are handled and where the guard should live.
</task>
```

## Citing local files

When you include code, cite it the same way the `<file>` element is shaped: a path plus an explicit line range. The shape is:

```xml
<file path="RELATIVE/OR/ABSOLUTE/PATH" start="FIRST_LINE" end="LAST_LINE">
... the lines themselves ...
</file>
```

- `path` is the file, relative to the enclosing `<repo path="...">` (or absolute — both are fine as long as you are consistent).
- `start` and `end` are 1-based, inclusive line numbers, matching what the file actually contains.
- The text between the tags should be the real lines for that range, so a citation and its content never drift apart.

This format does double duty: it tells the model exactly where a snippet came from, and it gives the client a stable key (path + commit + line range) for caching and de-duplication.

## Token-budget priority order

The model has a fixed context window (`max_context`). The client is responsible for making everything fit. When the packed envelope is too big, the client trims — and it should trim in a deliberate order, dropping or shrinking the **lowest-value** material first so the highest-value material survives.

Recommended order, from "protect at all costs" down to "drop first":

1. **The task (`<task>`) and the latest user turn — never dropped.** If this does not fit, the request is malformed; the model has nothing to do without it.
2. **The `<runtime>` header — tiny, keep it.** It costs almost nothing and makes the prompt self-describing.
3. **Directly-cited repo files the task explicitly references — high priority.** If the user is asking about `model.py:1551`, those lines must stay.
4. **Supporting repo snippets that are merely relevant — shrink first.** Narrow the line ranges, keep fewer files, or keep just the most relevant hits. This is the biggest, most compressible bucket.
5. **The `<session_summary>` — shrink or drop early.** A summary is already lossy; making it shorter (or dropping it on a fresh-topic turn) usually costs the least.

The justification is simple: the task is the only thing the model *must* have, and recent/explicitly-referenced material is what the user is actually thinking about. Old session summary text and loosely-relevant background code are the most replaceable, so they yield first. When trimming a bucket, prefer **shrinking** (narrower line ranges, shorter summary) over **dropping** entirely, because a smaller-but-present snippet is usually more useful than nothing.

A practical client will estimate tokens per section, sum them, and if over budget, walk this list from the bottom up — shrinking, then dropping — until the envelope fits with headroom for the model's reply (`max_tokens`).

## The runtime does not own any of this

This must be stated plainly: the tinygrad runtime does **not** own memory, repo context, or session state. It does not summarize. It does not pick files. It does not evict old turns. It does not enforce a token budget for you — it only fails cleanly if the final token count exceeds `max_context`.

Every responsibility above the tokenizer line is the client's:

- truncating history,
- summarizing the past,
- selecting which repo snippets to include,
- compacting tool results,
- and staying under the prompt token budget.

If the client sends an oversized prompt, the runtime's job is to reject it with a structured error (see Phase R2 in the roadmap), not to silently truncate or to crash on a tensor shape. The truncation decision is the client's, and the client must make it *before* sending.

## End-to-end example

A fully-packed `message.content` string, with every section present. Note that this entire block is the value of a single `content` field in the JSON request — the tags are just text inside it.

```text
<runtime>
  <model id="qwen3-8b-q4k" max_context="4096"/>
</runtime>

<session_summary>
User is hardening the tinygrad OpenAI-compatible server. We established that
the runtime only sees tokens and has no prompt guard yet. Prior turn confirmed
max_context comes from GGUF metadata.
</session_summary>

<repo path="/home/ubuntu/tinygrad-arkey" commit="a1b2c3d">
  <file path="tinygrad/llm/model.py" start="1551" end="1560">
    def generate(self, tokens, max_tokens):
      x = Tensor(tokens + [0] * (self.max_context - len(tokens)))
      start = self.get_start_pos(tokens)
      ...
  </file>
  <file path="tinygrad/llm/cli.py" start="88" end="96">
    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
      ...
  </file>
</repo>

<task>
Oversized prompts currently blow up on the padding line in generate(). Explain
where a prompt-token overflow guard should live and what error the runtime
should return instead.
</task>
```

And the JSON request that carries it (the runtime sees only this):

```json
{
  "model": "qwen3-8b-q4k",
  "messages": [
    { "role": "user", "content": "<runtime>\n  <model id=\"qwen3-8b-q4k\" max_context=\"4096\"/>\n</runtime>\n\n<session_summary>...</session_summary>\n\n<repo path=\"/home/ubuntu/tinygrad-arkey\" commit=\"a1b2c3d\">\n  <file path=\"tinygrad/llm/model.py\" start=\"1551\" end=\"1560\">...</file>\n</repo>\n\n<task>Oversized prompts currently blow up...</task>" }
  ],
  "temperature": 0,
  "max_tokens": 512
}
```

## Acceptance checklist (Phase R6)

This spec is considered to satisfy R6 when:

- [ ] It names the required/recommended sections: `<runtime>`, `<session_summary>`, `<repo>`/`<file>`, `<task>`.
- [ ] It describes the token-budget priority order (what to keep, what to shrink, what to drop first).
- [ ] It states explicitly that the runtime treats `message.content` as plain text and does not parse the XML.
- [ ] It defines the citation format for local files (path + inclusive line range, matching `<file path="..." start="..." end="...">`).
- [ ] It states that the runtime does not own memory/repo/session and that the client must enforce truncation/summarization before sending.

Verdict on completion: `R6_PASS_CONTEXT_ENVELOPE_SPEC`.
