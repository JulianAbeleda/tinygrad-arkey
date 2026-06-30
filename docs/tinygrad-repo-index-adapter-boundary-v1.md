# Tinygrad Repo Index Adapter Boundary v1

Date: 2026-06-30

Status: This is a forward-looking specification, not yet-implemented behavior. It is part of the runtime / client separation effort described in `tinygrad-runtime-client-separation-roadmap-20260630.md`, and it fills in the "Phase R7: Repo Index Adapter Boundary" deliverable. It describes a small interface that lives **outside** tinygrad — in the client — for turning a code repository into prompt context. The tinygrad runtime is not involved and does not depend on any of this.

## Why repo indexing is deliberately not in the runtime

It is tempting to teach the inference server how to read a repo: point it at a folder, let it search files, let it pull in the relevant lines. We are deliberately *not* doing that.

The tinygrad runtime has one job: take tokens, run the model, return tokens. Fast, predictable, portable. The moment it also has to walk a git tree, build embeddings, watch files for changes, and decide which snippets matter, it stops being a clean inference server and becomes a half-built agent framework. That is exactly what the roadmap's Non-Goals section rules out.

Repo work is also a *client* concern by nature. Different clients want different things: one wants BM25 keyword search, another wants vector embeddings, another just wants "the file the user has open." Git state, ignore rules, language-aware chunking, and which commit to read are all product decisions, not inference decisions. Keeping them on the client side means each client can do what it needs without ever forking or patching the runtime.

So the rule is simple: **the runtime sees only the packed prompt text. Everything that produced that text — indexing, search, ranking, packing — happens above the boundary, in the client.**

## The interface

A repo-index adapter is just a small set of functions the client calls. There is no network protocol required between these functions and tinygrad; they run inside (or alongside) the client and ultimately hand the client a string to put in `message.content`.

### `index_repo(path, commit?)`

**Job:** read a repository and build (or refresh) a searchable index of it.

- **Inputs:** `path` — the local repo root. `commit` (optional) — the git commit to index at; if omitted, the adapter uses the current working-tree state but should still record the resolved commit so results are reproducible.
- **Output:** a handle or identifier for the built index (and ideally the resolved commit), plus enough metadata to know whether this index can be reused later. Building the index is the expensive step (walking files, chunking, computing embeddings or keyword tables), so it is meant to be cached and reused — see the cache rule below.

### `search_repo(query, filters)`

**Job:** find the parts of the indexed repo most relevant to a query.

- **Inputs:** `query` — natural-language or keyword text describing what the user is asking about. `filters` — optional constraints such as path globs, file types, max number of hits, or "only this directory."
- **Output:** a ranked list of hits. Each hit identifies a file, a line range, and the snippet text — i.e. exactly the information needed to emit a `<file path="..." start="..." end="...">` citation later. The hit may also carry a relevance score so the packer can decide what to keep when space is tight.

### `pack_context(hits, token_budget)`

**Job:** turn a set of search hits into the actual prompt text, sized to fit.

- **Inputs:** `hits` — the (usually ranked) results from `search_repo`. `token_budget` — how many tokens this context is allowed to occupy.
- **Output:** a string of context in the envelope format (the `<repo>` / `<file>` sections described in the envelope spec), trimmed to fit the budget by shrinking line ranges and dropping the lowest-ranked hits. This is where the token-budget priority order from the envelope spec is applied. The output is ready to drop into `message.content`.

### `invalidate_repo(path)`

**Job:** mark a repo's index as stale so it gets rebuilt.

- **Inputs:** `path` — the repo whose index should be discarded.
- **Output:** nothing meaningful beyond confirmation. Call this when the working tree changes (new commit, edited files) so the next `index_repo`/`search_repo` does not serve stale snippets. Combined with the commit-based cache key below, this is how the client keeps citations honest.

## The output is envelope text, not a new format

This adapter does not invent a transport. Its end product is **`message.content` text in the XML-style envelope format**, which the client then sends to `/v1/chat/completions` as ordinary JSON.

In other words, `pack_context` emits the `<repo>` / `<file>` blocks defined in `tinygrad-client-context-envelope-v1.md`. The line-citation shape (`path` + inclusive `start`/`end`) is the same one specified there. Read that spec for the exact tag layout, the citation rules, and the token-budget priority order; this adapter is simply the machinery that produces those sections from a real repository.

## Who can implement it

Because the interface has no tinygrad runtime dependency, it can live anywhere the client finds convenient:

- inside the **proprietary app** directly,
- as an **OpenCode-style references** layer (the external repo/reference mechanism that already lives outside the runtime),
- or as a **separate local service** the client calls.

Any of these is valid. The only contract that matters is that the output is envelope-formatted `message.content` text. The runtime cannot tell — and does not need to tell — which of these produced the prompt.

## Cache ownership

Indexing a repo is expensive, so it should be cached and reused. The reuse rule comes straight from the roadmap's Cache Strategy table:

> **repo/context index cache** — owned by the **client** — reuse by **repo path + git commit + index config**.

That means an index is safe to reuse only when the repo path, the git commit, and the indexing configuration (chunking, embedding model, filters, etc.) all match. Change the commit and the cache key changes, so you rebuild — which is exactly why `index_repo` records the resolved commit and why `invalidate_repo` exists.

Critically, **this cache belongs to the client, not the runtime.** It sits next to the four caches the roadmap distinguishes, and it is the one the runtime explicitly does not own. Do not conflate it with the runtime's model-file cache, compiled-kernel cache, or prompt/KV prefix cache — those are runtime-owned and keyed differently. Keeping repo-index caching on the client side is what lets the runtime stay a pure inference server.

## Acceptance checklist (Phase R7)

This spec is considered to satisfy R7 when:

- [ ] The interface has **no tinygrad runtime dependency** — it runs entirely on the client side.
- [ ] It can be implemented by the proprietary app, an OpenCode references layer, or a separate local service.
- [ ] Its output is `message.content` text for `/v1/chat/completions`, in the envelope format defined by `tinygrad-client-context-envelope-v1.md`.
- [ ] It defines the four functions: `index_repo(path, commit?)`, `search_repo(query, filters)`, `pack_context(hits, token_budget)`, `invalidate_repo(path)`.
- [ ] The repo/context index cache is client-owned and reused by repo path + git commit + index config.

Verdict on completion: `R7_PASS_REPO_CONTEXT_BOUNDARY`.
