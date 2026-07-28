# Other Extra Surface

This directory holds tracked `extra/` files that do not yet fit the current domain map:

- `audit`
- `tools`
- `llm`
- `llm_research`
- `debug`
- `profiling`
- `drivers`
- `setup`

`other` is a temporary classification, not a permanent ownership tier and not a place to avoid review. Every file
placed here must have a ledger row naming its purpose, branch owner, consumers, validation command, retention criterion,
and next disposition. The valid dispositions are promotion into a named domain, movement to `dev` or `exp`, or deletion
with a compact conclusion and recovery commit.

Do not add new production runtime dependencies from `tinygrad/**` into this directory. If a file becomes production
runtime code, it must move to its `tinygrad/**` domain owner or to a named supported optional backend.
