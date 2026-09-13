# unravel-agent-providers

OpenAI-compatible model providers and capability-aware model discovery
for Unravel applications.

## Overview

This crate provides:

- **OpenAI-compatible Chat Completions** adapter with tools, streaming,
  usage, and multimodal image payloads.
- **Capability-aware model discovery** from Models.dev, generic `/models`
  endpoints, and local Ollama.
- **Instance-owned TTL caches** keyed by endpoint + account identity —
  no process-global mutable state, no secrets in cache keys.
- **Credential redaction** in all Debug/log/error output.
- **Structured error mapping** (permanent vs. transient) with
  retryability classification.

## Quick start

```rust
use unravel_agent_runtime::{Message, Model, ModelRequest, Session};
use unravel_agent_providers::{OpenAiCompatProvider, ProviderSpec};

# async fn run() -> unravel_agent_runtime::Result<()> {
let spec = ProviderSpec::openai("sk-...".to_string());
let provider = OpenAiCompatProvider::new(spec, "gpt-4o-mini")?;

let mut session = Session::new("demo");
session.messages.push(Message::user_text("Hello!"));

let request = ModelRequest::new("demo", session.messages);
let response = provider.complete(request).await?;
println!("{}", response.content);
# Ok(())
# }
```

## Provider specification

Built-in specs provide defaults for common providers:

| Provider | Spec constructor | Key env | Default endpoint |
|---|---|---|---|
| OpenAI | `ProviderSpec::openai(key)` | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| NVIDIA | `ProviderSpec::nvidia(key)` | `NVIDIA_API_KEY` | `https://integrate.api.nvidia.com/v1` |
| OpenRouter | `ProviderSpec::openrouter(key)` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| Ollama (local) | `ProviderSpec::ollama()` | `OLLAMA_API_KEY` (optional) | `http://localhost:11434/v1` |

Custom OpenAI-compatible endpoints:

```rust
use unravel_agent_providers::{ProviderSpec, OpenAiCompatProvider};

let spec = ProviderSpec::custom(
    "my-provider",
    "My Provider",
    "https://api.my-provider.com/v1".to_string(),
    None, // or Some(Secret::new("key"))
);
let provider = OpenAiCompatProvider::new(spec, "my-model")?;
```

### Endpoint precedence

1. Explicit override (`spec.with_base_url(...)`)
2. Environment variable (e.g. `OPENAI_BASE_URL`)
3. Built-in default

### API key precedence

1. Explicit key (`spec.with_key(...)`)
2. Environment variable (e.g. `OPENAI_API_KEY`)
3. None (valid for keyless providers like local Ollama)

## Discovery

Discovery describes capability metadata. It does **not** guarantee
invocation success. A model that reports `tool_call: true` may still
fail at runtime.

```rust
use unravel_agent_providers::{Discovery, DiscoveryOptions, ProviderSpec};

# async fn run() -> unravel_agent_providers::ProviderResult<()> {
let spec = ProviderSpec::openai("sk-...".to_string());
let discovery = Discovery::new();
let options = DiscoveryOptions::default();
let models = discovery.discover(&spec, &options).await?;

for model in &models {
    println!("{}: tools={:?}, deprecated={}, from_endpoint={}",
        model.model_id, model.tool_support, model.deprecated, model.from_endpoint);
}
# Ok(())
# }
```

### Tool workflow filtering

Use `DiscoveryOptions::for_tools()` to filter out deprecated and
explicitly non-tool-capable models. Unknown capability is preserved
(not silently upgraded to supported).

### Explicit model overrides

Add explicit model IDs that are included even if not discovered:

```rust
let options = DiscoveryOptions {
    explicit_model_ids: vec!["my-custom-model".to_string()],
    ..DiscoveryOptions::default()
};
```

### Cache behavior

- Instance-owned (no process-global state).
- TTL: 5 minutes by default (`Discovery::with_ttl` to customize).
- Keyed by: endpoint + account identity + protocol + discovery options.
- No secrets in cache keys (uses a hash of the API key).

## Ollama discovery

Ollama exposes a native `/api/tags` endpoint (separate from the
OpenAI-compatible `/v1/models`). The `/v1` base URL is used for the
chat completions runtime, while `/api/tags` (at the root, not under
`/v1`) is used for native discovery.

```rust
use unravel_agent_providers::discovery::fetch_ollama_tags;

# async fn run() -> unravel_agent_providers::ProviderResult<()> {
let tags = fetch_ollama_tags("http://localhost:11434/v1").await?;
for tag in &tags {
    println!("Ollama model: {tag}");
}
# Ok(())
# }
```

## Error handling

Errors are classified as permanent or transient:

- **Permanent**: auth failure, invalid request, unsupported protocol,
  deprecated/non-tool model, malformed response. Not retried.
- **Transient**: rate limit (with Retry-After), server error, network
  failure. Retried by the agent loop.

All error messages are redacted — no API keys, bearer tokens, or
secret-bearing URLs leak into error output.

## Credential safety

- Secrets are wrapped in `Secret` with redacted `Debug` and `Display`.
- No plaintext persistence.
- No secrets in cache keys (uses a hash).
- No secrets in error messages (defense-in-depth redaction).
- Explicit API key over env variable, but env is supported as fallback.

## HTTP transport

- `reqwest` with `rustls` (no native TLS dependency).
- No redirects (`redirect::Policy::none()`).
- Bounded response body (4 MiB for JSON, 16 MiB for SSE streams).
- Per-request timeouts.
- SSE fragmentation handling with proper UTF-8 boundary safety.

## Streaming

The adapter implements both `complete` and `stream` on the core `Model`
trait. Streaming handles:

- Fragmented SSE events across chunk boundaries.
- Multi-byte UTF-8 characters split across chunks.
- Interleaved tool-call assembly with stable IDs.
- `[DONE]` sentinel detection.
- Final finish reason reporting.
- Usage reporting (no fabricated counts).
- Truncated tool calls are never executed (non-dispatchable finish
  reasons are rejected by the loop).

## Attribution

See `NOTICE` for AGPL attribution of adapted Khadim code.

## Application adapters

Use `with_chat_options(ChatOptions { headers, timeout })` for application-owned
headers and request timeouts. OpenRouter attribution and Copilot dynamic headers
belong to the consuming application; the shared provider has no branded
defaults. Custom header values are excluded from debug output and redacted from
HTTP error bodies. No-redirect transport and response size limits still apply.

Applications with their own persisted history and tool-argument policies can use
`complete_chat` and `stream_chat` with a `ChatRequest`. These methods share the
canonical provider's HTTP and SSE implementation but return `ChatResponse` with
raw argument strings. They do not repair arguments or reconcile history.
Validate calls before dispatching them through the runtime. A missing finish
reason remains `None`, and an unfinished stream returns no tool calls.

The canonical `Model` implementation continues to reject unresolved tool
outcomes, missing finish reasons, and invalid or non-object tool arguments.
`ChatStreamEvent` exposes start, actual deltas, and terminal response snapshots,
including usage that arrives after the finish event. Unknown usage remains
unknown; token counts use `u64`, including optional cache read and write counts.

## Limitations

- Only `OpenAiChatCompletions` protocol is implemented. `OpenAiResponses`
  and `AnthropicMessages` are recognized but rejected with a typed
  error (follow-up work).
- No OAuth implementation (explicit API key or env variable only).
- No automatic live API calls in tests (all tests use mock HTTP
  servers).
- Discovery ≠ invocation: capability metadata does not guarantee
  runtime success.