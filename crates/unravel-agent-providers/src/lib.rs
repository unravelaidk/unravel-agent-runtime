//! OpenAI-compatible model providers and capability-aware model discovery
//! for Unravel applications.
//!
//! This crate delivers the first real provider vertical slice:
//! OpenAI-compatible Chat Completions with tools, streaming, usage, and
//! multimodal image payloads — plus robust model discovery from
//! [Models.dev](https://models.dev), generic `/models` endpoints, and
//! local Ollama.
//!
//! # Quick start
//!
//! ```no_run
//! use unravel_agent_runtime::{Message, Model, ModelRequest, Session};
//! use unravel_agent_providers::{OpenAiCompatProvider, ProviderSpec};
//!
//! # async fn run() -> unravel_agent_runtime::Result<()> {
//! let spec = ProviderSpec::openai("sk-...".to_string());
//! let provider = OpenAiCompatProvider::new(spec, "gpt-4o-mini")?;
//!
//! let mut session = Session::new("demo");
//! session.messages.push(Message::user_text("Hello!"));
//!
//! let request = ModelRequest::new("demo", session.messages);
//! let response = provider.complete(request).await?;
//! println!("{}", response.content);
//! # Ok(())
//! # }
//! ```
//!
//! # Discovery
//!
//! Discovery fetches metadata from Models.dev and optionally merges it
//! with a provider's authenticated `/models` endpoint. Discovery
//! **describes** capabilities; it does not guarantee invocation success.
//! A model that reports `tool_call: true` may still fail at runtime —
//! the adapter maps those failures to typed errors.
//!
//! # Attribution
//!
//! See `NOTICE` for AGPL attribution of adapted Khadim code.

mod cache;
pub mod catalog;
mod chat;
mod completions;
pub mod discovery;
mod error;
mod images;
mod messages;
mod secret;
mod spec;
mod transport;

pub use cache::{Clock, SystemClock, TestClock};
pub use chat::{
    ChatEventSink, ChatOptions, ChatRequest, ChatResponse, ChatStreamEvent, ChatToolCall,
};
pub use completions::OpenAiCompatProvider;
pub use discovery::{
    fetch_ollama_tags, CatalogModel, CatalogProvider, DiscoveredModel, Discovery, DiscoveryOptions,
    ModalitySupport, ToolSupport,
};
pub use error::{ProviderError, ProviderResult};
pub use images::validate_image_messages;
pub use secret::Secret;
pub use spec::{
    KeyPolicy, Protocol, ProviderSpec, BUILTIN_NVIDIA, BUILTIN_OLLAMA, BUILTIN_OPENAI,
    BUILTIN_OPENROUTER,
};

// Re-export core types commonly needed alongside provider construction.
pub use unravel_agent_runtime::{
    FinishReason, Model, ModelError, ModelRequest, ModelResponse, Usage,
};
