//! Domain-independent model, tool, and agent-loop primitives.
//!
//! [`AgentLoop::run`] owns an ordinary run: prompts, bounded model turns,
//! classified retries, tool dispatch, cancellation and strict interrupted-history
//! reconciliation. [`AgentLoop::turn`] exposes that same execution engine to
//! applications that own their prompts, workflow hooks and recovery policy.
//!
//! A [`Model`] returns a fully assembled [`ModelResponse`]. Streaming deltas
//! carry observable payloads but never execute partial tool calls. A [`Tool`]
//! returns text and structured metadata. Tools execute sequentially unless both
//! the tool and [`LoopConfig`] explicitly opt into bounded parallel execution.
//! Completed parallel results retain transcript order even when a caller drops
//! the pending turn future.
//!
//! Strict validation is the default. [`ResponseValidation::ToolFeedback`] lets
//! application-owned turns preserve unavailable-tool feedback and historical
//! call-ID reuse; it does not relax `run`'s strict persisted-history reconciliation.
//! Such workflows should use `turn` and retain their own recovery policy.
//! Existing unknown outcomes always require explicit resolution before execution.
//!
//! Run the credential-free model → tool → model example from the repository root:
//!
//! ```text
//! cargo run --locked -p unravel-agent-runtime --example tool_roundtrip
//! ```
//!
//! Shared Chat Completions transport and model discovery live in the sibling
//! `unravel-agent-providers` crate. Repository tools, robot policies, journals,
//! application prompts and UI projections remain outside this crate.

mod error;
mod event;
mod model;
mod orchestrator;
mod session;
mod stop;
mod tool;

pub use error::{Error, ModelError, Result, Retryability};
pub use event::{Event, EventSink, NoopEventSink};
pub use model::{
    DeltaSink, FinishReason, Model, ModelRequest, ModelResponse, NoopDeltaSink, Reasoning,
    Sampling, StreamAssembler, StreamDelta, Usage,
};
pub use orchestrator::{AgentLoop, LoopConfig, ResponseValidation, TurnOutcome};
pub use session::{Content, ContentPart, ImageSource, Message, Session};
pub use stop::StopToken;
pub use tool::{validate_tool_calls, Tool, ToolCall, ToolDefinition, ToolOutput, ToolRegistry};
