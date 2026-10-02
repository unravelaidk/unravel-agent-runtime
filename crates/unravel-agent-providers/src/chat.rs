//! OpenAI Chat Completions wire boundary for applications with their own history
//! and tool-argument policies. Unlike `ModelResponse`, these tool arguments are
//! unvalidated strings. Never dispatch them without application validation.

use crate::{ProviderError, ProviderResult};
use serde_json::Value;
use std::time::Duration;
use unravel_agent_runtime::{
    FinishReason, ModelResponse, Reasoning, Sampling, StreamDelta, ToolCall, Usage,
};

/// Concrete transport overrides. Redirects and response limits remain enforced.
#[derive(Clone)]
pub struct ChatOptions {
    /// Extra request headers, applied after bearer authentication. Applications
    /// own provider-specific identity and dynamic headers. Values are redacted
    /// from HTTP errors and never exposed by this type's `Debug` implementation.
    pub headers: reqwest::header::HeaderMap,
    /// Overall HTTP request timeout, including streaming response consumption.
    pub timeout: Duration,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            headers: reqwest::header::HeaderMap::new(),
            timeout: Duration::from_secs(120),
        }
    }
}

impl std::fmt::Debug for ChatOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatOptions")
            .field("header_count", &self.headers.len())
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// An already-transformed OpenAI conversation. This boundary deliberately does
/// not normalize IDs, repair arguments, or reconcile application history.
/// Image URL parts are validated before HTTP using the canonical PNG/JPEG byte,
/// pixel, frame, and remote URL limits. This applies to every message role.
#[derive(Debug, Clone, Default)]
pub struct ChatRequest {
    pub messages: Vec<Value>,
    /// `None` omits tools and tool_choice; `Some([])` sends an empty tool list
    /// with automatic tool selection, for endpoints requiring that wire shape.
    pub tools: Option<Vec<Value>>,
    pub max_output: Option<u32>,
    pub sampling: Option<Sampling>,
}

/// A complete wire tool call, retaining the original argument string for an
/// application's parser. This is not a validated runtime `ToolCall`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChatToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// A wire response. A missing finish reason remains explicit; the canonical
/// `Model` implementation rejects it before returning a runtime response.
#[derive(Debug, Clone, Default)]
pub struct ChatResponse {
    pub content: String,
    pub tool_calls: Vec<ChatToolCall>,
    pub usage: Usage,
    pub finish_reason: Option<FinishReason>,
    pub reasoning: Option<Reasoning>,
}

impl ChatResponse {
    pub(crate) fn into_model_response(self) -> ProviderResult<ModelResponse> {
        let finish_reason = self.finish_reason.ok_or_else(|| {
            ProviderError::malformed("completion did not include a finish reason")
        })?;
        let mut tool_calls = Vec::with_capacity(self.tool_calls.len());
        for call in self.tool_calls {
            if call.id.is_empty() || call.name.is_empty() {
                return Err(ProviderError::invalid(
                    "tool call is missing a non-empty id or function name",
                ));
            }
            let arguments: Value = serde_json::from_str(&call.arguments)
                .map_err(|_| ProviderError::invalid("tool call arguments are not valid JSON"))?;
            if !arguments.is_object() {
                return Err(ProviderError::invalid(
                    "tool call arguments must be a JSON object",
                ));
            }
            tool_calls.push(ToolCall {
                id: call.id,
                name: call.name,
                arguments,
            });
        }
        Ok(ModelResponse {
            content: self.content,
            tool_calls,
            usage: self.usage,
            finish_reason,
            reasoning: self.reasoning,
        })
    }
}

/// Provider lifecycle plus the same deltas emitted by the canonical `Model`.
/// `Finished` precedes any trailing usage-only events, just as on the wire.
pub enum ChatStreamEvent<'a> {
    Start,
    Delta(StreamDelta),
    Finished(&'a ChatResponse),
}

pub trait ChatEventSink: Send {
    fn on_event(&mut self, event: ChatStreamEvent<'_>);
}
