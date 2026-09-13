//! Provider-neutral model contracts.
//!
//! A [`Model`] implementation talks to a single provider (OpenAI, Anthropic, a
//! local server, a fake). The core loop only ever calls [`Model::complete`]
//! or [`Model::stream`]; streaming is an optional optimization whose default
//! implementation falls back to `complete`. Everything a provider returns is
//! assembled into a single [`ModelResponse`] before any tool executes, so
//! tool dispatch never sees partial streamed arguments.

use crate::{Message, Result, ToolCall, ToolDefinition};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Sampling parameters for a model request.
///
/// All fields are optional so a provider that does not support a knob can
/// simply ignore it instead of being forced to forward an unsupported value.
/// The loop never sets a field a provider has not opted into.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Sampling {
    /// Sampling temperature. `None` means the provider default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus (top-p) sampling. `None` means the provider default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Maximum number of completion tokens. `None` means the provider default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

impl Sampling {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

/// A provider-neutral model request.
///
/// `session_id` lets a provider correlate requests within a conversation and
/// `max_output` bounds the response. `sampling` carries optional knobs; the
/// loop never forces a sampling value the provider has not asked for.
#[derive(Debug, Clone)]
pub struct ModelRequest {
    /// Identifier of the owning session, forwarded for provider-side tracing.
    pub session_id: String,
    /// Full conversation history sent to the model.
    pub messages: Vec<Message>,
    /// Tool schemas the model may call.
    pub tools: Vec<ToolDefinition>,
    /// Bounded response budget (completion tokens). `None` means use the
    /// provider default.
    pub max_output: Option<u32>,
    /// Optional sampling parameters.
    pub sampling: Option<Sampling>,
    /// Per-attempt deadline for the model call or stream. The loop enforces
    /// this with a `tokio::time::timeout`; a provider that ignores it still
    /// gets bounded because the outer `select!` cancels the future.
    pub deadline: Option<Duration>,
}

impl ModelRequest {
    pub fn new(session_id: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            session_id: session_id.into(),
            messages,
            tools: Vec::new(),
            max_output: None,
            sampling: None,
            deadline: None,
        }
    }
}

/// Token usage reported by a model response.
///
/// Every field is optional: a provider that does not report usage returns
/// [`Usage::unknown()`]. The loop never invents token counts.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Usage {
    /// Prompt tokens consumed, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    /// Completion tokens generated, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    /// Total tokens, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    /// Prompt tokens read from a provider cache, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    /// Prompt tokens written to a provider cache, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
}

impl Usage {
    pub fn new(prompt: u64, completion: u64) -> Self {
        Self {
            prompt_tokens: Some(prompt),
            completion_tokens: Some(completion),
            total_tokens: prompt.checked_add(completion),
            ..Self::default()
        }
    }

    /// Sentinel for "the provider did not report usage".
    pub fn unknown() -> Self {
        Self::default()
    }

    /// `true` when no usage figure was reported.
    pub fn is_unknown(&self) -> bool {
        self.prompt_tokens.is_none()
            && self.completion_tokens.is_none()
            && self.total_tokens.is_none()
            && self.cache_read_tokens.is_none()
            && self.cache_write_tokens.is_none()
    }
}

/// Why the model stopped generating.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum FinishReason {
    /// The model called a tool (or tools) and stopped for tool execution.
    ToolCall,
    /// The model reached a natural stop.
    #[default]
    Stop,
    /// The configured max output budget was reached.
    MaxOutput,
    /// Content-filter or safety policy stopped generation.
    ContentFilter,
    /// The provider reported an unknown or unrecognised reason.
    Other(String),
}

impl FinishReason {
    pub fn is_tool_call(&self) -> bool {
        matches!(self, FinishReason::ToolCall)
    }

    /// Returns `true` when this finish reason permits the response to be
    /// used as a final answer or for tool dispatch. `Stop` and `ToolCall`
    /// are dispatchable; `MaxOutput` (truncated), `ContentFilter`, and
    /// `Other` are not — the loop rejects them to avoid acting on
    /// incomplete or filtered output.
    pub fn is_dispatchable(&self) -> bool {
        matches!(self, FinishReason::Stop | FinishReason::ToolCall)
    }
}

/// Optional provider-visible reasoning metadata.
///
/// Providers that do not expose reasoning leave this as `None`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(transparent)]
pub struct Reasoning {
    pub text: Option<String>,
}

/// A fully assembled model response.
///
/// Tool calls must contain complete, provider-validated arguments, never
/// partial stream fragments. The loop checks IDs and applies its configured
/// tool-availability policy only after successful terminal completion and a
/// cancellation check.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ModelResponse {
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Token usage. [`Usage::unknown()`] when the provider omits it.
    #[serde(default)]
    pub usage: Usage,
    /// Why the model stopped generating.
    #[serde(default)]
    pub finish_reason: FinishReason,
    /// Optional reasoning trace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
}

impl ModelResponse {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            ..Default::default()
        }
    }
}

/// A single streamed fragment of a model response.
///
/// Deltas are *never* executed as tools. The loop assembles them into a
/// [`ModelResponse`] and only dispatches tool calls from the complete
/// response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StreamDelta {
    /// A fragment of assistant text.
    Text { text: String },
    /// Start or continuation of a tool call's arguments.
    ///
    /// `index` identifies the tool-call slot across fragments (0-based).
    /// `id` and `name` appear on the first fragment for a slot and are
    /// carried into the assembled call. `arguments` is a JSON string
    /// fragment that the assembler concatenates before parsing.
    ToolCall {
        index: u32,
        id: Option<String>,
        name: Option<String>,
        arguments: String,
    },
    /// Usage reported during or at the end of the stream.
    Usage { usage: Usage },
    /// Reasoning trace fragment.
    Reasoning { text: String },
}

/// Receives streaming deltas from a model.
///
/// The loop passes a `DeltaSink` to [`Model::stream`]. Implementations call
/// [`on_delta`](DeltaSink::on_delta) for each fragment, then return the
/// assembled [`ModelResponse`]. Deltas are informational; tools execute only
/// from the returned complete response.
pub trait DeltaSink: Send {
    fn on_delta(&mut self, delta: StreamDelta);
}

/// A no-op sink used by the default `stream` fallback and tests that don't
/// care about deltas.
#[derive(Debug, Default)]
pub struct NoopDeltaSink;

impl DeltaSink for NoopDeltaSink {
    fn on_delta(&mut self, _delta: StreamDelta) {}
}

/// A provider-neutral model.
///
/// Implementors only **need** to implement [`complete`]; [`stream`] has a
/// default that falls back to `complete`. Streaming is an optional
/// optimization — the loop assembles every delta into a single
/// [`ModelResponse`] before any tool executes, so a non-streaming provider
/// and a streaming provider are interchangeable.
///
/// [`complete`]: Model::complete
/// [`stream`]: Model::stream
#[async_trait]
pub trait Model: Send + Sync {
    /// Produce a complete response in one shot.
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse>;

    /// Produce a stream of deltas followed by the assembled response.
    ///
    /// The default implementation forwards to [`complete`](Model::complete)
    /// and emits no deltas. Override this (and set
    /// [`can_stream`](Model::can_stream) to `true`) to stream deltas from
    /// providers that support it.
    async fn stream(
        &self,
        request: ModelRequest,
        _sink: &mut (dyn DeltaSink + Send),
    ) -> Result<ModelResponse> {
        self.complete(request).await
    }

    /// Whether this model streams via [`stream`](Model::stream) or just uses
    /// the `complete` fallback. The default is `false`.
    fn can_stream(&self) -> bool {
        false
    }
}

/// Assembles [`StreamDelta`]s into a [`ModelResponse`].
///
/// The loop uses this when consuming a model stream. Tool-call arguments are
/// concatenated by `index` and parsed as JSON only on terminal completion;
/// partial or malformed fragments never produce a dispatchable call.
#[derive(Debug, Default)]
pub struct StreamAssembler {
    content: String,
    reasoning: String,
    usage: Usage,
    /// Indexed tool-call accumulator: (id, name, concatenated arguments).
    tool_calls: std::collections::BTreeMap<u32, (String, String, String)>,
}

impl StreamAssembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold a delta into the assembled response.
    pub fn push(&mut self, delta: StreamDelta) {
        match delta {
            StreamDelta::Text { text } => self.content.push_str(&text),
            StreamDelta::Reasoning { text } => self.reasoning.push_str(&text),
            StreamDelta::Usage { usage } => self.usage = usage,
            StreamDelta::ToolCall {
                index,
                id,
                name,
                arguments,
            } => {
                let entry = self.tool_calls.entry(index).or_default();
                if let Some(id) = id {
                    entry.0 = id;
                }
                if let Some(name) = name {
                    entry.1 = name;
                }
                entry.2.push_str(&arguments);
            }
        }
    }

    /// Finalise the assembled response.
    ///
    /// Tool-call arguments are parsed as JSON. A fragment that fails to parse
    /// produces a raw JSON string value, never a silently dropped call.
    /// Providers must validate assembled arguments before returning a
    /// dispatchable response; this helper does not infer tool schemas.
    pub fn finish(self) -> ModelResponse {
        let mut tool_calls = Vec::with_capacity(self.tool_calls.len());
        for (_, (id, name, args)) in self.tool_calls {
            let arguments = if args.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::from_str(&args)
                    .unwrap_or_else(|_| serde_json::Value::String(args.clone()))
            };
            tool_calls.push(ToolCall {
                id,
                name,
                arguments,
            });
        }
        let finish_reason = if tool_calls.is_empty() {
            FinishReason::Stop
        } else {
            FinishReason::ToolCall
        };
        ModelResponse {
            content: self.content,
            tool_calls,
            usage: self.usage,
            finish_reason,
            reasoning: if self.reasoning.is_empty() {
                None
            } else {
                Some(Reasoning {
                    text: Some(self.reasoning),
                })
            },
        }
    }
}

#[cfg(test)]
mod usage_tests {
    use super::Usage;
    use serde_json::json;

    #[test]
    fn usage_round_trip_keeps_large_and_partial_reports_without_zero_filling() {
        let legacy = json!({
            "prompt_tokens": 4_294_967_296u64,
            "completion_tokens": 12,
            "total_tokens": 4_294_967_308u64,
        });
        let usage: Usage = serde_json::from_value(legacy.clone()).unwrap();
        assert_eq!(serde_json::to_value(&usage).unwrap(), legacy);

        let cached = json!({"cache_read_tokens": u64::MAX, "cache_write_tokens": 0});
        let usage: Usage = serde_json::from_value(cached.clone()).unwrap();
        assert!(!usage.is_unknown());
        assert_eq!(serde_json::to_value(&usage).unwrap(), cached);
    }
}
