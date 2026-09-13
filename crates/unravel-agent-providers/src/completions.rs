//! OpenAI-compatible Chat Completions transport and canonical model adapter.
//!
//! The HTTP/SSE implementation is shared by the strict runtime `Model` and
//! applications using the explicit raw-argument `complete_chat` / `stream_chat`
//! boundary. See NOTICE for the original ServoLoop and Khadim provenance.

use crate::chat::{
    ChatEventSink, ChatOptions, ChatRequest, ChatResponse, ChatStreamEvent, ChatToolCall,
};
use crate::error::{redact_url, ProviderError, ProviderResult};
use crate::messages::{to_openai_messages, to_openai_tools};
use crate::spec::ProviderSpec;
use crate::transport::{
    build_client, extract_retry_after, is_done, join_url, map_http_error_with_secret,
    process_sse_chunk, read_bounded_json, read_bounded_text,
};
use crate::Discovery;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use unravel_agent_runtime::{
    DeltaSink, FinishReason, Model, ModelRequest, ModelResponse, Reasoning, StreamDelta, Usage,
};

/// An OpenAI-compatible provider with a bounded, no-redirect HTTP client.
pub struct OpenAiCompatProvider {
    spec: ProviderSpec,
    model_id: String,
    client: reqwest::Client,
    discovery: Option<Arc<Discovery>>,
    options: ChatOptions,
}

impl std::fmt::Debug for OpenAiCompatProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatProvider")
            .field("provider", &self.spec.id)
            .field("model", &self.model_id)
            .field("endpoint", &redact_url(&self.spec.resolve_endpoint()))
            .field("has_discovery", &self.discovery.is_some())
            .field("options", &self.options)
            .finish()
    }
}

impl OpenAiCompatProvider {
    /// Validate the protocol and key policy and create the secured transport.
    pub fn new(spec: ProviderSpec, model_id: impl Into<String>) -> ProviderResult<Self> {
        spec.validate()?;
        Ok(Self {
            spec,
            model_id: model_id.into(),
            client: build_client()?,
            discovery: None,
            options: ChatOptions::default(),
        })
    }

    /// Set concrete transport overrides without relaxing response bounds or
    /// redirect policy. Application identity headers belong here, not in defaults.
    pub fn with_chat_options(mut self, options: ChatOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_discovery(mut self, discovery: Arc<Discovery>) -> Self {
        self.discovery = Some(discovery);
        self
    }

    pub fn spec(&self) -> &ProviderSpec {
        &self.spec
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    fn build_payload(&self, request: &ChatRequest, stream: bool) -> Value {
        let mut payload = serde_json::json!({
            "model": self.model_id,
            "messages": request.messages,
            "stream": stream,
        });
        if let Some(tools) = &request.tools {
            payload["tools"] = serde_json::json!(tools);
            payload["tool_choice"] = serde_json::json!("auto");
        }
        if stream {
            payload["stream_options"] = serde_json::json!({"include_usage": true});
        }
        if let Some(max_output) = request.max_output {
            payload["max_tokens"] = serde_json::json!(max_output);
        }
        if let Some(sampling) = &request.sampling {
            if let Some(temp) = sampling.temperature {
                payload["temperature"] = serde_json::json!(temp);
            }
            if let Some(top_p) = sampling.top_p {
                payload["top_p"] = serde_json::json!(top_p);
            }
            if let Some(max_tokens) = sampling.max_tokens {
                payload["max_tokens"] = serde_json::json!(max_tokens);
            }
        }
        payload
    }

    /// Both public boundaries use this same authenticated, bounded request path.
    async fn send(&self, request: &ChatRequest, stream: bool) -> ProviderResult<reqwest::Response> {
        let payload = self.build_payload(request, stream);
        let url = join_url(&self.spec.resolve_endpoint(), "chat/completions");
        let mut builder = self
            .client
            .post(&url)
            .json(&payload)
            .timeout(self.options.timeout);
        let key = self.spec.resolve_key();
        if let Some(key) = &key {
            builder = builder.bearer_auth(key.as_str());
        }
        let response = builder
            .headers(self.options.headers.clone())
            .send()
            .await
            .map_err(|e| ProviderError::transient(format!("request failed: {e}")))?;
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        let retry_after = if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            extract_retry_after(&response)
        } else {
            None
        };
        let mut body = read_bounded_text(response).await;
        // Custom authentication headers can be echoed without an identifying
        // prefix. Treat every configured value as potentially sensitive.
        for value in self.options.headers.values() {
            if let Ok(value) = value.to_str() {
                if !value.is_empty() {
                    body = body.replace(value, "[REDACTED]");
                    if let Some((scheme, token)) = value.split_once(' ') {
                        let token = token.trim();
                        if scheme.eq_ignore_ascii_case("bearer") && !token.is_empty() {
                            body = body.replace(token, "[REDACTED]");
                        }
                    }
                }
            }
        }
        let mut error =
            map_http_error_with_secret(status, &body, key.as_ref().map(|key| key.as_str()));
        if let (
            Some(delay),
            ProviderError::Transient {
                redacted_detail, ..
            },
        ) = (retry_after, &error)
        {
            error = ProviderError::transient_with_retry(redacted_detail.clone(), delay);
        }
        Err(error)
    }

    /// Complete an application-owned wire history. Arguments remain raw strings;
    /// validation and any legacy repair must happen before runtime dispatch.
    pub async fn complete_chat(&self, request: ChatRequest) -> ProviderResult<ChatResponse> {
        self.complete_chat_inner(request, false).await
    }

    async fn complete_chat_inner(
        &self,
        request: ChatRequest,
        canonical: bool,
    ) -> ProviderResult<ChatResponse> {
        let response = self.send(&request, false).await?;
        parse_chat_completion(&read_bounded_json(response).await?, canonical)
    }

    /// Stream an application-owned wire history through the same SSE parser as
    /// `Model::stream`. A stream without a finish reason returns no tool calls;
    /// callers can inspect `finish_reason` without inventing terminal success.
    pub async fn stream_chat(
        &self,
        request: ChatRequest,
        sink: &mut (dyn ChatEventSink + Send),
    ) -> ProviderResult<ChatResponse> {
        self.stream_chat_inner(request, sink, false).await
    }

    async fn stream_chat_inner(
        &self,
        request: ChatRequest,
        sink: &mut (dyn ChatEventSink + Send),
        canonical: bool,
    ) -> ProviderResult<ChatResponse> {
        use futures_util::StreamExt;
        let response = self.send(&request, true).await?;
        sink.on_event(ChatStreamEvent::Start);
        let mut buffer = Vec::new();
        let mut total_bytes = 0usize;
        let mut result = ChatResponse::default();
        let mut calls = BTreeMap::<u32, ChatToolCall>::new();
        let mut stream = response.bytes_stream();
        'chunks: while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|e| ProviderError::transient(format!("stream chunk error: {e}")))?;
            let (events, _) = process_sse_chunk(&mut buffer, &chunk, &mut total_bytes)?;
            for event in events {
                if is_done(&event.data) {
                    if canonical && result.finish_reason.is_none() {
                        return Err(ProviderError::malformed(
                            "SSE stream ended with [DONE] but no finish reason",
                        ));
                    }
                    break 'chunks;
                }
                let payload: Value = serde_json::from_str(&event.data).map_err(|e| {
                    ProviderError::malformed(format!("failed to parse SSE JSON: {e}"))
                })?;
                if payload.get("error").is_some() {
                    return Err(ProviderError::invalid(
                        "SSE stream returned an error payload",
                    ));
                }
                if let Some(raw_usage) = response_usage(&payload) {
                    let usage = parse_usage_value(raw_usage);
                    result.usage = usage.clone();
                    sink.on_event(ChatStreamEvent::Delta(StreamDelta::Usage { usage }));
                }
                let Some(choice) = payload
                    .get("choices")
                    .and_then(Value::as_array)
                    .and_then(|c| c.first())
                else {
                    continue;
                };
                let delta = &choice["delta"];
                let text = delta
                    .get("content")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty());
                let reasoning = extract_reasoning_text(delta);
                if result.finish_reason.is_some()
                    && (text.is_some()
                        || reasoning.is_some()
                        || delta.get("tool_calls").is_some()
                        || choice
                            .get("finish_reason")
                            .is_some_and(|reason| !reason.is_null()))
                {
                    return Err(ProviderError::malformed(
                        "SSE stream mutated completion after finish reason",
                    ));
                }
                if let Some(text) = text {
                    result.content.push_str(text);
                    sink.on_event(ChatStreamEvent::Delta(StreamDelta::Text {
                        text: text.to_owned(),
                    }));
                }
                if let Some(text) = reasoning {
                    result
                        .reasoning
                        .get_or_insert_with(Reasoning::default)
                        .text
                        .get_or_insert_with(String::new)
                        .push_str(&text);
                    sink.on_event(ChatStreamEvent::Delta(StreamDelta::Reasoning { text }));
                }
                if canonical
                    && delta
                        .get("tool_calls")
                        .is_some_and(|calls| !calls.is_array())
                {
                    return Err(ProviderError::malformed(
                        "streamed tool_calls must be an array",
                    ));
                }
                if let Some(fragments) = delta.get("tool_calls").and_then(Value::as_array) {
                    for call in fragments {
                        // Canonical streams retain the runtime assembler's slot
                        // semantics; the raw boundary preserves legacy first-ID
                        // and missing-index behavior without repairing arguments.
                        let index = call
                            .get("index")
                            .and_then(Value::as_u64)
                            .map(|i| i as u32)
                            .unwrap_or_else(|| if canonical { 0 } else { calls.len() as u32 });
                        let id = call.get("id").and_then(Value::as_str);
                        let name = call
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(Value::as_str);
                        let arguments = call
                            .get("function")
                            .and_then(|f| f.get("arguments"))
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        let entry = calls.entry(index).or_default();
                        if let Some(id) = id {
                            if canonical || entry.id.is_empty() {
                                entry.id = id.to_owned();
                            }
                        }
                        if let Some(name) = name {
                            if canonical || entry.name.is_empty() {
                                entry.name = name.to_owned();
                            }
                        }
                        entry.arguments.push_str(arguments);
                        sink.on_event(ChatStreamEvent::Delta(StreamDelta::ToolCall {
                            index,
                            id: id.map(str::to_owned),
                            name: name.map(str::to_owned),
                            arguments: arguments.to_owned(),
                        }));
                    }
                }
                if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                    result.finish_reason = Some(parse_finish_reason(reason));
                    result
                        .tool_calls
                        .extend(std::mem::take(&mut calls).into_values());
                    sink.on_event(ChatStreamEvent::Finished(&result));
                }
            }
        }
        if !buffer.is_empty() {
            return Err(ProviderError::malformed(
                "SSE stream ended with an incomplete event",
            ));
        }
        if canonical && result.finish_reason.is_none() {
            return Err(ProviderError::malformed(
                "SSE stream ended without a finish reason",
            ));
        }
        Ok(result)
    }

    #[cfg(test)]
    fn parse_completion(body: &Value) -> ProviderResult<ModelResponse> {
        parse_chat_completion(body, true)?.into_model_response()
    }
}

fn parse_chat_completion(body: &Value, canonical: bool) -> ProviderResult<ChatResponse> {
    let choice = body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .ok_or_else(|| ProviderError::malformed("response did not include any choices"))?;
    let message = choice
        .get("message")
        .ok_or_else(|| ProviderError::malformed("choice did not include a message"))?;
    let mut tool_calls = Vec::new();
    let raw_calls = message.get("tool_calls");
    if canonical && raw_calls.is_some_and(|calls| !calls.is_array()) {
        return Err(ProviderError::malformed(
            "message tool_calls must be an array",
        ));
    }
    if let Some(calls) = raw_calls.and_then(Value::as_array) {
        tool_calls.reserve(calls.len());
        for call in calls {
            let id = call.get("id").and_then(Value::as_str);
            let name = call
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str);
            let arguments = call
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(Value::as_str);
            match (id, name, arguments) {
                (Some(id), Some(name), Some(arguments)) => tool_calls.push(ChatToolCall {
                    id: id.to_owned(),
                    name: name.to_owned(),
                    arguments: arguments.to_owned(),
                }),
                _ if canonical => {
                    return Err(ProviderError::malformed(
                        "tool call is missing a string id, function name, or arguments",
                    ))
                }
                _ => {
                    // Preserve legacy whole-array deserialization behavior at
                    // the raw boundary, without weakening canonical validation.
                    tool_calls.clear();
                    break;
                }
            }
        }
    }
    Ok(ChatResponse {
        content: message
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        tool_calls,
        usage: parse_usage(body),
        finish_reason: choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .map(parse_finish_reason),
        reasoning: extract_reasoning_text(message).map(|text| Reasoning { text: Some(text) }),
    })
}

/// Reasoning aliases used by OpenAI-compatible providers, in precedence order.
fn extract_reasoning_text(message: &Value) -> Option<String> {
    ["reasoning_content", "reasoning", "reasoning_text"]
        .iter()
        .find_map(|field| {
            message
                .get(*field)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
        })
}

fn response_usage(body: &Value) -> Option<&Value> {
    body.get("usage").filter(|v| !v.is_null()).or_else(|| {
        body.get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
            .and_then(|choice| choice.get("usage"))
            .filter(|v| !v.is_null())
    })
}

fn parse_usage(body: &Value) -> Usage {
    response_usage(body)
        .map(parse_usage_value)
        .unwrap_or_else(Usage::unknown)
}

fn parse_usage_value(usage: &Value) -> Usage {
    Usage {
        prompt_tokens: usage.get("prompt_tokens").and_then(Value::as_u64),
        completion_tokens: usage.get("completion_tokens").and_then(Value::as_u64),
        total_tokens: usage.get("total_tokens").and_then(Value::as_u64),
        cache_read_tokens: usage
            .get("prompt_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(Value::as_u64),
        cache_write_tokens: usage
            .get("cache_creation_input_tokens")
            .and_then(Value::as_u64),
    }
}

fn parse_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "tool_calls" | "function_call" => FinishReason::ToolCall,
        "length" => FinishReason::MaxOutput,
        "content_filter" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_owned()),
    }
}

fn canonical_request(request: ModelRequest) -> ProviderResult<ChatRequest> {
    let messages = to_openai_messages(&request.messages)?;
    let tools = if request.tools.is_empty() {
        None
    } else {
        Some(to_openai_tools(&request.tools))
    };
    Ok(ChatRequest {
        messages,
        tools,
        max_output: request.max_output,
        sampling: request.sampling,
    })
}

struct CanonicalSink<'a>(&'a mut (dyn DeltaSink + Send));

impl ChatEventSink for CanonicalSink<'_> {
    fn on_event(&mut self, event: ChatStreamEvent<'_>) {
        if let ChatStreamEvent::Delta(delta) = event {
            self.0.on_delta(delta);
        }
    }
}

#[async_trait::async_trait]
impl Model for OpenAiCompatProvider {
    async fn complete(
        &self,
        request: ModelRequest,
    ) -> unravel_agent_runtime::Result<ModelResponse> {
        self.complete_chat_inner(canonical_request(request)?, true)
            .await?
            .into_model_response()
            .map_err(Into::into)
    }

    async fn stream(
        &self,
        request: ModelRequest,
        sink: &mut (dyn DeltaSink + Send),
    ) -> unravel_agent_runtime::Result<ModelResponse> {
        self.stream_chat_inner(canonical_request(request)?, &mut CanonicalSink(sink), true)
            .await?
            .into_model_response()
            .map_err(Into::into)
    }

    fn can_stream(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_finish_reason_mappings() {
        assert_eq!(parse_finish_reason("stop"), FinishReason::Stop);
        assert_eq!(parse_finish_reason("tool_calls"), FinishReason::ToolCall);
        assert_eq!(parse_finish_reason("function_call"), FinishReason::ToolCall);
        assert_eq!(parse_finish_reason("length"), FinishReason::MaxOutput);
        assert_eq!(
            parse_finish_reason("content_filter"),
            FinishReason::ContentFilter
        );
        assert_eq!(
            parse_finish_reason("weird"),
            FinishReason::Other("weird".to_string())
        );
    }

    #[test]
    fn parse_usage_from_body() {
        let body = serde_json::json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        });
        let usage = parse_usage(&body);
        assert_eq!(usage.prompt_tokens, Some(10));
        assert_eq!(usage.completion_tokens, Some(5));
        assert_eq!(usage.total_tokens, Some(15));
    }

    #[test]
    fn parse_usage_unknown_when_absent() {
        let body = serde_json::json!({"choices": []});
        let usage = parse_usage(&body);
        assert!(usage.is_unknown());
    }

    #[test]
    fn parse_completion_with_tool_calls() {
        let body = serde_json::json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": "{\"x\": 42}",
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
        });
        let response = OpenAiCompatProvider::parse_completion(&body).unwrap();
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].id, "call_1");
        assert_eq!(response.tool_calls[0].name, "echo");
        assert_eq!(response.tool_calls[0].arguments, json!({"x": 42}));
        assert_eq!(response.finish_reason, FinishReason::ToolCall);
        assert_eq!(response.usage.prompt_tokens, Some(10));
    }

    #[test]
    fn parse_completion_text_only() {
        let body = serde_json::json!({
            "choices": [{
                "message": {
                    "content": "hello world",
                },
                "finish_reason": "stop",
            }],
        });
        let response = OpenAiCompatProvider::parse_completion(&body).unwrap();
        assert_eq!(response.content, "hello world");
        assert_eq!(response.finish_reason, FinishReason::Stop);
        assert!(response.usage.is_unknown());
    }

    #[test]
    fn parse_completion_rejects_malformed_tool_shape() {
        let body = json!({
            "choices": [{
                "message": {"tool_calls": [{"function": {"name": "echo", "arguments": "{}"}}]},
                "finish_reason": "tool_calls"
            }]
        });
        assert!(OpenAiCompatProvider::parse_completion(&body).is_err());

        let wrong_type = json!({
            "choices": [{
                "message": {"tool_calls": "not-an-array"},
                "finish_reason": "stop"
            }]
        });
        assert!(OpenAiCompatProvider::parse_completion(&wrong_type).is_err());
    }

    #[test]
    fn parse_completion_with_reasoning() {
        let body = serde_json::json!({
            "choices": [{
                "message": {
                    "content": "answer",
                    "reasoning_content": "step by step",
                },
                "finish_reason": "stop",
            }],
        });
        let response = OpenAiCompatProvider::parse_completion(&body).unwrap();
        assert_eq!(
            response.reasoning.as_ref().and_then(|r| r.text.as_deref()),
            Some("step by step")
        );
    }

    #[test]
    fn parse_completion_missing_choices_is_error() {
        let body = serde_json::json!({"error": "bad"});
        let result = OpenAiCompatProvider::parse_completion(&body);
        assert!(result.is_err());
    }
}
