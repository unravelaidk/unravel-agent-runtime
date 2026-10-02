//! Canonical-to-OpenAI message transformation.
//!
//! Converts [`unravel_agent_runtime::Message`] into the OpenAI Chat Completions
//! wire format. Real image blocks (base64 data URLs and remote URLs) are
//! preserved — never dropped or replaced with a textual marker.
//! Reasoning traces and tool-call IDs are carried through intact.
//! `ToolUnknown` messages are rejected — the caller must reconcile them
//! before sending.

use crate::error::ProviderError;
use crate::images::validate_image_messages;
use crate::ProviderResult;
use serde_json::{json, Value};
use unravel_agent_runtime::{ContentPart, ImageSource, Message, ToolDefinition};

/// Convert canonical messages into OpenAI Chat Completions wire format.
///
/// Returns an error if the history contains a `ToolUnknown` message —
/// the model must never see an implied success or an unresolved unknown.
pub fn to_openai_messages(messages: &[Message]) -> ProviderResult<Vec<Value>> {
    validate_image_messages(messages)?;
    let mut converted = Vec::with_capacity(messages.len());

    for message in messages {
        match message {
            Message::System { content } => {
                converted.push(json!({
                    "role": "system",
                    "content": content,
                }));
            }
            Message::User { content } => {
                let wire = user_content_to_wire(content);
                converted.push(json!({
                    "role": "user",
                    "content": wire,
                }));
            }
            Message::Assistant {
                content,
                tool_calls,
                reasoning,
            } => {
                let mut value = json!({
                    "role": "assistant",
                    "content": if content.is_empty() { Value::Null } else { json!(content) },
                });

                if !tool_calls.is_empty() {
                    let wire_calls: Vec<Value> = tool_calls
                        .iter()
                        .map(|call| {
                            json!({
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": serde_json::to_string(&call.arguments)
                                        .unwrap_or_else(|_| "{}".to_string()),
                                },
                            })
                        })
                        .collect();
                    value["tool_calls"] = Value::Array(wire_calls);
                }

                // Preserve reasoning trace for providers that support it.
                if let Some(r) = reasoning {
                    if let Some(text) = &r.text {
                        if !text.is_empty() {
                            value["reasoning_content"] = json!(text);
                        }
                    }
                }

                converted.push(value);
            }
            Message::Tool {
                call_id,
                name,
                content,
                is_error,
            } => {
                let tool_content = if *is_error {
                    json!({
                        "content": content,
                        "is_error": true,
                    })
                } else {
                    json!(content)
                };
                converted.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": tool_content,
                }));
            }
            Message::ToolUnknown { call_id, .. } => {
                return Err(ProviderError::UnresolvedToolUnknown {
                    call_id: call_id.clone(),
                });
            }
        }
    }

    Ok(converted)
}

/// Convert canonical user content (potentially multimodal) into the
/// OpenAI wire format.
///
/// - Text-only content becomes a plain string (the simplest wire form).
/// - Content with any image part becomes an array of content blocks.
/// - Image parts produce `{"type": "image_url", "image_url": {"url": …}}`
///   with the actual data URL or remote URL — never a textual marker.
fn user_content_to_wire(content: &unravel_agent_runtime::Content) -> Value {
    // If only text parts, use the simple string form.
    if content
        .parts
        .iter()
        .all(|p| matches!(p, ContentPart::Text { .. }))
    {
        let text: String = content
            .parts
            .iter()
            .filter_map(|p| {
                if let ContentPart::Text { text } = p {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");
        return json!(text);
    }

    // Multimodal: array of content blocks.
    let blocks: Vec<Value> = content
        .parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text } => json!({
                "type": "text",
                "text": text,
            }),
            ContentPart::Image { media_type, source } => {
                let url = image_source_to_url(source, media_type.as_deref());
                json!({
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                    },
                })
            }
        })
        .collect();

    Value::Array(blocks)
}

/// Convert an [`ImageSource`] into the URL expected by OpenAI-compatible
/// providers.
///
/// - `Base64` → `data:{media_type};base64,{data}` (a data URL).
/// - `Url` → the URL as-is (the provider resolves it).
fn image_source_to_url(source: &ImageSource, media_type: Option<&str>) -> String {
    match source {
        ImageSource::Base64 { data } => {
            let mime = media_type.unwrap_or("image/png");
            format!("data:{mime};base64,{data}")
        }
        ImageSource::Url { url } => url.clone(),
    }
}

/// Convert tool definitions into the OpenAI tools wire format.
pub fn to_openai_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use unravel_agent_runtime::{Content, ImageSource, Reasoning, ToolCall};

    #[test]
    fn system_message_becomes_system_role() {
        let messages = vec![Message::System {
            content: "you are helpful".into(),
        }];
        let wire = to_openai_messages(&messages).unwrap();
        assert_eq!(wire.len(), 1);
        assert_eq!(wire[0]["role"], "system");
        assert_eq!(wire[0]["content"], "you are helpful");
    }

    #[test]
    fn user_text_becomes_simple_string() {
        let messages = vec![Message::user_text("hello")];
        let wire = to_openai_messages(&messages).unwrap();
        assert_eq!(wire[0]["role"], "user");
        assert_eq!(wire[0]["content"], "hello");
    }

    #[test]
    fn user_image_url_preserved() {
        let content = Content::from_parts(vec![
            ContentPart::text("check this"),
            ContentPart::Image {
                media_type: None,
                source: ImageSource::Url {
                    url: "https://example.com/image.jpg".into(),
                },
            },
        ]);
        let messages = vec![Message::User { content }];
        let wire = to_openai_messages(&messages).unwrap();
        let content = wire[0]["content"].as_array().unwrap();
        assert_eq!(
            content[1]["image_url"]["url"],
            "https://example.com/image.jpg"
        );
    }

    #[test]
    fn assistant_with_tool_calls() {
        let messages = vec![Message::Assistant {
            content: "Let me check".into(),
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "echo".into(),
                arguments: json!({"x": 42}),
            }],
            reasoning: None,
        }];
        let wire = to_openai_messages(&messages).unwrap();
        assert_eq!(wire[0]["role"], "assistant");
        assert_eq!(wire[0]["content"], "Let me check");
        let calls = wire[0]["tool_calls"].as_array().unwrap();
        assert_eq!(calls[0]["id"], "call_1");
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "echo");
        assert_eq!(calls[0]["function"]["arguments"], "{\"x\":42}");
    }

    #[test]
    fn assistant_preserves_reasoning() {
        let messages = vec![Message::Assistant {
            content: "answer".into(),
            tool_calls: vec![],
            reasoning: Some(Reasoning {
                text: Some("step by step".into()),
            }),
        }];
        let wire = to_openai_messages(&messages).unwrap();
        assert_eq!(wire[0]["reasoning_content"], "step by step");
    }

    #[test]
    fn assistant_empty_content_becomes_null() {
        let messages = vec![Message::Assistant {
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "echo".into(),
                arguments: json!({}),
            }],
            reasoning: None,
        }];
        let wire = to_openai_messages(&messages).unwrap();
        assert!(wire[0]["content"].is_null());
        assert!(wire[0]["tool_calls"].is_array());
    }

    #[test]
    fn tool_result_becomes_tool_role() {
        let messages = vec![Message::Tool {
            call_id: "call_1".into(),
            name: "echo".into(),
            content: "42".into(),
            is_error: false,
        }];
        let wire = to_openai_messages(&messages).unwrap();
        assert_eq!(wire[0]["role"], "tool");
        assert_eq!(wire[0]["tool_call_id"], "call_1");
        assert_eq!(wire[0]["name"], "echo");
        assert_eq!(wire[0]["content"], "42");
    }

    #[test]
    fn tool_error_result_includes_is_error() {
        let messages = vec![Message::Tool {
            call_id: "call_1".into(),
            name: "echo".into(),
            content: "failed".into(),
            is_error: true,
        }];
        let wire = to_openai_messages(&messages).unwrap();
        assert_eq!(wire[0]["content"]["is_error"], true);
    }

    #[test]
    fn tool_unknown_is_rejected() {
        let messages = vec![Message::ToolUnknown {
            call_id: "call_1".into(),
            name: "echo".into(),
            reason: "interrupted".into(),
        }];
        let err = to_openai_messages(&messages).unwrap_err();
        assert!(matches!(err, ProviderError::UnresolvedToolUnknown { .. }));
    }

    #[test]
    fn to_openai_tools_correct_format() {
        let tools = vec![ToolDefinition {
            name: "echo".into(),
            description: "Echo a value".into(),
            parameters: json!({"type": "object"}),
        }];
        let wire = to_openai_tools(&tools);
        assert_eq!(wire[0]["type"], "function");
        assert_eq!(wire[0]["function"]["name"], "echo");
        assert_eq!(wire[0]["function"]["description"], "Echo a value");
    }

    #[test]
    fn full_roundtrip_assistant_tool_result() {
        let messages = vec![
            Message::user_text("go"),
            Message::Assistant {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "echo".into(),
                    arguments: json!({}),
                }],
                reasoning: None,
            },
            Message::Tool {
                call_id: "call_1".into(),
                name: "echo".into(),
                content: "ok".into(),
                is_error: false,
            },
        ];
        let wire = to_openai_messages(&messages).unwrap();
        assert_eq!(wire.len(), 3);
        assert_eq!(wire[0]["role"], "user");
        assert_eq!(wire[1]["role"], "assistant");
        assert_eq!(wire[2]["role"], "tool");
        assert_eq!(wire[2]["tool_call_id"], "call_1");
    }
}
