//! Credential-free demonstration of the real model → tool → model runtime path.
//! A fixture sensor also exercises typed, ephemeral model input without hardware.
use async_trait::async_trait;
use serde_json::{json, Value};
use std::{sync::Arc, time::Duration};
use unravel_agent_runtime::{
    AgentLoop, Content, ContentPart, Error, Event, EventSink, FinishReason, ImageSource, Message,
    Model, ModelRequest, ModelResponse, Result, Session, StopToken, Tool, ToolCall, ToolDefinition,
    ToolObservation, ToolOutput, ToolRegistry,
};

const FRAME: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC";

struct ScriptedModel;

#[async_trait]
impl Model for ScriptedModel {
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse> {
        if let Some(content) = request.messages.iter().find_map(|message| match message {
            Message::Tool { name, content, .. } if name == "word_count" => Some(content),
            _ => None,
        }) {
            let observation = request.messages.last().expect("sensor observation");
            assert!(matches!(
                observation,
                Message::User { content } if matches!(
                    content.parts.as_slice(),
                    [
                        ContentPart::Text { text },
                        ContentPart::Image {
                            media_type: Some(media_type),
                            source: ImageSource::Base64 { data },
                        }
                    ] if text.contains("Untrusted")
                        && text.contains("fixture_sensor")
                        && text.contains("preview")
                        && media_type == "image/png" && data == FRAME
                )
            ));
            assert!(matches!(
                &request.messages[request.messages.len() - 2],
                Message::Tool { call_id, .. } if call_id == "preview"
            ));
            return Ok(ModelResponse {
                content: format!("The tool counted {content} words."),
                finish_reason: FinishReason::Stop,
                ..ModelResponse::default()
            });
        }
        let text = request
            .messages
            .iter()
            .rev()
            .find_map(|message| match message {
                Message::User { content } => Some(content.to_text()),
                _ => None,
            })
            .ok_or_else(|| Error::Model("missing user text".into()))?;
        Ok(ModelResponse {
            tool_calls: vec![
                ToolCall {
                    id: "count-words".into(),
                    name: "word_count".into(),
                    arguments: json!({ "text": text }),
                },
                ToolCall {
                    id: "preview".into(),
                    name: "fixture_sensor".into(),
                    arguments: json!({}),
                },
            ],
            finish_reason: FinishReason::ToolCall,
            ..ModelResponse::default()
        })
    }
}

struct WordCount;

#[async_trait]
impl Tool for WordCount {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "word_count".into(),
            description: "Count whitespace-separated words in text".into(),
            parameters: json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"],
                "additionalProperties": false,
            }),
        }
    }

    async fn execute(&self, arguments: Value) -> Result<ToolOutput> {
        let text = arguments
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Tool("text must be a string".into()))?;
        Ok(ToolOutput::text(
            text.split_whitespace().count().to_string(),
        ))
    }
}

struct FixtureSensor;

#[async_trait]
impl Tool for FixtureSensor {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "fixture_sensor".into(),
            description: "Supply a known one-pixel PNG fixture, without hardware".into(),
            parameters: json!({"type": "object"}),
        }
    }

    async fn execute(&self, _arguments: Value) -> Result<ToolOutput> {
        Ok(
            ToolOutput::with_metadata("fixture captured", json!({"width": 1, "height": 1}))
                .with_observation(ToolObservation::new(
                    Content::from_parts(vec![ContentPart::Image {
                        media_type: Some("image/png".into()),
                        source: ImageSource::Base64 { data: FRAME.into() },
                    }]),
                    Duration::from_secs(60),
                )),
        )
    }
}

struct PrintEvents;
impl EventSink for PrintEvents {
    fn emit(&self, event: Event) {
        let serialized = serde_json::to_string(&event).expect("serializable runtime event");
        assert!(!serialized.contains(FRAME));
        println!("{serialized}");
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut tools = ToolRegistry::new();
    tools.register(WordCount)?;
    tools.register(FixtureSensor)?;
    let runtime = AgentLoop::new(Arc::new(ScriptedModel), tools, "Use the word-count tool.");
    let mut session = Session::new("tool-roundtrip");
    let answer = runtime
        .run(
            &mut session,
            "shared execution keeps applications independent",
            &PrintEvents,
            &StopToken::new(),
        )
        .await?;
    assert_eq!(answer, "The tool counted 5 words.");
    let serialized = serde_json::to_string(&session).expect("serializable session");
    assert!(!serialized.contains(FRAME));
    assert!(!session.messages.iter().any(|message| matches!(
        message, Message::User { content }
            if content.parts.iter().any(|part| matches!(part, ContentPart::Image { .. }))
    )));
    println!("{answer}");
    Ok(())
}
