//! Credential-free demonstration of the real model → tool → model runtime path.
//! The scripted model keeps this example deterministic; the tool executes normally.
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use unravel_agent_runtime::{
    AgentLoop, Error, Event, EventSink, FinishReason, Message, Model, ModelRequest, ModelResponse,
    Result, Session, StopToken, Tool, ToolCall, ToolDefinition, ToolOutput, ToolRegistry,
};

struct ScriptedModel;

#[async_trait]
impl Model for ScriptedModel {
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse> {
        if let Some(Message::Tool { content, .. }) = request.messages.last() {
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
            tool_calls: vec![ToolCall {
                id: "count-words".into(),
                name: "word_count".into(),
                arguments: json!({ "text": text }),
            }],
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

struct PrintEvents;
impl EventSink for PrintEvents {
    fn emit(&self, event: Event) {
        println!(
            "{}",
            serde_json::to_string(&event).expect("serializable runtime event")
        );
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut tools = ToolRegistry::new();
    tools.register(WordCount)?;
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
    println!("{answer}");
    Ok(())
}
