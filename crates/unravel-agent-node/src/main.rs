use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::HashMap, sync::Arc};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    sync::{mpsc, oneshot, Mutex},
};
use unravel_agent_providers::{OpenAiCompatProvider, ProviderSpec, Secret};
use unravel_agent_runtime::{
    AgentLoop, Error, Event, LoopConfig, Session, StopToken, Tool, ToolDefinition, ToolOutput,
    ToolRegistry,
};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProviderOptions {
    kind: String,
    model: String,
    api_key: Option<String>,
    base_url: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolOptions {
    name: String,
    description: String,
    parameters: Value,
    #[serde(default)]
    parallel_safe: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RunOptions {
    provider: ProviderOptions,
    system_prompt: String,
    prompt: String,
    session: Session,
    #[serde(default)]
    tools: Vec<ToolOptions>,
    max_turns: Option<usize>,
    prefer_streaming: Option<bool>,
}

type PendingTools = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<ToolOutput, Error>>>>>;

struct JsTool {
    definition: ToolDefinition,
    parallel_safe: bool,
    output: mpsc::UnboundedSender<Value>,
    pending: PendingTools,
    next_id: Arc<std::sync::atomic::AtomicU64>,
}

#[async_trait]
impl Tool for JsTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    fn parallel_safe(&self) -> bool {
        self.parallel_safe
    }

    async fn execute(&self, arguments: Value) -> Result<ToolOutput, Error> {
        use std::sync::atomic::Ordering;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);
        if self
            .output
            .send(json!({
                "type": "tool_request", "id": id, "name": self.definition.name,
                "arguments": arguments
            }))
            .is_err()
        {
            self.pending.lock().await.remove(&id);
            return Err(Error::InvalidInput("Node bridge disconnected".into()));
        }
        rx.await
            .map_err(|_| Error::InvalidInput("Node bridge disconnected".into()))?
    }
}

fn provider(options: ProviderOptions) -> Result<OpenAiCompatProvider, String> {
    let mut spec = match options.kind.as_str() {
        "openai" => ProviderSpec::openai(
            options
                .api_key
                .unwrap_or_else(|| std::env::var("OPENAI_API_KEY").unwrap_or_default()),
        ),
        "nvidia" => ProviderSpec::nvidia(
            options
                .api_key
                .unwrap_or_else(|| std::env::var("NVIDIA_API_KEY").unwrap_or_default()),
        ),
        "openrouter" => ProviderSpec::openrouter(
            options
                .api_key
                .unwrap_or_else(|| std::env::var("OPENROUTER_API_KEY").unwrap_or_default()),
        ),
        "ollama" => {
            let spec = ProviderSpec::ollama();
            match options.api_key {
                Some(key) => spec.with_key(Secret::new(key)),
                None => spec,
            }
        }
        "custom" => ProviderSpec::custom(
            "custom",
            "Custom",
            options
                .base_url
                .clone()
                .ok_or("custom provider requires baseUrl")?,
            options.api_key.map(Secret::new),
        ),
        _ => return Err("unsupported provider kind".into()),
    };
    if let Some(url) = options.base_url {
        spec = spec.with_base_url(url);
    }
    OpenAiCompatProvider::new(spec, options.model).map_err(|e| e.to_string())
}

#[tokio::main]
async fn main() {
    let (output, mut outgoing) = mpsc::unbounded_channel::<Value>();
    let writer = tokio::spawn(async move {
        let mut stdout = tokio::io::stdout();
        while let Some(value) = outgoing.recv().await {
            let mut bytes = value.to_string().into_bytes();
            bytes.push(b'\n');
            if stdout.write_all(&bytes).await.is_err() || stdout.flush().await.is_err() {
                break;
            }
        }
    });

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let Ok(Some(line)) = lines.next_line().await else {
        return;
    };
    let mut options: RunOptions = match serde_json::from_str(&line) {
        Ok(options) => options,
        Err(_) => {
            let _ = output.send(json!({"type": "error", "error": "invalid run options"}));
            drop(output);
            let _ = writer.await;
            return;
        }
    };
    let model = match provider(options.provider) {
        Ok(model) => model,
        Err(error) => {
            let _ = output.send(json!({"type": "error", "error": error}));
            drop(output);
            let _ = writer.await;
            return;
        }
    };

    let pending = Arc::new(Mutex::new(HashMap::<u64, oneshot::Sender<_>>::new()));
    let next_id = Arc::new(std::sync::atomic::AtomicU64::new(1));
    let mut registry = ToolRegistry::new();
    for tool in options.tools {
        if let Err(error) = registry.register(JsTool {
            definition: ToolDefinition {
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
            },
            parallel_safe: tool.parallel_safe,
            output: output.clone(),
            pending: pending.clone(),
            next_id: next_id.clone(),
        }) {
            let _ = output.send(json!({"type": "error", "error": error.to_string()}));
            drop(output);
            let _ = writer.await;
            return;
        }
    }
    let mut config = LoopConfig::default();
    if let Some(max_turns) = options.max_turns {
        config.max_turns = max_turns;
    }
    if let Some(prefer_streaming) = options.prefer_streaming {
        config.prefer_streaming = prefer_streaming;
    }
    let agent =
        AgentLoop::new(Arc::new(model), registry, options.system_prompt).with_config(config);
    let events = {
        let output = output.clone();
        move |event: Event| {
            let _ = output.send(json!({"type": "event", "event": event}));
        }
    };
    let stop = StopToken::new();
    let result = {
        let run = agent.run(&mut options.session, options.prompt, &events, &stop);
        tokio::pin!(run);
        loop {
            tokio::select! {
                result = &mut run => break result,
                line = lines.next_line() => {
                    match line {
                        Ok(Some(line)) => {
                            if let Ok(value) = serde_json::from_str::<Value>(&line) {
                                if value["type"] == "tool_result" {
                                    if let Some(id) = value["id"].as_u64() {
                                        if let Some(tx) = pending.lock().await.remove(&id) {
                                            let result = if let Some(error) = value["error"].as_str() {
                                                Err(Error::InvalidInput(error.to_owned()))
                                            } else {
                                                Ok(ToolOutput::with_metadata(
                                                    value["content"].as_str().unwrap_or_default(),
                                                    value.get("metadata").cloned().unwrap_or(Value::Null),
                                                ))
                                            };
                                            let _ = tx.send(result);
                                        }
                                    }
                                } else if value["type"] == "stop" {
                                    stop.stop();
                                }
                            }
                        }
                        _ => { stop.stop(); }
                    }
                }
            }
        }
    };
    let message = match result {
        Ok(value) => json!({"type": "result", "output": value, "session": options.session}),
        Err(error) => {
            json!({"type": "error", "error": error.to_string(), "session": options.session})
        }
    };
    let _ = output.send(message);
    drop(events);
    drop(agent);
    drop(output);
    let _ = writer.await;
}
