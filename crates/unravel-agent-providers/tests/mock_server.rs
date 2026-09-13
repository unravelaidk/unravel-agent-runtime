//! Integration tests using a local mock HTTP server.
//!
//! These tests exercise the real provider implementation and the real
//! `AgentLoop` against a mock server — no live APIs, no real
//! credentials, no `process.env` mutation, no sleep-based
//! synchronization.
//!
//! Test tiers (per test-coding-agent skill):
//! - **Session tier**: real `AgentLoop` + real `OpenAiCompatProvider`
//!   (HTTP adapter) + local scripted HTTP server + safe fake tool.
//!   Asserts exact requests, messages, events, and model call counts.
//! - **Unit tier**: pure-function tests in `src/` modules (message
//!   transforms, SSE parsing, catalog parsing, etc.).
//!
//! Anti-patterns refused:
//! - No sleeping to synchronize — `tokio::sync::Notify` for readiness.
//! - No real network — mock HTTP server via `TcpListener`.
//! - No `process.env` mutation — inject explicit catalog URLs.
//! - No mocking internal collaborators — test through the public
//!   `AgentLoop::run` interface.
//! - No asserting on implementation shape — assert on messages, events,
//!   and side effects.

use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Notify;
use unravel_agent_providers::{
    CatalogModel, CatalogProvider, Discovery, DiscoveryOptions, ModalitySupport,
    OpenAiCompatProvider, ProviderSpec, Secret, TestClock, ToolSupport,
};
use unravel_agent_runtime::{
    AgentLoop, Content, ContentPart, DeltaSink, Event, EventSink, FinishReason, ImageSource,
    LoopConfig, Message, Model, ModelRequest, NoopEventSink, Session, StopToken, StreamDelta, Tool,
    ToolDefinition, ToolOutput, ToolRegistry,
};

// ───────────────────────────────────────────────────────────────────
// Mock HTTP server
// ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn catalog_accepts_reported_4614121_byte_payload_through_fetch_and_discovery() {
    let mut body = serde_json::to_vec(&json!({"large-catalog-fixture": {
        "name":"Large Catalog Fixture", "npm":"@ai-sdk/openai-compatible",
        "env":[], "models":{"fixture-model":{"name":"Fixture", "tool_call":true}}
    }}))
    .unwrap();
    body.resize(4_614_121, b' ');
    let server = MockServer::start(Box::new(move |_, path, _, _| {
        if path == "/v1/models" {
            MockResponse::json(200, json!({"data":[]}))
        } else {
            MockResponse {
                status: 200,
                headers: vec![("content-type".into(), "application/json".into())],
                body: body.clone(),
                hold_open: None,
            }
        }
    }))
    .await;
    let url = format!("http://{}/catalog.json", server.addr);
    let providers = unravel_agent_providers::catalog::fetch_catalog(Some(&url))
        .await
        .unwrap();
    assert_eq!(providers[0].id, "large-catalog-fixture");
    let spec = ProviderSpec::custom("large-catalog-fixture", "Fixture", server.base_url(), None);
    let options = DiscoveryOptions {
        models_dev_url_override: Some(url),
        ..Default::default()
    };
    let models = Discovery::new().discover(&spec, &options).await.unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].model_id, "fixture-model");
    assert!(!models[0].from_endpoint);
}

#[tokio::test]
async fn catalog_still_rejects_declared_bodies_above_16_mib() {
    let server = MockServer::start(Box::new(|_, _, _, _| MockResponse {
        status: 200,
        headers: vec![],
        body: vec![b' '; 16 * 1024 * 1024 + 1],
        hold_open: None,
    }))
    .await;
    let url = format!("http://{}/catalog.json", server.addr);
    let error = unravel_agent_providers::catalog::fetch_catalog(Some(&url))
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("16777216 byte limit"), "{error}");
    assert!(error.contains("declared 16777217"), "{error}");
}

#[tokio::test]
async fn ordinary_provider_json_keeps_the_4_mib_limit() {
    let server = MockServer::start(Box::new(|_, _, _, _| MockResponse {
        status: 200,
        headers: vec![],
        body: vec![b' '; 4_614_121],
        hold_open: None,
    }))
    .await;
    let provider = OpenAiCompatProvider::new(
        ProviderSpec::custom("bounded", "Bounded", server.base_url(), None),
        "test",
    )
    .unwrap();
    let error = provider
        .complete(ModelRequest::new(
            "bounded",
            vec![Message::user_text("test")],
        ))
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("4194304 byte limit"), "{error}");
}

#[tokio::test]
async fn catalog_stops_chunked_download_at_limit_without_waiting_for_eof() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/catalog.json", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await?;
        let mut request = [0u8; 4096];
        if socket.read(&mut request).await? == 0 {
            return Ok(());
        }
        socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n").await?;
        let chunk = vec![b' '; 64 * 1024];
        for _ in 0..257 {
            socket.write_all(b"10000\r\n").await?;
            socket.write_all(&chunk).await?;
            socket.write_all(b"\r\n").await?;
        }
        // Deliberately never terminate the response: the byte cap, not EOF
        // or the HTTP timeout, must stop the download.
        std::future::pending::<std::io::Result<()>>().await
    });
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        unravel_agent_providers::catalog::fetch_catalog(Some(&url)),
    )
    .await;
    server.abort();
    let error = result
        .expect("byte limit must stop before EOF")
        .unwrap_err()
        .to_string();
    assert!(error.contains("16777216 byte limit"), "{error}");
    assert!(error.contains("received"), "{error}");
}

/// A handler function that receives (method, path, body, headers) and
/// returns a mock response.
type Handler =
    Box<dyn Fn(&str, &str, &str, &HashMap<String, String>) -> MockResponse + Send + Sync>;

struct MockResponse {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    hold_open: Option<Arc<Notify>>,
}

impl MockResponse {
    fn json(status: u16, body: Value) -> Self {
        Self {
            status,
            headers: vec![("content-type".into(), "application/json".into())],
            body: serde_json::to_vec(&body).unwrap(),
            hold_open: None,
        }
    }

    fn sse(status: u16, events: Vec<String>) -> Self {
        let mut body = Vec::new();
        for event in events {
            body.extend_from_slice(format!("data: {event}\n\n").as_bytes());
        }
        Self {
            status,
            headers: vec![("content-type".into(), "text/event-stream".into())],
            body,
            hold_open: None,
        }
    }

    fn error(status: u16, message: &str) -> Self {
        Self::json(status, json!({"error": {"message": message}}))
    }

    fn hold_open(mut self, release: Arc<Notify>) -> Self {
        self.hold_open = Some(release);
        self
    }
}

/// A mock HTTP server on a local ephemeral port.
struct MockServer {
    addr: SocketAddr,
    requests: Arc<Mutex<Vec<CapturedRequest>>>,
    _handle: tokio::task::JoinHandle<()>,
}

/// A captured HTTP request for inspection in tests.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CapturedRequest {
    method: String,
    path: String,
    body: String,
    headers: HashMap<String, String>,
}

impl MockServer {
    async fn start(handler: Handler) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handler = Arc::new(handler);
        let requests: Arc<Mutex<Vec<CapturedRequest>>> = Arc::new(Mutex::new(Vec::new()));
        let requests_clone = requests.clone();

        let handle = tokio::spawn(async move {
            loop {
                let (mut sock, _) = match listener.accept().await {
                    Ok(v) => v,
                    Err(_) => break,
                };
                let handler = handler.clone();
                let requests = requests_clone.clone();
                tokio::spawn(async move {
                    let _ = handle_request(&mut sock, &handler, &requests).await;
                });
            }
        });

        Self {
            addr,
            requests,
            _handle: handle,
        }
    }

    fn base_url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }

    /// All captured requests, in arrival order.
    fn captured_requests(&self) -> Vec<CapturedRequest> {
        self.requests.lock().unwrap().clone()
    }

    /// Number of requests to `/v1/chat/completions`.
    fn completions_call_count(&self) -> usize {
        self.captured_requests()
            .iter()
            .filter(|r| r.path.ends_with("/v1/chat/completions"))
            .count()
    }

    /// Number of requests to `/v1/models`.
    fn models_call_count(&self) -> usize {
        self.captured_requests()
            .iter()
            .filter(|r| r.path.ends_with("/v1/models"))
            .count()
    }
}

async fn handle_request(
    sock: &mut tokio::net::TcpStream,
    handler: &Arc<Handler>,
    requests: &Arc<Mutex<Vec<CapturedRequest>>>,
) -> std::io::Result<()> {
    let mut buf = vec![0u8; 65536];
    let n = sock.read(&mut buf).await?;
    let raw = String::from_utf8_lossy(&buf[..n]).to_string();

    let (method, path) = parse_request_line(&raw);
    let headers = parse_headers(&raw);
    let body = extract_body(&raw);

    // Capture the request for test inspection.
    requests.lock().unwrap().push(CapturedRequest {
        method: method.clone(),
        path: path.clone(),
        body: body.clone(),
        headers: headers.clone(),
    });

    let response = handler(&method, &path, &body, &headers);

    let mut response_str = format!("HTTP/1.1 {} OK\r\n", response.status);
    for (key, value) in &response.headers {
        response_str.push_str(&format!("{key}: {value}\r\n"));
    }
    response_str.push_str(&format!("content-length: {}\r\n", response.body.len()));
    response_str.push_str("\r\n");

    sock.write_all(response_str.as_bytes()).await?;
    sock.write_all(&response.body).await?;
    sock.flush().await?;
    if let Some(release) = response.hold_open {
        release.notified().await;
    }
    Ok(())
}

fn parse_request_line(raw: &str) -> (String, String) {
    let first_line = raw.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    (
        parts.first().map(|s| s.to_string()).unwrap_or_default(),
        parts.get(1).map(|s| s.to_string()).unwrap_or_default(),
    )
}

fn parse_headers(raw: &str) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    for line in raw.lines().skip(1) {
        if line.is_empty() {
            break;
        }
        if let Some((key, value)) = line.split_once(": ") {
            headers.insert(key.to_lowercase(), value.trim().to_string());
        }
    }
    headers
}

fn extract_body(raw: &str) -> String {
    if let Some(idx) = raw.find("\r\n\r\n") {
        raw[idx + 4..].to_string()
    } else {
        String::new()
    }
}

// ───────────────────────────────────────────────────────────────────
// Test harness: wires AgentLoop + OpenAiCompatProvider + mock server
// ───────────────────────────────────────────────────────────────────

/// A safe fake tool that echoes its arguments.
struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "echo".into(),
            description: "Echo a value back".into(),
            parameters: json!({
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            }),
        }
    }

    async fn execute(&self, arguments: Value) -> unravel_agent_runtime::Result<ToolOutput> {
        let text = arguments
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("(no text)");
        Ok(ToolOutput::text(format!("echo:{text}")))
    }
}

/// Event collector that captures all emitted events.
#[derive(Default)]
struct EventCollector {
    events: Mutex<Vec<Event>>,
}

impl EventCollector {
    fn new() -> Self {
        Self::default()
    }

    fn events(&self) -> Vec<Event> {
        self.events.lock().unwrap().clone()
    }

    fn count_of(&self, predicate: impl Fn(&Event) -> bool) -> usize {
        self.events().iter().filter(|e| predicate(e)).count()
    }
}

impl EventSink for EventCollector {
    fn emit(&self, event: Event) {
        self.events.lock().unwrap().push(event);
    }
}

/// Build a provider pointed at the mock server.
fn make_provider(server: &MockServer, model: &str) -> OpenAiCompatProvider {
    let spec = ProviderSpec::custom(
        "mock",
        "Mock Provider",
        server.base_url(),
        Some(Secret::new("test-key")),
    );
    OpenAiCompatProvider::new(spec, model).unwrap()
}

/// Build an AgentLoop with the echo tool registered.
fn make_agent_loop(model: Arc<dyn Model>) -> AgentLoop {
    let mut tools = ToolRegistry::new();
    tools.register(EchoTool).unwrap();
    AgentLoop::new(model, tools, "You are a helpful test assistant.").with_config(LoopConfig {
        // Zero retry delay so tests don't sleep.
        retry_base_delay: Duration::ZERO,
        max_model_attempts: 3,
        ..LoopConfig::default()
    })
}

// ───────────────────────────────────────────────────────────────────
// Session-tier tests: real AgentLoop + HTTP adapter + mock server
// ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn session_text_roundtrip_through_agent_loop() {
    let server = MockServer::start(Box::new(|_method, _path, _body, _headers| {
        MockResponse::json(
            200,
            json!({
                "choices": [{
                    "message": {"content": "Hello from the model!"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
            }),
        )
    }))
    .await;

    let provider = Arc::new(make_provider(&server, "test-model"));
    let agent = make_agent_loop(provider);
    let mut session = Session::new("s1");
    let events = EventCollector::new();
    let stop = StopToken::new();

    let result = agent
        .run(&mut session, "Say hello", &events, &stop)
        .await
        .unwrap();

    // Assert on observable output.
    assert_eq!(result, "Hello from the model!");
    assert_eq!(server.completions_call_count(), 1);

    // Session messages: system, user, assistant.
    assert_eq!(session.messages.len(), 3);
    assert!(matches!(&session.messages[0], Message::System { .. }));
    assert!(matches!(&session.messages[1], Message::User { .. }));
    assert!(
        matches!(&session.messages[2], Message::Assistant { content, .. } if content == "Hello from the model!")
    );

    // Events: RunStarted, TurnStarted, ModelAttempt, RunCompleted.
    assert_eq!(
        events.count_of(|e| matches!(e, Event::RunStarted { .. })),
        1
    );
    assert_eq!(
        events.count_of(|e| matches!(e, Event::RunCompleted { .. })),
        1
    );
}

#[tokio::test]
async fn session_tool_call_roundtrip_through_agent_loop() {
    // Scripted responses: first a tool call, then a final answer.
    let call_count = Arc::new(AtomicU32::new(0));
    let call_count_clone = call_count.clone();

    let server = MockServer::start(Box::new(move |_method, _path, body, _headers| {
        let n = call_count_clone.fetch_add(1, Ordering::SeqCst);
        let payload: Value = serde_json::from_str(body).unwrap();

        if n == 0 {
            // First call: model requests a tool.
            // Verify tools were sent.
            assert!(payload["tools"].is_array());
            assert_eq!(payload["tools"][0]["function"]["name"], "echo");
            MockResponse::json(
                200,
                json!({
                    "choices": [{
                        "message": {
                            "content": null,
                            "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": "{\"text\": \"hello\"}",
                                },
                            }],
                        },
                        "finish_reason": "tool_calls",
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }),
            )
        } else {
            // Second call: verify the tool result is in the messages.
            let messages = payload["messages"].as_array().unwrap();
            assert!(
                messages.iter().any(|m| m["role"] == "tool"),
                "tool result must be in the follow-up request"
            );
            MockResponse::json(
                200,
                json!({
                    "choices": [{
                        "message": {"content": "The echo tool returned: echo:hello"},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 8},
                }),
            )
        }
    }))
    .await;

    let provider = Arc::new(make_provider(&server, "test-model"));
    let agent = make_agent_loop(provider);
    let mut session = Session::new("s1");
    let events = EventCollector::new();
    let stop = StopToken::new();

    let result = agent
        .run(&mut session, "Echo hello", &events, &stop)
        .await
        .unwrap();

    // Assert on observable output.
    assert_eq!(result, "The echo tool returned: echo:hello");
    assert_eq!(server.completions_call_count(), 2);

    // Session: system, user, assistant(tool_call), tool_result, assistant(final).
    assert_eq!(session.messages.len(), 5);
    assert!(
        matches!(&session.messages[2], Message::Assistant { tool_calls, .. } if tool_calls.len() == 1)
    );
    assert!(
        matches!(&session.messages[3], Message::Tool { content, .. } if content == "echo:hello")
    );
    assert!(
        matches!(&session.messages[4], Message::Assistant { content, .. } if content.contains("echo:hello"))
    );

    // Events: ToolStarted + ToolCompleted.
    assert_eq!(
        events.count_of(|e| matches!(e, Event::ToolStarted { .. })),
        1
    );
    assert_eq!(
        events.count_of(|e| matches!(
            e,
            Event::ToolCompleted {
                is_error: false,
                ..
            }
        )),
        1
    );
}

#[tokio::test]
async fn session_image_payload_preserved_in_request() {
    let server = MockServer::start(Box::new(|_method, _path, body, _headers| {
        let payload: Value = serde_json::from_str(body).unwrap();
        let messages = payload["messages"].as_array().unwrap();
        // Find the user message (not the system message).
        let user_msg = messages
            .iter()
            .find(|m| m["role"] == "user")
            .expect("user message must be present");
        let content = &user_msg["content"];

        // Must be an array (multimodal), not a plain string.
        assert!(
            content.is_array(),
            "image content must be an array, got: {content}"
        );
        let blocks = content.as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "image_url");

        // The image URL must be a data URL with the actual base64 data.
        let url = blocks[1]["image_url"]["url"].as_str().unwrap();
        assert!(
            url.starts_with("data:image/png;base64,"),
            "must be a data URL"
        );
        assert!(
            url.contains("iVBORw0KGgo="),
            "must contain the actual base64 data"
        );

        MockResponse::json(
            200,
            json!({
                "choices": [{"message": {"content": "I see an image"}, "finish_reason": "stop"}],
            }),
        )
    }))
    .await;

    let provider = Arc::new(make_provider(&server, "test-model"));
    let provider_for_request = provider.clone();
    let _agent = make_agent_loop(provider);
    let mut session = Session::new("s1");

    // We need to push a multimodal user message before the loop adds
    // the system prompt + user text. So we pre-build the session with
    // the multimodal content.
    session.messages.push(Message::System {
        content: "test system prompt".into(),
    });
    session.messages.push(Message::User {
        content: Content::from_parts(vec![
            ContentPart::text("look at this"),
            ContentPart::Image {
                media_type: Some("image/png".into()),
                source: ImageSource::Base64 {
                    data: "iVBORw0KGgo=".into(),
                },
            },
        ]),
    });

    // We can't use agent.run because it pushes a user text message.
    // Instead, build the request directly and call complete.
    let request = ModelRequest::new("s1", session.messages);
    let response = provider_for_request.complete(request).await.unwrap();
    assert_eq!(response.content, "I see an image");
}

#[tokio::test]
async fn session_auth_failure_permanent_no_retry() {
    let server = MockServer::start(Box::new(|_method, _path, _body, _headers| {
        MockResponse::error(401, "Invalid API key")
    }))
    .await;

    let provider = Arc::new(make_provider(&server, "test-model"));
    let agent = make_agent_loop(provider);
    let mut session = Session::new("s1");
    let events = EventCollector::new();
    let stop = StopToken::new();

    let err = agent
        .run(&mut session, "Hello", &events, &stop)
        .await
        .unwrap_err();

    // Auth failure is permanent — exactly 1 call, no retry.
    assert_eq!(server.completions_call_count(), 1);
    assert!(!err.is_retryable());

    // RunFailed event emitted.
    assert_eq!(events.count_of(|e| matches!(e, Event::RunFailed { .. })), 1);
}

#[tokio::test]
async fn session_rate_limit_transient_retries() {
    let call_count = Arc::new(AtomicU32::new(0));
    let call_count_clone = call_count.clone();

    let server = MockServer::start(Box::new(move |_method, _path, _body, _headers| {
        let n = call_count_clone.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            MockResponse::error(429, "Rate limited")
        } else {
            MockResponse::json(
                200,
                json!({
                    "choices": [{"message": {"content": "recovered"}, "finish_reason": "stop"}],
                }),
            )
        }
    }))
    .await;

    let provider = Arc::new(make_provider(&server, "test-model"));
    let agent = make_agent_loop(provider);
    let mut session = Session::new("s1");
    let events = EventCollector::new();
    let stop = StopToken::new();

    let result = agent
        .run(&mut session, "Hello", &events, &stop)
        .await
        .unwrap();

    assert_eq!(result, "recovered");
    // Exactly 2 calls: first rate-limited, second succeeds.
    assert_eq!(server.completions_call_count(), 2);

    // ModelRetry event emitted.
    assert_eq!(
        events.count_of(|e| matches!(e, Event::ModelRetry { .. })),
        1
    );
}

#[tokio::test]
async fn session_max_output_finish_reason_rejected_by_loop() {
    // The model returns a "length" finish reason (truncated). The loop
    // must reject it — truncated output must not be dispatched.
    let server = MockServer::start(Box::new(|_method, _path, _body, _headers| {
        MockResponse::json(
            200,
            json!({
                "choices": [{
                    "message": {"content": "truncated..."},
                    "finish_reason": "length",
                }],
            }),
        )
    }))
    .await;

    let provider = Arc::new(make_provider(&server, "test-model"));
    let agent = make_agent_loop(provider);
    let mut session = Session::new("s1");
    let stop = StopToken::new();

    let err = agent
        .run(&mut session, "Hello", &NoopEventSink, &stop)
        .await
        .unwrap_err();

    // The loop rejects non-dispatchable finish reasons.
    assert!(err.to_string().contains("finish") || err.to_string().contains("non-dispatchable"));
}

#[tokio::test]
async fn session_content_filter_rejected_by_loop() {
    let server = MockServer::start(Box::new(|_method, _path, _body, _headers| {
        MockResponse::json(
            200,
            json!({
                "choices": [{
                    "message": {"content": "filtered"},
                    "finish_reason": "content_filter",
                }],
            }),
        )
    }))
    .await;

    let provider = Arc::new(make_provider(&server, "test-model"));
    let agent = make_agent_loop(provider);
    let mut session = Session::new("s1");
    let stop = StopToken::new();

    let err = agent
        .run(&mut session, "Hello", &NoopEventSink, &stop)
        .await
        .unwrap_err();

    assert!(err.to_string().contains("finish") || err.to_string().contains("non-dispatchable"));
}

#[tokio::test]
async fn session_streaming_text_through_agent_loop() {
    let events = vec![
        r#"{"choices":[{"delta":{"content":"Hello"}}]}"#.to_string(),
        r#"{"choices":[{"delta":{"content":" world"}}]}"#.to_string(),
        r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#.to_string(),
        r#"{"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}"#.to_string(),
        "[DONE]".to_string(),
    ];

    let server = MockServer::start(Box::new(move |_method, _path, _body, _headers| {
        MockResponse::sse(200, events.clone())
    }))
    .await;

    let provider = Arc::new(make_provider(&server, "test-model"));
    let agent = make_agent_loop(provider).with_config(LoopConfig {
        prefer_streaming: true,
        retry_base_delay: Duration::ZERO,
        ..LoopConfig::default()
    });
    let mut session = Session::new("s1");
    let stop = StopToken::new();

    let result = agent
        .run(&mut session, "Hi", &NoopEventSink, &stop)
        .await
        .unwrap();

    assert_eq!(result, "Hello world");
    assert_eq!(server.completions_call_count(), 1);
}

#[tokio::test]
async fn session_streaming_tool_call_assembly() {
    let events: [&str; 8] = [
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"echo","arguments":""}}]}}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"text"}}]}}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\": \"hi\"}"}}]}}]}"#,
        r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        "[DONE]",
        // Second call: final answer after tool result.
        r#"{"choices":[{"delta":{"content":"done"}}]}"#,
        r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        "[DONE]",
    ];

    let call_count = Arc::new(AtomicU32::new(0));
    let call_count_clone = call_count.clone();

    let server = MockServer::start(Box::new(move |_method, _path, _body, _headers| {
        let n = call_count_clone.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            MockResponse::sse(200, events[..5].iter().map(|s| s.to_string()).collect())
        } else {
            MockResponse::sse(200, events[5..].iter().map(|s| s.to_string()).collect())
        }
    }))
    .await;

    let provider = Arc::new(make_provider(&server, "test-model"));
    let agent = make_agent_loop(provider).with_config(LoopConfig {
        prefer_streaming: true,
        retry_base_delay: Duration::ZERO,
        ..LoopConfig::default()
    });
    let mut session = Session::new("s1");
    let stop = StopToken::new();

    let result = agent
        .run(&mut session, "Echo hi", &NoopEventSink, &stop)
        .await
        .unwrap();

    assert_eq!(result, "done");
    assert_eq!(server.completions_call_count(), 2);

    // Verify tool call was assembled and executed.
    assert!(session.messages.iter().any(|m| matches!(
        m,
        Message::Tool { content, .. } if content == "echo:hi"
    )));
}

#[tokio::test]
async fn session_malformed_streamed_tool_args_not_dispatched() {
    // The model streams a tool call with malformed JSON arguments.
    // The provider boundary must reject it before the AgentLoop can dispatch
    // any tool. In particular, there must be no partial streamed execution.
    let events: [&str; 3] = [
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"echo","arguments":"not-valid-json"}}]}}]}"#,
        r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        "[DONE]",
    ];

    let call_count = Arc::new(AtomicU32::new(0));
    let call_count_clone = call_count.clone();

    let server = MockServer::start(Box::new(move |_method, _path, _body, _headers| {
        let n = call_count_clone.fetch_add(1, Ordering::SeqCst);
        assert_eq!(
            n, 0,
            "malformed tool args must stop before a follow-up request"
        );
        MockResponse::sse(200, events.iter().map(|s| s.to_string()).collect())
    }))
    .await;

    let provider = Arc::new(make_provider(&server, "test-model"));
    let agent = make_agent_loop(provider).with_config(LoopConfig {
        prefer_streaming: true,
        retry_base_delay: Duration::ZERO,
        ..LoopConfig::default()
    });
    let mut session = Session::new("s1");
    let stop = StopToken::new();

    let err = agent
        .run(&mut session, "Echo", &NoopEventSink, &stop)
        .await
        .unwrap_err();

    assert!(
        !err.is_retryable(),
        "malformed tool arguments are permanent"
    );
    assert_eq!(server.completions_call_count(), 1);
    assert!(!session
        .messages
        .iter()
        .any(|m| matches!(m, Message::Tool { .. })));
}

// ───────────────────────────────────────────────────────────────────
// Provider-level tests (no AgentLoop, but through real HTTP adapter)
// ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn provider_complete_text_roundtrip() {
    let server = MockServer::start(Box::new(|_method, _path, body, _headers| {
        let payload: Value = serde_json::from_str(body).unwrap();
        assert_eq!(payload["model"], "test-model");
        assert_eq!(payload["messages"][0]["role"], "user");
        assert_eq!(payload["messages"][0]["content"], "Hello");

        MockResponse::json(
            200,
            json!({
                "choices": [{
                    "message": {"content": "Hi there!"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }),
        )
    }))
    .await;

    let provider = make_provider(&server, "test-model");
    let mut session = Session::new("test");
    session.messages.push(Message::user_text("Hello"));
    let request = ModelRequest::new("test", session.messages);

    let response = provider.complete(request).await.unwrap();
    assert_eq!(response.content, "Hi there!");
    assert_eq!(response.finish_reason, FinishReason::Stop);
    assert_eq!(response.usage.prompt_tokens, Some(5));
    assert_eq!(response.usage.completion_tokens, Some(3));
    assert_eq!(response.usage.total_tokens, Some(8));
}

#[tokio::test]
async fn provider_auth_failure_does_not_leak_key() {
    let server = MockServer::start(Box::new(|_method, _path, _body, headers| {
        // Verify the auth header is present.
        let auth = headers.get("authorization").unwrap();
        assert!(auth.starts_with("Bearer "));

        // The error body intentionally does NOT contain our key.
        MockResponse::error(401, "Invalid API key")
    }))
    .await;

    let provider = make_provider(&server, "test-model");
    let mut session = Session::new("test");
    session.messages.push(Message::user_text("Hello"));
    let request = ModelRequest::new("test", session.messages);

    let err = provider.complete(request).await.unwrap_err();
    let err_msg = err.to_string();

    // The actual API key ("test-key") must never appear in the error.
    assert!(
        !err_msg.contains("test-key"),
        "error message must not contain the API key: {err_msg}"
    );
    assert!(!err.is_retryable());
}

#[tokio::test]
async fn provider_credential_redaction_against_echoed_key_in_body() {
    // Simulate a misbehaving server that echoes the raw credential in
    // unstructured text (not a Bearer token or JSON key/value).
    let server = MockServer::start(Box::new(|_method, _path, _body, headers| {
        let auth = headers.get("authorization").cloned().unwrap_or_default();
        assert_eq!(auth, "Bearer test-key");
        MockResponse::error(401, "Unauthorized: leaked credential test-key")
    }))
    .await;

    let provider = make_provider(&server, "test-model");
    let mut session = Session::new("test");
    session.messages.push(Message::user_text("Hello"));
    let request = ModelRequest::new("test", session.messages);

    let err = provider.complete(request).await.unwrap_err();
    let err_msg = err.to_string();

    // The bearer token "test-key" must be redacted from the error.
    assert!(
        !err_msg.contains("test-key"),
        "error must not contain the actual API key even when server echoes it: {err_msg}"
    );
    // The error should still mention "Unauthorized" or "401".
    assert!(err_msg.contains("401") || err_msg.contains("auth"));
}

#[tokio::test]
async fn provider_rate_limit_is_transient() {
    let server = MockServer::start(Box::new(|_method, _path, _body, _headers| {
        let mut resp = MockResponse::error(429, "Rate limited");
        resp.headers.push(("retry-after".into(), "5".into()));
        resp
    }))
    .await;

    let provider = make_provider(&server, "test-model");
    let mut session = Session::new("test");
    session.messages.push(Message::user_text("Hello"));
    let request = ModelRequest::new("test", session.messages);

    let err = provider.complete(request).await.unwrap_err();
    assert!(err.is_retryable(), "rate limit must be retryable");
}

#[tokio::test]
async fn provider_server_error_is_transient() {
    let server = MockServer::start(Box::new(|_method, _path, _body, _headers| {
        MockResponse::error(500, "Internal server error")
    }))
    .await;

    let provider = make_provider(&server, "test-model");
    let mut session = Session::new("test");
    session.messages.push(Message::user_text("Hello"));
    let request = ModelRequest::new("test", session.messages);

    let err = provider.complete(request).await.unwrap_err();
    assert!(err.is_retryable(), "server error must be retryable");
}

#[tokio::test]
async fn provider_bad_request_is_permanent() {
    let server = MockServer::start(Box::new(|_method, _path, _body, _headers| {
        MockResponse::error(400, "Invalid model")
    }))
    .await;

    let provider = make_provider(&server, "test-model");
    let mut session = Session::new("test");
    session.messages.push(Message::user_text("Hello"));
    let request = ModelRequest::new("test", session.messages);

    let err = provider.complete(request).await.unwrap_err();
    assert!(!err.is_retryable(), "bad request must not be retryable");
}

#[tokio::test]
async fn provider_tool_unknown_rejected() {
    let server = MockServer::start(Box::new(|_method, _path, _body, _headers| {
        MockResponse::json(
            200,
            json!({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}),
        )
    }))
    .await;

    let provider = make_provider(&server, "test-model");
    let mut session = Session::new("test");
    session.messages.push(Message::ToolUnknown {
        call_id: "call_1".into(),
        name: "echo".into(),
        reason: "interrupted".into(),
    });
    let request = ModelRequest::new("test", session.messages);

    let err = provider.complete(request).await.unwrap_err();
    assert!(err.to_string().contains("unresolved") || err.to_string().contains("ToolUnknown"));
}

#[tokio::test]
async fn provider_debug_does_not_leak_key() {
    let spec = ProviderSpec::custom(
        "test",
        "Test",
        "http://127.0.0.1:1/v1",
        Some(Secret::new("sk-super-secret-key-12345")),
    );
    let provider = OpenAiCompatProvider::new(spec, "test-model").unwrap();
    let debug = format!("{:?}", provider);
    assert!(!debug.contains("sk-super-secret-key-12345"));
    assert!(!debug.contains("super-secret"));
}

#[tokio::test]
async fn provider_keyless_works_without_key() {
    let server = MockServer::start(Box::new(|_method, _path, _body, headers| {
        // Verify no authorization header is present.
        assert!(
            !headers.contains_key("authorization"),
            "keyless provider must not send an auth header"
        );
        MockResponse::json(
            200,
            json!({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}),
        )
    }))
    .await;

    let spec = ProviderSpec::custom("local", "Local", server.base_url(), None::<Secret>);
    let provider = OpenAiCompatProvider::new(spec, "test-model").unwrap();
    let mut session = Session::new("test");
    session.messages.push(Message::user_text("Hi"));
    let request = ModelRequest::new("test", session.messages);

    let response = provider.complete(request).await.unwrap();
    assert_eq!(response.content, "ok");
}

#[tokio::test]
async fn provider_malformed_response_is_permanent() {
    let server = MockServer::start(Box::new(|_method, _path, _body, _headers| {
        MockResponse::json(200, json!({"unexpected": true}))
    }))
    .await;

    let provider = make_provider(&server, "test-model");
    let mut session = Session::new("test");
    session.messages.push(Message::user_text("Hi"));
    let request = ModelRequest::new("test", session.messages);

    let err = provider.complete(request).await.unwrap_err();
    assert!(
        !err.is_retryable(),
        "malformed response must not be retryable"
    );
}

// ───────────────────────────────────────────────────────────────────
// Streaming tests (through real HTTP adapter, no AgentLoop)
// ───────────────────────────────────────────────────────────────────

/// A delta sink that collects all deltas for inspection.
#[derive(Default)]
struct CollectingSink {
    deltas: Vec<StreamDelta>,
}

impl DeltaSink for CollectingSink {
    fn on_delta(&mut self, delta: StreamDelta) {
        self.deltas.push(delta);
    }
}

#[tokio::test]
async fn streaming_text_and_usage() {
    let events = vec![
        r#"{"choices":[{"delta":{"content":"Hello"}}]}"#.to_string(),
        r#"{"choices":[{"delta":{"content":" world"}}]}"#.to_string(),
        r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#.to_string(),
        r#"{"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}"#.to_string(),
        "[DONE]".to_string(),
    ];

    let server = MockServer::start(Box::new(move |_method, _path, _body, _headers| {
        MockResponse::sse(200, events.clone())
    }))
    .await;

    let provider = make_provider(&server, "test-model");
    let mut session = Session::new("test");
    session.messages.push(Message::user_text("Hi"));
    let request = ModelRequest::new("test", session.messages);

    let mut sink = CollectingSink::default();
    let response = provider.stream(request, &mut sink).await.unwrap();

    assert_eq!(response.content, "Hello world");
    assert_eq!(response.finish_reason, FinishReason::Stop);
    assert_eq!(response.usage.total_tokens, Some(4));

    let text_deltas: Vec<_> = sink
        .deltas
        .iter()
        .filter(|d| matches!(d, StreamDelta::Text { .. }))
        .collect();
    assert_eq!(text_deltas.len(), 2);

    let usage_deltas: Vec<_> = sink
        .deltas
        .iter()
        .filter(|d| matches!(d, StreamDelta::Usage { .. }))
        .collect();
    assert_eq!(usage_deltas.len(), 1);
}

#[tokio::test]
async fn streaming_reasoning_preserved() {
    let events = vec![
        r#"{"choices":[{"delta":{"reasoning_content":"thinking..."}}]}"#.to_string(),
        r#"{"choices":[{"delta":{"content":"answer"}}]}"#.to_string(),
        r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#.to_string(),
        "[DONE]".to_string(),
    ];

    let server = MockServer::start(Box::new(move |_method, _path, _body, _headers| {
        MockResponse::sse(200, events.clone())
    }))
    .await;

    let provider = make_provider(&server, "test-model");
    let mut session = Session::new("test");
    session.messages.push(Message::user_text("think"));
    let request = ModelRequest::new("test", session.messages);

    let mut sink = CollectingSink::default();
    let response = provider.stream(request, &mut sink).await.unwrap();

    assert_eq!(response.content, "answer");
    assert_eq!(
        response.reasoning.as_ref().and_then(|r| r.text.as_deref()),
        Some("thinking...")
    );
    assert!(sink
        .deltas
        .iter()
        .any(|d| matches!(d, StreamDelta::Reasoning { text } if text == "thinking...")));
}

#[tokio::test]
async fn streaming_fragmented_chunks() {
    let events = vec![
        r#"{"choices":[{"delta":{"content":"frag"}}]}"#.to_string(),
        r#"{"choices":[{"delta":{"content":"mented"}}]}"#.to_string(),
        r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#.to_string(),
        "[DONE]".to_string(),
    ];

    let server = MockServer::start(Box::new(move |_method, _path, _body, _headers| {
        MockResponse::sse(200, events.clone())
    }))
    .await;

    let provider = make_provider(&server, "test-model");
    let mut session = Session::new("test");
    session.messages.push(Message::user_text("Hi"));
    let request = ModelRequest::new("test", session.messages);

    let mut sink = CollectingSink::default();
    let response = provider.stream(request, &mut sink).await.unwrap();
    assert_eq!(response.content, "fragmented");
}

#[tokio::test]
async fn session_stream_without_finish_does_not_dispatch_tool() {
    let server = MockServer::start(Box::new(|_, _, _, _| {
        MockResponse::sse(
            200,
            vec![r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"echo","arguments":"{\"text\":\"x\"}"}}]}}]}"#.into()],
        )
    }))
    .await;
    let agent =
        make_agent_loop(Arc::new(make_provider(&server, "test-model"))).with_config(LoopConfig {
            prefer_streaming: true,
            ..LoopConfig::default()
        });
    let mut session = Session::new("partial");
    let events = EventCollector::new();
    let result = agent
        .run(&mut session, "call echo", &events, &StopToken::new())
        .await;
    assert!(result.is_err());
    assert_eq!(
        events.count_of(|e| matches!(e, Event::ToolStarted { .. })),
        0
    );
}

#[tokio::test]
async fn session_done_without_finish_does_not_dispatch_tool() {
    let server = MockServer::start(Box::new(|_, _, _, _| {
        MockResponse::sse(
            200,
            vec![r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"echo","arguments":"{\"text\":\"x\"}"}}]}}]}"#.into(), "[DONE]".into()],
        )
    }))
    .await;
    let agent =
        make_agent_loop(Arc::new(make_provider(&server, "test-model"))).with_config(LoopConfig {
            prefer_streaming: true,
            ..LoopConfig::default()
        });
    let mut session = Session::new("done-no-finish");
    let events = EventCollector::new();
    assert!(agent
        .run(&mut session, "call echo", &events, &StopToken::new())
        .await
        .is_err());
    assert_eq!(
        events.count_of(|e| matches!(e, Event::ToolStarted { .. })),
        0
    );
}

#[tokio::test]
async fn session_done_stops_reading_a_kept_open_connection() {
    let release = Arc::new(Notify::new());
    let release_for_handler = release.clone();
    let server = MockServer::start(Box::new(move |_, _, _, _| {
        MockResponse::sse(
            200,
            vec![
                r#"{"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}"#.into(),
                "[DONE]".into(),
            ],
        )
        .hold_open(release_for_handler.clone())
    }))
    .await;
    let agent =
        make_agent_loop(Arc::new(make_provider(&server, "test-model"))).with_config(LoopConfig {
            prefer_streaming: true,
            ..LoopConfig::default()
        });
    let mut session = Session::new("done-open");
    let events = EventCollector::new();
    let result = tokio::time::timeout(
        Duration::from_secs(1),
        agent.run(&mut session, "hello", &events, &StopToken::new()),
    )
    .await;
    release.notify_waiters();
    assert_eq!(result.unwrap().unwrap(), "done");
}

// ───────────────────────────────────────────────────────────────────
// Discovery tests (no real network, no env mutation)
// ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn discovery_merges_endpoint_models() {
    let server = MockServer::start(Box::new(|_method, path, _body, _headers| {
        if path.ends_with("/v1/models") {
            MockResponse::json(
                200,
                json!({"data": [{"id": "gpt-4o-mini"}, {"id": "gpt-4o"}]}),
            )
        } else {
            MockResponse::error(404, "not found")
        }
    }))
    .await;

    let spec = ProviderSpec::custom(
        "test-provider",
        "Test Provider",
        server.base_url(),
        Some(Secret::new("key")),
    );

    let options = DiscoveryOptions {
        include_models_dev: false,
        ..DiscoveryOptions::default()
    };
    let discovery = Discovery::new();
    let models = discovery.discover(&spec, &options).await.unwrap();

    assert_eq!(models.len(), 2);
    assert!(models.iter().any(|m| m.model_id == "gpt-4o-mini"));
    assert!(models.iter().any(|m| m.model_id == "gpt-4o"));
    assert!(models.iter().all(|m| m.from_endpoint));
    // Without catalog metadata, tool support is Unknown (not Yes).
    assert!(models
        .iter()
        .all(|m| m.tool_support == ToolSupport::Unknown));
}

#[tokio::test]
async fn discovery_caches_results_no_duplicate_requests() {
    let call_count = Arc::new(AtomicU32::new(0));
    let call_count_clone = call_count.clone();

    let server = MockServer::start(Box::new(move |_method, path, _body, _headers| {
        if path.ends_with("/v1/models") {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
            MockResponse::json(200, json!({"data": [{"id": "cached-model"}]}))
        } else {
            MockResponse::error(404, "")
        }
    }))
    .await;

    let spec = ProviderSpec::custom(
        "test-provider",
        "Test Provider",
        server.base_url(),
        Some(Secret::new("key")),
    );

    let options = DiscoveryOptions {
        include_models_dev: false,
        ..DiscoveryOptions::default()
    };
    let discovery = Discovery::new();

    // First call hits the endpoint.
    let models1 = discovery.discover(&spec, &options).await.unwrap();
    assert_eq!(models1.len(), 1);
    assert_eq!(call_count.load(Ordering::SeqCst), 1);
    assert_eq!(server.models_call_count(), 1);

    // Second call uses cache — no new request.
    let models2 = discovery.discover(&spec, &options).await.unwrap();
    assert_eq!(models2.len(), 1);
    assert_eq!(call_count.load(Ordering::SeqCst), 1);
    assert_eq!(server.models_call_count(), 1);
}

#[tokio::test]
async fn discovery_cache_expires_with_test_clock() {
    let clock = Arc::new(TestClock::new(Instant::now()));
    let call_count = Arc::new(AtomicU32::new(0));
    let call_count_clone = call_count.clone();

    let server = MockServer::start(Box::new(move |_method, path, _body, _headers| {
        if path.ends_with("/v1/models") {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
            MockResponse::json(200, json!({"data": [{"id": "model"}]}))
        } else {
            MockResponse::error(404, "")
        }
    }))
    .await;

    let spec = ProviderSpec::custom(
        "test-provider",
        "Test Provider",
        server.base_url(),
        Some(Secret::new("key")),
    );

    let options = DiscoveryOptions {
        include_models_dev: false,
        ..DiscoveryOptions::default()
    };

    // Use a short TTL with the test clock — no sleeping.
    let discovery = Discovery::with_clock(Duration::from_millis(100), clock.clone());

    // First call hits endpoint.
    let _models1 = discovery.discover(&spec, &options).await.unwrap();
    assert_eq!(call_count.load(Ordering::SeqCst), 1);

    // Second call uses cache (before expiry).
    let _models2 = discovery.discover(&spec, &options).await.unwrap();
    assert_eq!(call_count.load(Ordering::SeqCst), 1);

    // Advance past TTL — no sleeping.
    clock.advance(Duration::from_millis(150));

    // Third call hits endpoint again (cache expired).
    let _models3 = discovery.discover(&spec, &options).await.unwrap();
    assert_eq!(call_count.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn discovery_cache_isolates_by_account_identity() {
    let server = MockServer::start(Box::new(|_method, path, _body, _headers| {
        if path.ends_with("/v1/models") {
            MockResponse::json(200, json!({"data": [{"id": "model"}]}))
        } else {
            MockResponse::error(404, "")
        }
    }))
    .await;

    let spec1 = ProviderSpec::custom(
        "test",
        "Test",
        server.base_url(),
        Some(Secret::new("key-a")),
    );
    let spec2 = ProviderSpec::custom(
        "test",
        "Test",
        server.base_url(),
        Some(Secret::new("key-b")),
    );

    let options = DiscoveryOptions {
        include_models_dev: false,
        ..DiscoveryOptions::default()
    };
    let discovery = Discovery::new();

    // Different account identities must not share cache entries.
    // Both should hit the endpoint (2 separate requests).
    let _models1 = discovery.discover(&spec1, &options).await.unwrap();
    let _models2 = discovery.discover(&spec2, &options).await.unwrap();
    assert_eq!(server.models_call_count(), 2);

    // Now spec1's cache hit should not serve spec2.
    let _models1_cached = discovery.discover(&spec1, &options).await.unwrap();
    assert_eq!(server.models_call_count(), 2); // spec1 served from cache
    let _models2_cached = discovery.discover(&spec2, &options).await.unwrap();
    assert_eq!(server.models_call_count(), 2); // spec2 also served from cache
}

#[tokio::test]
async fn discovery_cache_isolates_by_endpoint() {
    let server_a = MockServer::start(Box::new(|_method, path, _body, _headers| {
        if path.ends_with("/v1/models") {
            MockResponse::json(200, json!({"data": [{"id": "model-a"}]}))
        } else {
            MockResponse::error(404, "")
        }
    }))
    .await;

    let server_b = MockServer::start(Box::new(|_method, path, _body, _headers| {
        if path.ends_with("/v1/models") {
            MockResponse::json(200, json!({"data": [{"id": "model-b"}]}))
        } else {
            MockResponse::error(404, "")
        }
    }))
    .await;

    let spec_a = ProviderSpec::custom(
        "test",
        "Test",
        server_a.base_url(),
        Some(Secret::new("key")),
    );
    let spec_b = ProviderSpec::custom(
        "test",
        "Test",
        server_b.base_url(),
        Some(Secret::new("key")),
    );

    let options = DiscoveryOptions {
        include_models_dev: false,
        ..DiscoveryOptions::default()
    };
    let discovery = Discovery::new();

    let models_a = discovery.discover(&spec_a, &options).await.unwrap();
    let models_b = discovery.discover(&spec_b, &options).await.unwrap();

    assert!(models_a.iter().any(|m| m.model_id == "model-a"));
    assert!(models_b.iter().any(|m| m.model_id == "model-b"));
    // Different endpoints — both hit their respective servers.
    assert_eq!(server_a.models_call_count(), 1);
    assert_eq!(server_b.models_call_count(), 1);
}

#[tokio::test]
async fn discovery_cache_isolates_provider_metadata_on_same_endpoint() {
    let endpoint = MockServer::start(Box::new(|_, path, _, _| {
        if path.ends_with("/v1/models") {
            MockResponse::json(200, json!({"data": []}))
        } else {
            MockResponse::error(404, "not found")
        }
    }))
    .await;
    let catalog = MockServer::start(Box::new(|_, _, _, _| MockResponse::json(200, json!({
        "provider-a": {"name": "A", "models": {"shared": {"name": "A shared", "tool_call": true}}},
        "provider-b": {"name": "B", "models": {"shared": {"name": "B shared", "tool_call": false}}}
    })))).await;
    let options = DiscoveryOptions {
        models_dev_url_override: Some(format!("http://{}/catalog.json", catalog.addr)),
        ..Default::default()
    };
    let spec_a = ProviderSpec::custom("provider-a", "A", endpoint.base_url(), None::<Secret>);
    let spec_b = ProviderSpec::custom("provider-b", "B", endpoint.base_url(), None::<Secret>);
    let discovery = Discovery::new();
    let models_a = discovery.discover(&spec_a, &options).await.unwrap();
    let models_b = discovery.discover(&spec_b, &options).await.unwrap();
    assert_eq!(models_a[0].name, "A shared");
    assert_eq!(models_a[0].tool_support, ToolSupport::Yes);
    assert_eq!(models_b[0].name, "B shared");
    assert_eq!(models_b[0].tool_support, ToolSupport::No);
}

#[tokio::test]
async fn discovery_explicit_model_ids_offline() {
    // Unreachable endpoint — only explicit IDs should be returned.
    let spec = ProviderSpec::custom(
        "offline-provider",
        "Offline Provider",
        "http://127.0.0.1:1/v1",
        None::<Secret>,
    );

    let options = DiscoveryOptions {
        include_models_dev: false,
        explicit_model_ids: vec!["custom-model-1".into(), "custom-model-2".into()],
        ..DiscoveryOptions::default()
    };
    let discovery = Discovery::new();
    let models = discovery.discover(&spec, &options).await.unwrap();

    assert_eq!(models.len(), 2);
    assert!(models.iter().any(|m| m.model_id == "custom-model-1"));
    assert!(models.iter().any(|m| m.model_id == "custom-model-2"));
    assert!(models.iter().all(|m| !m.from_endpoint));
    assert!(models
        .iter()
        .all(|m| m.tool_support == ToolSupport::Unknown));
}

#[tokio::test]
async fn discovery_cache_includes_explicit_ids_and_catalog_url() {
    let endpoint = MockServer::start(Box::new(|_, path, _, _| {
        if path.ends_with("/v1/models") {
            MockResponse::json(200, json!({"data": []}))
        } else {
            MockResponse::error(404, "not found")
        }
    }))
    .await;
    let catalog_a = MockServer::start(Box::new(|_, _, _, _| {
        MockResponse::json(
            200,
            json!({"provider": {"name": "P", "models": {
                "catalog-a": {"name": "A"}
            }}}),
        )
    }))
    .await;
    let catalog_b = MockServer::start(Box::new(|_, _, _, _| {
        MockResponse::json(
            200,
            json!({"provider": {"name": "P", "models": {
                "catalog-b": {"name": "B"}
            }}}),
        )
    }))
    .await;
    let spec = ProviderSpec::custom("provider", "P", endpoint.base_url(), None::<Secret>);
    let discovery = Discovery::new();

    let first = DiscoveryOptions {
        include_models_dev: true,
        explicit_model_ids: vec!["first".into()],
        models_dev_url_override: Some(format!("http://{}/api.json", catalog_a.addr)),
        ..Default::default()
    };
    let second = DiscoveryOptions {
        include_models_dev: true,
        explicit_model_ids: vec!["second".into()],
        models_dev_url_override: Some(format!("http://{}/api.json", catalog_b.addr)),
        ..Default::default()
    };
    let a = discovery.discover(&spec, &first).await.unwrap();
    let b = discovery.discover(&spec, &second).await.unwrap();
    assert!(a
        .iter()
        .any(|m| m.model_id == "first" && m.model_id != "second"));
    assert!(a.iter().any(|m| m.model_id == "catalog-a"));
    assert!(b
        .iter()
        .any(|m| m.model_id == "second" && m.model_id != "first"));
    assert!(b.iter().any(|m| m.model_id == "catalog-b"));
}

#[tokio::test]
async fn discovery_ollama_main_api_uses_native_tags() {
    let server = MockServer::start(Box::new(|_, path, _, headers| {
        assert_eq!(path, "/api/tags");
        assert_eq!(
            headers.get("authorization").map(String::as_str),
            Some("Bearer ollama-key")
        );
        MockResponse::json(200, json!({"models": [{"name": "llama3:8b"}]}))
    }))
    .await;
    let spec = ProviderSpec::ollama()
        .with_base_url(format!("http://{}/v1", server.addr))
        .with_key(Secret::new("ollama-key"));
    let options = DiscoveryOptions {
        include_models_dev: false,
        ..Default::default()
    };
    let models = Discovery::new().discover(&spec, &options).await.unwrap();
    assert_eq!(
        models
            .iter()
            .map(|m| m.model_id.as_str())
            .collect::<Vec<_>>(),
        vec!["llama3:8b"]
    );
}

#[tokio::test]
async fn discovery_total_failure_is_typed_and_not_cached() {
    let spec = ProviderSpec::custom(
        "offline",
        "Offline",
        "http://127.0.0.1:1/v1",
        None::<Secret>,
    );
    let options = DiscoveryOptions {
        models_dev_url_override: Some("http://127.0.0.1:1/catalog.json".into()),
        ..Default::default()
    };
    let err = Discovery::new()
        .discover(&spec, &options)
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        unravel_agent_providers::ProviderError::DiscoveryFailed { .. }
    ));
}

#[tokio::test]
async fn discovery_rejects_empty_explicit_model_id() {
    let spec = ProviderSpec::custom(
        "offline",
        "Offline",
        "http://127.0.0.1:1/v1",
        None::<Secret>,
    );
    let options = DiscoveryOptions {
        include_models_dev: false,
        explicit_model_ids: vec!["  ".into()],
        ..Default::default()
    };
    let err = Discovery::new()
        .discover(&spec, &options)
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        unravel_agent_providers::ProviderError::Invalid { .. }
    ));
}

#[tokio::test]
async fn discovery_filters_deprecated_and_non_tool_for_tool_workflow() {
    // We can't easily test Models.dev filtering through the real fetch
    // path without a mock catalog server. Instead, test the filter
    // logic directly with known models.
    let models = vec![
        discovered_model("good", ToolSupport::Yes, false),
        discovered_model("deprecated", ToolSupport::Yes, true),
        discovered_model("no-tools", ToolSupport::No, false),
        discovered_model("unknown", ToolSupport::Unknown, false),
    ];

    let mut filtered = models.clone();
    filtered.retain(|m| !m.deprecated && m.tool_support != ToolSupport::No);

    assert_eq!(filtered.len(), 2);
    assert!(filtered.iter().any(|m| m.model_id == "good"));
    assert!(filtered.iter().any(|m| m.model_id == "unknown"));
    // Unknown is preserved (not silently filtered).
}

fn discovered_model(
    id: &str,
    tool_support: ToolSupport,
    deprecated: bool,
) -> unravel_agent_providers::DiscoveredModel {
    unravel_agent_providers::DiscoveredModel {
        provider_id: "test".into(),
        model_id: id.into(),
        name: id.into(),
        tool_support,
        deprecated,
        reasoning: false,
        modalities: ModalitySupport::default(),
        context_window: 0,
        max_output: 0,
        from_endpoint: true,
    }
}

// ───────────────────────────────────────────────────────────────────
// Models.dev fixture test via mock server (real fetch path)
// ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn models_dev_fixture_via_mock_server() {
    // Serve a realistic Models.dev catalog from a mock server.
    let fixture = json!({
        "test-provider": {
            "id": "test-provider",
            "name": "Test Provider",
            "npm": "@ai-sdk/openai-compatible",
            "api": "http://localhost:8080/v1",
            "env": ["TEST_API_KEY"],
            "models": {
                "good-model": {
                    "id": "good-model",
                    "name": "Good Model",
                    "tool_call": true,
                    "attachment": true,
                    "reasoning": false,
                    "modalities": {
                        "input": ["text", "image"],
                        "output": ["text"]
                    },
                    "limit": {"context": 128000, "output": 16384},
                    "cost": {"input": 0.15, "output": 0.6, "cache_read": 0.075}
                },
                "deprecated-model": {
                    "id": "deprecated-model",
                    "name": "Deprecated Model",
                    "tool_call": true,
                    "status": "deprecated",
                    "modalities": {"input": ["text"], "output": ["text"]}
                },
                "no-tool-model": {
                    "id": "no-tool-model",
                    "name": "No Tool Model",
                    "tool_call": false,
                    "modalities": {"input": ["text"], "output": ["text"]}
                },
                "unknown-tool-model": {
                    "name": "Unknown Tool Model",
                    "modalities": {"input": ["text"], "output": ["text"]}
                }
            }
        }
    });

    let catalog_server = MockServer::start(Box::new(move |_method, _path, _body, _headers| {
        MockResponse::json(200, fixture.clone())
    }))
    .await;

    // Parse the catalog through the real fetch path, using the mock
    // server's URL as the explicit override (no env mutation).
    let catalog_url = format!("http://{}/api.json", catalog_server.addr);
    let catalog = unravel_agent_providers::catalog::fetch_catalog(Some(&catalog_url))
        .await
        .unwrap();

    assert_eq!(catalog.len(), 1);
    let provider = &catalog[0];
    assert_eq!(provider.id, "test-provider");
    assert_eq!(provider.models.len(), 4);

    // Verify tool support preservation.
    let good = provider
        .models
        .iter()
        .find(|m| m.model_id == "good-model")
        .unwrap();
    assert_eq!(good.tool_support, ToolSupport::Yes);
    assert!(!good.deprecated);
    assert!(good.modalities.image_input);
    assert_eq!(good.context_window, 128000);
    assert!((good.cost_input - 0.15).abs() < 0.001);

    let deprecated = provider
        .models
        .iter()
        .find(|m| m.model_id == "deprecated-model")
        .unwrap();
    assert_eq!(deprecated.tool_support, ToolSupport::Yes);
    assert!(deprecated.deprecated);

    let no_tool = provider
        .models
        .iter()
        .find(|m| m.model_id == "no-tool-model")
        .unwrap();
    assert_eq!(no_tool.tool_support, ToolSupport::No);

    let unknown = provider
        .models
        .iter()
        .find(|m| m.model_id == "unknown-tool-model")
        .unwrap();
    assert_eq!(unknown.tool_support, ToolSupport::Unknown);
}

#[tokio::test]
async fn models_dev_catalog_merge_preserves_provenance() {
    // Test the merge_with_catalog function directly with a realistic
    // fixture to verify provenance (from_endpoint vs catalog-only).
    let spec = ProviderSpec::custom(
        "test-provider",
        "Test Provider",
        "http://localhost:8080/v1",
        Some(Secret::new("key")),
    );

    // Model on the endpoint but NOT in the catalog → from_endpoint=true, Unknown.
    // Model in the catalog but NOT on the endpoint → from_endpoint=false.
    // Model in both → from_endpoint=true, enriched with catalog metadata.
    let endpoint_models = vec!["endpoint-only".to_string(), "shared".to_string()];

    let catalog = vec![CatalogProvider {
        npm: "@ai-sdk/openai".into(),
        env: vec![],
        id: "test-provider".to_string(),
        name: "Test Provider".to_string(),
        base_url: Some("http://localhost:8080/v1".into()),
        api_key_required: true,
        models: vec![
            CatalogModel {
                provider_id: "test-provider".to_string(),
                model_id: "shared".to_string(),
                name: "Shared Model".to_string(),
                tool_support: ToolSupport::Yes,
                deprecated: false,
                reasoning: false,
                attachment: true,
                modalities: ModalitySupport {
                    text_input: true,
                    image_input: true,
                    text_output: true,
                },
                context_window: 128000,
                max_output: 16384,
                cost_input: 0.15,
                cost_output: 0.6,
                cost_cache_read: 0.075,
            },
            CatalogModel {
                provider_id: "test-provider".to_string(),
                model_id: "catalog-only".to_string(),
                name: "Catalog Only Model".to_string(),
                tool_support: ToolSupport::No,
                deprecated: false,
                reasoning: false,
                attachment: false,
                modalities: ModalitySupport::default(),
                context_window: 0,
                max_output: 0,
                cost_input: 0.0,
                cost_output: 0.0,
                cost_cache_read: 0.0,
            },
        ],
    }];

    // Use the public merge function.
    let merged =
        unravel_agent_providers::discovery::merge_with_catalog(&spec, &endpoint_models, &catalog);

    assert_eq!(merged.len(), 3);

    // endpoint-only: from_endpoint=true, Unknown tool support.
    let eo = merged
        .iter()
        .find(|m| m.model_id == "endpoint-only")
        .unwrap();
    assert!(eo.from_endpoint);
    assert_eq!(eo.tool_support, ToolSupport::Unknown);

    // shared: from_endpoint=true, enriched with Yes.
    let sh = merged.iter().find(|m| m.model_id == "shared").unwrap();
    assert!(sh.from_endpoint);
    assert_eq!(sh.tool_support, ToolSupport::Yes);
    assert!(sh.modalities.image_input);
    assert_eq!(sh.context_window, 128000);

    // catalog-only: from_endpoint=false.
    let co = merged
        .iter()
        .find(|m| m.model_id == "catalog-only")
        .unwrap();
    assert!(!co.from_endpoint);
    assert_eq!(co.tool_support, ToolSupport::No);
}

// ───────────────────────────────────────────────────────────────────
// Ollama tests
// ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn ollama_tags_url_normalization() {
    let server = MockServer::start(Box::new(|_method, path, _body, _headers| {
        // The tags endpoint should be at /api/tags, not /v1/api/tags.
        assert_eq!(path, "/api/tags");
        MockResponse::json(
            200,
            json!({"models": [{"name": "llama3:8b"}, {"name": "qwen2:7b"}]}),
        )
    }))
    .await;

    let base_url = format!("http://{}/v1", server.addr);
    let tags = unravel_agent_providers::fetch_ollama_tags(&base_url)
        .await
        .unwrap();
    assert_eq!(tags, vec!["llama3:8b", "qwen2:7b"]);
}

#[tokio::test]
async fn ollama_tags_url_no_v1_suffix() {
    let server = MockServer::start(Box::new(|_method, path, _body, _headers| {
        assert_eq!(path, "/api/tags");
        MockResponse::json(200, json!({"models": [{"name": "mistral:7b"}]}))
    }))
    .await;

    let base_url = format!("http://{}", server.addr);
    let tags = unravel_agent_providers::fetch_ollama_tags(&base_url)
        .await
        .unwrap();
    assert_eq!(tags, vec!["mistral:7b"]);
}

#[tokio::test]
async fn chat_boundary_preserves_raw_arguments_and_large_cached_usage() {
    use unravel_agent_providers::{ChatOptions, ChatRequest};
    let server = MockServer::start(Box::new(|method, path, body, headers| {
        assert_eq!(method, "POST");
        assert_eq!(path, "/v1/chat/completions");
        assert_eq!(headers["authorization"], "Bearer override-key");
        assert_eq!(headers["x-initiator"], "agent");
        let body: Value = serde_json::from_str(body).unwrap();
        assert_eq!(body["tools"], json!([]));
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["temperature"], 0.25);
        assert!(body.get("store").is_none());
        assert!(body.get("reasoning_effort").is_none());
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["messages"][0]["content"][0]["image_url"]["url"], "data:image/png;base64,aW1n");
        assert_eq!(body["messages"][1]["reasoning_content"], "retained trace");
        MockResponse::json(200, json!({
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_text": "new trace",
                    "tool_calls": [{"id": "call_1", "function": {"name": "echo", "arguments": "{\"text\":"}}]
                },
                "finish_reason": "tool_calls",
                "usage": {
                    "prompt_tokens": 4294967310_u64, "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 4294967300_u64}
                }
            }]
        }))
    })).await;
    let mut options = ChatOptions::default();
    options
        .headers
        .insert("authorization", "Bearer override-key".parse().unwrap());
    options
        .headers
        .insert("x-initiator", "agent".parse().unwrap());
    let provider = make_provider(&server, "test-model").with_chat_options(options);
    let request = ChatRequest {
        messages: vec![
            json!({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1n"}}]}),
            json!({"role": "assistant", "content": "", "reasoning_content": "retained trace"}),
        ],
        tools: Some(vec![]),
        sampling: Some(unravel_agent_runtime::Sampling::new().with_temperature(0.25)),
        ..Default::default()
    };
    let response = provider.complete_chat(request).await.unwrap();
    assert_eq!(response.tool_calls[0].arguments, "{\"text\":");
    assert_eq!(response.usage.prompt_tokens, Some(4294967310));
    assert_eq!(response.usage.cache_read_tokens, Some(4294967300));
    assert_eq!(response.usage.total_tokens, None);
    assert_eq!(response.usage.cache_write_tokens, None);
    assert_eq!(
        response.reasoning.unwrap().text.as_deref(),
        Some("new trace")
    );
}

#[tokio::test]
async fn custom_authentication_is_redacted_from_errors_and_debug() {
    use unravel_agent_providers::{ChatOptions, ChatRequest};
    let server = MockServer::start(Box::new(|_, _, _, _| {
        MockResponse::error(
            401,
            "credential custom-api-secret; override-token; original-key",
        )
    }))
    .await;
    let mut options = ChatOptions::default();
    options
        .headers
        .insert("x-api-key", "custom-api-secret".parse().unwrap());
    options
        .headers
        .insert("authorization", "bEaReR override-token".parse().unwrap());
    let provider = OpenAiCompatProvider::new(
        ProviderSpec::custom(
            "test",
            "Test",
            server.base_url(),
            Some(Secret::new("original-key")),
        ),
        "test-model",
    )
    .unwrap()
    .with_chat_options(options.clone());
    let error = provider
        .complete_chat(ChatRequest::default())
        .await
        .unwrap_err();
    let diagnostics = format!("{provider:?} {options:?} {error} {error:?}");
    for secret in ["custom-api-secret", "override-token", "original-key"] {
        assert!(!diagnostics.contains(secret), "{diagnostics}");
    }
    assert!(matches!(
        error,
        unravel_agent_providers::ProviderError::Auth { .. }
    ));
}

#[tokio::test]
async fn raw_chat_stream_preserves_fragments_and_terminal_usage_without_dispatch_validation() {
    use unravel_agent_providers::{ChatEventSink, ChatRequest, ChatStreamEvent};
    #[derive(Default)]
    struct Sink {
        deltas: Vec<StreamDelta>,
        terminal_arguments: Vec<String>,
        started: bool,
    }
    impl ChatEventSink for Sink {
        fn on_event(&mut self, event: ChatStreamEvent<'_>) {
            match event {
                ChatStreamEvent::Start => self.started = true,
                ChatStreamEvent::Delta(delta) => self.deltas.push(delta),
                ChatStreamEvent::Finished(response) => {
                    self.terminal_arguments = response
                        .tool_calls
                        .iter()
                        .map(|c| c.arguments.clone())
                        .collect();
                }
            }
        }
    }
    let server = MockServer::start(Box::new(|_, _, body, _| {
        let body: Value = serde_json::from_str(body).unwrap();
        assert_eq!(body["stream_options"], json!({"include_usage": true}));
        assert!(body.get("temperature").is_none());
        MockResponse::sse(200, vec![
            json!({"choices":[{"delta":{"reasoning":"think"}}]}).to_string(),
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"echo","arguments":"{\"text\":"}}]}}]}).to_string(),
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"broken"}}]}}]}).to_string(),
            json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}]}).to_string(),
            json!({"usage":null,"choices":[{"usage":{"prompt_tokens":4294967310_u64,"prompt_tokens_details":{"cached_tokens":4}}}]}).to_string(),
            "[DONE]".to_string(),
        ])
    })).await;
    let provider = make_provider(&server, "test-model");
    let mut sink = Sink::default();
    let response = provider
        .stream_chat(ChatRequest::default(), &mut sink)
        .await
        .unwrap();
    assert!(sink.started);
    assert_eq!(sink.terminal_arguments, ["{\"text\":broken"]);
    assert_eq!(response.tool_calls[0].arguments, "{\"text\":broken");
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCall));
    assert_eq!(response.usage.prompt_tokens, Some(4294967310));
    assert_eq!(response.usage.cache_read_tokens, Some(4));
    assert!(matches!(&sink.deltas[0], StreamDelta::Reasoning { text } if text == "think"));
    assert!(
        matches!(&sink.deltas[2], StreamDelta::ToolCall { arguments, .. } if arguments == "broken")
    );
    assert!(
        matches!(&sink.deltas[3], StreamDelta::Usage { usage } if usage.prompt_tokens == Some(4294967310))
    );
    let mut canonical_sink = CollectingSink::default();
    let error = provider
        .stream(ModelRequest::new("strict", vec![]), &mut canonical_sink)
        .await
        .unwrap_err();
    assert!(
        !error.is_retryable(),
        "canonical Model must still reject malformed arguments"
    );
}

#[tokio::test]
async fn raw_chat_options_cannot_enable_redirects() {
    use unravel_agent_providers::{ChatOptions, ChatRequest};
    let target = MockServer::start(Box::new(|_, _, _, _| {
        MockResponse::json(
            200,
            json!({"choices":[{"message":{"content":"leaked"},"finish_reason":"stop"}]}),
        )
    }))
    .await;
    let location = format!("{}/chat/completions", target.base_url());
    let server = MockServer::start(Box::new(move |_, _, _, _| MockResponse {
        status: 307,
        headers: vec![("location".into(), location.clone())],
        body: vec![],
        hold_open: None,
    }))
    .await;
    let mut options = ChatOptions::default();
    options
        .headers
        .insert("x-api-key", "private".parse().unwrap());
    let provider = make_provider(&server, "test-model").with_chat_options(options);
    assert!(provider
        .complete_chat(ChatRequest::default())
        .await
        .is_err());
    assert_eq!(
        target.completions_call_count(),
        0,
        "custom credentials must never follow redirects"
    );
}

#[tokio::test]
async fn unfinished_raw_stream_cannot_return_dispatchable_calls() {
    use unravel_agent_providers::{ChatEventSink, ChatRequest, ChatStreamEvent};
    struct Sink;
    impl ChatEventSink for Sink {
        fn on_event(&mut self, _: ChatStreamEvent<'_>) {}
    }
    let server = MockServer::start(Box::new(|_, _, _, _| MockResponse::sse(200, vec![
        json!({"choices":[{"delta":{"content":"partial","tool_calls":[{"index":0,"id":"c1","function":{"name":"echo","arguments":"{}"}}]}}]}).to_string(),
        "[DONE]".into(),
    ]))).await;
    let response = make_provider(&server, "test-model")
        .stream_chat(ChatRequest::default(), &mut Sink)
        .await
        .unwrap();
    assert_eq!(response.content, "partial");
    assert_eq!(response.finish_reason, None);
    assert!(response.tool_calls.is_empty());
}

#[tokio::test]
async fn raw_null_tool_list_does_not_relax_canonical_response_validation() {
    use unravel_agent_providers::ChatRequest;
    let server = MockServer::start(Box::new(|_, _, _, _| MockResponse::json(200, json!({
        "choices": [{"message": {"content": "answer", "tool_calls": null}, "finish_reason": "stop"}]
    })))).await;
    let provider = make_provider(&server, "test-model");
    let raw = provider
        .complete_chat(ChatRequest::default())
        .await
        .unwrap();
    assert_eq!(raw.content, "answer");
    assert!(raw.tool_calls.is_empty());
    let error = provider
        .complete(ModelRequest::new("strict", vec![]))
        .await
        .unwrap_err();
    assert!(!error.is_retryable());
}

#[tokio::test]
async fn session_changed_terminal_reason_never_dispatches_tools() {
    let server = MockServer::start(Box::new(|_, _, _, _| {
        MockResponse::sse(
            200,
            vec![
                json!({"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "call_1",
                "function": {"name": "echo", "arguments": "{\"text\":\"x\"}"}
            }]}, "finish_reason": "length"}]})
                .to_string(),
                json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}).to_string(),
                "[DONE]".into(),
            ],
        )
    }))
    .await;
    let agent =
        make_agent_loop(Arc::new(make_provider(&server, "test-model"))).with_config(LoopConfig {
            prefer_streaming: true,
            max_model_attempts: 1,
            max_turns: 1,
            ..LoopConfig::default()
        });
    let mut session = Session::new("changed-terminal");
    let events = EventCollector::new();
    assert!(agent
        .run(&mut session, "call echo", &events, &StopToken::new())
        .await
        .is_err());
    assert_eq!(
        events.count_of(|event| matches!(event, Event::ToolStarted { .. })),
        0
    );
}

#[tokio::test]
async fn session_malformed_streamed_batch_never_dispatches_valid_prefix() {
    let server = MockServer::start(Box::new(|_, _, _, _| {
        MockResponse::sse(
            200,
            vec![
                json!({"choices": [{"delta": {"tool_calls": [{
                    "index": 0, "id": "call_1",
                    "function": {"name": "echo", "arguments": "{\"text\":\"x\"}"}
                }]}}]})
                .to_string(),
                json!({"choices": [{"delta": {"tool_calls": {
                "index": 1, "id": "malformed-call"
            }}, "finish_reason": "tool_calls"}]})
                .to_string(),
                "[DONE]".into(),
            ],
        )
    }))
    .await;
    let agent =
        make_agent_loop(Arc::new(make_provider(&server, "test-model"))).with_config(LoopConfig {
            prefer_streaming: true,
            max_model_attempts: 1,
            max_turns: 1,
            ..LoopConfig::default()
        });
    let mut session = Session::new("malformed-batch");
    let events = EventCollector::new();
    assert!(agent
        .run(&mut session, "call echo", &events, &StopToken::new())
        .await
        .is_err());
    assert_eq!(
        events.count_of(|event| matches!(event, Event::ToolStarted { .. })),
        0
    );
}
