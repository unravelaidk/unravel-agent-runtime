use crate::{Content, Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, sync::Arc, time::Duration};
use tokio::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Untrusted sensor content delivered only to the next model turn.
///
/// Observations never enter session history or event metadata and cannot be
/// replayed after resume. Retries reuse identical content until monotonic
/// expiry; a zero or unrepresentable lifetime is already stale.
#[derive(Debug, Clone)]
pub struct ToolObservation {
    pub content: Content,
    pub(crate) expires_at: Instant,
}

impl ToolObservation {
    pub fn new(content: Content, valid_for: Duration) -> Self {
        let now = Instant::now();
        Self {
            content,
            expires_at: now.checked_add(valid_for).unwrap_or(now),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolOutput {
    /// Persisted tool-role text. Keep sensor bytes and private annotations in
    /// `observation`, not here.
    pub content: String,
    /// Safe event metadata. Never include raw observations or image bytes.
    pub metadata: Value,
    /// Ephemeral model input, separate from persisted text and metadata.
    pub observation: Option<ToolObservation>,
}

impl ToolOutput {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            metadata: Value::Null,
            observation: None,
        }
    }

    pub fn with_metadata(content: impl Into<String>, metadata: Value) -> Self {
        Self {
            content: content.into(),
            metadata,
            observation: None,
        }
    }

    pub fn with_observation(mut self, observation: ToolObservation) -> Self {
        self.observation = Some(observation);
        self
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    /// Opt into concurrent execution with other parallel-safe tools.
    /// Tools with ordering dependencies or shared side effects should leave
    /// this disabled. The loop also requires an explicit concurrency limit.
    fn parallel_safe(&self) -> bool {
        false
    }
    async fn execute(&self, arguments: Value) -> Result<ToolOutput>;
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T: Tool + 'static>(&mut self, tool: T) -> Result<()> {
        self.register_arc(Arc::new(tool))
    }

    pub fn register_arc(&mut self, tool: Arc<dyn Tool>) -> Result<()> {
        let name = tool.definition().name;
        if self.tools.contains_key(&name) {
            return Err(Error::InvalidInput(format!(
                "tool `{name}` is already registered"
            )));
        }
        self.tools.insert(name, tool);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|tool| tool.definition()).collect()
    }

    /// Return the set of registered tool names for batch validation.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(|s| s.as_str())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    pub(crate) fn parallel_safe(&self, name: &str) -> bool {
        self.tools
            .get(name)
            .is_some_and(|tool| tool.parallel_safe())
    }
}

/// Validate a batch of tool calls before any side effects.
///
/// Checks:
/// - Every call ID is non-empty and unique within the batch.
/// - Every call ID is unique across the entire session history (no reusing
///   IDs from prior turns).
/// - Every tool name is registered.
///
/// Returns the first validation error or `Ok(())`. Call this before
/// executing *any* tool so an invalid batch never partially executes.
pub fn validate_tool_calls(
    calls: &[ToolCall],
    registry: &ToolRegistry,
    existing_ids: &std::collections::HashSet<String>,
) -> Result<()> {
    validate_tool_call_batch(calls, Some(registry), Some(existing_ids))
}

/// Feedback mode still requires unambiguous IDs within the current batch,
/// but may omit tool availability and historical-ID checks.
pub(crate) fn validate_tool_call_batch(
    calls: &[ToolCall],
    registry: Option<&ToolRegistry>,
    existing_ids: Option<&std::collections::HashSet<String>>,
) -> Result<()> {
    let mut seen_ids = std::collections::HashSet::new();
    for call in calls {
        if call.id.is_empty() {
            return Err(Error::InvalidInput("tool call has an empty id".into()));
        }
        if !seen_ids.insert(call.id.clone()) {
            return Err(Error::InvalidInput(format!(
                "duplicate tool call id `{}`",
                call.id
            )));
        }
        if existing_ids.is_some_and(|ids| ids.contains(&call.id)) {
            return Err(Error::InvalidInput(format!(
                "tool call id `{}` was already used in a prior turn",
                call.id
            )));
        }
        if registry.is_some_and(|registry| !registry.contains(&call.name)) {
            return Err(Error::InvalidInput(format!(
                "requested tool `{}` is not registered",
                call.name
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tool_tests {
    use super::*;
    use serde_json::json;

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "echo".into(),
                description: "echo".into(),
                parameters: json!({"type": "object"}),
            }
        }

        async fn execute(&self, arguments: Value) -> Result<ToolOutput> {
            Ok(ToolOutput::text(arguments.to_string()))
        }
    }

    #[test]
    fn validate_accepts_unique_registered_calls() {
        let mut reg = ToolRegistry::new();
        reg.register(EchoTool).unwrap();
        let calls = vec![
            ToolCall {
                id: "a".into(),
                name: "echo".into(),
                arguments: json!({}),
            },
            ToolCall {
                id: "b".into(),
                name: "echo".into(),
                arguments: json!({}),
            },
        ];
        let existing = std::collections::HashSet::new();
        assert!(validate_tool_calls(&calls, &reg, &existing).is_ok());
    }

    #[test]
    fn validate_rejects_empty_id() {
        let mut reg = ToolRegistry::new();
        reg.register(EchoTool).unwrap();
        let calls = vec![ToolCall {
            id: String::new(),
            name: "echo".into(),
            arguments: json!({}),
        }];
        let existing = std::collections::HashSet::new();
        assert!(validate_tool_calls(&calls, &reg, &existing).is_err());
    }

    #[test]
    fn validate_rejects_duplicate_id() {
        let mut reg = ToolRegistry::new();
        reg.register(EchoTool).unwrap();
        let calls = vec![
            ToolCall {
                id: "dup".into(),
                name: "echo".into(),
                arguments: json!({}),
            },
            ToolCall {
                id: "dup".into(),
                name: "echo".into(),
                arguments: json!({}),
            },
        ];
        let existing = std::collections::HashSet::new();
        assert!(validate_tool_calls(&calls, &reg, &existing).is_err());
    }

    #[test]
    fn validate_rejects_unregistered_name() {
        let reg = ToolRegistry::new();
        let calls = vec![ToolCall {
            id: "a".into(),
            name: "nope".into(),
            arguments: json!({}),
        }];
        let existing = std::collections::HashSet::new();
        assert!(validate_tool_calls(&calls, &reg, &existing).is_err());
    }

    #[test]
    fn validate_rejects_reused_prior_turn_id() {
        let mut reg = ToolRegistry::new();
        reg.register(EchoTool).unwrap();
        let calls = vec![ToolCall {
            id: "call-1".into(),
            name: "echo".into(),
            arguments: json!({}),
        }];
        let mut existing = std::collections::HashSet::new();
        existing.insert("call-1".into());
        let err = validate_tool_calls(&calls, &reg, &existing).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }
}
