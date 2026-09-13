use crate::StreamDelta;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    RunStarted {
        session_id: String,
    },
    TurnStarted {
        turn: usize,
    },
    ModelAttempt {
        turn: usize,
        attempt: u32,
    },
    ModelRetry {
        turn: usize,
        attempt: u32,
        error: String,
        retryable: bool,
    },
    ModelStreamStarted {
        turn: usize,
    },
    ModelStreamDelta {
        turn: usize,
        kind: String,
        /// Absent only in persisted events written before delta payloads.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        delta: Option<StreamDelta>,
    },
    ToolStarted {
        turn: usize,
        call_id: String,
        tool: String,
    },
    ToolCompleted {
        turn: usize,
        call_id: String,
        tool: String,
        is_error: bool,
        /// Tool output, ordinary error feedback, or an interruption diagnostic.
        /// Absent only in persisted events written before output payloads.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        metadata: Value,
    },
    Reconciled {
        unknown: usize,
    },
    RunCompleted {
        session_id: String,
        output: String,
    },
    RunFailed {
        session_id: String,
        error: String,
    },
}

pub trait EventSink: Send + Sync {
    fn emit(&self, event: Event);
}

impl<F> EventSink for F
where
    F: Fn(Event) + Send + Sync,
{
    fn emit(&self, event: Event) {
        self(event);
    }
}

#[derive(Debug, Default)]
pub struct NoopEventSink;

impl EventSink for NoopEventSink {
    fn emit(&self, _event: Event) {}
}

#[cfg(test)]
mod tests {
    use super::Event;
    use serde_json::json;

    #[test]
    fn legacy_journal_events_round_trip_without_inventing_missing_payloads() {
        let delta = json!({
            "type": "model_stream_delta",
            "turn": 2,
            "kind": "tool_call",
        });
        let event: Event = serde_json::from_value(delta.clone()).unwrap();
        assert!(matches!(
            &event,
            Event::ModelStreamDelta { delta: None, .. }
        ));
        assert_eq!(serde_json::to_value(event).unwrap(), delta);

        let completed = json!({
            "type": "tool_completed",
            "turn": 2,
            "call_id": "prior-call",
            "tool": "lookup",
            "is_error": false,
            "metadata": null,
        });
        let event: Event = serde_json::from_value(completed.clone()).unwrap();
        assert!(matches!(&event, Event::ToolCompleted { content: None, .. }));
        assert_eq!(serde_json::to_value(event).unwrap(), completed);
    }
}
