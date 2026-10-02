//! Conversation sessions with multimodal content and interrupted-history
//! reconciliation.

use crate::{Error, Result, ToolCall, ToolObservation};
use serde::{Deserialize, Serialize};

/// A typed content part for a multimodal message.
///
/// At least text and image source representations are supported. The loop
/// never fetches media or introduces provider-specific payloads; an
/// [`ImageSource`] carries only the data the caller supplied.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    Image {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
        source: ImageSource,
    },
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        ContentPart::Text { text: text.into() }
    }
}

/// An image source representation. No fetch is performed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageSource {
    /// Raw base64-encoded bytes with the media type in the enclosing part.
    Base64 { data: String },
    /// A URL the provider may resolve. The loop does not fetch it.
    Url { url: String },
}

/// Multimodal content for user and assistant messages.
///
/// Provides an ergonomic text-only path: a single text part can be built
/// with [`Content::text`] and read with [`Content::as_text`].
///
/// # Legacy compatibility
///
/// For backward compatibility with the pre-multimodal format, a bare JSON
/// string (e.g. `"hello"`) deserializes as a single text part. The new
/// structured form is `{"parts": [{"type": "text", "text": "hello"}]}`.
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct Content {
    pub parts: Vec<ContentPart>,
}

impl<'de> Deserialize<'de> for Content {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;

        // First try to deserialize as the structured form.
        #[derive(Deserialize)]
        struct Structured {
            parts: Vec<ContentPart>,
        }

        let value = serde_json::Value::deserialize(deserializer)?;
        // If it's a string, treat as legacy single-text-part content.
        if let Some(text) = value.as_str() {
            return Ok(Content::text(text));
        }
        // Otherwise parse as the structured form.
        let structured = serde_json::from_value::<Structured>(value)
            .map_err(|e| D::Error::custom(e.to_string()))?;
        Ok(Content {
            parts: structured.parts,
        })
    }
}

impl Content {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create text-only content.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            parts: vec![ContentPart::text(text)],
        }
    }

    /// Create content from a list of parts.
    pub fn from_parts(parts: Vec<ContentPart>) -> Self {
        Self { parts }
    }

    /// If the content is a single text part, return its text. Returns `None`
    /// for empty or multimodal content.
    pub fn as_text(&self) -> Option<&str> {
        match self.parts.as_slice() {
            [ContentPart::Text { text }] => Some(text),
            _ => None,
        }
    }

    /// Concatenate all text parts, ignoring non-text parts.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        for part in &self.parts {
            if let ContentPart::Text { text } = part {
                out.push_str(text);
            }
        }
        out
    }

    pub fn push_text(&mut self, text: impl Into<String>) {
        self.parts.push(ContentPart::text(text));
    }

    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }
}

impl From<String> for Content {
    fn from(text: String) -> Self {
        Content::text(text)
    }
}

impl From<&str> for Content {
    fn from(text: &str) -> Self {
        Content::text(text)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    System {
        content: String,
    },
    User {
        #[serde(default)]
        content: Content,
    },
    Assistant {
        #[serde(default)]
        content: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
        /// Reasoning trace from the model, preserved so providers that
        /// support chain-of-thought can replay it in subsequent turns.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning: Option<crate::Reasoning>,
    },
    Tool {
        call_id: String,
        name: String,
        content: String,
        is_error: bool,
    },
    /// A tool call whose outcome is unknown because the previous run was
    /// interrupted before the result was recorded. Reconciliation inserts
    /// this before the next model request so the model never sees an
    /// implied success. The caller must reconcile (investigate or halt)
    /// before resuming.
    ToolUnknown {
        call_id: String,
        name: String,
        reason: String,
    },
}

impl Message {
    /// Convenience: construct a user text message (single text part).
    pub fn user_text(text: impl Into<String>) -> Self {
        Message::User {
            content: Content::text(text),
        }
    }

    /// Convenience: construct an assistant text message with no tool calls.
    pub fn assistant_text(text: impl Into<String>) -> Self {
        Message::Assistant {
            content: text.into(),
            tool_calls: Vec::new(),
            reasoning: None,
        }
    }

    /// Construct an assistant message from a model response, preserving
    /// reasoning so a provider can replay it in subsequent turns.
    pub fn assistant_from_response(response: &crate::ModelResponse) -> Self {
        Message::Assistant {
            content: response.content.clone(),
            tool_calls: response.tool_calls.clone(),
            reasoning: response.reasoning.clone(),
        }
    }
}

/// Persisted conversation plus one-use, in-memory observation scratch.
///
/// Adjacent [`crate::AgentLoop::turn`] calls consume pending observations once.
/// Serialization, cloning, reconciliation and fresh runs never carry them over.
#[derive(Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(skip)]
    pub(crate) pending_observations: Vec<(usize, ToolObservation)>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("id", &self.id)
            .field("messages", &self.messages)
            .finish_non_exhaustive()
    }
}

impl Clone for Session {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            messages: self.messages.clone(),
            pending_observations: Vec::new(),
        }
    }
}

impl Session {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            messages: Vec::new(),
            pending_observations: Vec::new(),
        }
    }

    /// Reconcile interrupted tool histories before the next model request.
    ///
    /// Every assistant tool call without a matching `Tool` result becomes an
    /// explicit `ToolUnknown` result. This is idempotent: a call already
    /// marked unknown is not duplicated, and calls with a `Tool` result are
    /// left untouched. Orphan `Tool` results (no matching call) are removed
    /// with diagnostic evidence preserved as a system note.
    ///
    /// **This never invokes tools and never implies a success.** After
    /// reconciliation the caller must resolve unknown outcomes (investigate
    /// and replace the `ToolUnknown` with a real `Tool` result) before
    /// resuming; resuming blindly would risk replaying an operation whose
    /// effect is unknown.
    ///
    /// Calling this establishes a resume boundary and discards observations.
    ///
    /// Returns the number of *newly* marked unknowns (not pre-existing ones).
    pub fn reconcile_tool_history(&mut self) -> Result<usize> {
        self.pending_observations.clear();
        // Collect (assistant-message-index, call_id, name) for every tool
        // call that does not yet have a Tool result or ToolUnknown marker.
        let mut unknown_count = 0usize;
        let mut result_call_ids: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        // First pass: index existing results and detect orphan results.
        let mut orphan_results: Vec<usize> = Vec::new();
        for (idx, msg) in self.messages.iter().enumerate() {
            if let Message::Tool { call_id, .. } | Message::ToolUnknown { call_id, .. } = msg {
                result_call_ids.insert(call_id.clone());
            }
            if let Message::Tool { call_id, .. } = msg {
                // Check if there is a preceding assistant message with this
                // call_id.
                let has_call = self.messages[..idx].iter().any(|m| {
                    matches!(m, Message::Assistant { tool_calls, .. }
                        if tool_calls.iter().any(|c| &c.id == call_id))
                });
                if !has_call {
                    orphan_results.push(idx);
                }
            }
        }

        // Remove orphan tool results, replacing the first with a diagnostic
        // system note to retain evidence.
        if !orphan_results.is_empty() {
            let first = orphan_results[0];
            if let Message::Tool {
                call_id,
                name,
                content,
                ..
            } = &self.messages[first]
            {
                self.messages[first] = Message::System {
                    content: format!(
                        "reconciliation: removed orphan tool result for `{name}` \
                         (call_id `{call_id}`) — no matching assistant call. \
                         content: {content}"
                    ),
                };
            }
            // Remove the rest in reverse order to keep indices stable.
            for &idx in orphan_results[1..].iter().rev() {
                self.messages.remove(idx);
            }
        }

        // Second pass: for each assistant tool call without a result, insert
        // a ToolUnknown marker immediately after the assistant message (or
        // after the last contiguous tool result block following it).
        let mut insert_at: Vec<(usize, ToolCall)> = Vec::new();
        for (idx, msg) in self.messages.iter().enumerate() {
            if let Message::Assistant { tool_calls, .. } = msg {
                for call in tool_calls {
                    if !result_call_ids.contains(&call.id) {
                        insert_at.push((idx, call.clone()));
                    }
                }
            }
        }

        // Insert in reverse index order so earlier insertions don't shift
        // later indices.
        insert_at.sort_by_key(|entry| std::cmp::Reverse(entry.0));
        for (idx, call) in insert_at {
            let unknown = Message::ToolUnknown {
                call_id: call.id.clone(),
                name: call.name.clone(),
                reason: "previous run interrupted before the result was recorded".into(),
            };
            self.messages.insert(idx + 1, unknown);
            unknown_count += 1;
            result_call_ids.insert(call.id);
        }

        // Detect duplicate tool results (same call_id appearing twice).
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut duplicates: Vec<usize> = Vec::new();
        for (idx, msg) in self.messages.iter().enumerate() {
            if let Some(call_id) = match msg {
                Message::Tool { call_id, .. } => Some(call_id.clone()),
                _ => None,
            } {
                if !seen.insert(call_id) {
                    duplicates.push(idx);
                }
            }
        }
        if !duplicates.is_empty() {
            return Err(Error::Reconciliation(format!(
                "duplicate tool result(s) at message index {:?}",
                duplicates
            )));
        }

        Ok(unknown_count)
    }

    /// Returns `true` if the history contains any `ToolUnknown` messages.
    /// The loop halts with [`Error::Reconciliation`] when this is `true`,
    /// requiring the caller to replace each unknown with a resolved result
    /// before resuming.
    pub fn has_unresolved_unknowns(&self) -> bool {
        self.messages
            .iter()
            .any(|m| matches!(m, Message::ToolUnknown { .. }))
    }

    /// Returns the set of all tool-call IDs that appear in assistant messages
    /// across the entire history (all prior turns). Used by the loop to
    /// validate that a new batch of tool calls does not reuse a call ID
    /// from a prior turn.
    pub fn existing_tool_call_ids(&self) -> std::collections::HashSet<String> {
        let mut ids = std::collections::HashSet::new();
        for msg in &self.messages {
            if let Message::Assistant { tool_calls, .. } = msg {
                for call in tool_calls {
                    ids.insert(call.id.clone());
                }
            }
        }
        ids
    }
}

#[cfg(test)]
mod session_tests {
    use super::*;
    use crate::Reasoning;
    use serde_json::json;

    #[test]
    fn content_text_round_trip() {
        let c = Content::text("hello");
        assert_eq!(c.as_text(), Some("hello"));
        let serialized = serde_json::to_string(&c).unwrap();
        let back: Content = serde_json::from_str(&serialized).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn content_multimodal_round_trip() {
        let c = Content::from_parts(vec![
            ContentPart::text("look at this"),
            ContentPart::Image {
                media_type: Some("image/png".into()),
                source: ImageSource::Base64 {
                    data: "iVBORw0KGgo=".into(),
                },
            },
        ]);
        assert_eq!(c.as_text(), None);
        assert_eq!(c.to_text(), "look at this");
        let serialized = serde_json::to_string(&c).unwrap();
        let back: Content = serde_json::from_str(&serialized).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn legacy_user_string_migrates_to_content() {
        // A legacy serialized User message with a string content field
        // deserializes into the new Content-typed field as a single text
        // part (backward-compatible migration path).
        let legacy = serde_json::json!({
            "role": "user",
            "content": "hello world"
        });
        let msg: Message = serde_json::from_value(legacy).unwrap();
        match msg {
            Message::User { content } => {
                assert_eq!(content.as_text(), Some("hello world"));
                assert_eq!(content.parts.len(), 1);
            }
            _ => panic!("expected User message"),
        }
    }

    #[test]
    fn content_structured_form_round_trips() {
        let c = Content::from_parts(vec![ContentPart::text("part1"), ContentPart::text("part2")]);
        let serialized = serde_json::to_string(&c).unwrap();
        let back: Content = serde_json::from_str(&serialized).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn reconcile_marks_missing_results_as_unknown() {
        let mut session = Session::new("s1");
        session.messages.push(Message::System {
            content: "sys".into(),
        });
        session.messages.push(Message::user_text("go"));
        session.messages.push(Message::Assistant {
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "call-1".into(),
                name: "echo".into(),
                arguments: json!({}),
            }],
            reasoning: None,
        });
        let count = session.reconcile_tool_history().unwrap();
        assert_eq!(count, 1);
        assert!(matches!(
            &session.messages[3],
            Message::ToolUnknown { call_id, .. } if call_id == "call-1"
        ));
    }

    #[test]
    fn reconcile_is_idempotent() {
        let mut session = Session::new("s1");
        session.messages.push(Message::Assistant {
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "call-1".into(),
                name: "echo".into(),
                arguments: json!({}),
            }],
            reasoning: None,
        });
        session.reconcile_tool_history().unwrap();
        let count = session.reconcile_tool_history().unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn reconcile_leaves_completed_results_alone() {
        let mut session = Session::new("s1");
        session.messages.push(Message::Assistant {
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "call-1".into(),
                name: "echo".into(),
                arguments: json!({}),
            }],
            reasoning: None,
        });
        session.messages.push(Message::Tool {
            call_id: "call-1".into(),
            name: "echo".into(),
            content: "42".into(),
            is_error: false,
        });
        let count = session.reconcile_tool_history().unwrap();
        assert_eq!(count, 0);
        assert!(matches!(
            &session.messages[1],
            Message::Tool { content, .. } if content == "42"
        ));
    }

    #[test]
    fn reconcile_removes_orphan_tool_result() {
        let mut session = Session::new("s1");
        session.messages.push(Message::Tool {
            call_id: "orphan".into(),
            name: "echo".into(),
            content: "x".into(),
            is_error: false,
        });
        session.reconcile_tool_history().unwrap();
        // The orphan should be replaced with a system diagnostic note.
        assert!(matches!(
            &session.messages[0],
            Message::System { content } if content.contains("reconciliation")
        ));
    }

    #[test]
    fn reconcile_detects_duplicate_tool_results() {
        let mut session = Session::new("s1");
        session.messages.push(Message::Assistant {
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "call-1".into(),
                name: "echo".into(),
                arguments: json!({}),
            }],
            reasoning: None,
        });
        session.messages.push(Message::Tool {
            call_id: "call-1".into(),
            name: "echo".into(),
            content: "first".into(),
            is_error: false,
        });
        session.messages.push(Message::Tool {
            call_id: "call-1".into(),
            name: "echo".into(),
            content: "second".into(),
            is_error: false,
        });
        let err = session.reconcile_tool_history().unwrap_err();
        assert!(matches!(err, Error::Reconciliation(_)));
    }

    #[test]
    fn reconcile_mismatched_result_no_replay() {
        let mut session = Session::new("s1");
        session.messages.push(Message::user_text("go"));
        session.messages.push(Message::Assistant {
            content: "I will call echo".into(),
            tool_calls: vec![ToolCall {
                id: "call-1".into(),
                name: "echo".into(),
                arguments: json!({}),
            }],
            reasoning: None,
        });
        session.messages.push(Message::Tool {
            call_id: "wrong-id".into(),
            name: "echo".into(),
            content: "wrong".into(),
            is_error: false,
        });
        session.reconcile_tool_history().unwrap();
        assert!(
            session
                .messages
                .iter()
                .any(|m| matches!(m, Message::ToolUnknown { call_id, .. } if call_id == "call-1")),
            "missing call-1 should be marked unknown"
        );
        assert!(
            session
                .messages
                .iter()
                .any(|m| matches!(m, Message::System { content } if content.contains("orphan"))),
            "orphan result should be replaced with a diagnostic note"
        );
    }

    #[test]
    fn reconcile_multiple_unknown_calls() {
        let mut session = Session::new("s1");
        session.messages.push(Message::Assistant {
            content: String::new(),
            tool_calls: vec![
                ToolCall {
                    id: "call-1".into(),
                    name: "echo".into(),
                    arguments: json!({}),
                },
                ToolCall {
                    id: "call-2".into(),
                    name: "echo".into(),
                    arguments: json!({}),
                },
            ],
            reasoning: None,
        });
        let count = session.reconcile_tool_history().unwrap();
        assert_eq!(count, 2);
        let unknown_ids: Vec<&str> = session
            .messages
            .iter()
            .filter_map(|m| match m {
                Message::ToolUnknown { call_id, .. } => Some(call_id.as_str()),
                _ => None,
            })
            .collect();
        assert!(unknown_ids.contains(&"call-1"));
        assert!(unknown_ids.contains(&"call-2"));
    }

    #[test]
    fn has_unresolved_unknowns_detects_tool_unknown() {
        let mut session = Session::new("s1");
        session.messages.push(Message::ToolUnknown {
            call_id: "x".into(),
            name: "echo".into(),
            reason: "test".into(),
        });
        assert!(session.has_unresolved_unknowns());
        let mut session2 = Session::new("s2");
        session2.messages.push(Message::Tool {
            call_id: "x".into(),
            name: "echo".into(),
            content: "ok".into(),
            is_error: false,
        });
        assert!(!session2.has_unresolved_unknowns());
    }

    #[test]
    fn existing_tool_call_ids_collects_from_all_turns() {
        let mut session = Session::new("s1");
        session.messages.push(Message::Assistant {
            content: String::new(),
            tool_calls: vec![
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
            ],
            reasoning: None,
        });
        session.messages.push(Message::Tool {
            call_id: "a".into(),
            name: "echo".into(),
            content: "1".into(),
            is_error: false,
        });
        session.messages.push(Message::Tool {
            call_id: "b".into(),
            name: "echo".into(),
            content: "2".into(),
            is_error: false,
        });
        session.messages.push(Message::Assistant {
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "c".into(),
                name: "echo".into(),
                arguments: json!({}),
            }],
            reasoning: None,
        });
        let ids = session.existing_tool_call_ids();
        assert!(ids.contains("a"));
        assert!(ids.contains("b"));
        assert!(ids.contains("c"));
    }

    #[test]
    fn reasoning_serde_round_trip() {
        let msg = Message::Assistant {
            content: "thinking".into(),
            tool_calls: vec![],
            reasoning: Some(Reasoning {
                text: Some("because I said so".into()),
            }),
        };
        let serialized = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&serialized).unwrap();
        assert_eq!(msg, back);
        // Verify the reasoning field is in the serialized JSON.
        assert!(serialized.contains("reasoning"));
        assert!(serialized.contains("because I said so"));
    }

    #[test]
    fn message_serde_round_trip() {
        let msgs = vec![
            Message::System {
                content: "sys".into(),
            },
            Message::user_text("hello"),
            Message::assistant_text("hi"),
            Message::Assistant {
                content: "thinking".into(),
                tool_calls: vec![ToolCall {
                    id: "call-0".into(),
                    name: "echo".into(),
                    arguments: json!({}),
                }],
                reasoning: Some(Reasoning {
                    text: Some("step by step".into()),
                }),
            },
            Message::Tool {
                call_id: "call-1".into(),
                name: "echo".into(),
                content: "42".into(),
                is_error: false,
            },
            Message::ToolUnknown {
                call_id: "call-2".into(),
                name: "echo".into(),
                reason: "interrupted".into(),
            },
        ];
        for msg in &msgs {
            let serialized = serde_json::to_string(msg).unwrap();
            let back: Message = serde_json::from_str(&serialized).unwrap();
            assert_eq!(msg, &back);
        }
    }
}
