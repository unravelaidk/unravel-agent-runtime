//! Provider-local error mapping with credential redaction.
//!
//! Errors from HTTP responses are classified as permanent or transient
//! and carry redacted detail — no API keys, bearer tokens, or
//! secret-bearing URLs leak into error messages.

use crate::spec::Protocol;
use std::time::Duration;
use unravel_agent_runtime::{Error, ModelError};

pub type ProviderResult<T> = std::result::Result<T, ProviderError>;

/// A typed provider error.
///
/// All variants carry redacted detail. Credentials, authorization
/// headers, and secret-bearing URLs are never included.
#[derive(Debug, Clone)]
pub enum ProviderError {
    /// The protocol is not implemented by this crate.
    UnsupportedProtocol { protocol: Protocol },
    /// The provider requires an API key but none was provided.
    MissingKey { provider_id: String },
    /// Authentication or authorization failure (permanent).
    Auth { redacted_detail: String },
    /// Invalid request or unsupported feature (permanent).
    Invalid { redacted_detail: String },
    /// Transient failure (overloaded, timeout, network). May succeed on
    /// retry.
    Transient {
        redacted_detail: String,
        retry_after: Option<Duration>,
    },
    /// The provider reported a model that is explicitly not tool-capable
    /// for a tool workflow.
    ModelNotToolCapable { model_id: String },
    /// The model is deprecated.
    ModelDeprecated { model_id: String },
    /// The provider returned an unparseable or malformed response.
    MalformedResponse { redacted_detail: String },
    /// Neither the provider endpoint nor the catalog could provide a result.
    DiscoveryFailed { redacted_detail: String },
    /// A tool result references an unresolved `ToolUnknown` call.
    UnresolvedToolUnknown { call_id: String },
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderError::UnsupportedProtocol { protocol } => {
                write!(f, "protocol `{protocol}` is not implemented in this crate")
            }
            ProviderError::MissingKey { provider_id } => {
                write!(
                    f,
                    "provider `{provider_id}` requires an API key but none was provided"
                )
            }
            ProviderError::Auth { redacted_detail } => {
                write!(f, "authentication failure: {redacted_detail}")
            }
            ProviderError::Invalid { redacted_detail } => {
                write!(f, "invalid request: {redacted_detail}")
            }
            ProviderError::Transient {
                redacted_detail,
                retry_after,
            } => {
                if let Some(d) = retry_after {
                    write!(
                        f,
                        "transient failure (retry after {d:?}): {redacted_detail}"
                    )
                } else {
                    write!(f, "transient failure: {redacted_detail}")
                }
            }
            ProviderError::ModelNotToolCapable { model_id } => {
                write!(f, "model `{model_id}` is explicitly not tool-capable")
            }
            ProviderError::ModelDeprecated { model_id } => {
                write!(f, "model `{model_id}` is deprecated")
            }
            ProviderError::MalformedResponse { redacted_detail } => {
                write!(f, "malformed provider response: {redacted_detail}")
            }
            ProviderError::DiscoveryFailed { redacted_detail } => {
                write!(f, "model discovery failed: {redacted_detail}")
            }
            ProviderError::UnresolvedToolUnknown { call_id } => {
                write!(
                    f,
                    "tool result references unresolved ToolUnknown call `{call_id}`"
                )
            }
        }
    }
}

impl std::error::Error for ProviderError {}

impl ProviderError {
    pub fn unsupported_protocol(protocol: Protocol) -> Self {
        ProviderError::UnsupportedProtocol { protocol }
    }

    pub fn missing_key(provider_id: impl Into<String>) -> Self {
        ProviderError::MissingKey {
            provider_id: provider_id.into(),
        }
    }

    pub fn auth(detail: impl Into<String>) -> Self {
        ProviderError::Auth {
            redacted_detail: redact(detail),
        }
    }

    pub fn invalid(detail: impl Into<String>) -> Self {
        ProviderError::Invalid {
            redacted_detail: redact(detail),
        }
    }

    pub fn transient(detail: impl Into<String>) -> Self {
        ProviderError::Transient {
            redacted_detail: redact(detail),
            retry_after: None,
        }
    }

    pub fn transient_with_retry(detail: impl Into<String>, retry_after: Duration) -> Self {
        ProviderError::Transient {
            redacted_detail: redact(detail),
            retry_after: Some(retry_after),
        }
    }

    pub fn malformed(detail: impl Into<String>) -> Self {
        ProviderError::MalformedResponse {
            redacted_detail: redact(detail),
        }
    }

    pub fn discovery_failed(detail: impl Into<String>) -> Self {
        ProviderError::DiscoveryFailed {
            redacted_detail: redact(detail),
        }
    }

    /// Convert this provider error into the core [`Error`] with proper
    /// retryability classification.
    pub fn into_core_error(self) -> Error {
        match self {
            ProviderError::UnsupportedProtocol { .. }
            | ProviderError::MissingKey { .. }
            | ProviderError::Auth { .. }
            | ProviderError::Invalid { .. }
            | ProviderError::ModelNotToolCapable { .. }
            | ProviderError::ModelDeprecated { .. }
            | ProviderError::UnresolvedToolUnknown { .. }
            | ProviderError::MalformedResponse { .. } => Error::ModelTyped(ModelError::new(
                self.to_string(),
                unravel_agent_runtime::Retryability::Permanent,
            )),
            ProviderError::DiscoveryFailed { .. } => {
                Error::ModelTyped(ModelError::transient(self.to_string()))
            }
            ProviderError::Transient {
                retry_after: Some(d),
                ..
            } => Error::ModelTyped(ModelError::rate_limit(self.to_string(), d)),
            ProviderError::Transient { .. } => {
                Error::ModelTyped(ModelError::transient(self.to_string()))
            }
        }
    }
}

impl From<ProviderError> for Error {
    fn from(e: ProviderError) -> Error {
        e.into_core_error()
    }
}

/// Redact potential secrets from a string.
///
/// Replaces bearer tokens, `Authorization` header values, and API keys
/// that commonly appear in error responses. This is defense-in-depth —
/// providers should not echo keys back, but we strip them if they do.
pub(crate) fn redact(input: impl Into<String>) -> String {
    let s = input.into();
    // Redact Bearer tokens.
    let s = redact_bearer(&s);
    // Redact "api_key": "..." and "apiKey": "..." patterns.
    let s = redact_json_key(&s, "\"api_key\":");
    let s = redact_json_key(&s, "\"apiKey\":");
    // Redact "authorization": "..." patterns.
    redact_json_key(&s, "\"authorization\":")
}

/// Redact a known secret value from a string.
///
/// This is the strongest redaction: it replaces the actual key value
/// wherever it appears in the body, regardless of context. Use this
/// when the provider knows its own key and the server might echo it
/// back in an unstructured error body.
pub(crate) fn redact_secret(input: &str, secret: &str) -> String {
    if secret.is_empty() || secret.len() < 4 {
        return input.to_string();
    }
    input.replace(secret, "…")
}

/// Redact a Bearer token: everything after "Bearer " up to the next
/// whitespace or end of string.
fn redact_bearer(s: &str) -> String {
    if let Some(idx) = s.find("Bearer ") {
        let before = &s[..idx + 7]; // "Bearer ".len() == 7
        let rest = &s[idx + 7..];
        let end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
        let token = &rest[..end];
        if token.len() > 3 {
            let after = &rest[end..];
            return format!("{before}…{after}");
        }
    }
    s.to_string()
}

/// Redact a JSON key-value pattern like `"api_key": "value"`.
/// Replaces the quoted value with `…`.
fn redact_json_key(s: &str, prefix: &str) -> String {
    if let Some(idx) = s.find(prefix) {
        let after_prefix = &s[idx + prefix.len()..];
        // Skip whitespace after the colon.
        let trimmed = after_prefix.trim_start();
        let ws_len = after_prefix.len() - trimmed.len();
        // Expect a quoted string value.
        if let Some(inner) = trimmed.strip_prefix('"') {
            // Find the closing quote.
            let end = inner.find('"').unwrap_or(inner.len());
            let value = &inner[..end];
            if value.len() > 3 {
                let before = &s[..idx + prefix.len()];
                let after_value = &inner[end + 1..]; // skip the closing quote
                let ws = &after_prefix[..ws_len];
                return format!("{before}{ws}\"…\"{after_value}");
            }
        }
    }
    s.to_string()
}

/// Redact a URL that might contain embedded credentials or query-param
/// API keys.
pub fn redact_url(url: &str) -> String {
    // Strip query parameters that might contain keys.
    if let Some(idx) = url.find('?') {
        format!("{}?…", &url[..idx])
    } else {
        url.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auth_error_is_permanent() {
        let err = ProviderError::auth("invalid api key").into_core_error();
        assert!(!err.is_retryable());
    }

    #[test]
    fn transient_error_is_retryable() {
        let err = ProviderError::transient("overloaded").into_core_error();
        assert!(err.is_retryable());
    }

    #[test]
    fn transient_with_retry_is_retryable() {
        let err = ProviderError::transient_with_retry("rate limited", Duration::from_secs(5))
            .into_core_error();
        assert!(err.is_retryable());
    }

    #[test]
    fn redact_strips_bearer_tokens() {
        let redacted = redact("HTTP 401: Bearer sk-abc123def456 invalid");
        assert!(!redacted.contains("sk-abc123def456"));
        assert!(redacted.contains("…"));
    }

    #[test]
    fn redact_strips_api_key_json() {
        let redacted = redact(r#"{"api_key": "sk-secret123", "error": "bad"}"#);
        assert!(!redacted.contains("sk-secret123"));
    }

    #[test]
    fn redact_url_strips_query_params() {
        let redacted = redact_url("https://api.example.com/v1/models?key=secret123");
        assert!(!redacted.contains("secret123"));
        assert!(redacted.contains("…"));
    }

    #[test]
    fn debug_does_not_leak_secrets() {
        let err = ProviderError::auth("Bearer sk-leaked-key-12345 invalid");
        let debug = format!("{:?}", err);
        assert!(!debug.contains("sk-leaked-key-12345"));
    }
}
