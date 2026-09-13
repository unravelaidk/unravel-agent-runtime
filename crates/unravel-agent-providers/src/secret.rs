//! Credential handling with redacted Debug output.
//!
//! Secrets are never persisted and never appear in `Debug` formatting,
//! log output, or error messages. A [`Secret`] wraps an API key and
//! only exposes it through explicit access.

use std::fmt;

/// An API key or bearer token that is redacted in all Debug output.
///
/// The inner value is only accessible via [`as_str`](Secret::as_str) or
/// [`into_string`](Secret::into_string), which require an explicit call.
/// The `Debug` impl prints `Secret("…")` regardless of the actual value.
#[derive(Clone)]
pub struct Secret(String);

impl Secret {
    /// Wrap an API key.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Wrap an API key only if the string is non-empty.
    pub fn from_optional(value: Option<String>) -> Option<Self> {
        value.filter(|s| !s.is_empty()).map(Self::new)
    }

    /// Access the raw key. Callers must ensure this is never logged.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume and return the raw key.
    pub fn into_string(self) -> String {
        self.0
    }

    /// `true` if the key is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            f.write_str("Secret(<empty>)")
        } else {
            f.write_str("Secret(…)")
        }
    }
}

impl fmt::Display for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_redacts_value() {
        let secret = Secret::new("sk-deadbeef1234567890");
        let debug = format!("{:?}", secret);
        assert!(!debug.contains("deadbeef"));
        assert!(debug.contains("Secret"));
    }

    #[test]
    fn display_is_redacted() {
        let secret = Secret::new("sk-secret-key");
        assert_eq!(format!("{}", secret), "…");
    }

    #[test]
    fn from_optional_skips_empty() {
        assert!(Secret::from_optional(None).is_none());
        assert!(Secret::from_optional(Some(String::new())).is_none());
        assert!(Secret::from_optional(Some("key".to_string())).is_some());
    }
}
