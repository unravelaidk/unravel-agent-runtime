//! Provider specification: typed protocol selection, default endpoints,
//! environment variable names, and keyless policy.

use crate::secret::Secret;
use std::borrow::Cow;
use std::fmt;

/// The wire protocol used to talk to a provider.
///
/// Only [`Protocol::OpenAiChatCompletions`] is implemented in this slice.
/// Other protocols are recognized so callers get a typed rejection instead
/// of a silent fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Protocol {
    /// OpenAI-compatible `/v1/chat/completions` with tool calls and SSE
    /// streaming.
    OpenAiChatCompletions,
    /// OpenAI Responses API (`/v1/responses`). Follow-up work, not this slice.
    OpenAiResponses,
    /// Anthropic Messages API (`/v1/messages`). Follow-up work, not this slice.
    AnthropicMessages,
}

impl Protocol {
    /// `true` when this protocol is implemented by the adapter in this
    /// crate.
    pub fn is_supported(&self) -> bool {
        matches!(self, Protocol::OpenAiChatCompletions)
    }
}

impl fmt::Display for Protocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Protocol::OpenAiChatCompletions => f.write_str("openai-chat-completions"),
            Protocol::OpenAiResponses => f.write_str("openai-responses"),
            Protocol::AnthropicMessages => f.write_str("anthropic-messages"),
        }
    }
}

/// Whether a provider requires an API key for discovery and invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyPolicy {
    /// The provider requires an API key for both discovery and invocation.
    Required,
    /// The provider is keyless — typically a local server (Ollama).
    /// A key may still be supplied for cloud-hosted variants but is
    /// never required.
    Keyless,
}

/// A typed provider specification.
///
/// Carries the protocol, endpoint, environment variable names, and key
/// policy. No static model IDs — discovery is remote. Built-in specs
/// ([`BUILTIN_OPENAI`], [`BUILTIN_NVIDIA`], [`BUILTIN_OPENROUTER`],
/// [`BUILTIN_OLLAMA`]) provide defaults; callers can override the
/// endpoint via `with_base_url`.
#[derive(Debug, Clone)]
pub struct ProviderSpec {
    /// Short provider identifier (e.g. `"openai"`, `"nvidia"`).
    pub id: Cow<'static, str>,
    /// Human-readable name (e.g. `"OpenAI"`).
    pub display_name: Cow<'static, str>,
    /// Wire protocol.
    pub protocol: Protocol,
    /// Default base URL including any `/v1` path prefix.
    pub default_endpoint: &'static str,
    /// Environment variable names for the API key, checked in order.
    pub env_key_names: &'static [&'static str],
    /// Environment variable name for base URL override.
    pub env_base_url_name: Option<&'static str>,
    /// Whether a key is required.
    pub key_policy: KeyPolicy,
    /// Explicit base URL override (takes precedence over env and
    /// default).
    endpoint_override: Option<String>,
    /// Explicit API key (takes precedence over env).
    key: Option<Secret>,
}

impl ProviderSpec {
    /// Create a spec with an explicit API key.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: &'static str,
        display_name: &'static str,
        protocol: Protocol,
        default_endpoint: &'static str,
        env_key_names: &'static [&'static str],
        env_base_url_name: Option<&'static str>,
        key_policy: KeyPolicy,
        key: Secret,
    ) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            protocol,
            default_endpoint,
            env_key_names,
            env_base_url_name,
            key_policy,
            endpoint_override: None,
            key: Some(key),
        }
    }

    /// OpenAI spec with an explicit API key.
    pub fn openai(key: impl Into<String>) -> Self {
        Self::new(
            BUILTIN_OPENAI.id,
            BUILTIN_OPENAI.display_name,
            BUILTIN_OPENAI.protocol,
            BUILTIN_OPENAI.default_endpoint,
            BUILTIN_OPENAI.env_key_names,
            BUILTIN_OPENAI.env_base_url_name,
            BUILTIN_OPENAI.key_policy,
            Secret::new(key),
        )
    }

    /// NVIDIA spec with an explicit API key.
    pub fn nvidia(key: impl Into<String>) -> Self {
        Self::new(
            BUILTIN_NVIDIA.id,
            BUILTIN_NVIDIA.display_name,
            BUILTIN_NVIDIA.protocol,
            BUILTIN_NVIDIA.default_endpoint,
            BUILTIN_NVIDIA.env_key_names,
            BUILTIN_NVIDIA.env_base_url_name,
            BUILTIN_NVIDIA.key_policy,
            Secret::new(key),
        )
    }

    /// OpenRouter spec with an explicit API key.
    pub fn openrouter(key: impl Into<String>) -> Self {
        Self::new(
            BUILTIN_OPENROUTER.id,
            BUILTIN_OPENROUTER.display_name,
            BUILTIN_OPENROUTER.protocol,
            BUILTIN_OPENROUTER.default_endpoint,
            BUILTIN_OPENROUTER.env_key_names,
            BUILTIN_OPENROUTER.env_base_url_name,
            BUILTIN_OPENROUTER.key_policy,
            Secret::new(key),
        )
    }

    /// Ollama (local, keyless) spec.
    pub fn ollama() -> Self {
        Self {
            id: BUILTIN_OLLAMA.id.into(),
            display_name: BUILTIN_OLLAMA.display_name.into(),
            protocol: BUILTIN_OLLAMA.protocol,
            default_endpoint: BUILTIN_OLLAMA.default_endpoint,
            env_key_names: BUILTIN_OLLAMA.env_key_names,
            env_base_url_name: BUILTIN_OLLAMA.env_base_url_name,
            key_policy: BUILTIN_OLLAMA.key_policy,
            endpoint_override: None,
            key: None,
        }
    }

    /// A custom OpenAI-compatible spec for a self-hosted or unlisted
    /// provider. The endpoint must include the `/v1` prefix if the
    /// server expects it.
    pub fn custom(
        id: impl Into<Cow<'static, str>>,
        display_name: impl Into<Cow<'static, str>>,
        endpoint: impl Into<String>,
        key: Option<Secret>,
    ) -> Self {
        let key_policy = if key.is_some() {
            KeyPolicy::Required
        } else {
            KeyPolicy::Keyless
        };
        Self {
            id: id.into(),
            display_name: display_name.into(),
            protocol: Protocol::OpenAiChatCompletions,
            default_endpoint: "",
            env_key_names: &[],
            env_base_url_name: None,
            key_policy,
            endpoint_override: Some(endpoint.into()),
            key,
        }
    }

    /// Override the base URL (takes precedence over env and default).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.endpoint_override = Some(url.into());
        self
    }

    /// Override the API key (takes precedence over env).
    pub fn with_key(mut self, key: Secret) -> Self {
        self.key = Some(key);
        self
    }

    /// Resolve the effective base URL.
    ///
    /// Precedence: explicit override > env variable > default.
    pub fn resolve_endpoint(&self) -> String {
        if let Some(ref url) = self.endpoint_override {
            return url.clone();
        }
        if let Some(env_name) = self.env_base_url_name {
            if let Ok(url) = std::env::var(env_name) {
                if !url.is_empty() {
                    return url;
                }
            }
        }
        self.default_endpoint.to_string()
    }

    /// Resolve the API key.
    ///
    /// Precedence: explicit key > env variables (checked in order).
    /// Returns `None` when no key is available, which is valid for
    /// [`KeyPolicy::Keyless`] providers.
    pub fn resolve_key(&self) -> Option<Secret> {
        if let Some(ref key) = self.key {
            return Some(key.clone());
        }
        for name in self.env_key_names {
            if let Ok(value) = std::env::var(name) {
                if !value.is_empty() {
                    return Some(Secret::new(value));
                }
            }
        }
        None
    }

    /// `true` when the provider has a usable key (explicit or env).
    pub fn has_key(&self) -> bool {
        self.resolve_key().is_some()
    }

    /// Validate that the spec is usable: supported protocol, and a key
    /// when one is required.
    pub fn validate(&self) -> crate::ProviderResult<()> {
        if !self.protocol.is_supported() {
            return Err(crate::ProviderError::unsupported_protocol(self.protocol));
        }
        if self.key_policy == KeyPolicy::Required && !self.has_key() {
            return Err(crate::ProviderError::missing_key(self.id.to_string()));
        }
        Ok(())
    }

    /// A stable, secret-free identity for cache partitioning.
    ///
    /// Uses the provider ID and a hash of the API key (so different
    /// accounts get different cache partitions) without exposing the
    /// key itself.
    pub(crate) fn cache_identity(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        if let Some(key) = self.resolve_key() {
            key.as_str().hash(&mut hasher);
        }
        let key_hash = hasher.finish();
        format!("{}:{:016x}", self.id, key_hash)
    }
}

/// Built-in OpenAI spec defaults.
pub const BUILTIN_OPENAI: ProviderSpecBuiltin = ProviderSpecBuiltin {
    id: "openai",
    display_name: "OpenAI",
    protocol: Protocol::OpenAiChatCompletions,
    default_endpoint: "https://api.openai.com/v1",
    env_key_names: &["OPENAI_API_KEY"],
    env_base_url_name: Some("OPENAI_BASE_URL"),
    key_policy: KeyPolicy::Required,
};

/// Built-in NVIDIA spec defaults.
pub const BUILTIN_NVIDIA: ProviderSpecBuiltin = ProviderSpecBuiltin {
    id: "nvidia",
    display_name: "NVIDIA",
    protocol: Protocol::OpenAiChatCompletions,
    default_endpoint: "https://integrate.api.nvidia.com/v1",
    env_key_names: &["NVIDIA_API_KEY"],
    env_base_url_name: Some("NVIDIA_BASE_URL"),
    key_policy: KeyPolicy::Required,
};

/// Built-in OpenRouter spec defaults.
pub const BUILTIN_OPENROUTER: ProviderSpecBuiltin = ProviderSpecBuiltin {
    id: "openrouter",
    display_name: "OpenRouter",
    protocol: Protocol::OpenAiChatCompletions,
    default_endpoint: "https://openrouter.ai/api/v1",
    env_key_names: &["OPENROUTER_API_KEY"],
    env_base_url_name: Some("OPENROUTER_BASE_URL"),
    key_policy: KeyPolicy::Required,
};

/// Built-in Ollama spec defaults (local, keyless).
pub const BUILTIN_OLLAMA: ProviderSpecBuiltin = ProviderSpecBuiltin {
    id: "ollama",
    display_name: "Ollama",
    protocol: Protocol::OpenAiChatCompletions,
    default_endpoint: "http://localhost:11434/v1",
    env_key_names: &["OLLAMA_API_KEY"],
    env_base_url_name: Some("OLLAMA_BASE_URL"),
    key_policy: KeyPolicy::Keyless,
};

/// A const-friendly subset of [`ProviderSpec`] used for built-in
/// defaults. Convert to a full `ProviderSpec` via the `new` constructor.
#[derive(Debug, Clone, Copy)]
pub struct ProviderSpecBuiltin {
    pub id: &'static str,
    pub display_name: &'static str,
    pub protocol: Protocol,
    pub default_endpoint: &'static str,
    pub env_key_names: &'static [&'static str],
    pub env_base_url_name: Option<&'static str>,
    pub key_policy: KeyPolicy,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Endpoint resolution (no env mutation) ─────────────────────

    #[test]
    fn resolve_endpoint_explicit_override() {
        // Explicit override always wins — no env variable needed.
        let spec = ProviderSpec::custom(
            "test",
            "Test",
            "https://default.example/v1",
            Some(Secret::new("key")),
        )
        .with_base_url("https://override.example/v1");
        assert_eq!(spec.resolve_endpoint(), "https://override.example/v1");
    }

    #[test]
    fn resolve_endpoint_default_for_custom_spec() {
        let spec = ProviderSpec::custom(
            "test",
            "Test",
            "https://default.example/v1",
            Some(Secret::new("key")),
        );
        assert_eq!(spec.resolve_endpoint(), "https://default.example/v1");
    }

    #[test]
    fn resolve_endpoint_default_for_builtin() {
        // Built-in specs have a hardcoded default endpoint.
        let spec = ProviderSpec::openai("key");
        // If env is unset, we get the default. The important assertion
        // is that the endpoint is a valid URL.
        let endpoint = spec.resolve_endpoint();
        assert!(
            endpoint.starts_with("https://"),
            "expected HTTPS endpoint, got {endpoint}"
        );
    }

    // ── Key resolution (no env mutation) ──────────────────────────

    #[test]
    fn resolve_key_explicit_always_wins() {
        // Explicit key always wins — no env variable needed.
        let spec = ProviderSpec::custom(
            "test",
            "Test",
            "https://example.com/v1",
            Some(Secret::new("explicit-key")),
        );
        assert_eq!(spec.resolve_key().unwrap().as_str(), "explicit-key");
    }

    #[test]
    fn resolve_key_none_for_keyless_custom() {
        let spec = ProviderSpec::custom("test", "Test", "https://example.com/v1", None::<Secret>);
        assert!(spec.resolve_key().is_none());
    }

    // ── Validation ────────────────────────────────────────────────

    #[test]
    fn validate_rejects_unsupported_protocol() {
        let spec = ProviderSpec {
            id: "test".into(),
            display_name: "Test".into(),
            protocol: Protocol::AnthropicMessages,
            default_endpoint: "https://example.com",
            env_key_names: &[],
            env_base_url_name: None,
            key_policy: KeyPolicy::Required,
            endpoint_override: None,
            key: Some(Secret::new("key")),
        };
        let err = spec.validate().unwrap_err();
        assert!(matches!(
            err,
            crate::ProviderError::UnsupportedProtocol { .. }
        ));
    }

    #[test]
    fn validate_rejects_missing_key_when_required() {
        // Use a nonexistent env key name so no env variable can satisfy it.
        let spec = ProviderSpec {
            id: "test-required".into(),
            display_name: "Test Required".into(),
            protocol: Protocol::OpenAiChatCompletions,
            default_endpoint: "https://example.com/v1",
            env_key_names: &["NONEXISTENT_TEST_KEY_12345"],
            env_base_url_name: None,
            key_policy: KeyPolicy::Required,
            endpoint_override: None,
            key: None,
        };
        let err = spec.validate().unwrap_err();
        assert!(matches!(err, crate::ProviderError::MissingKey { .. }));
    }

    #[test]
    fn validate_allows_keyless_without_key() {
        let spec = ProviderSpec::ollama();
        spec.validate().unwrap();
    }

    #[test]
    fn validate_allows_custom_with_key() {
        let spec = ProviderSpec::custom(
            "test",
            "Test",
            "https://example.com/v1",
            Some(Secret::new("key")),
        );
        spec.validate().unwrap();
    }

    // ── Cache identity ────────────────────────────────────────────

    #[test]
    fn cache_identity_is_stable_and_secret_free() {
        let spec = ProviderSpec::openai("sk-test-key");
        let identity = spec.cache_identity();
        assert!(!identity.contains("sk-test-key"));
        assert!(identity.starts_with("openai:"));
        // Same key -> same identity.
        let spec2 = ProviderSpec::openai("sk-test-key");
        assert_eq!(identity, spec2.cache_identity());
        // Different key -> different identity.
        let spec3 = ProviderSpec::openai("sk-different-key");
        assert_ne!(identity, spec3.cache_identity());
    }

    #[test]
    fn cache_identity_keyless_is_just_provider_id() {
        let spec =
            ProviderSpec::custom("local", "Local", "http://localhost:8080/v1", None::<Secret>);
        let identity = spec.cache_identity();
        assert!(identity.starts_with("local:"));
        // Keyless identity is stable.
        let spec2 =
            ProviderSpec::custom("local", "Local", "http://localhost:8080/v1", None::<Secret>);
        assert_eq!(identity, spec2.cache_identity());
    }

    // ── Keyless / custom ──────────────────────────────────────────

    #[test]
    fn ollama_keyless_policy() {
        let spec = ProviderSpec::ollama();
        assert_eq!(spec.key_policy, KeyPolicy::Keyless);
    }

    #[test]
    fn custom_spec_keyless_when_no_key() {
        let spec =
            ProviderSpec::custom("local", "Local", "http://localhost:8080/v1", None::<Secret>);
        assert_eq!(spec.key_policy, KeyPolicy::Keyless);
        assert!(!spec.has_key());
    }

    #[test]
    fn custom_spec_required_when_key_present() {
        let spec = ProviderSpec::custom(
            "local",
            "Local",
            "http://localhost:8080/v1",
            Some(Secret::new("key")),
        );
        assert_eq!(spec.key_policy, KeyPolicy::Required);
        assert!(spec.has_key());
    }
}
