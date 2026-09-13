//! Model discovery: generic `/models`, Ollama native `/api/tags`, and
//! Models.dev metadata merging.
//!
//! Discovery **describes** capabilities; it does not guarantee invocation
//! success. A model that reports `tool_call: true` may still fail at
//! runtime. Tests and docs make this explicit.

use crate::cache::{DiscoveryCacheKey, TtlCache};
use crate::catalog;
use crate::error::{ProviderError, ProviderResult};
use crate::spec::ProviderSpec;
use crate::transport::{
    build_client, join_url, map_http_error, read_bounded_json, read_bounded_text,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

/// Default cache TTL for discovery results (5 minutes).
const DEFAULT_CACHE_TTL: Duration = Duration::from_secs(300);

/// Whether a model supports tool calls.
///
/// Discovery preserves uncertainty: [`Unknown`](ToolSupport::Unknown) is
/// never silently upgraded to [`Yes`](ToolSupport::Yes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolSupport {
    /// The catalog or endpoint explicitly reports tool support.
    Yes,
    /// The catalog or endpoint explicitly reports no tool support.
    No,
    /// Tool support is unknown (field absent or not reported).
    Unknown,
}

/// Input/output modality support.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModalitySupport {
    pub text_input: bool,
    pub image_input: bool,
    pub text_output: bool,
}

/// A model entry from the Models.dev catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogModel {
    pub provider_id: String,
    pub model_id: String,
    pub name: String,
    pub tool_support: ToolSupport,
    pub deprecated: bool,
    pub reasoning: bool,
    pub attachment: bool,
    pub modalities: ModalitySupport,
    pub context_window: u64,
    pub max_output: u64,
    pub cost_input: f64,
    pub cost_output: f64,
    pub cost_cache_read: f64,
}

/// A provider entry from the Models.dev catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogProvider {
    pub id: String,
    pub name: String,
    pub base_url: Option<String>,
    pub api_key_required: bool,
    /// Catalog adapter package. An unknown package is not proof of protocol compatibility.
    #[serde(default)]
    pub npm: String,
    /// Credential environment variable references, never credential values.
    #[serde(default)]
    pub env: Vec<String>,
    pub models: Vec<CatalogModel>,
}

/// A discovered model ready for use.
///
/// Merges catalog metadata with live `/models` discovery. The
/// `from_endpoint` flag indicates whether the model was seen on the
/// provider's live endpoint (provenance).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredModel {
    pub provider_id: String,
    pub model_id: String,
    pub name: String,
    pub tool_support: ToolSupport,
    pub deprecated: bool,
    pub reasoning: bool,
    pub modalities: ModalitySupport,
    pub context_window: u64,
    pub max_output: u64,
    /// `true` if the model was seen on the provider's live `/models`
    /// endpoint. `false` if it only comes from the static catalog.
    pub from_endpoint: bool,
}

/// Options for discovery.
#[derive(Debug, Clone)]
pub struct DiscoveryOptions {
    /// Whether to fetch and merge Models.dev metadata.
    pub include_models_dev: bool,
    /// Whether to filter out deprecated and explicitly non-tool-capable
    /// models. When `true`, only models with `tool_support != No` and
    /// `deprecated == false` are returned.
    pub filter_for_tools: bool,
    /// Explicit model IDs to include even if not discovered. These are
    /// added with `ToolSupport::Unknown` and `from_endpoint: false`.
    pub explicit_model_ids: Vec<String>,
    /// Override the Models.dev catalog URL.
    pub models_dev_url_override: Option<String>,
    /// Maximum time to wait for each discovery request.
    pub timeout: Duration,
}

impl Default for DiscoveryOptions {
    fn default() -> Self {
        Self {
            include_models_dev: true,
            filter_for_tools: false,
            explicit_model_ids: Vec::new(),
            models_dev_url_override: None,
            timeout: Duration::from_secs(10),
        }
    }
}

impl DiscoveryOptions {
    /// Options for a tool workflow: filter deprecated and non-tool models.
    pub fn for_tools() -> Self {
        Self {
            include_models_dev: true,
            filter_for_tools: true,
            explicit_model_ids: Vec::new(),
            models_dev_url_override: None,
            timeout: Duration::from_secs(10),
        }
    }
}

/// The model discovery service.
///
/// Instance-owned (no process-global state). Uses a TTL cache keyed by
/// endpoint + account identity + protocol + discovery options. No
/// secrets in cache keys or Debug output. The clock is injectable for
/// deterministic cache expiry tests.
pub struct Discovery {
    cache: TtlCache<DiscoveryCacheKey, Vec<DiscoveredModel>>,
}

impl std::fmt::Debug for Discovery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Discovery")
            .field("cache", &self.cache)
            .finish()
    }
}

impl Discovery {
    /// Create a new discovery service with the default cache TTL.
    pub fn new() -> Self {
        Self {
            cache: TtlCache::new(DEFAULT_CACHE_TTL),
        }
    }

    /// Create a new discovery service with a custom cache TTL.
    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            cache: TtlCache::new(ttl),
        }
    }

    /// Create a new discovery service with a custom cache TTL and
    /// an injectable clock for deterministic expiry tests.
    pub fn with_clock(ttl: Duration, clock: Arc<dyn crate::cache::Clock>) -> Self {
        Self {
            cache: TtlCache::with_clock(ttl, clock),
        }
    }

    /// Discover models for a given provider spec.
    ///
    /// 1. Fetch the provider's `/models` endpoint (if accessible).
    /// 2. Optionally fetch and merge Models.dev metadata.
    /// 3. Filter deprecated and non-tool models if `filter_for_tools`.
    /// 4. Add explicit model IDs from options.
    /// 5. Cache the result keyed by endpoint + account + options.
    pub async fn discover(
        &self,
        spec: &ProviderSpec,
        options: &DiscoveryOptions,
    ) -> ProviderResult<Vec<DiscoveredModel>> {
        let endpoint = spec.resolve_endpoint();
        let explicit_model_ids = validate_explicit_model_ids(&options.explicit_model_ids)?;
        // Resolve this once: besides avoiding inconsistent fallback behavior,
        // this makes an env-provided catalog URL part of cache identity.
        let catalog_url = catalog::resolve_catalog_url(options.models_dev_url_override.as_deref());
        let key = DiscoveryCacheKey::new(
            &spec.id,
            &endpoint,
            &spec.cache_identity(),
            &spec.protocol.to_string(),
            options.include_models_dev,
            options.filter_for_tools,
            explicit_model_ids.clone(),
            &catalog_url,
            options.timeout,
        );

        if let Some(cached) = self.cache.get(&key) {
            return Ok(cached);
        }

        // 1. Fetch from the provider's endpoint. Ollama's native API is the
        // authoritative discovery API; its OpenAI-compatible /v1/models is
        // not consistently implemented by Ollama installations.
        let endpoint_result = if spec.id == "ollama" {
            fetch_ollama_tags_with_options(&endpoint, spec, options.timeout).await
        } else {
            fetch_endpoint_models(spec, options.timeout).await
        };
        let endpoint_models = endpoint_result.as_ref().ok().cloned().unwrap_or_default();

        // 2. Fetch Models.dev catalog and merge.
        let catalog_result = if options.include_models_dev {
            Some(fetch_catalog_at(&catalog_url).await)
        } else {
            None
        };
        let mut models = if let Some(Ok(catalog)) = catalog_result.as_ref() {
            merge_with_catalog(spec, &endpoint_models, catalog)
        } else {
            endpoint_models
                .into_iter()
                .map(|id| DiscoveredModel {
                    provider_id: spec.id.to_string(),
                    model_id: id.clone(),
                    name: id,
                    tool_support: ToolSupport::Unknown,
                    deprecated: false,
                    reasoning: false,
                    modalities: ModalitySupport::default(),
                    context_window: 0,
                    max_output: 0,
                    from_endpoint: true,
                })
                .collect()
        };

        // 3. Add explicit model IDs.
        for explicit_id in &explicit_model_ids {
            if !models.iter().any(|m| &m.model_id == explicit_id) {
                models.push(DiscoveredModel {
                    provider_id: spec.id.to_string(),
                    model_id: explicit_id.clone(),
                    name: explicit_id.clone(),
                    tool_support: ToolSupport::Unknown,
                    deprecated: false,
                    reasoning: false,
                    modalities: ModalitySupport::default(),
                    context_window: 0,
                    max_output: 0,
                    from_endpoint: false,
                });
            }
        }

        // 4. Filter for tool workflows.
        if options.filter_for_tools {
            models.retain(|m| !m.deprecated && m.tool_support != ToolSupport::No);
        }

        // An empty result is not a successful discovery when every source
        // failed. Do not cache that failure, so a later retry can recover.
        let endpoint_failed = endpoint_result.is_err();
        let catalog_failed =
            options.include_models_dev && catalog_result.as_ref().is_some_and(Result::is_err);
        let all_sources_failed = endpoint_failed && (!options.include_models_dev || catalog_failed);
        if models.is_empty() && all_sources_failed && explicit_model_ids.is_empty() {
            return Err(ProviderError::discovery_failed(
                "provider endpoint and catalog were unavailable",
            ));
        }

        // 5. Cache and return.
        self.cache.insert(key, models.clone());
        Ok(models)
    }

    /// Clear the cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

fn validate_explicit_model_ids(ids: &[String]) -> ProviderResult<Vec<String>> {
    let mut normalized = Vec::with_capacity(ids.len());
    for id in ids {
        let trimmed = id.trim();
        if trimmed.is_empty() {
            return Err(ProviderError::invalid(
                "explicit model IDs must not be empty or whitespace",
            ));
        }
        if !normalized.iter().any(|existing| existing == trimmed) {
            normalized.push(trimmed.to_string());
        }
    }
    Ok(normalized)
}

async fn fetch_catalog_at(url: &str) -> ProviderResult<Vec<CatalogProvider>> {
    catalog::fetch_catalog(Some(url)).await
}

impl Default for Discovery {
    fn default() -> Self {
        Self::new()
    }
}

/// Fetch model IDs from a provider's OpenAI-compatible `/models` endpoint.
async fn fetch_endpoint_models(
    spec: &ProviderSpec,
    timeout: Duration,
) -> ProviderResult<Vec<String>> {
    let endpoint = spec.resolve_endpoint();
    let url = join_url(&endpoint, "models");

    let client = build_client()?;
    let mut request = client.get(&url).timeout(timeout);

    let key = spec.resolve_key();
    if let Some(key) = &key {
        request = request.bearer_auth(key.as_str());
    }

    let response = request
        .send()
        .await
        .map_err(|e| ProviderError::transient(format!("models endpoint request failed: {e}")))?;

    let status = response.status();
    if !status.is_success() {
        let body = read_bounded_text(response).await;
        return Err(crate::transport::map_http_error_with_secret(
            status,
            &body,
            key.as_ref().map(|key| key.as_str()),
        ));
    }

    let body = read_bounded_json(response).await?;
    parse_models_list(&body)
}

/// Parse an OpenAI-compatible `/models` response (`{"data": [{"id": ...}]}`).
fn parse_models_list(body: &serde_json::Value) -> ProviderResult<Vec<String>> {
    let data = body
        .get("data")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ProviderError::malformed("models response missing 'data' array"))?;

    let models = data
        .iter()
        .filter_map(|item| {
            item.get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .collect();

    Ok(models)
}

/// Merge live endpoint model IDs with Models.dev catalog metadata.
///
/// - Models present on the endpoint get `from_endpoint: true` and
///   inherit catalog metadata if available.
/// - Models only in the catalog (not on the endpoint) get
///   `from_endpoint: false`.
/// - Unknown capability is preserved as [`ToolSupport::Unknown`].
pub fn merge_with_catalog(
    spec: &ProviderSpec,
    endpoint_models: &[String],
    catalog: &[CatalogProvider],
) -> Vec<DiscoveredModel> {
    // Find the catalog entry for this provider.
    let catalog_provider = catalog.iter().find(|p| p.id == spec.id);

    let mut result = Vec::new();

    // First, add models from the endpoint, enriched with catalog metadata.
    for model_id in endpoint_models {
        let catalog_model =
            catalog_provider.and_then(|p| p.models.iter().find(|m| m.model_id == *model_id));

        let discovered = if let Some(cm) = catalog_model {
            DiscoveredModel {
                provider_id: spec.id.to_string(),
                model_id: cm.model_id.clone(),
                name: cm.name.clone(),
                tool_support: cm.tool_support,
                deprecated: cm.deprecated,
                reasoning: cm.reasoning,
                modalities: cm.modalities.clone(),
                context_window: cm.context_window,
                max_output: cm.max_output,
                from_endpoint: true,
            }
        } else {
            // No catalog metadata — preserve as unknown.
            DiscoveredModel {
                provider_id: spec.id.to_string(),
                model_id: model_id.clone(),
                name: model_id.clone(),
                tool_support: ToolSupport::Unknown,
                deprecated: false,
                reasoning: false,
                modalities: ModalitySupport::default(),
                context_window: 0,
                max_output: 0,
                from_endpoint: true,
            }
        };
        result.push(discovered);
    }

    // Then, add catalog-only models (not on the endpoint).
    if let Some(cp) = catalog_provider {
        for cm in &cp.models {
            if !endpoint_models.iter().any(|id| id == &cm.model_id) {
                result.push(DiscoveredModel {
                    provider_id: spec.id.to_string(),
                    model_id: cm.model_id.clone(),
                    name: cm.name.clone(),
                    tool_support: cm.tool_support,
                    deprecated: cm.deprecated,
                    reasoning: cm.reasoning,
                    modalities: cm.modalities.clone(),
                    context_window: cm.context_window,
                    max_output: cm.max_output,
                    from_endpoint: false,
                });
            }
        }
    }

    result
}

/// Fetch models from Ollama's native `/api/tags` endpoint.
///
/// Ollama exposes a non-OpenAI tags endpoint at `/api/tags` (separate
/// from the OpenAI-compatible `/v1/models`). This function fetches and
/// normalizes the tag list, returning model IDs suitable for use with
/// the `/v1/chat/completions` runtime endpoint.
pub async fn fetch_ollama_tags(base_url: &str) -> ProviderResult<Vec<String>> {
    fetch_ollama_tags_request(base_url, None, Duration::from_secs(10)).await
}

async fn fetch_ollama_tags_with_options(
    base_url: &str,
    spec: &ProviderSpec,
    timeout: Duration,
) -> ProviderResult<Vec<String>> {
    fetch_ollama_tags_request(base_url, spec.resolve_key().as_ref(), timeout).await
}

async fn fetch_ollama_tags_request(
    base_url: &str,
    key: Option<&crate::secret::Secret>,
    timeout: Duration,
) -> ProviderResult<Vec<String>> {
    // The native tags endpoint is at /api/tags (no /v1 prefix).
    let tags_url = join_ollama_tags_url(base_url);

    let client = build_client()?;
    let mut request = client.get(&tags_url).timeout(timeout);
    if let Some(key) = key {
        request = request.bearer_auth(key.as_str());
    }
    let response = request
        .send()
        .await
        .map_err(|e| ProviderError::transient(format!("Ollama tags request failed: {e}")))?;

    let status = response.status();
    if !status.is_success() {
        let body = read_bounded_text(response).await;
        return Err(map_http_error(status, &body));
    }

    let body = read_bounded_json(response).await?;
    parse_ollama_tags(&body)
}

/// Parse an Ollama `/api/tags` response (`{"models": [{"name": ...}]}`).
fn parse_ollama_tags(body: &serde_json::Value) -> ProviderResult<Vec<String>> {
    let models = body
        .get("models")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ProviderError::malformed("Ollama tags response missing 'models' array"))?;

    let result = models
        .iter()
        .filter_map(|item| {
            item.get("name")
                .and_then(|v| v.as_str())
                .or_else(|| item.get("model").and_then(|v| v.as_str()))
                .map(|s| s.to_string())
        })
        .collect();

    Ok(result)
}

/// Join an Ollama base URL with the `/api/tags` path.
///
/// If the base URL ends with `/v1`, the tags endpoint is at the parent
/// path's `/api/tags` (Ollama exposes tags at the root, not under
/// `/v1`).
fn join_ollama_tags_url(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    // Strip /v1 suffix to get the root.
    let root = base.strip_suffix("/v1").unwrap_or(base);
    format!("{root}/api/tags")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_models_list_standard() {
        let body = serde_json::json!({
            "data": [
                {"id": "gpt-4o-mini", "object": "model"},
                {"id": "gpt-4o", "object": "model"},
            ]
        });
        let models = parse_models_list(&body).unwrap();
        assert_eq!(models, vec!["gpt-4o-mini", "gpt-4o"]);
    }

    #[test]
    fn parse_models_list_empty() {
        let body = serde_json::json!({"data": []});
        let models = parse_models_list(&body).unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn parse_models_list_missing_data_is_error() {
        let body = serde_json::json!({"error": "bad"});
        assert!(parse_models_list(&body).is_err());
    }

    #[test]
    fn parse_ollama_tags_standard() {
        let body = serde_json::json!({
            "models": [
                {"name": "llama3:8b"},
                {"name": "qwen2:7b"},
            ]
        });
        let models = parse_ollama_tags(&body).unwrap();
        assert_eq!(models, vec!["llama3:8b", "qwen2:7b"]);
    }

    #[test]
    fn parse_ollama_tags_uses_model_field_fallback() {
        let body = serde_json::json!({
            "models": [
                {"model": "mistral:7b"},
            ]
        });
        let models = parse_ollama_tags(&body).unwrap();
        assert_eq!(models, vec!["mistral:7b"]);
    }

    #[test]
    fn join_ollama_tags_url_strips_v1() {
        assert_eq!(
            join_ollama_tags_url("http://localhost:11434/v1"),
            "http://localhost:11434/api/tags"
        );
    }

    #[test]
    fn join_ollama_tags_url_no_v1_suffix() {
        assert_eq!(
            join_ollama_tags_url("http://localhost:11434"),
            "http://localhost:11434/api/tags"
        );
    }

    #[test]
    fn join_ollama_tags_url_trailing_slash() {
        assert_eq!(
            join_ollama_tags_url("http://localhost:11434/v1/"),
            "http://localhost:11434/api/tags"
        );
    }

    #[test]
    fn merge_with_catalog_enriches_endpoint_models() {
        let spec = ProviderSpec::openai("key");
        let endpoint_models = vec!["gpt-4o-mini".to_string()];
        let catalog = vec![CatalogProvider {
            npm: "@ai-sdk/openai".into(),
            env: vec![],
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: None,
            api_key_required: true,
            models: vec![CatalogModel {
                provider_id: "openai".to_string(),
                model_id: "gpt-4o-mini".to_string(),
                name: "GPT-4o mini".to_string(),
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
            }],
        }];
        let result = merge_with_catalog(&spec, &endpoint_models, &catalog);
        assert_eq!(result.len(), 1);
        assert!(result[0].from_endpoint);
        assert_eq!(result[0].tool_support, ToolSupport::Yes);
        assert!(result[0].modalities.image_input);
        assert_eq!(result[0].context_window, 128000);
    }

    #[test]
    fn merge_with_catalog_preserves_unknown_for_endpoint_only() {
        let spec = ProviderSpec::openai("key");
        let endpoint_models = vec!["custom-model".to_string()];
        let catalog: Vec<CatalogProvider> = vec![];
        let result = merge_with_catalog(&spec, &endpoint_models, &catalog);
        assert_eq!(result.len(), 1);
        assert!(result[0].from_endpoint);
        assert_eq!(result[0].tool_support, ToolSupport::Unknown);
    }

    #[test]
    fn merge_with_catalog_adds_catalog_only_models() {
        let spec = ProviderSpec::openai("key");
        let endpoint_models: Vec<String> = vec![];
        let catalog = vec![CatalogProvider {
            npm: "@ai-sdk/openai".into(),
            env: vec![],
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            base_url: None,
            api_key_required: true,
            models: vec![CatalogModel {
                provider_id: "openai".to_string(),
                model_id: "gpt-4o".to_string(),
                name: "GPT-4o".to_string(),
                tool_support: ToolSupport::Yes,
                deprecated: false,
                reasoning: false,
                attachment: true,
                modalities: ModalitySupport::default(),
                context_window: 128000,
                max_output: 16384,
                cost_input: 0.0,
                cost_output: 0.0,
                cost_cache_read: 0.0,
            }],
        }];
        let result = merge_with_catalog(&spec, &endpoint_models, &catalog);
        assert_eq!(result.len(), 1);
        assert!(!result[0].from_endpoint);
        assert_eq!(result[0].model_id, "gpt-4o");
    }

    #[test]
    fn filter_for_tools_removes_deprecated_and_no_tool() {
        let models = vec![
            DiscoveredModel {
                provider_id: "test".into(),
                model_id: "good".into(),
                name: "good".into(),
                tool_support: ToolSupport::Yes,
                deprecated: false,
                reasoning: false,
                modalities: ModalitySupport::default(),
                context_window: 0,
                max_output: 0,
                from_endpoint: true,
            },
            DiscoveredModel {
                provider_id: "test".into(),
                model_id: "deprecated".into(),
                name: "deprecated".into(),
                tool_support: ToolSupport::Yes,
                deprecated: true,
                reasoning: false,
                modalities: ModalitySupport::default(),
                context_window: 0,
                max_output: 0,
                from_endpoint: true,
            },
            DiscoveredModel {
                provider_id: "test".into(),
                model_id: "no-tools".into(),
                name: "no-tools".into(),
                tool_support: ToolSupport::No,
                deprecated: false,
                reasoning: false,
                modalities: ModalitySupport::default(),
                context_window: 0,
                max_output: 0,
                from_endpoint: true,
            },
            DiscoveredModel {
                provider_id: "test".into(),
                model_id: "unknown".into(),
                name: "unknown".into(),
                tool_support: ToolSupport::Unknown,
                deprecated: false,
                reasoning: false,
                modalities: ModalitySupport::default(),
                context_window: 0,
                max_output: 0,
                from_endpoint: true,
            },
        ];

        let mut filtered = models.clone();
        filtered.retain(|m| !m.deprecated && m.tool_support != ToolSupport::No);

        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().any(|m| m.model_id == "good"));
        assert!(filtered.iter().any(|m| m.model_id == "unknown"));
    }

    #[test]
    fn discovery_cache_key_isolates_by_endpoint() {
        let key1 = DiscoveryCacheKey::new(
            "openai",
            "https://api.openai.com/v1",
            "openai:abc",
            "openai-chat-completions",
            true,
            false,
            Vec::new(),
            "https://models.dev/api.json",
            Duration::from_secs(10),
        );
        let key2 = DiscoveryCacheKey::new(
            "openai",
            "https://api.openai.com/v1",
            "openai:xyz",
            "openai-chat-completions",
            true,
            false,
            Vec::new(),
            "https://models.dev/api.json",
            Duration::from_secs(10),
        );
        assert_ne!(key1, key2, "different account identities must not collide");
    }

    #[test]
    fn discovery_options_for_tools_filters() {
        let opts = DiscoveryOptions::for_tools();
        assert!(opts.filter_for_tools);
        assert!(opts.include_models_dev);
    }
}
