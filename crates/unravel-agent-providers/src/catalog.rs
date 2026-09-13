//! Models.dev catalog fetch and parse.
//!
//! Fetches the default `https://models.dev/api.json` (or an explicit
//! override) and parses the actual shape including modality, context,
//! output, and cost metadata. Preserves provenance and capability
//! uncertainty — unknown capability is never silently upgraded to
//! supported.

use crate::discovery::{CatalogModel, CatalogProvider, ModalitySupport, ToolSupport};
use crate::error::{ProviderError, ProviderResult};
use crate::transport::{
    build_client, map_http_error, read_bounded_json_with_limit, read_bounded_text,
};
use serde::Deserialize;
use std::collections::BTreeMap;

/// Default Models.dev catalog URL.
pub const DEFAULT_MODELS_DEV_URL: &str = "https://models.dev/api.json";

/// Environment variable for overriding the catalog URL.
pub const MODELS_DEV_URL_ENV: &str = "SERVOLOOP_MODELS_DEV_URL";

/// Catalogs contain metadata for many providers, unlike one provider response.
/// Allow growth beyond the ordinary 4 MiB cap while keeping downloads bounded.
const MAX_CATALOG_BYTES: usize = 16 * 1024 * 1024;

/// Resolve the effective catalog URL once for a discovery operation.
pub fn resolve_catalog_url(url: Option<&str>) -> String {
    url.map(str::to_owned)
        .or_else(|| std::env::var(MODELS_DEV_URL_ENV).ok())
        .unwrap_or_else(|| DEFAULT_MODELS_DEV_URL.to_string())
}

/// Fetch the Models.dev catalog.
///
/// Uses the default URL or an explicit override via
/// [`MODELS_DEV_URL_ENV`](MODELS_DEV_URL_ENV) or the `url` parameter.
/// The `url` parameter takes precedence over the env variable.
pub async fn fetch_catalog(url: Option<&str>) -> ProviderResult<Vec<CatalogProvider>> {
    let endpoint = resolve_catalog_url(url);

    let client = build_client()?;
    let response = client
        .get(&endpoint)
        .send()
        .await
        .map_err(|e| ProviderError::transient(format!("catalog fetch failed: {e}")))?;

    let status = response.status();
    if !status.is_success() {
        let body = read_bounded_text(response).await;
        return Err(map_http_error(status, &body));
    }

    let body = read_bounded_json_with_limit(response, MAX_CATALOG_BYTES).await?;
    parse_catalog(&body)
}

/// Parse a Models.dev catalog JSON body into typed providers and models.
///
/// Preserves provenance (provider ID + model ID). Filters deprecated
/// models and models that explicitly report `tool_call: false` is
/// recorded as [`ToolSupport::No`]. Unknown capability (field absent or
/// null) is preserved as [`ToolSupport::Unknown`].
pub fn parse_catalog(body: &serde_json::Value) -> ProviderResult<Vec<CatalogProvider>> {
    let providers: BTreeMap<String, RawProvider> =
        serde_json::from_value(body.clone()).map_err(|e| {
            ProviderError::malformed(format!("failed to parse Models.dev catalog: {e}"))
        })?;

    let catalog = providers
        .into_iter()
        .filter_map(|(id, provider)| parse_provider(&id, provider))
        .collect();

    Ok(catalog)
}

/// Parse a single provider entry.
fn parse_provider(id: &str, provider: RawProvider) -> Option<CatalogProvider> {
    let models: Vec<CatalogModel> = provider
        .models
        .into_iter()
        .map(|(key, model)| parse_model(id, &key, &model))
        .collect();

    if models.is_empty() {
        return None;
    }

    Some(CatalogProvider {
        id: id.to_string(),
        name: provider.name,
        base_url: provider.api,
        api_key_required: !provider.env.is_empty(),
        npm: provider.npm,
        env: provider.env,
        models,
    })
}

/// Parse a single model entry from the Models.dev catalog.
fn parse_model(provider_id: &str, key: &str, model: &RawModel) -> CatalogModel {
    // Use the explicit `id` field if present, otherwise the map key.
    let model_id = if model.id.is_empty() {
        key.to_string()
    } else {
        model.id.clone()
    };

    // Determine tool support from the actual `tool_call` field.
    let tool_support = match &model.tool_call {
        Some(true) => ToolSupport::Yes,
        Some(false) => ToolSupport::No,
        None => ToolSupport::Unknown,
    };

    // Determine status (deprecated, beta, etc.).
    let deprecated = model
        .status
        .as_deref()
        .map(|s| s == "deprecated")
        .unwrap_or(false);

    // Parse modalities.
    let modalities = parse_modalities(model);

    // Parse limits.
    let context_window = model.limit.as_ref().and_then(|l| l.context).unwrap_or(0);
    let max_output = model.limit.as_ref().and_then(|l| l.output).unwrap_or(0);

    // Parse cost.
    let (cost_input, cost_output, cost_cache_read) = model
        .cost
        .as_ref()
        .map(|c| {
            (
                c.input.unwrap_or(0.0),
                c.output.unwrap_or(0.0),
                c.cache_read.unwrap_or(0.0),
            )
        })
        .unwrap_or((0.0, 0.0, 0.0));

    CatalogModel {
        provider_id: provider_id.to_string(),
        model_id,
        name: model.name.clone(),
        tool_support,
        deprecated,
        reasoning: model.reasoning.unwrap_or(false),
        attachment: model.attachment.unwrap_or(false),
        modalities,
        context_window,
        max_output,
        cost_input,
        cost_output,
        cost_cache_read,
    }
}

/// Parse input/output modalities from the raw model.
fn parse_modalities(model: &RawModel) -> ModalitySupport {
    let (input, output) = model
        .modalities
        .as_ref()
        .map(|m| {
            (
                m.input.clone().unwrap_or_default(),
                m.output.clone().unwrap_or_default(),
            )
        })
        .unwrap_or_default();

    let supports_text_input = input.iter().any(|m| m == "text");
    let supports_image_input = input.iter().any(|m| m == "image");
    let supports_text_output = output.iter().any(|m| m == "text");

    ModalitySupport {
        text_input: supports_text_input,
        image_input: supports_image_input,
        text_output: supports_text_output,
    }
}

// ── Raw deserialization types matching the Models.dev JSON shape ──

#[derive(Debug, Deserialize)]
struct RawProvider {
    name: String,
    #[serde(default)]
    #[allow(dead_code)]
    npm: String,
    api: Option<String>,
    #[serde(default)]
    env: Vec<String>,
    #[serde(default)]
    models: BTreeMap<String, RawModel>,
}

#[derive(Debug, Deserialize)]
struct RawModel {
    #[serde(default)]
    id: String,
    name: String,
    #[serde(default)]
    tool_call: Option<bool>,
    status: Option<String>,
    #[serde(default)]
    reasoning: Option<bool>,
    #[serde(default)]
    attachment: Option<bool>,
    modalities: Option<RawModalities>,
    limit: Option<RawLimit>,
    cost: Option<RawCost>,
}

#[derive(Debug, Deserialize)]
struct RawModalities {
    input: Option<Vec<String>>,
    output: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct RawLimit {
    context: Option<u64>,
    #[allow(dead_code)]
    input: Option<u64>,
    output: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct RawCost {
    input: Option<f64>,
    output: Option<f64>,
    cache_read: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_catalog() -> serde_json::Value {
        serde_json::json!({
            "openai": {
                "id": "openai",
                "name": "OpenAI",
                "npm": "@ai-sdk/openai",
                "env": ["OPENAI_API_KEY"],
                "models": {
                    "gpt-4o-mini": {
                        "id": "gpt-4o-mini",
                        "name": "GPT-4o mini",
                        "tool_call": true,
                        "attachment": true,
                        "reasoning": false,
                        "modalities": {
                            "input": ["text", "image"],
                            "output": ["text"]
                        },
                        "limit": {
                            "context": 128000,
                            "output": 16384
                        },
                        "cost": {
                            "input": 0.15,
                            "output": 0.6,
                            "cache_read": 0.075
                        }
                    },
                    "gpt-3.5-turbo": {
                        "id": "gpt-3.5-turbo",
                        "name": "GPT-3.5 Turbo",
                        "tool_call": false,
                        "status": "deprecated",
                        "modalities": {
                            "input": ["text"],
                            "output": ["text"]
                        }
                    },
                    "gpt-unknown": {
                        "name": "Unknown Tool Model",
                        "modalities": {
                            "input": ["text"],
                            "output": ["text"]
                        }
                    }
                }
            },
            "nvidia": {
                "id": "nvidia",
                "name": "NVIDIA",
                "npm": "@ai-sdk/openai-compatible",
                "api": "https://integrate.api.nvidia.com/v1",
                "env": ["NVIDIA_API_KEY"],
                "models": {
                    "meta/llama-3.1-405b-instruct": {
                        "id": "meta/llama-3.1-405b-instruct",
                        "name": "Llama 3.1 405B",
                        "tool_call": true,
                        "modalities": {
                            "input": ["text"],
                            "output": ["text"]
                        },
                        "limit": {
                            "context": 128000,
                            "output": 4096
                        }
                    }
                }
            },
            "ollama": {
                "id": "ollama",
                "name": "Ollama",
                "npm": "@ai-sdk/openai-compatible",
                "api": "http://localhost:11434/v1",
                "env": [],
                "models": {}
            }
        })
    }

    #[test]
    fn parse_catalog_basic() {
        let catalog = parse_catalog(&sample_catalog()).unwrap();
        // Ollama has no models, so it's filtered out.
        assert_eq!(catalog.len(), 2);
    }

    #[test]
    fn parse_catalog_preserves_tool_support() {
        let catalog = parse_catalog(&sample_catalog()).unwrap();
        let openai = catalog.iter().find(|p| p.id == "openai").unwrap();

        let mini = openai
            .models
            .iter()
            .find(|m| m.model_id == "gpt-4o-mini")
            .unwrap();
        assert_eq!(mini.tool_support, ToolSupport::Yes);
        assert!(!mini.deprecated);

        let turbo = openai
            .models
            .iter()
            .find(|m| m.model_id == "gpt-3.5-turbo")
            .unwrap();
        assert_eq!(turbo.tool_support, ToolSupport::No);
        assert!(turbo.deprecated);

        let unknown = openai
            .models
            .iter()
            .find(|m| m.model_id == "gpt-unknown")
            .unwrap();
        assert_eq!(unknown.tool_support, ToolSupport::Unknown);
    }

    #[test]
    fn parse_catalog_preserves_modalities() {
        let catalog = parse_catalog(&sample_catalog()).unwrap();
        let openai = catalog.iter().find(|p| p.id == "openai").unwrap();
        let mini = openai
            .models
            .iter()
            .find(|m| m.model_id == "gpt-4o-mini")
            .unwrap();
        assert!(mini.modalities.text_input);
        assert!(mini.modalities.image_input);
        assert!(mini.modalities.text_output);
    }

    #[test]
    fn parse_catalog_preserves_limits_and_cost() {
        let catalog = parse_catalog(&sample_catalog()).unwrap();
        let openai = catalog.iter().find(|p| p.id == "openai").unwrap();
        let mini = openai
            .models
            .iter()
            .find(|m| m.model_id == "gpt-4o-mini")
            .unwrap();
        assert_eq!(mini.context_window, 128000);
        assert_eq!(mini.max_output, 16384);
        assert!((mini.cost_input - 0.15).abs() < 0.001);
        assert!((mini.cost_output - 0.6).abs() < 0.001);
    }

    #[test]
    fn parse_catalog_preserves_provider_endpoint() {
        let catalog = parse_catalog(&sample_catalog()).unwrap();
        let nvidia = catalog.iter().find(|p| p.id == "nvidia").unwrap();
        assert_eq!(
            nvidia.base_url.as_deref(),
            Some("https://integrate.api.nvidia.com/v1")
        );
        assert!(nvidia.api_key_required);
    }

    #[test]
    fn parse_catalog_empty_provider_filtered() {
        let catalog = parse_catalog(&sample_catalog()).unwrap();
        assert!(!catalog.iter().any(|p| p.id == "ollama"));
    }

    #[test]
    fn parse_catalog_uses_key_when_id_empty() {
        let body = serde_json::json!({
            "test": {
                "name": "Test",
                "env": ["TEST_API_KEY"],
                "models": {
                    "model-by-key": {
                        "name": "Model By Key",
                        "tool_call": true
                    }
                }
            }
        });
        let catalog = parse_catalog(&body).unwrap();
        let provider = &catalog[0];
        assert_eq!(provider.models[0].model_id, "model-by-key");
    }
}
