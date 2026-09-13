//! HTTP transport: reqwest with rustls, no redirects, bounded
//! response/SSE limits, and deadline enforcement.

use crate::error::{ProviderError, ProviderResult};
use std::time::Duration;

/// Maximum response body size for non-streaming JSON responses (4 MiB).
const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;

/// Maximum total SSE stream size (16 MiB). Prevents unbounded memory
/// growth from a misbehaving server.
const MAX_SSE_TOTAL_BYTES: usize = 16 * 1024 * 1024;

/// Default per-request timeout.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

/// Build a reqwest client with rustls, no redirects, and a timeout.
pub fn build_client() -> ProviderResult<reqwest::Client> {
    reqwest::Client::builder()
        .use_rustls_tls()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(DEFAULT_TIMEOUT)
        .build()
        .map_err(|e| ProviderError::transient(format!("failed to build HTTP client: {e}")))
}

/// Join a base URL with a path, respecting `/v1` and path prefixes.
///
/// Examples:
/// - `join("https://api.openai.com/v1", "chat/completions")` →
///   `https://api.openai.com/v1/chat/completions`
/// - `join("https://api.openai.com/v1/", "chat/completions")` →
///   `https://api.openai.com/v1/chat/completions`
/// - `join("https://custom.example.com/proxy/v1", "models")` →
///   `https://custom.example.com/proxy/v1/models`
pub fn join_url(base: &str, path: &str) -> String {
    let base = base.trim_end_matches('/');
    if path.is_empty() {
        return base.to_string();
    }
    let path = path.trim_start_matches('/');
    format!("{base}/{path}")
}

/// Read a bounded JSON response body.
///
/// Returns an error if the body exceeds [`MAX_RESPONSE_BYTES`].
pub async fn read_bounded_json(response: reqwest::Response) -> ProviderResult<serde_json::Value> {
    read_bounded_json_with_limit(response, MAX_RESPONSE_BYTES).await
}

/// Resource-specific JSON limit, enforced before appending each received
/// chunk as well as against Content-Length. Missing lengths cannot bypass it.
pub(crate) async fn read_bounded_json_with_limit(
    mut response: reqwest::Response,
    max_bytes: usize,
) -> ProviderResult<serde_json::Value> {
    let content_length = response.content_length();
    if let Some(len) = content_length {
        if len > max_bytes as u64 {
            return Err(ProviderError::malformed(format!(
                "response body exceeds {} byte limit (declared {})",
                max_bytes, len
            )));
        }
    }

    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| ProviderError::transient(format!("failed to read response body: {e}")))?
    {
        let received = bytes.len().saturating_add(chunk.len());
        if received > max_bytes {
            return Err(ProviderError::malformed(format!(
                "response body exceeds {} byte limit (received {})",
                max_bytes, received
            )));
        }
        bytes.extend_from_slice(&chunk);
    }

    serde_json::from_slice::<serde_json::Value>(&bytes)
        .map_err(|e| ProviderError::malformed(format!("failed to parse JSON response: {e}")))
}

/// Read a bounded text response body (for error diagnostics).
pub async fn read_bounded_text(response: reqwest::Response) -> String {
    let bytes = response.bytes().await;
    match bytes {
        Ok(b) if b.len() <= MAX_RESPONSE_BYTES => String::from_utf8_lossy(&b).to_string(),
        Ok(b) => format!("<response body too large: {} bytes>", b.len()),
        Err(e) => format!("<failed to read body: {e}>"),
    }
}

/// Map an HTTP error status to a typed [`ProviderError`].
///
/// - 401/403 → permanent auth
/// - 400/404/422 → permanent invalid
/// - 429 → transient rate-limit (with Retry-After if present)
/// - 5xx → transient
/// - other → transient (conservative)
pub fn map_http_error(status: reqwest::StatusCode, body: &str) -> ProviderError {
    map_http_error_with_secret(status, body, None)
}

/// Map an HTTP error while also removing the credential used for the request.
///
/// A provider may echo a credential in arbitrary, non-Bearer text. Pattern
/// redaction alone cannot protect against that, so callers that have the key
/// must pass it through this boundary.
pub fn map_http_error_with_secret(
    status: reqwest::StatusCode,
    body: &str,
    secret: Option<&str>,
) -> ProviderError {
    let redacted = crate::error::redact(body);
    let redacted = secret
        .map(|secret| crate::error::redact_secret(&redacted, secret))
        .unwrap_or(redacted);
    let status_code = status.as_u16();

    match status_code {
        401 | 403 => ProviderError::auth(format!("HTTP {status_code}: {redacted}")),
        400 | 404 | 422 => ProviderError::invalid(format!("HTTP {status_code}: {redacted}")),
        429 => {
            // Retry-After is handled by the caller (needs header access).
            ProviderError::transient(format!("HTTP 429: rate limited: {redacted}"))
        }
        500..=599 => {
            ProviderError::transient(format!("HTTP {status_code}: server error: {redacted}"))
        }
        _ => ProviderError::transient(format!("HTTP {status_code}: unexpected status: {redacted}")),
    }
}

/// Extract Retry-After from a response, if present.
pub fn extract_retry_after(response: &reqwest::Response) -> Option<Duration> {
    response
        .headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| {
            // Try parsing as seconds (delta-seconds per HTTP spec).
            s.trim().parse::<u64>().ok().map(Duration::from_secs)
        })
}

/// An SSE event parsed from a byte stream.
#[derive(Debug, Clone, PartialEq)]
pub struct SseEvent {
    pub data: String,
}

/// Parse SSE events from a UTF-8 byte buffer, returning remaining unparsed bytes.
///
/// SSE events are separated by `\n\n`. Each event may have multiple
/// `data:` lines that are joined with `\n`. A `data: [DONE]` sentinel
/// is preserved as-is for the caller to handle.
///
/// This handles fragmentation: partial events remain in the buffer
/// until the next chunk completes them. Invalid UTF-8 in a complete event
/// is returned as a protocol error rather than replaced.
pub fn parse_sse_events(buffer: &mut Vec<u8>) -> ProviderResult<Vec<SseEvent>> {
    let mut events = Vec::new();

    loop {
        // SSE events are separated by \n\n (or \r\n\r\n per the spec).
        let lf = buffer.windows(2).position(|w| w == b"\n\n");
        let crlf = buffer.windows(4).position(|w| w == b"\r\n\r\n");
        let (idx, sep_len) = match (lf, crlf) {
            (Some(lf), Some(crlf)) if crlf < lf => (crlf, 4),
            (Some(lf), _) => (lf, 2),
            (None, Some(crlf)) => (crlf, 4),
            (None, None) => break,
        };

        let raw = std::str::from_utf8(&buffer[..idx])
            .map_err(|e| ProviderError::malformed(format!("SSE event was not valid UTF-8: {e}")))?
            .to_string();
        buffer.drain(..idx + sep_len);

        // Collect data: lines. Also handle \r\n line endings within events.
        let data: Vec<String> = raw
            .lines()
            .filter_map(|line| {
                let line = line.strip_suffix('\r').unwrap_or(line);
                line.strip_prefix("data:")
                    .map(|rest| rest.trim_start().to_string())
            })
            .collect();

        if !data.is_empty() {
            events.push(SseEvent {
                data: data.join("\n"),
            });
        }
    }

    Ok(events)
}

/// Process a chunk of SSE bytes: append to buffer, parse complete events.
///
/// Returns the events and `true` if the total stream size has been
/// exceeded (caller should stop).
pub fn process_sse_chunk(
    buffer: &mut Vec<u8>,
    chunk: &[u8],
    total_bytes: &mut usize,
) -> ProviderResult<(Vec<SseEvent>, bool)> {
    buffer.extend_from_slice(chunk);
    *total_bytes += chunk.len();

    if *total_bytes > MAX_SSE_TOTAL_BYTES {
        return Err(ProviderError::malformed(format!(
            "SSE stream exceeded {} byte limit",
            MAX_SSE_TOTAL_BYTES
        )));
    }

    let events = parse_sse_events(buffer)?;
    Ok((events, false))
}

/// Check if an SSE data payload is the `[DONE]` sentinel.
pub fn is_done(data: &str) -> bool {
    data.trim() == "[DONE]"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_url_basic() {
        assert_eq!(
            join_url("https://api.openai.com/v1", "chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn join_url_trailing_slash() {
        assert_eq!(
            join_url("https://api.openai.com/v1/", "chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn join_url_path_prefix() {
        assert_eq!(
            join_url("https://custom.example.com/proxy/v1", "models"),
            "https://custom.example.com/proxy/v1/models"
        );
    }

    #[test]
    fn join_url_empty_path() {
        assert_eq!(
            join_url("https://api.openai.com/v1", ""),
            "https://api.openai.com/v1"
        );
    }

    #[test]
    fn join_url_leading_slash_in_path() {
        assert_eq!(
            join_url("https://api.openai.com/v1", "/chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn parse_sse_single_event() {
        let mut buf = b"data: {\"hello\":true}\n\n".to_vec();
        let events = parse_sse_events(&mut buf).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "{\"hello\":true}");
        assert!(buf.is_empty());
    }

    #[test]
    fn parse_sse_done_sentinel() {
        let mut buf = b"data: [DONE]\n\n".to_vec();
        let events = parse_sse_events(&mut buf).unwrap();
        assert_eq!(events.len(), 1);
        assert!(is_done(&events[0].data));
    }

    #[test]
    fn parse_sse_multiline_data() {
        let mut buf = b"data: line1\ndata: line2\n\n".to_vec();
        let events = parse_sse_events(&mut buf).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2");
    }

    #[test]
    fn parse_sse_fragmented() {
        let mut buf = b"data: {\"par".to_vec();
        let events = parse_sse_events(&mut buf).unwrap();
        assert!(events.is_empty());
        assert!(!buf.is_empty());

        buf.extend_from_slice(b"tial\":true}\n\n");
        let events = parse_sse_events(&mut buf).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "{\"partial\":true}");
    }

    #[test]
    fn parse_sse_multiple_events() {
        let mut buf = b"data: first\n\ndata: second\n\n".to_vec();
        let events = parse_sse_events(&mut buf).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "first");
        assert_eq!(events[1].data, "second");
    }

    #[test]
    fn parse_sse_ignores_non_data_lines() {
        let mut buf = b"event: ping\ndata: hello\n:id: 42\n\n".to_vec();
        let events = parse_sse_events(&mut buf).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn parse_sse_handles_crlf() {
        let mut buf = b"data: hello\r\n\r\n".to_vec();
        let events = parse_sse_events(&mut buf).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn parse_sse_utf8_fragment() {
        // A multi-byte UTF-8 character split across chunks. The boundary
        // falls inside the bytes of 'é' (U+00E9 = 0xC3 0xA9).
        let mut buf = Vec::new();
        let mut total = 0usize;

        // First chunk: valid ASCII + first byte of é.
        let chunk1 = b"data: caf\xc3";
        let (events, exceeded) = process_sse_chunk(&mut buf, chunk1, &mut total).unwrap();
        assert!(events.is_empty());
        assert!(!exceeded);

        // Second chunk: second byte of é + rest, terminated.
        let chunk2 = b"\xa9 world\n\n";
        let (events, _) = process_sse_chunk(&mut buf, chunk2, &mut total).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "café world");
    }

    #[test]
    fn parse_sse_uses_earliest_mixed_separator() {
        let mut buf = b"data: first\r\n\r\ndata: second\n\n".to_vec();
        let events = parse_sse_events(&mut buf).unwrap();
        assert_eq!(
            events
                .iter()
                .map(|event| event.data.as_str())
                .collect::<Vec<_>>(),
            ["first", "second"]
        );
        assert!(buf.is_empty());
    }

    #[test]
    fn process_sse_chunk_enforces_limit() {
        let mut buf = Vec::new();
        let mut total = MAX_SSE_TOTAL_BYTES;
        let big_chunk = vec![b'x'; 100];
        let result = process_sse_chunk(&mut buf, &big_chunk, &mut total);
        assert!(result.is_err());
    }

    #[test]
    fn map_http_error_401_is_auth() {
        let err = map_http_error(reqwest::StatusCode::UNAUTHORIZED, "bad key");
        assert!(matches!(err, ProviderError::Auth { .. }));
    }

    #[test]
    fn map_http_error_429_is_transient() {
        let err = map_http_error(reqwest::StatusCode::TOO_MANY_REQUESTS, "slow down");
        assert!(matches!(err, ProviderError::Transient { .. }));
    }

    #[test]
    fn map_http_error_500_is_transient() {
        let err = map_http_error(reqwest::StatusCode::INTERNAL_SERVER_ERROR, "internal error");
        assert!(matches!(err, ProviderError::Transient { .. }));
    }

    #[test]
    fn map_http_error_400_is_invalid() {
        let err = map_http_error(reqwest::StatusCode::BAD_REQUEST, "bad request");
        assert!(matches!(err, ProviderError::Invalid { .. }));
    }

    #[test]
    fn map_http_error_redacts_bearer_in_body() {
        let err = map_http_error(
            reqwest::StatusCode::UNAUTHORIZED,
            "Bearer sk-secret-key-12345 is invalid",
        );
        let msg = err.to_string();
        assert!(!msg.contains("sk-secret-key-12345"));
    }

    #[test]
    fn redact_url_strips_query() {
        let r = crate::error::redact_url("https://api.example.com/models?api_key=secret");
        assert!(!r.contains("secret"));
    }
}
