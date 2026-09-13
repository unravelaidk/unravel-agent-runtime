//! Instance-owned TTL cache for discovery results.
//!
//! Keyed by endpoint + account identity + protocol + discovery options.
//! No process-global mutable state. No secrets in cache keys or Debug
//! output. Entries expire after a configurable TTL.
//!
//! The clock is injectable via a [`Clock`] trait so tests can advance
//! time deterministically without `thread::sleep`.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// A clock source for cache expiry checks.
///
/// The default implementation uses `Instant::now()`. Tests inject a
/// controllable clock to advance time without sleeping.
pub trait Clock: Send + Sync {
    /// Return the current instant.
    fn now(&self) -> Instant;
}

/// The default wall-clock implementation.
#[derive(Debug, Clone, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// A controllable clock for tests. The instant is advanced explicitly
/// via [`advance`](TestClock::advance) — no sleeping.
#[derive(Debug)]
pub struct TestClock {
    inner: Mutex<Instant>,
}

impl TestClock {
    /// Create a test clock starting at the given instant.
    pub fn new(start: Instant) -> Self {
        Self {
            inner: Mutex::new(start),
        }
    }

    /// Advance the clock by the given duration.
    pub fn advance(&self, dur: Duration) {
        if let Ok(mut now) = self.inner.lock() {
            *now += dur;
        }
    }
}

impl Clock for TestClock {
    fn now(&self) -> Instant {
        self.inner
            .lock()
            .map(|g| *g)
            .unwrap_or_else(|_| Instant::now())
    }
}

/// A cache entry with an expiry timestamp.
#[derive(Debug, Clone)]
struct Entry<V> {
    value: V,
    expires_at: Instant,
}

/// An instance-owned TTL cache.
///
/// Each `Discovery` instance owns its own cache — there are no
/// process-global statics. Keys are derived from endpoint + account
/// identity + protocol + discovery options, never from raw credentials.
/// The clock is injectable for deterministic expiry tests.
pub struct TtlCache<K, V> {
    entries: Mutex<HashMap<K, Entry<V>>>,
    ttl: Duration,
    clock: Arc<dyn Clock>,
}

impl<K: std::hash::Hash + std::cmp::Eq + Clone, V: Clone> TtlCache<K, V> {
    /// Create a new cache with the given TTL and the system clock.
    pub fn new(ttl: Duration) -> Self {
        Self::with_clock(ttl, Arc::new(SystemClock))
    }

    /// Create a new cache with the given TTL and a custom clock.
    pub fn with_clock(ttl: Duration, clock: Arc<dyn Clock>) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            ttl,
            clock,
        }
    }

    /// Get a cached value if it exists and has not expired.
    pub fn get(&self, key: &K) -> Option<V> {
        let entries = self.entries.lock().ok()?;
        let entry = entries.get(key)?;
        if entry.expires_at <= self.clock.now() {
            return None;
        }
        Some(entry.value.clone())
    }

    /// Insert a value with the cache's TTL.
    pub fn insert(&self, key: K, value: V) {
        if let Ok(mut entries) = self.entries.lock() {
            let expires_at = self.clock.now() + self.ttl;
            entries.insert(key, Entry { value, expires_at });
        }
    }

    /// Remove all entries.
    pub fn clear(&self) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.clear();
        }
    }

    /// Remove expired entries.
    #[allow(dead_code)]
    pub fn evict_expired(&self) {
        let now = self.clock.now();
        if let Ok(mut entries) = self.entries.lock() {
            entries.retain(|_, entry| entry.expires_at > now);
        }
    }

    /// Number of entries (including expired ones not yet evicted).
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.lock().map(|e| e.len()).unwrap_or(0)
    }

    /// Whether the cache is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<K, V> std::fmt::Debug for TtlCache<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.entries.lock().map(|e| e.len()).unwrap_or(0);
        f.debug_struct("TtlCache")
            .field("ttl", &self.ttl)
            .field("len", &len)
            .finish()
    }
}

/// A cache key for discovery results.
///
/// Contains no secrets — only the endpoint, account identity (a hash),
/// protocol, and relevant discovery options.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DiscoveryCacheKey {
    pub provider_id: String,
    pub endpoint: String,
    pub account_identity: String,
    pub protocol: String,
    pub include_models_dev: bool,
    pub filter_for_tools: bool,
    pub explicit_model_ids: Vec<String>,
    pub models_dev_url: String,
    pub timeout: Duration,
}

impl DiscoveryCacheKey {
    /// Build a cache key from the relevant parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider_id: &str,
        endpoint: &str,
        account_identity: &str,
        protocol: &str,
        include_models_dev: bool,
        filter_for_tools: bool,
        explicit_model_ids: Vec<String>,
        models_dev_url: &str,
        timeout: Duration,
    ) -> Self {
        Self {
            provider_id: provider_id.to_string(),
            endpoint: endpoint.to_string(),
            account_identity: account_identity.to_string(),
            protocol: protocol.to_string(),
            include_models_dev,
            filter_for_tools,
            explicit_model_ids,
            models_dev_url: digest(models_dev_url),
            timeout,
        }
    }
}

fn digest(value: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_stores_and_retrieves() {
        let cache: TtlCache<String, String> = TtlCache::new(Duration::from_secs(60));
        cache.insert("key1".to_string(), "value1".to_string());
        assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));
    }

    #[test]
    fn cache_miss_returns_none() {
        let cache: TtlCache<String, String> = TtlCache::new(Duration::from_secs(60));
        assert_eq!(cache.get(&"missing".to_string()), None);
    }

    #[test]
    fn cache_expires_after_ttl_with_test_clock() {
        let clock = Arc::new(TestClock::new(Instant::now()));
        let cache: TtlCache<String, String> =
            TtlCache::with_clock(Duration::from_millis(50), clock.clone());
        cache.insert("key1".to_string(), "value1".to_string());
        // Advance past TTL — no sleeping.
        clock.advance(Duration::from_millis(60));
        assert_eq!(cache.get(&"key1".to_string()), None);
    }

    #[test]
    fn cache_isolates_by_key() {
        let cache: TtlCache<String, String> = TtlCache::new(Duration::from_secs(60));
        cache.insert("a".to_string(), "value_a".to_string());
        cache.insert("b".to_string(), "value_b".to_string());
        assert_eq!(cache.get(&"a".to_string()), Some("value_a".to_string()));
        assert_eq!(cache.get(&"b".to_string()), Some("value_b".to_string()));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cache_clear_removes_all() {
        let cache: TtlCache<String, String> = TtlCache::new(Duration::from_secs(60));
        cache.insert("a".to_string(), "1".to_string());
        cache.insert("b".to_string(), "2".to_string());
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn cache_evict_expired_removes_only_expired() {
        let clock = Arc::new(TestClock::new(Instant::now()));
        let cache: TtlCache<String, String> =
            TtlCache::with_clock(Duration::from_millis(50), clock.clone());
        cache.insert("expired".to_string(), "1".to_string());
        clock.advance(Duration::from_millis(60));
        cache.insert("fresh".to_string(), "2".to_string());
        cache.evict_expired();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&"fresh".to_string()), Some("2".to_string()));
    }

    #[test]
    fn cache_debug_does_not_leak_values() {
        let cache: TtlCache<String, String> = TtlCache::new(Duration::from_secs(60));
        cache.insert("key".to_string(), "secret-value".to_string());
        let debug = format!("{:?}", cache);
        assert!(!debug.contains("secret-value"));
    }

    #[test]
    fn cache_entry_still_valid_at_boundary() {
        let clock = Arc::new(TestClock::new(Instant::now()));
        let cache: TtlCache<String, String> =
            TtlCache::with_clock(Duration::from_millis(100), clock.clone());
        cache.insert("key".to_string(), "value".to_string());
        // Advance exactly to TTL boundary — entry expires at `now + ttl`,
        // and `get` returns None when `expires_at <= clock.now()`.
        clock.advance(Duration::from_millis(100));
        assert_eq!(cache.get(&"key".to_string()), None);
    }

    #[test]
    fn cache_entry_valid_before_expiry() {
        let clock = Arc::new(TestClock::new(Instant::now()));
        let cache: TtlCache<String, String> =
            TtlCache::with_clock(Duration::from_millis(100), clock.clone());
        cache.insert("key".to_string(), "value".to_string());
        clock.advance(Duration::from_millis(99));
        assert_eq!(cache.get(&"key".to_string()), Some("value".to_string()));
    }

    #[test]
    fn discovery_cache_key_has_no_secrets() {
        let key = DiscoveryCacheKey::new(
            "openai",
            "https://api.openai.com/v1",
            "openai:abcdef0123456789",
            "openai-chat-completions",
            true,
            false,
            Vec::new(),
            "https://models.dev/api.json",
            Duration::from_secs(10),
        );
        let debug = format!("{:?}", key);
        assert!(debug.contains("api.openai.com"));
        assert!(!debug.contains("sk-"));
    }
}
