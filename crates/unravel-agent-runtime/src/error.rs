use std::time::Duration;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// Retryability classification for model failures.
///
/// Only [`Retryability::Transient`] errors are retried. Permanent errors
/// (authentication, validation, unsupported request) fail the run immediately
/// so the caller can reconcile instead of burning attempts on an unrecoverable
/// condition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Retryability {
    /// A permanent failure that will not succeed on retry. The loop must not
    /// replay tools or resubmit the request.
    Permanent,
    /// A transient failure (overloaded, timeout, rate limit) that may succeed
    /// on a later attempt. When set, `retry_after` is the server-suggested
    /// minimum delay before the next attempt. `None` means use the backoff
    /// schedule.
    Transient { retry_after: Option<Duration> },
}

impl Retryability {
    pub fn permanent() -> Self {
        Retryability::Permanent
    }

    pub fn transient() -> Self {
        Retryability::Transient { retry_after: None }
    }

    pub fn transient_after(retry_after: Duration) -> Self {
        Retryability::Transient {
            retry_after: Some(retry_after),
        }
    }

    /// Returns `true` when the caller is permitted to retry the operation.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Retryability::Transient { .. })
    }
}

/// A typed model error with retryability classification.
///
/// Construct this from a provider response to let the loop classify, bound,
/// and interrupt retries. The free-form `message` carries provider detail;
/// `retryability` drives the retry policy.
#[derive(Debug, Clone, Error)]
#[error("{message}")]
pub struct ModelError {
    message: String,
    retryability: Retryability,
}

impl ModelError {
    pub fn new(message: impl Into<String>, retryability: Retryability) -> Self {
        Self {
            message: message.into(),
            retryability,
        }
    }

    /// Permanent authentication or authorization failure.
    pub fn auth(message: impl Into<String>) -> Self {
        Self::new(message, Retryability::Permanent)
    }

    /// Permanent invalid-request or unsupported-feature failure.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(message, Retryability::Permanent)
    }

    /// Transient failure with no server-suggested delay.
    pub fn transient(message: impl Into<String>) -> Self {
        Self::new(message, Retryability::Transient { retry_after: None })
    }

    /// Transient rate-limit failure with a server-suggested `retry_after`.
    pub fn rate_limit(message: impl Into<String>, retry_after: Duration) -> Self {
        Self::new(
            message,
            Retryability::Transient {
                retry_after: Some(retry_after),
            },
        )
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn retryability(&self) -> &Retryability {
        &self.retryability
    }

    pub fn is_retryable(&self) -> bool {
        self.retryability.is_retryable()
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("{0}")]
    ModelTyped(#[from] ModelError),
    #[error("tool error: {0}")]
    Tool(String),
    #[error("{0}")]
    Policy(String),
    #[error("run stopped")]
    Stopped,
    #[error("maximum turn count reached ({0})")]
    MaxTurns(usize),
    #[error("model deadline exceeded")]
    DeadlineExceeded,
    #[error("session reconciliation failed: {0}")]
    Reconciliation(String),
}

impl Error {
    /// Returns `true` when this is a retryable model error.
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::ModelTyped(me) => me.is_retryable(),
            // Generic string model errors are treated as non-retryable by
            // default (unknown classification — be conservative).
            Error::Model(_) => false,
            _ => false,
        }
    }
}
