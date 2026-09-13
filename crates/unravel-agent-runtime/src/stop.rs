//! A wakeable, race-free cancellation token backed by `tokio::sync::watch`.
//!
//! Unlike a polling atomic, `cancelled()` returns a future that resolves the
//! moment `stop()` is called, so the loop can interrupt pending model calls,
//! backoff sleeps, and tool futures without a lost-wakeup race.

use std::sync::Arc;

/// A cancellable stop signal.
///
/// Clone to share; call [`stop`](StopToken::stop) to cancel; await
/// [`cancelled`](StopToken::cancelled) to wait for cancellation. The token
/// starts un-cancelled.
#[derive(Clone)]
pub struct StopToken {
    tx: Arc<tokio::sync::watch::Sender<bool>>,
    rx: tokio::sync::watch::Receiver<bool>,
}

impl std::fmt::Debug for StopToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StopToken")
            .field("stopped", &self.is_stopped())
            .finish()
    }
}

impl Default for StopToken {
    fn default() -> Self {
        Self::new()
    }
}

impl StopToken {
    /// Create a new un-cancelled token.
    pub fn new() -> Self {
        let (tx, rx) = tokio::sync::watch::channel(false);
        Self {
            tx: Arc::new(tx),
            rx,
        }
    }

    /// Signal cancellation. All clones see the cancellation.
    pub fn stop(&self) {
        let _ = self.tx.send(true);
    }

    /// Polling check: `true` if `stop()` has been called.
    pub fn is_stopped(&self) -> bool {
        *self.rx.borrow()
    }

    /// A future that resolves when `stop()` is called. Use in
    /// `tokio::select!` to race cancellation against other work.
    ///
    /// ```ignore
    /// tokio::select! {
    ///     biased;
    ///     _ = stop.cancelled() => return Err(Error::Stopped),
    ///     res = model_work => res,
    /// }
    /// ```
    pub async fn cancelled(&self) {
        // If already stopped, return immediately.
        if *self.rx.borrow() {
            return;
        }
        // Otherwise wait for a change. Using `wait_for` avoids missing an
        // already-applied change after `borrow`.
        let mut rx = self.rx.clone();
        let _ = rx.wait_for(|&v| v).await;
    }
}

#[cfg(test)]
mod stop_token_tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn cancelled_resolves_after_stop() {
        let token = StopToken::new();
        let t2 = token.clone();
        assert!(!token.is_stopped());
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            t2.stop();
        });
        token.cancelled().await;
        assert!(token.is_stopped());
    }

    #[tokio::test]
    async fn already_stopped_returns_immediately() {
        let token = StopToken::new();
        token.stop();
        token.cancelled().await;
        assert!(token.is_stopped());
    }
}
