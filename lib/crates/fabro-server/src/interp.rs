//! Shared process-env interpolation helpers for server-scope settings.
//!
//! Server-scope `InterpString` fields resolve `{{ env.* }}` tokens against
//! the server's own process environment. This module owns the single
//! process-env lookup facade and the canonical resolve helpers; do not add
//! per-module copies.

use std::path::PathBuf;

use anyhow::Context;
use fabro_types::settings::InterpString;

/// Resolve a server-scope `InterpString` against the process environment.
#[expect(
    clippy::disallowed_methods,
    reason = "raw source shown in the error message when resolution fails"
)]
pub(crate) fn resolve_interp(value: &InterpString) -> anyhow::Result<String> {
    value
        .resolve(process_env_var)
        .map(|resolved| resolved.value)
        .with_context(|| format!("failed to resolve {}", value.as_source()))
}

/// [`resolve_interp`], parsed into a filesystem path.
pub(crate) fn resolve_interp_path(value: &InterpString) -> anyhow::Result<PathBuf> {
    Ok(PathBuf::from(resolve_interp(value)?))
}

/// The server-owned process-env lookup facade for `{{ env.* }}`
/// interpolation and server configuration/secret reads.
#[expect(
    clippy::disallowed_methods,
    reason = "server-scope interpolation and configuration own this process-env lookup facade"
)]
pub(crate) fn process_env_var(name: &str) -> Option<String> {
    std::env::var(name).ok()
}
