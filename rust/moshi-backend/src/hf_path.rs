// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

//! Model path resolution (all weight/tokenizer fields use the same rules):
//!
//! 1. **Local file** — if the string is an existing path after env expansion, it is used as-is.
//! 2. **Explicit Hub** — `org/repo:path/in/repo` (single colon separates repo id from file path).
//! 3. **Hub via `hf_repo`** — if the string is not (1) or (2) and config **`hf_repo`** is set, resolve
//!    as `{hf_repo}:{value}` (the value is the path of the file inside that repo, e.g. `model.safetensors`).
//!
//! For assets in a **different** repo than `hf_repo`, use form (2), e.g. `kyutai/stt-1b-en_fr-candle:model.safetensors`.
//!
//! [`Cache`](hf_hub::Cache) is consulted so already-downloaded files are found without network when possible.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Parse `namespace/repo:filename` or `namespace/repo:subdir/file`.
pub fn parse_hf_repo_colon_file(s: &str) -> Option<(String, String)> {
    let s = s.trim();
    if s.starts_with("http://") || s.starts_with("https://") {
        return None;
    }
    let (repo_id, path_in_repo) = s.split_once(':')?;
    if repo_id.is_empty() || path_in_repo.is_empty() {
        return None;
    }
    if repo_id.len() == 1
        && repo_id.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && (path_in_repo.starts_with('\\') || path_in_repo.starts_with('/'))
    {
        return None;
    }
    if repo_id.matches('/').count() != 1 {
        return None;
    }
    let (org, name) = repo_id.split_once('/')?;
    if org.is_empty() || name.is_empty() {
        return None;
    }
    Some((repo_id.to_string(), path_in_repo.to_string()))
}

fn path_buf_to_string(p: PathBuf) -> Result<String> {
    p.into_os_string().into_string().map_err(|_| anyhow::anyhow!("path is not valid UTF-8"))
}

/// Resolved path in the local Hugging Face hub cache, if present (`HF_HOME` / default cache).
pub fn hub_cache_file(repo_id: &str, path_in_repo: &str) -> Option<PathBuf> {
    hf_hub::Cache::from_env().model(repo_id.to_string()).get(path_in_repo)
}

/// True until the string is an existing local filesystem path.
///
/// Hub-style values (`org/repo:file`, or basename + `hf_repo`) are not valid `Path`s until
/// [`resolve_model_file`] rewrites them to the hub cache (or downloads). We intentionally ignore
/// “already in HF cache” here so `download_from_hub` still runs and performs that rewrite.
pub fn path_needs_resolution(path: &str) -> bool {
    !Path::new(path.trim()).exists()
}

async fn hub_fetch(
    api: &hf_hub::api::tokio::Api,
    repo_id: &str,
    path_in_repo: &str,
    path: &mut String,
) -> Result<()> {
    if let Some(p) = hub_cache_file(repo_id, path_in_repo) {
        *path = path_buf_to_string(p)?;
        return Ok(());
    }
    let repo = api.model(repo_id.to_string());
    let downloaded_path = repo
        .get(path_in_repo)
        .await
        .with_context(|| format!("failed to get '{path_in_repo}' from {repo_id}"))?;
    *path = path_buf_to_string(downloaded_path)?;
    Ok(())
}

/// Resolve a config path: local file, explicit `org/repo:path`, or `{hf_repo}:{path}` when `hf_repo` is set.
pub async fn resolve_model_file(
    api: &hf_hub::api::tokio::Api,
    path: &mut String,
    hf_repo: Option<&str>,
) -> Result<()> {
    if Path::new(path.as_str()).exists() {
        return Ok(());
    }

    if let Some(rest) = path.strip_prefix("hf://") {
        let parts: Vec<&str> = rest.split('/').collect();
        if parts.len() < 3 || parts[0].is_empty() || parts[1].is_empty() || parts[2].is_empty() {
            anyhow::bail!("invalid hf:// path '{}': expected hf://org/repo/path/to/file", path);
        }
        let repo_id = format!("{}/{}", parts[0], parts[1]);
        let path_in_repo = parts[2..].join("/");
        return hub_fetch(api, &repo_id, &path_in_repo, path).await;
    }

    if let Some((repo_id, path_in_repo)) = parse_hf_repo_colon_file(path.as_str()) {
        return hub_fetch(api, &repo_id, &path_in_repo, path).await;
    }

    let repo_id = hf_repo.filter(|r| !r.is_empty()).ok_or_else(|| {
        anyhow::anyhow!(
            "not a local file: '{}'. Use an existing path, `org/repo:path_in_repo`, or set config field `hf_repo` (value is then read as path inside that repo)",
            path
        )
    })?;

    let path_in_repo = path.trim().to_string();
    if path_in_repo.is_empty() {
        anyhow::bail!("empty path_in_repo for hf_repo resolution");
    }
    hub_fetch(api, repo_id, &path_in_repo, path).await
}
