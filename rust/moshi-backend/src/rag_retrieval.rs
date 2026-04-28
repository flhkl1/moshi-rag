// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

/// Which bundled reference prompt template to send to the retrieval LLM.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptStyle {
    #[default]
    Original,
    Simplified,
}

/// One OpenAI-compatible retrieval endpoint (reference LLM).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RagLlmProfile {
    pub id: String,
    pub base_url: String,
    pub model: String,
    #[serde(default)]
    pub api_key: Option<String>,
    /// Exactly one profile must have `default: true` when multiple profiles are configured.
    #[serde(rename = "default", default)]
    pub is_default: bool,
    /// Bundled reference prompt variant for this endpoint (`config.json` / env).
    #[serde(default, alias = "reference_prompt")]
    pub prompt_style: PromptStyle,
}

fn default_profile_index(profiles: &[RagLlmProfile]) -> usize {
    if profiles.len() <= 1 {
        return 0;
    }
    let defaults: Vec<usize> =
        profiles.iter().enumerate().filter(|(_, p)| p.is_default).map(|(i, _)| i).collect();
    if defaults.len() != 1 {
        panic!(
            "rag_llm_profiles: exactly one profile must have \"default\": true when using multiple profiles (found {})",
            defaults.len()
        );
    }
    defaults[0]
}

/// Shared state for which retrieval profile is active (WebSocket can switch `active`).
pub struct RagRetrievalEndpoints {
    profiles: Vec<RagLlmProfile>,
    active_default: AtomicUsize,
    active_by_slot: Mutex<HashMap<usize, usize>>,
    /// Index of the fallback profile (always queried in parallel with the active profile when they differ).
    default_idx: usize,
}

impl RagRetrievalEndpoints {
    pub fn from_profiles(
        profiles: Option<Vec<RagLlmProfile>>,
        default_active_id: Option<&str>,
    ) -> Arc<Self> {
        let profiles = profiles.unwrap_or_default();
        let default_idx = default_profile_index(&profiles);
        let mut idx = default_idx;
        if profiles.len() >= 2 {
            if let Some(did) = default_active_id {
                if let Some(i) = profiles.iter().position(|p| p.id == did) {
                    idx = i;
                }
            }
        }
        if profiles.is_empty() {
            tracing::info!(
                "rag_llm_profiles: none (reference LLM from LLM_BASE_URL / LLM_MODEL_NAME env)"
            );
        } else {
            let ids: Vec<&str> = profiles.iter().map(|p| p.id.as_str()).collect();
            tracing::info!(count = profiles.len(), ?ids, "rag_llm_profiles loaded from config");
            if profiles.len() >= 2 {
                let fallback = profiles.get(default_idx).map(|p| p.id.as_str()).unwrap_or("?");
                let initial = profiles.get(idx).map(|p| p.id.as_str()).unwrap_or("?");
                tracing::info!(%fallback, %initial, "retrieval LLM WebSocket switching enabled (fallback + initial active)");
            } else {
                tracing::info!("only one rag_llm_profile; UI switching requires >=2 profiles");
            }
        }
        Arc::new(Self {
            profiles,
            active_default: AtomicUsize::new(idx),
            active_by_slot: Mutex::new(HashMap::new()),
            default_idx,
        })
    }

    pub fn prompt_style_at(&self, index: usize) -> PromptStyle {
        if self.profiles.is_empty() {
            return PromptStyle::Original;
        }
        let i = index.min(self.profiles.len() - 1);
        self.profiles[i].prompt_style
    }

    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    pub fn active_index_for_slot(&self, slot_id: usize) -> usize {
        if self.profiles.is_empty() {
            return 0;
        }
        let idx = self
            .active_by_slot
            .lock()
            .ok()
            .and_then(|m| m.get(&slot_id).copied())
            .unwrap_or_else(|| self.active_default.load(Ordering::SeqCst));
        idx.min(self.profiles.len() - 1)
    }

    pub fn default_index(&self) -> usize {
        if self.profiles.is_empty() {
            return 0;
        }
        self.default_idx.min(self.profiles.len() - 1)
    }

    pub fn credentials_at(&self, index: usize) -> Result<(String, String, String), String> {
        if self.profiles.is_empty() {
            return Err("no rag_llm_profiles".to_string());
        }
        let i = index.min(self.profiles.len() - 1);
        let p = &self.profiles[i];
        let key =
            p.api_key.clone().unwrap_or_else(|| std::env::var("LLM_API_KEY").unwrap_or_default());
        Ok((p.base_url.clone(), p.model.clone(), key))
    }

    pub fn default_id_for_ui(&self) -> Option<String> {
        self.default_id_for_ui_slot(0)
    }

    pub fn default_id_for_ui_slot(&self, slot_id: usize) -> Option<String> {
        if self.profiles.len() < 2 {
            return None;
        }
        let i = self.active_index_for_slot(slot_id);
        Some(self.profiles[i].id.clone())
    }

    pub fn set_active_id(&self, id: &str) -> Result<(), String> {
        self.set_active_id_for_slot(0, id)
    }

    pub fn set_active_id_for_slot(&self, slot_id: usize, id: &str) -> Result<(), String> {
        if self.profiles.len() < 2 {
            return Err("retrieval switching disabled".to_string());
        }
        let i = self
            .profiles
            .iter()
            .position(|p| p.id == id)
            .ok_or_else(|| format!("unknown retrieval_backend_id {id:?}"))?;
        let mut guard = self
            .active_by_slot
            .lock()
            .map_err(|_| "retrieval slot map lock poisoned".to_string())?;
        guard.insert(slot_id, i);
        Ok(())
    }

    pub fn reset_active_slot(&self, slot_id: usize) {
        if let Ok(mut guard) = self.active_by_slot.lock() {
            guard.remove(&slot_id);
        }
    }

    pub fn resolve_llm_credentials_for_slot(
        &self,
        slot_id: usize,
    ) -> Result<(String, String, String), String> {
        if self.profiles.len() >= 2 {
            return self.credentials_at(self.active_index_for_slot(slot_id));
        }
        if self.profiles.len() == 1 {
            return self.credentials_at(0);
        }
        let base_url = std::env::var("LLM_BASE_URL").map_err(|e| e.to_string())?;
        let model_name = std::env::var("LLM_MODEL_NAME").map_err(|e| e.to_string())?;
        let api_key = std::env::var("LLM_API_KEY").unwrap_or_default();
        Ok((base_url, model_name, api_key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_uses_second_profile_after_switch() {
        let profiles = vec![
            RagLlmProfile {
                id: "a".into(),
                base_url: "https://a/v1".into(),
                model: "ma".into(),
                api_key: None,
                is_default: true,
                prompt_style: PromptStyle::Original,
            },
            RagLlmProfile {
                id: "b".into(),
                base_url: "https://b/v1".into(),
                model: "mb".into(),
                api_key: None,
                is_default: false,
                prompt_style: PromptStyle::Original,
            },
        ];
        let r = RagRetrievalEndpoints::from_profiles(Some(profiles), None);
        r.set_active_id("b").unwrap();
        let (url, model, _) = r.resolve_llm_credentials_for_slot(0).unwrap();
        assert_eq!(url, "https://b/v1");
        assert_eq!(model, "mb");
    }
}
