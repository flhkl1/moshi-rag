// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::{Context, Result};
use axum::extract::ws;
use futures_util::{
    stream::{SplitSink, SplitStream, StreamExt},
    SinkExt,
};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub instance_name: String,
    #[serde(default)]
    pub hf_repo: String,
    pub lm_model_file: String,
    pub log_dir: String,
    pub text_tokenizer_file: String,
    pub mimi_model_file: String,
    pub mimi_num_codebooks: usize,
    pub lm_config: Option<moshi::lm_generate_multistream::Config>,
    pub batch_size: usize,
    #[serde(default = "default_false")]
    pub use_cpu_for_mimi: bool,
    pub asr_delay_in_tokens: Option<usize>,
    // Optional
    #[serde(default)]
    pub power_threshold: Option<f64>,
    #[serde(default)]
    pub init_active_speaker: Option<String>,
    // Retrieval context preparation
    #[serde(default)]
    pub stt_lm_model_file: Option<String>,
    #[serde(default)]
    pub stt_text_tokenizer_file: Option<String>,
    #[serde(default)]
    pub stt_mimi_model_file: Option<String>,
    #[serde(default)]
    pub stt_wait_time: Option<f32>,
    #[serde(default)]
    pub vad_window_size: Option<usize>,
    #[serde(default)]
    pub vad_threshold: Option<f32>,
    // Retrieval and post-processing
    #[serde(default)]
    pub rag_token_id: Option<u32>,
    #[serde(default = "default_zero_f32")]
    pub rag_timeout: f32,
    #[serde(default)]
    pub arc_encoder_tokenizer_path: Option<String>,
    #[serde(default)]
    pub arc_encoder_model_file: Option<String>,
    #[serde(default = "default_zero_usize")]
    pub moshi_gpu_id: usize,
    #[serde(default = "default_zero_usize")]
    pub stt_gpu_id: usize,
    /// Optional list of retrieval (reference) LLM profiles from `MOSHI_RETRIEVAL_LLMS_JSON` only.
    #[serde(default, skip_deserializing)]
    pub rag_llm_profiles: Option<Vec<crate::rag_retrieval::RagLlmProfile>>,
    /// Deprecated for config-driven profiles; ignored from `config.json`.
    #[serde(default, skip_deserializing)]
    pub rag_llm_default_id: Option<String>,
}

fn default_zero_usize() -> usize {
    0
}

fn default_zero_f32() -> f32 {
    0.0
}

fn default_false() -> bool {
    false
}

fn parse_prompt_style_from_json_value(
    v: &serde_json::Value,
) -> Result<crate::rag_retrieval::PromptStyle> {
    let s = v.as_str().ok_or_else(|| anyhow::anyhow!("prompt_style must be a JSON string"))?;
    match s.trim().to_ascii_lowercase().as_str() {
        "original" => Ok(crate::rag_retrieval::PromptStyle::Original),
        "simplified" => Ok(crate::rag_retrieval::PromptStyle::Simplified),
        x => anyhow::bail!("prompt_style must be \"original\" or \"simplified\", got {:?}", x),
    }
}

fn parse_default_flag_from_json_value(v: &serde_json::Value) -> Result<bool> {
    match v {
        serde_json::Value::Null => Ok(false),
        serde_json::Value::Bool(b) => Ok(*b),
        serde_json::Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => anyhow::bail!("default must be true/false or \"true\"/\"false\", got {:?}", s),
        },
        _ => anyhow::bail!("default must be true/false or \"true\"/\"false\""),
    }
}

pub(crate) fn parse_retrieval_llms_json(config: &mut Config) -> Result<()> {
    if let Ok(raw) = std::env::var("MOSHI_RETRIEVAL_LLMS_JSON") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let value: serde_json::Value = serde_json::from_str(trimmed)
                .with_context(|| "invalid JSON in MOSHI_RETRIEVAL_LLMS_JSON")?;
            #[derive(serde::Deserialize)]
            struct RagLlmProfileEnv {
                id: String,
                base_url: String,
                model: String,
                #[serde(default)]
                api_key: Option<String>,
                #[serde(rename = "default", default)]
                raw_is_default: serde_json::Value,
                #[serde(default, alias = "reference_prompt")]
                prompt_style: Option<crate::rag_retrieval::PromptStyle>,
            }

            let (profiles_value, profile_default_prompt) = match value {
                serde_json::Value::Array(_) => (value, crate::rag_retrieval::PromptStyle::Original),
                serde_json::Value::Object(mut map) => {
                    let profile_default_prompt = match map
                        .remove("prompt_style")
                        .or_else(|| map.remove("reference_prompt"))
                    {
                        None => crate::rag_retrieval::PromptStyle::Original,
                        Some(v) => parse_prompt_style_from_json_value(&v)?,
                    };
                    let profiles_val = map.remove("profiles").ok_or_else(|| {
                        anyhow::anyhow!(
                            "MOSHI_RETRIEVAL_LLMS_JSON: object form requires a \"profiles\" array (legacy form is a bare JSON array of profile objects)"
                        )
                    })?;
                    if !map.is_empty() {
                        let keys: Vec<String> = map.keys().cloned().collect();
                        anyhow::bail!(
                            "MOSHI_RETRIEVAL_LLMS_JSON: unknown top-level keys: {}",
                            keys.join(", ")
                        );
                    }
                    (profiles_val, profile_default_prompt)
                }
                _ => anyhow::bail!(
                    "MOSHI_RETRIEVAL_LLMS_JSON: must be an array of profiles, or an object with \"profiles\" (and optional \"prompt_style\")"
                ),
            };
            let raw_profiles: Vec<RagLlmProfileEnv> = serde_json::from_value(profiles_value).with_context(|| {
                "invalid profile objects in MOSHI_RETRIEVAL_LLMS_JSON (expected id, base_url, model, optional api_key, optional default, optional prompt_style)"
            })?;
            let profiles: Vec<crate::rag_retrieval::RagLlmProfile> = raw_profiles
                .into_iter()
                .map(|r| {
                    Ok(crate::rag_retrieval::RagLlmProfile {
                        id: r.id,
                        base_url: r.base_url,
                        model: r.model,
                        api_key: r.api_key,
                        is_default: parse_default_flag_from_json_value(&r.raw_is_default)?,
                        prompt_style: r.prompt_style.unwrap_or(profile_default_prompt),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if profiles.len() >= 2 {
                let n_default = profiles.iter().filter(|p| p.is_default).count();
                if n_default != 1 {
                    anyhow::bail!(
                        "MOSHI_RETRIEVAL_LLMS_JSON: exactly one profile must have \"default\": true when using multiple profiles (found {})",
                        n_default
                    );
                }
            }
            let n = profiles.len();
            config.rag_llm_profiles = Some(profiles);
            tracing::info!(
                "MOSHI_RETRIEVAL_LLMS_JSON: loaded {n} rag_llm_profile(s) from environment (overrides config)"
            );
        }
    }
    Ok(())
}

impl Config {
    pub fn load<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let config = std::fs::read_to_string(p)?;
        let mut config: Self = serde_json::from_str(&config)?;
        config.log_dir = crate::utils::replace_env_vars(&config.log_dir);
        config.text_tokenizer_file = crate::utils::replace_env_vars(&config.text_tokenizer_file);
        config.mimi_model_file = crate::utils::replace_env_vars(&config.mimi_model_file);
        config.lm_model_file = crate::utils::replace_env_vars(&config.lm_model_file);
        if let Some(ref mut s) = config.stt_lm_model_file {
            *s = crate::utils::replace_env_vars(s);
        }
        if let Some(ref mut s) = config.stt_text_tokenizer_file {
            *s = crate::utils::replace_env_vars(s);
        }
        if let Some(ref mut s) = config.stt_mimi_model_file {
            *s = crate::utils::replace_env_vars(s);
        }
        if let Some(ref mut s) = config.arc_encoder_tokenizer_path {
            *s = crate::utils::replace_env_vars(s);
        }
        if let Some(ref mut s) = config.arc_encoder_model_file {
            *s = crate::utils::replace_env_vars(s);
        }

        parse_retrieval_llms_json(&mut config)?;

        Ok(config)
    }

    pub fn use_rag(&self) -> bool {
        self.use_stt() && self.arc_encoder_tokenizer_path.is_some()
    }

    pub fn use_stt(&self) -> bool {
        self.stt_lm_model_file.is_some()
            && self.stt_text_tokenizer_file.is_some()
            && self.stt_mimi_model_file.is_some()
    }

    /// True if any configured model path is not yet an existing file (same rule for main, STT, and ARC).
    pub fn requires_model_download(&self) -> bool {
        let mut paths: Vec<&str> = vec![
            self.lm_model_file.as_str(),
            self.mimi_model_file.as_str(),
            self.text_tokenizer_file.as_str(),
        ];
        if self.use_stt() {
            paths.extend([
                self.stt_lm_model_file.as_deref().unwrap(),
                self.stt_mimi_model_file.as_deref().unwrap(),
                self.stt_text_tokenizer_file.as_deref().unwrap(),
            ]);
        }
        if self.use_rag() {
            if let Some(p) = self.arc_encoder_tokenizer_path.as_deref() {
                paths.push(p);
            }
            if let Some(p) = self.arc_encoder_model_file.as_deref() {
                paths.push(p);
            }
        }
        paths.iter().any(|p| crate::hf_path::path_needs_resolution(p))
    }
}

fn rms_db(frame: &[f32]) -> f32 {
    let n = frame.len();
    if n == 0 {
        return f32::NEG_INFINITY;
    }
    let sum_sq: f32 = frame.iter().map(|x| x * x).sum();
    let rms_sq = sum_sq / n as f32;
    10.0 * (rms_sq + 1e-16_f32).log10()
}

fn apply_power_threshold_frame(frame: &mut [f32], thresh_db: f32) {
    if rms_db(frame) < thresh_db {
        frame.fill(0.0);
    }
}

fn apply_power_threshold_frames(samples: &mut [f32], frame_size: usize, thresh_db: f32) {
    let mut i = 0;
    while i < samples.len() {
        let end = (i + frame_size).min(samples.len());
        if end == i {
            break;
        }
        apply_power_threshold_frame(&mut samples[i..end], thresh_db);
        i = end;
    }
}

pub type AppState = Arc<AppStateInner>;

/// Unified state for standalone: standard (single LM) or RAG (LM + STT) or batched (standard or RAG).
pub enum AppStateVariant {
    Standard(Arc<AppStateInner>),
    Rag(Arc<AppStateRag>),
    Batched(Arc<BatchedState>),
}

pub struct AppStateInner {
    pub lm_model: moshi::lm::LmModel,
    pub mimi_model: moshi::mimi::Mimi,
    pub text_tokenizer: sentencepiece::SentencePieceProcessor,
    pub device: candle::Device,
    pub stt_device: candle::Device,
    pub config: Config,
}

impl AppStateInner {
    fn text(
        &self,
        prev_text_token: u32,
        text_token: u32,
        config: &moshi::lm_generate_multistream::Config,
    ) -> Option<String> {
        decode_text_piece(&self.text_tokenizer, prev_text_token, text_token, config)
    }
}

fn decode_text_piece(
    tokenizer: &sentencepiece::SentencePieceProcessor,
    prev_text_token: u32,
    text_token: u32,
    config: &moshi::lm_generate_multistream::Config,
) -> Option<String> {
    if text_token != config.text_start_token
        && text_token != config.text_pad_token
        && text_token != config.text_eop_token
    {
        if prev_text_token == config.text_start_token {
            tokenizer.decode_piece_ids(&[text_token]).ok()
        } else {
            let prev_ids = tokenizer.decode_piece_ids(&[prev_text_token]).ok();
            let ids = tokenizer.decode_piece_ids(&[prev_text_token, text_token]).ok();
            prev_ids.and_then(|prev_ids| {
                ids.map(|ids| {
                    if ids.len() > prev_ids.len() {
                        ids[prev_ids.len()..].to_string()
                    } else {
                        String::new()
                    }
                })
            })
        }
    } else {
        None
    }
}

/// RAG  state: main LM + Mimi (via inner) and STT LM + Mimi + tokenizers.
pub struct AppStateRag {
    /// RAG reuses AppState and adds STT.
    pub inner: AppState,
    pub stt_lm_model: moshi::lm::LmModel,
    pub stt_mimi_model: moshi::mimi::Mimi,
    pub stt_text_tokenizer: sentencepiece::SentencePieceProcessor,
}

/// Batched state: one shared model loop & channel pool. Used when batch_size > 1.
pub struct BatchedState {
    pub inner: Arc<AppStateInner>,
    pub pool: Arc<crate::batched_channels::BatchedStreamingChannels>,
    pub rag_retrieval: Arc<crate::rag_retrieval::RagRetrievalEndpoints>,
    _loop_handle: Option<std::thread::JoinHandle<()>>,
}

impl BatchedState {
    /// Create the pool and spawn the single model loop. Call when batch_size > 1.
    pub fn new(inner: Arc<AppStateInner>) -> Result<Self> {
        let batch_size = inner.lm_model.batch_size();
        if batch_size <= 1 {
            anyhow::bail!("batch_size > 1 required");
        }
        let mimi_config = inner.mimi_model.config();
        let frame_size = (mimi_config.sample_rate / mimi_config.frame_rate).ceil() as usize;
        let pool = crate::batched_channels::BatchedStreamingChannels::new(batch_size, frame_size);
        let pool = Arc::new(pool);
        let rag_retrieval = crate::rag_retrieval::RagRetrievalEndpoints::from_profiles(
            inner.config.rag_llm_profiles.clone(),
            inner.config.rag_llm_default_id.as_deref(),
        );
        let inner_clone = inner.clone();
        let pool_clone = pool.clone();
        let rag_for_loop = rag_retrieval.clone();
        let loop_handle = std::thread::spawn(move || {
            if let Err(e) = Self::run_loop(inner_clone, pool_clone, rag_for_loop) {
                // Max step_idx reached is a normal end-of-stream; close thread without logging.
                if !e.to_string().contains("max step_idx reached") {
                    tracing::error!(err = %e, "batched model loop error");
                }
            }
        });
        Ok(Self { inner, pool, rag_retrieval, _loop_handle: Some(loop_handle) })
    }

    fn run_loop(
        inner: Arc<AppStateInner>,
        pool: Arc<crate::batched_channels::BatchedStreamingChannels>,
        rag_retrieval: Arc<crate::rag_retrieval::RagRetrievalEndpoints>,
    ) -> Result<()> {
        let session_config = default_session_config_batched();
        let gen_config = inner
            .config
            .lm_config
            .clone()
            .unwrap_or_else(moshi::lm_generate_multistream::Config::v0_1);
        let batch_size = inner.lm_model.batch_size();
        let audio_lp: Vec<_> = (0..batch_size)
            .map(|_| {
                candle_transformers::generation::LogitsProcessor::from_sampling(
                    session_config.audio_seed,
                    candle_transformers::generation::Sampling::TopK {
                        k: session_config.audio_topk,
                        temperature: session_config.audio_temperature,
                    },
                )
            })
            .collect();
        let text_lp = candle_transformers::generation::LogitsProcessor::from_sampling(
            session_config.text_seed,
            candle_transformers::generation::Sampling::TopK {
                k: session_config.text_topk,
                temperature: session_config.text_temperature,
            },
        );
        let mut state = moshi::batched_lm_generate_multistream::State::new(
            inner.lm_model.clone(),
            inner.mimi_model.clone(),
            session_config.max_steps,
            audio_lp,
            text_lp,
            session_config.pad_mult,
            session_config.repetition_penalty,
            gen_config.clone(),
        )?;
        state.warmup(pool.frame_size)?;
        // Batched STT state (ASR model + Mimi) for per-slot transcriptions.
        let batch_size = inner.lm_model.batch_size();
        let mut stt_state = if inner.config.use_stt() {
            match (
                inner.config.stt_lm_model_file.as_ref(),
                inner.config.stt_mimi_model_file.as_ref(),
                inner.config.stt_text_tokenizer_file.as_ref(),
            ) {
                (Some(stt_lm_file), Some(stt_mimi_file), Some(stt_tok_file)) => {
                    tracing::info!(
                        "Loading batched STT LM and STT Mimi on {:?} (config stt_gpu_id={})",
                        inner.stt_device,
                        inner.config.stt_gpu_id
                    );
                    let dtype = if inner.stt_device.is_cuda() {
                        candle::DType::BF16
                    } else {
                        candle::DType::F32
                    };
                    let stt_lm = moshi::lm::load_asr_stt_1b_en_fr(
                        batch_size,
                        stt_lm_file,
                        dtype,
                        &inner.stt_device,
                    )?;
                    let mimi_device = if inner.config.use_cpu_for_mimi {
                        &candle::Device::Cpu
                    } else {
                        &inner.stt_device
                    };
                    tracing::info!(
                        "STT Mimi will use {:?} (use_cpu_for_mimi={})",
                        mimi_device,
                        inner.config.use_cpu_for_mimi
                    );
                    // STT Mimi uses 32 codebooks (stt-1b-en_fr-candle config).
                    let stt_mimi = moshi::mimi::load_b(
                        Some(batch_size),
                        stt_mimi_file,
                        Some(32),
                        mimi_device,
                    )?;
                    let asr_delay = inner.config.asr_delay_in_tokens.unwrap_or(0);
                    let temperature = 0.0;
                    let stt_asr_state = moshi::asr::State::new(
                        batch_size,
                        asr_delay,
                        temperature,
                        stt_mimi,
                        stt_lm,
                    )?;
                    let stt_text_tokenizer =
                        sentencepiece::SentencePieceProcessor::open(stt_tok_file)?;
                    tracing::info!(
                        "batched STT state initialized with STT LM on {:?} and STT Mimi on {:?}",
                        inner.stt_device,
                        mimi_device
                    );
                    Some((stt_asr_state, stt_text_tokenizer))
                }
                _ => {
                    tracing::warn!(
                        "RAG enabled but STT model paths missing; disabling batched STT"
                    );
                    None
                }
            }
        } else {
            None
        };
        let device = inner.device.clone();
        let rag_manager = inner.config.rag_token_id.map(|rag_token_id| {
            (crate::rag_manager::RagManager::new(rag_retrieval.clone()), rag_token_id)
        });
        let stt_wait_secs = inner.config.stt_wait_time.map(|t| t as f64).unwrap_or(0.0);
        let rag_timeout = inner.config.rag_timeout;
        let vad_window = inner.config.vad_window_size.unwrap_or(4);
        let vad_threshold = inner.config.vad_threshold.unwrap_or(0.5);
        let frame_rate = inner.mimi_model.config().frame_rate;
        let vad_wait_steps = (stt_wait_secs * frame_rate).ceil() as usize;
        let (init_speaker, first_speaker_str): (crate::turn_manager::TextRole, Option<&str>) =
            match inner.config.init_active_speaker.as_deref() {
                Some("user") => (crate::turn_manager::TextRole::User, Some("user")),
                Some("model") => (crate::turn_manager::TextRole::Model, Some("model")),
                _ => (crate::turn_manager::TextRole::Model, None),
            };
        let mut turn_managers: Option<Vec<Arc<Mutex<crate::turn_manager::TurnManager>>>> =
            if stt_state.is_some() {
                Some(
                    (0..batch_size)
                        .map(|_| {
                            Arc::new(Mutex::new(crate::turn_manager::TurnManager::new(
                                vad_window,
                                vad_threshold,
                                vad_wait_steps,
                                init_speaker,
                            )))
                        })
                        .collect(),
                )
            } else {
                None
            };

        tracing::info!("batched model loop started");
        loop {
            // Reset LM instance if necessary. Main LM reset inside pre_process.
            let (mut batch_pcm, mask, ref_channel_ids, reset_slots) = pool.pre_process(&mut state);

            // Reset other components.
            for &bid in &reset_slots {
                if let Err(e) =
                    state.set_prepend_condition_lut(bid, "first_speaker", first_speaker_str)
                {
                    tracing::debug!(?e, bid, "set_prepend_condition_lut (no condition provider)");
                }
                if let Some((ref mut stt_asr, _)) = stt_state {
                    if let Err(e) = stt_asr.reset_batch_idx(bid) {
                        tracing::debug!(?e, bid, "stt reset_batch_idx failed");
                    }
                }
                if let Some(ref mut turn_managers) = turn_managers {
                    if let Some(tm) = turn_managers.get(bid) {
                        tm.lock().unwrap().reset(init_speaker);
                    }
                }
            }

            // Check whether any retrieval task has completed and deliver result to the right slot.
            if let Some((ref rag_mgr, _)) = rag_manager {
                if let Some((slot_id, ref_text)) = rag_mgr.try_recv_result_slot() {
                    // Encoding of reference text is very fast and can definitely be done within the time of one frame.
                    if let Err(e) =
                        state.set_streaming_sum_condition(slot_id, "reference_with_time", &ref_text)
                    {
                        tracing::warn!(?e, slot_id, "set_streaming_sum_condition");
                    }
                    let out = if ref_text.trim().is_empty() {
                        StreamOut::TextByRole {
                            text: "[RET_FAILED]".to_string(),
                            role: TextRole::Model,
                        }
                    } else {
                        StreamOut::ReferenceText { text: ref_text }
                    };
                    if pool.send_to_slot(slot_id, out).is_err() {
                        tracing::debug!(slot_id, "failed to send RAG result to slot");
                    }
                }
            }

            // Main model loop
            let any_active = mask.iter().any(|&b| b);
            let mut msgs_for_pools: Vec<moshi::batched_lm_generate_multistream::StreamingOutMsg> =
                Vec::new();
            let batch_stt_pcm = batch_pcm.clone();
            if any_active {
                // Optional per-frame power thresholding (silence suppression) per slot.
                if let Some(thresh_db64) = inner.config.power_threshold {
                    let thresh_db = thresh_db64 as f32;
                    let frame_size = pool.frame_size;
                    for (b, &active) in mask.iter().enumerate().take(batch_size) {
                        if !active {
                            continue;
                        }
                        let start = b * frame_size;
                        let end = start + frame_size;
                        if end > batch_pcm.len() {
                            continue;
                        }
                        apply_power_threshold_frame(&mut batch_pcm[start..end], thresh_db);
                    }
                }

                // Batched STT transcription
                if let Some((ref mut stt_asr, ref _stt_tokenizer)) = stt_state {
                    // Run STT forward with unfiltered pcm.
                    let stt_dev = stt_asr.device().clone();
                    let batch_stt_pcm = candle::Tensor::new(batch_stt_pcm.as_slice(), &stt_dev)?
                        .reshape((batch_size, 1, pool.frame_size))?;
                    let stt_mask = moshi::StreamMask::new(mask.clone(), &stt_dev)?;
                    let stt_msgs =
                        stt_asr.step_pcm(batch_stt_pcm, None, &stt_mask, |_, _, _| ())?;
                    let stt_config = moshi::lm_generate_multistream::Config::v0_1_stt();

                    // Update VAD
                    for msg in &stt_msgs {
                        if let moshi::asr::AsrMsg::Step { prs, .. } = msg {
                            if let Some(ref mut turn_managers) = turn_managers {
                                for b in 0..batch_size {
                                    let vad_value = prs
                                        .get(BATCH_VAD_HEAD_INDEX)
                                        .and_then(|v| v.get(b).copied())
                                        .unwrap_or(0.0);
                                    if let Some(tm) = turn_managers.get(b) {
                                        tm.lock().unwrap().update_vad(vad_value);
                                    }
                                }
                            }
                        }
                    }

                    // Extracting predicted text tokens.
                    for msg in &stt_msgs {
                        if let moshi::asr::AsrMsg::Word { tokens, batch_idx, .. } = msg {
                            let mut prev = state
                                .last_stt_text_token(*batch_idx)
                                .unwrap_or(stt_config.text_start_token);
                            for &token in tokens {
                                msgs_for_pools.push(moshi::batched_lm_generate_multistream::StreamingOutMsg::TextToken {
                                    batch_idx: *batch_idx,
                                    prev_token: prev,
                                    token,
                                    role: moshi::batched_lm_generate_multistream::TextRole::User,
                                });
                                prev = token;
                            }
                            state.set_last_stt_text_token(*batch_idx, prev);
                        }
                    }
                }

                // Main model forward
                let pcm_tensor = candle::Tensor::from_vec(
                    batch_pcm,
                    (pool.batch_size, 1, pool.frame_size),
                    &device,
                )?;
                let stream_mask = moshi::StreamMask::new(mask, &device)?;
                let msgs = state.step_pcm(&pcm_tensor, &stream_mask)?;

                // Trigger RAG if any text token matches the RAG token.
                if let Some((ref rag_mgr, rag_token_id)) = rag_manager {
                    for msg in &msgs {
                        if let moshi::batched_lm_generate_multistream::StreamingOutMsg::TextToken {
                            batch_idx,
                            prev_token: _,
                            token,
                            ..
                        } = msg
                        {
                            if *token == rag_token_id {
                                let bid = *batch_idx;
                                if let Some(ref mut turn_managers) = turn_managers {
                                    if let Some(tm_arc) = turn_managers.get(bid) {
                                        // Make an atomic reference of the turn manager so it can be passed to the retrieval thread where it is called to provide context.
                                        let tm_ptr = Arc::clone(tm_arc);
                                        rag_mgr.trigger_background_generation_slot(
                                            bid,
                                            stt_wait_secs,
                                            rag_timeout,
                                            move || -> String {
                                                tm_ptr.lock().unwrap().get_context().to_string()
                                            },
                                        );
                                        let _ = pool.send_to_slot(
                                            bid,
                                            StreamOut::TextByRole {
                                                text: "[RET]".to_string(),
                                                role: crate::turn_manager::TextRole::Model,
                                            },
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                // Decoding text tokens and emitting messages.
                let stt_config = moshi::lm_generate_multistream::Config::v0_1_stt();
                msgs_for_pools.extend(msgs);
                for b in 0..batch_size {
                    let msgs_this_batch: Vec<
                        &moshi::batched_lm_generate_multistream::StreamingOutMsg,
                    > = msgs_for_pools
                        .iter()
                        .filter(|msg| {
                            matches!(
                                msg,
                                moshi::batched_lm_generate_multistream::StreamingOutMsg::TextToken {
                                    batch_idx,
                                    ..
                                }
                                | moshi::batched_lm_generate_multistream::StreamingOutMsg::Pcm {
                                    batch_idx,
                                    ..
                                } if *batch_idx == b
                            )
                        })
                        .collect();
                    let mut model_text = String::new();
                    let mut user_text = String::new();
                    for msg in &msgs_this_batch {
                        match msg {
                            moshi::batched_lm_generate_multistream::StreamingOutMsg::TextToken { role, prev_token, token, .. } => {
                                // Get tokenizer and config for main model or STT model.
                                let (tokenizer, config) = match role {
                                    moshi::batched_lm_generate_multistream::TextRole::Model => {
                                        (&inner.text_tokenizer, &gen_config)
                                    }
                                    moshi::batched_lm_generate_multistream::TextRole::User => {
                                        match stt_state {
                                            Some((_, ref stt_tokenizer)) => (stt_tokenizer, &stt_config),
                                            None => continue,
                                        }
                                    }
                                };
                                // Decode text piece.
                                if let Some(text) =
                                    decode_text_piece(tokenizer, *prev_token, *token, config)
                                {
                                    match role {
                                        moshi::batched_lm_generate_multistream::TextRole::Model => model_text.push_str(&text),
                                        moshi::batched_lm_generate_multistream::TextRole::User => user_text.push_str(&text),
                                    }
                                }
                            },
                            moshi::batched_lm_generate_multistream::StreamingOutMsg::Pcm { batch_idx, pcm } => {
                                pool.post_process(
                                    StreamOut::Pcm { pcm: pcm.clone() },
                                    *batch_idx,
                                    &ref_channel_ids,
                                )?;
                            }
                        }
                    }
                    if let Some(ref mut turn_managers) = turn_managers {
                        // Process spoken text with the turn manager, which organizes the transcript into a turn-based format.
                        let outputs: Vec<(String, crate::turn_manager::TextRole)> =
                            turn_managers.get(b).unwrap().lock().unwrap().handle_spoken_text(
                                Some(model_text.as_str()),
                                Some(user_text.as_str()),
                            );
                        // Send processed text to the channel pool.
                        for (text, role) in &outputs {
                            pool.post_process(
                                StreamOut::TextByRole { text: text.clone(), role: *role },
                                b,
                                &ref_channel_ids,
                            )?;
                        }
                    }
                }
            } else {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct SessionConfigReq {
    pub text_temperature: Option<f64>,
    pub text_topk: Option<usize>,
    pub audio_temperature: Option<f64>,
    pub audio_topk: Option<usize>,
    pub max_steps: Option<usize>,
    pub audio_seed: Option<u64>,
    pub text_seed: Option<u64>,
    pub email: Option<String>,
    pub pad_mult: Option<f32>,
    pub repetition_penalty_context: Option<usize>,
    pub repetition_penalty: Option<f32>,
}

#[derive(serde::Serialize, Debug, Clone)]
pub struct SessionConfig {
    pub text_temperature: f64,
    pub text_topk: usize,
    pub audio_temperature: f64,
    pub audio_topk: usize,
    pub max_steps: usize,
    pub audio_seed: u64,
    pub text_seed: u64,
    pub pad_mult: Option<f32>,
    pub repetition_penalty: Option<(usize, f32)>,
    pub email: Option<String>,
    pub user_feedback: Option<usize>,
}

#[allow(dead_code)]
#[derive(serde::Serialize, Debug, Clone)]
struct SessionSummary<'a> {
    #[serde(flatten)]
    session_config: &'a SessionConfig,
    last_step_idx: usize,
    transcript: String,
    addr: Option<String>,
    lm_model_file: &'a str,
    mimi_model_file: &'a str,
    #[serde(flatten)]
    lm_config: &'a Option<moshi::lm_generate_multistream::Config>,
}

impl SessionConfigReq {
    fn into_session_config(self) -> SessionConfig {
        use rand::Rng;

        let repetition_penalty = self.repetition_penalty_context.zip(self.repetition_penalty);
        SessionConfig {
            text_temperature: self.text_temperature.unwrap_or(0.8),
            text_topk: self.text_topk.unwrap_or(250),
            text_seed: self.text_seed.unwrap_or_else(|| rand::thread_rng().gen()),
            audio_temperature: self.audio_temperature.unwrap_or(0.8),
            audio_topk: self.audio_topk.unwrap_or(250),
            audio_seed: self.audio_seed.unwrap_or_else(|| rand::thread_rng().gen()),
            email: self.email,
            user_feedback: None,
            max_steps: self.max_steps.unwrap_or(4500).min(4500),
            pad_mult: self.pad_mult,
            repetition_penalty,
        }
    }
}

/// Hard-coded sampling config for the STT model.
fn stt_sampling_config() -> SessionConfig {
    SessionConfig {
        text_temperature: 0.001,
        text_topk: 50,
        audio_temperature: 0.001,
        audio_topk: 250,
        text_seed: 0,
        audio_seed: 0,
        email: None,
        user_feedback: None,
        max_steps: 4500,
        pad_mult: None,
        repetition_penalty: None,
    }
}

/// Default session config for the batched model loop (no per-connection params).
fn default_session_config_batched() -> SessionConfig {
    SessionConfigReq {
        text_temperature: None,
        text_topk: None,
        audio_temperature: None,
        audio_topk: None,
        max_steps: None,
        audio_seed: None,
        text_seed: None,
        email: None,
        pad_mult: None,
        repetition_penalty_context: None,
        repetition_penalty: None,
    }
    .into_session_config()
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct RetrievalBackendMeta {
    pub id: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct MetaData {
    text_temperature: f64,
    text_topk: usize,
    audio_temperature: f64,
    audio_topk: usize,
    pad_mult: f32,
    repetition_penalty_context: usize,
    repetition_penalty: f32,
    lm_model_file: String,
    mimi_model_file: String,
    build_info: crate::utils::BuildInfo,
    instance_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval_backends: Option<Vec<RetrievalBackendMeta>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval_backend_default: Option<String>,
}

/// Re-export for protocol (ColoredText by role).
pub use crate::turn_manager::TextRole;

#[derive(Debug, Clone)]
pub enum StreamOut {
    Ready,
    InputPcm {
        pcm_len: usize,
    },
    MetaData {
        metadata: Box<MetaData>,
    },
    StepStart {
        step: usize,
    },
    StepPostSampling {
        step: usize,
    },
    #[allow(dead_code)]
    Text {
        text: String,
    },
    /// For display of transcript with speaker (ColoredText: 0x07 + color_id + utf8).
    TextByRole {
        text: String,
        role: TextRole,
    },
    /// Reference text from RAG (ColoredReferenceText: 0x09 + color_id 4 + utf8).
    ReferenceText {
        text: String,
    },
    Pcm {
        pcm: Vec<f32>,
    },
}

// This must be an allowed value among 120, 240, 480, 960, 1920, and 2880.
// Using a different value would result in a BadArg "invalid argument" error when calling encode.
// https://opus-codec.org/docs/opus_api-1.2/group__opus__encoder.html#ga4ae9905859cd241ef4bb5c59cd5e5309
const OPUS_ENCODER_FRAME_SIZE: usize = 960;

#[derive(Debug, Clone, Copy)]
pub enum MsgType {
    Handshake,
    Audio,
    Text,
    Control,
    Metadata,
    Error,
    Ping,
    ColoredText,
    ReferenceText,
    ColoredReferenceText,
}

impl MsgType {
    pub fn from_u8(v: u8) -> Result<Self> {
        let s = match v {
            0 => MsgType::Handshake,
            1 => MsgType::Audio,
            2 => MsgType::Text,
            3 => MsgType::Control,
            4 => MsgType::Metadata,
            5 => MsgType::Error,
            6 => MsgType::Ping,
            7 => MsgType::ColoredText,
            8 => MsgType::ReferenceText,
            9 => MsgType::ColoredReferenceText,
            _ => anyhow::bail!("unexpected msg type {v}"),
        };
        Ok(s)
    }

    pub fn to_u8(self) -> u8 {
        match self {
            MsgType::Handshake => 0,
            MsgType::Audio => 1,
            MsgType::Text => 2,
            MsgType::Control => 3,
            MsgType::Metadata => 4,
            MsgType::Error => 5,
            MsgType::Ping => 6,
            MsgType::ColoredText => 7,
            MsgType::ReferenceText => 8,
            MsgType::ColoredReferenceText => 9,
        }
    }
}

pub struct MsgSender {
    pw: ogg::PacketWriter<'static, Vec<u8>>,
    encoder: opus::Encoder,
    out_pcm: std::collections::VecDeque<f32>,
    out_pcm_buf: Vec<u8>,
    total_data: usize,
    sender: SplitSink<ws::WebSocket, ws::Message>,
}

impl MsgSender {
    fn new(sender: SplitSink<ws::WebSocket, ws::Message>) -> Result<Self> {
        let encoder = opus::Encoder::new(24000, opus::Channels::Mono, opus::Application::Voip)?;
        // Not sure what the appropriate buffer size would be here.
        let out_pcm_buf = vec![0u8; 50_000];
        let out_pcm = std::collections::VecDeque::with_capacity(2 * OPUS_ENCODER_FRAME_SIZE);

        let all_data = Vec::new();
        let mut pw = ogg::PacketWriter::new(all_data);
        let mut head = Vec::new();
        crate::audio::write_opus_header(&mut head)?;
        pw.write_packet(head, 42, ogg::PacketWriteEndInfo::EndPage, 0)?;
        let mut tags = Vec::new();
        crate::audio::write_opus_tags(&mut tags)?;
        pw.write_packet(tags, 42, ogg::PacketWriteEndInfo::EndPage, 0)?;
        Ok(Self { pw, encoder, out_pcm, out_pcm_buf, total_data: 0, sender })
    }

    async fn send_error(&mut self, text: String) -> Result<()> {
        let msg: Vec<u8> = [&[MsgType::Error.to_u8()], text.as_bytes()].concat();
        let msg = ws::Message::Binary(msg.into());
        self.sender.send(msg).await?;
        Ok(())
    }

    async fn send_text(&mut self, text: String) -> Result<()> {
        let msg: Vec<u8> = [&[MsgType::Text.to_u8()], text.as_bytes()].concat();
        let msg = ws::Message::Binary(msg.into());
        self.sender.send(msg).await?;
        Ok(())
    }

    /// Wire: ColoredText (0x07) + color_id (4=model, 10=user) + utf8.
    async fn send_text_by_role(&mut self, text: String, role: TextRole) -> Result<()> {
        let color_id = match role {
            TextRole::Model => 4u8,
            TextRole::User => 10u8,
        };
        let msg: Vec<u8> = [&[MsgType::ColoredText.to_u8(), color_id], text.as_bytes()].concat();
        let msg = ws::Message::Binary(msg.into());
        self.sender.send(msg).await?;
        Ok(())
    }

    /// Wire: ColoredReferenceText (0x09) + color_id 4 + utf8.
    async fn send_reference_text(&mut self, text: String) -> Result<()> {
        const REFERENCE_COLOR_ID: u8 = 4;
        let msg: Vec<u8> =
            [&[MsgType::ColoredReferenceText.to_u8(), REFERENCE_COLOR_ID], text.as_bytes()]
                .concat();
        let msg = ws::Message::Binary(msg.into());
        self.sender.send(msg).await?;
        Ok(())
    }

    async fn send_ready(&mut self) -> Result<()> {
        // The payload is made of two fields.
        // 1. Protocol version (`u32`) - always 0 for now.
        // 2. Model version (`u32`).
        let msg: Vec<u8> = [&[MsgType::Handshake.to_u8()], [0u8; 8].as_slice()].concat();
        let msg = ws::Message::Binary(msg.into());
        self.sender.send(msg).await?;
        Ok(())
    }

    async fn send_metadata(&mut self, md: Box<MetaData>) -> Result<()> {
        let bytes = serde_json::to_vec(&md)?;
        let msg: Vec<u8> = [&[MsgType::Metadata.to_u8()], bytes.as_slice()].concat();
        let msg = ws::Message::Binary(msg.into());
        self.sender.send(msg).await?;
        Ok(())
    }

    async fn send_pcm(&mut self, pcm: Vec<f32>) -> Result<()> {
        self.out_pcm.extend(pcm.iter());
        self.total_data += pcm.len();
        let nchunks = self.out_pcm.len() / OPUS_ENCODER_FRAME_SIZE;
        for _chunk_id in 0..nchunks {
            let mut chunk = Vec::with_capacity(OPUS_ENCODER_FRAME_SIZE);
            for _i in 0..OPUS_ENCODER_FRAME_SIZE {
                let v = match self.out_pcm.pop_front() {
                    None => anyhow::bail!("unexpected err popping from pcms"),
                    Some(v) => v,
                };
                chunk.push(v)
            }
            let size = self.encoder.encode_float(&chunk, &mut self.out_pcm_buf)?;
            if size > 0 {
                let msg = self.out_pcm_buf[..size].to_vec();
                self.pw.write_packet(
                    msg,
                    42,
                    ogg::PacketWriteEndInfo::EndPage,
                    self.total_data as u64,
                )?
            } else {
                tracing::error!("OPUS SIZE 0")
            }
            let data = self.pw.inner_mut();
            if !data.is_empty() {
                let msg: Vec<u8> = [&[MsgType::Audio.to_u8()], data.as_slice()].concat();
                let msg = ws::Message::Binary(msg.into());
                self.sender.send(msg).await?;
                self.sender.flush().await?;
                data.clear();
            } else {
                tracing::error!("OGG SIZE 0")
            }
        }
        Ok(())
    }
}

pub struct StreamingModel {
    state_variant: AppStateVariant,
    device: candle::Device,
    config: moshi::lm_generate_multistream::Config,
    session_config: SessionConfig,
    rag_retrieval: std::sync::Arc<crate::rag_retrieval::RagRetrievalEndpoints>,
}

impl StreamingModel {
    fn run_with_state_asr(
        &self,
        state: &mut moshi::lm_generate_multistream::State,
        receiver: std::sync::mpsc::Receiver<Vec<f32>>,
        sender: tokio::sync::mpsc::UnboundedSender<StreamOut>,
        asr_delay_in_tokens: usize,
    ) -> Result<()> {
        use candle::IndexOp;

        let app_state = self.inner_state();

        let mut mimi = app_state.mimi_model.clone();
        let config = state.config().clone();

        mimi.reset_state();
        tracing::info!("processing loop");
        let mut prev_text_token = config.text_start_token;
        let mimi_device =
            if app_state.config.use_cpu_for_mimi { &candle::Device::Cpu } else { &self.device };
        mimi_device.synchronize()?;
        sender.send(StreamOut::Ready)?;
        while let Ok(in_pcm) = receiver.recv() {
            if in_pcm.is_empty() {
                continue;
            }
            let pcm_len = in_pcm.len();
            sender.send(StreamOut::InputPcm { pcm_len })?;
            let pcms = candle::Tensor::from_vec(in_pcm, (1, 1, pcm_len), mimi_device)?;
            let audio_tokens = mimi.encode_step(&pcms.into(), &().into())?;
            let audio_tokens = match audio_tokens.as_option() {
                None => continue,
                Some(audio_tokens) => audio_tokens,
            };
            let (_one, _codebooks, steps) = audio_tokens.dims3()?;

            for step in 0..steps {
                let codes = audio_tokens.i((0, .., step))?.to_vec1::<u32>()?;
                // For the ASR, we don't provide text tokens during the initial steps except the
                // initial one.
                if state.step_idx() > 0 && state.step_idx() < asr_delay_in_tokens {
                    let (t, _) = state.step_(None, &codes, None, None, None)?;
                    prev_text_token = t;
                } else {
                    sender.send(StreamOut::StepStart { step })?;
                    let text_token = state.step(prev_text_token, &codes, None, None)?;
                    sender.send(StreamOut::StepPostSampling { step })?;
                    if let Some(text) = app_state.text(prev_text_token, text_token, &config) {
                        sender.send(StreamOut::TextByRole { text, role: TextRole::Model })?;
                    }
                    prev_text_token = text_token;
                }
            }
        }
        tracing::info!("finished the processing loop");
        Ok(())
    }

    fn run_with_state(
        &self,
        state: &mut moshi::lm_generate_multistream::State,
        receiver: std::sync::mpsc::Receiver<Vec<f32>>,
        sender: tokio::sync::mpsc::UnboundedSender<StreamOut>,
    ) -> Result<()> {
        use candle::IndexOp;

        let app_state = self.inner_state();

        let mut mimi = app_state.mimi_model.clone();
        let config = state.config().clone();

        mimi.reset_state();
        tracing::info!("processing loop");
        let mut prev_text_token = config.text_start_token;
        let mut tensor_tokens = vec![];
        let mimi_device =
            if app_state.config.use_cpu_for_mimi { &candle::Device::Cpu } else { &self.device };
        mimi_device.synchronize()?;
        sender.send(StreamOut::Ready)?;
        while let Ok(in_pcm) = receiver.recv() {
            if in_pcm.is_empty() {
                continue;
            }
            let pcm_len = in_pcm.len();
            sender.send(StreamOut::InputPcm { pcm_len })?;
            let pcms = candle::Tensor::from_vec(in_pcm, (1, 1, pcm_len), mimi_device)?;
            let audio_tokens = mimi.encode_step(&pcms.into(), &().into())?;
            let audio_tokens = match audio_tokens.as_option() {
                None => continue,
                Some(audio_tokens) => audio_tokens,
            };
            let (_one, _codebooks, steps) = audio_tokens.dims3()?;

            for step in 0..steps {
                let codes = audio_tokens.i((0, .., step))?.to_vec1::<u32>()?;
                sender.send(StreamOut::StepStart { step })?;
                let text_token = state.step(prev_text_token, &codes, None, None)?;
                sender.send(StreamOut::StepPostSampling { step })?;
                if let Some(audio_tokens) = state.last_audio_tokens() {
                    let audio_tokens = {
                        let cb = app_state.config.mimi_num_codebooks;
                        candle::Tensor::from_slice(&audio_tokens[..cb], (1, cb, 1), mimi_device)?
                    };
                    tensor_tokens.push(audio_tokens.clone());
                    let pcm = mimi.decode_step(&audio_tokens.into(), &().into())?;
                    if let Some(pcm) = pcm.as_option() {
                        let pcm = pcm.i((0, 0))?.to_vec1::<f32>()?;
                        sender.send(StreamOut::Pcm { pcm })?;
                    }
                }
                if let Some(text) = app_state.text(prev_text_token, text_token, &config) {
                    sender.send(StreamOut::TextByRole { text, role: TextRole::Model })?;
                }
                prev_text_token = text_token;
            }
        }
        tracing::info!("finished the processing loop");
        Ok(())
    }

    fn run_with_state_mt(
        &self,
        state: &mut moshi::lm_generate_multistream::State,
        receiver: std::sync::mpsc::Receiver<Vec<f32>>,
        sender: tokio::sync::mpsc::UnboundedSender<StreamOut>,
    ) -> Result<()> {
        use candle::IndexOp;

        let app_state = self.inner_state();

        let mut mimi = app_state.mimi_model.clone();
        let config = state.config().clone();

        mimi.reset_state();
        tracing::info!("processing loop");
        let mut prev_text_token = config.text_start_token;
        let mut tensor_tokens = vec![];
        let (tx_i, rx_i) = std::sync::mpsc::channel::<(Vec<u32>, usize)>();
        let (tx_o, rx_o) = std::sync::mpsc::channel::<Vec<u32>>();
        let sender = Arc::new(sender);
        let status = std::thread::scope(|s| {
            s.spawn({
                let mut mimi = mimi.clone();
                let sender = sender.clone();
                move || {
                    'outer: while let Ok(in_pcm) = receiver.recv() {
                        if in_pcm.is_empty() {
                            continue;
                        }
                        let pcm_len = in_pcm.len();
                        sender.send(StreamOut::InputPcm { pcm_len })?;
                        let pcms = candle::Tensor::from_vec(
                            in_pcm,
                            (1, 1, pcm_len),
                            &candle::Device::Cpu,
                        )?;
                        let audio_tokens = mimi.encode_step(&pcms.into(), &().into())?;
                        let audio_tokens = match audio_tokens.as_option() {
                            None => continue,
                            Some(audio_tokens) => audio_tokens,
                        };
                        let (_one, _codebooks, steps) = audio_tokens.dims3()?;
                        for step in 0..steps {
                            let codes = audio_tokens.i((0, .., step))?.to_vec1::<u32>()?;
                            if tx_i.send((codes, step)).is_err() {
                                break 'outer;
                            }
                        }
                    }
                    Ok::<_, anyhow::Error>(())
                }
            });
            s.spawn({
                let cb = app_state.config.mimi_num_codebooks;
                let sender = sender.clone();
                move || {
                    while let Ok(audio_tokens) = rx_o.recv() {
                        let audio_tokens = {
                            candle::Tensor::from_slice(
                                &audio_tokens[..cb],
                                (1, cb, 1),
                                &candle::Device::Cpu,
                            )?
                        };
                        tensor_tokens.push(audio_tokens.clone());
                        let pcm = mimi.decode_step(&audio_tokens.into(), &().into())?;
                        if let Some(pcm) = pcm.as_option() {
                            let pcm = pcm.i((0, 0))?.to_vec1::<f32>()?;
                            sender.send(StreamOut::Pcm { pcm })?;
                        }
                    }
                    Ok::<_, anyhow::Error>(())
                }
            });
            sender.send(StreamOut::Ready)?;
            while let Ok((codes, step)) = rx_i.recv() {
                tracing::info!("received codes");
                sender.send(StreamOut::StepStart { step })?;
                let text_token = state.step(prev_text_token, &codes, None, None);
                sender.send(StreamOut::StepPostSampling { step })?;
                tracing::info!(?text_token, "codes");
                if text_token.is_err() {
                    drop(rx_i);
                    drop(tx_o);
                    break;
                }
                let text_token = text_token?;
                if let Some(audio_tokens) = state.last_audio_tokens() {
                    tx_o.send(audio_tokens)?
                }
                if let Some(text) = app_state.text(prev_text_token, text_token, &config) {
                    sender.send(StreamOut::TextByRole { text, role: TextRole::Model })?;
                }
                prev_text_token = text_token;
            }
            Ok::<_, anyhow::Error>(())
        });
        match status {
            Ok(()) => tracing::info!("finished the processing loop"),
            Err(err) => tracing::error!(?err, "processing loop"),
        };
        Ok(())
    }

    pub fn new(state: &AppState, session_config: SessionConfigReq) -> Self {
        let config = match state.config.lm_config.as_ref() {
            None => moshi::lm_generate_multistream::Config::v0_1(),
            Some(config) => config.clone(),
        };
        let session_config = session_config.into_session_config();
        let rag_retrieval = crate::rag_retrieval::RagRetrievalEndpoints::from_profiles(
            state.config.rag_llm_profiles.clone(),
            state.config.rag_llm_default_id.as_deref(),
        );
        Self {
            state_variant: AppStateVariant::Standard(state.clone()),
            device: state.device.clone(),
            config,
            session_config,
            rag_retrieval,
        }
    }

    /// Build a StreamingModel that runs the RAG loop.
    pub fn new_rag(
        rag_state: &std::sync::Arc<AppStateRag>,
        session_config: SessionConfigReq,
    ) -> Self {
        let mut sm = Self::new(&rag_state.inner, session_config);
        sm.state_variant = AppStateVariant::Rag(rag_state.clone());
        sm
    }

    /// Access the main (inner) state regardless of Standard or RAG or Batched.
    fn inner_state(&self) -> &AppStateInner {
        match &self.state_variant {
            AppStateVariant::Standard(inner) => inner,
            AppStateVariant::Rag(rag) => rag.inner.as_ref(),
            AppStateVariant::Batched(batched) => batched.inner.as_ref(),
        }
    }

    /// Shared metadata send for both Standard and RAG.
    fn send_initial_metadata(
        &self,
        sender: &tokio::sync::mpsc::UnboundedSender<StreamOut>,
    ) -> Result<()> {
        let inner = self.inner_state();
        let (repetition_penalty_context, repetition_penalty) =
            self.session_config.repetition_penalty.unwrap_or((32, 1.));
        let (retrieval_backends, retrieval_backend_default) =
            if inner.config.rag_llm_profiles.as_ref().is_some_and(|p| p.len() >= 2) {
                let profs = inner.config.rag_llm_profiles.as_ref().unwrap();
                let backends: Vec<RetrievalBackendMeta> =
                    profs.iter().map(|p| RetrievalBackendMeta { id: p.id.clone() }).collect();
                (Some(backends), self.rag_retrieval.default_id_for_ui())
            } else {
                (None, None)
            };
        let metadata = MetaData {
            text_temperature: self.session_config.text_temperature,
            text_topk: self.session_config.text_topk,
            audio_temperature: self.session_config.audio_temperature,
            audio_topk: self.session_config.audio_topk,
            pad_mult: self.session_config.pad_mult.unwrap_or(0.),
            repetition_penalty,
            repetition_penalty_context,
            lm_model_file: inner.config.lm_model_file.clone(),
            mimi_model_file: inner.config.mimi_model_file.clone(),
            build_info: crate::utils::BuildInfo::new(),
            instance_name: inner.config.instance_name.clone(),
            retrieval_backends,
            retrieval_backend_default,
        };
        sender.send(StreamOut::MetaData { metadata: Box::new(metadata) })?;
        Ok(())
    }

    /// Build main or STT LM state with shared LogitsProcessor setup.
    fn build_core_lm_state(
        &self,
        lm_model: moshi::lm::LmModel,
        gen_config: &moshi::lm_generate_multistream::Config,
        session_config: &SessionConfig,
    ) -> Result<moshi::lm_generate_multistream::State> {
        let audio_lp = candle_transformers::generation::LogitsProcessor::from_sampling(
            session_config.audio_seed,
            candle_transformers::generation::Sampling::TopK {
                k: session_config.audio_topk,
                temperature: session_config.audio_temperature,
            },
        );
        let text_lp = candle_transformers::generation::LogitsProcessor::from_sampling(
            session_config.text_seed,
            candle_transformers::generation::Sampling::TopK {
                k: session_config.text_topk,
                temperature: session_config.text_temperature,
            },
        );
        Ok(moshi::lm_generate_multistream::State::new(
            lm_model,
            session_config.max_steps,
            audio_lp,
            text_lp,
            session_config.pad_mult,
            session_config.repetition_penalty,
            None,
            gen_config.clone(),
        ))
    }

    pub fn run(
        &self,
        receiver: std::sync::mpsc::Receiver<Vec<f32>>,
        sender: tokio::sync::mpsc::UnboundedSender<StreamOut>,
        _addr: Option<String>,
    ) -> Result<()> {
        let app_state = self.inner_state();
        self.send_initial_metadata(&sender)?;
        let mut state = self.build_core_lm_state(
            app_state.lm_model.clone(),
            &self.config,
            &self.session_config,
        )?;

        // We want to log the output even if the run function returns an error.
        if app_state.config.use_cpu_for_mimi {
            self.run_with_state_mt(&mut state, receiver, sender)
        } else if let Some(asr_delay_in_tokens) = app_state.config.asr_delay_in_tokens {
            self.run_with_state_asr(&mut state, receiver, sender, asr_delay_in_tokens)
        } else {
            self.run_with_state(&mut state, receiver, sender)
        }
    }

    /// RAG loop. Only valid when constructed with `new_rag`.
    pub fn run_rag(
        &self,
        receiver: std::sync::mpsc::Receiver<Vec<f32>>,
        sender: tokio::sync::mpsc::UnboundedSender<StreamOut>,
        _addr: Option<String>,
    ) -> Result<()> {
        let rag = match &self.state_variant {
            AppStateVariant::Rag(r) => r,
            AppStateVariant::Standard(_) => panic!("run_rag requires StreamingModel::new_rag"),
            AppStateVariant::Batched(_) => panic!("run_rag not used with Batched"),
        };
        self.send_initial_metadata(&sender)?;
        let mut state = self.build_core_lm_state(
            self.inner_state().lm_model.clone(),
            &self.config,
            &self.session_config,
        )?;
        let stt_config = moshi::lm_generate_multistream::Config::v0_1_stt();
        let mut state_stt = self.build_core_lm_state(
            rag.stt_lm_model.clone(),
            &stt_config,
            &stt_sampling_config(),
        )?;
        let rag_manager = match self.inner_state().config.rag_token_id {
            Some(_) => Some(crate::rag_manager::RagManager::new(self.rag_retrieval.clone())),
            _ => None,
        };
        let rag_token_id = self.inner_state().config.rag_token_id;
        let result = self.run_with_state_rag(
            &mut state,
            &mut state_stt,
            receiver,
            sender,
            rag,
            rag_manager.as_ref(),
            rag_token_id,
        );
        if let Some(ref r) = rag_manager {
            r.cancel_pending();
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn run_with_state_rag(
        &self,
        state: &mut moshi::lm_generate_multistream::State,
        state_stt: &mut moshi::lm_generate_multistream::State,
        receiver: std::sync::mpsc::Receiver<Vec<f32>>,
        sender: tokio::sync::mpsc::UnboundedSender<StreamOut>,
        rag: &std::sync::Arc<AppStateRag>,
        rag_manager: Option<&crate::rag_manager::RagManager>,
        rag_token_id: Option<u32>,
    ) -> Result<()> {
        use candle::IndexOp;

        let mut mimi = self.inner_state().mimi_model.clone();
        let mut mimi_stt = rag.stt_mimi_model.clone();
        let mimi_device = if self.inner_state().config.use_cpu_for_mimi {
            &candle::Device::Cpu
        } else {
            &self.device
        };

        let frame_rate = mimi.config().frame_rate;
        let frame_size = (mimi.config().sample_rate / frame_rate).ceil() as usize;
        let stt_wait_secs =
            self.inner_state().config.stt_wait_time.map(|t| t as f64).unwrap_or(0.0);
        let vad_wait_steps = (stt_wait_secs * frame_rate).ceil() as usize;
        let vad_window = self.inner_state().config.vad_window_size.unwrap_or(4);
        let vad_threshold = self.inner_state().config.vad_threshold.unwrap_or(0.5);
        let (init_speaker, first_speaker_str): (crate::turn_manager::TextRole, Option<&str>) =
            match self.inner_state().config.init_active_speaker.as_deref() {
                Some("user") => (crate::turn_manager::TextRole::User, Some("user")),
                Some("model") => (crate::turn_manager::TextRole::Model, Some("model")),
                _ => (crate::turn_manager::TextRole::Model, None),
            };
        let turn_manager = Arc::new(Mutex::new(crate::turn_manager::TurnManager::new(
            vad_window,
            vad_threshold,
            vad_wait_steps,
            init_speaker,
        )));

        mimi.reset_state();
        mimi_stt.reset_state();
        mimi_device.synchronize()?;

        let stt_config = moshi::lm_generate_multistream::Config::v0_1_stt();
        let mut prev_text_token = self.config.text_start_token;
        let mut prev_text_token_stt = stt_config.text_start_token;
        let mut skip_frames = 1u32;

        let power_threshold = self.inner_state().config.power_threshold;
        let input_cb_main = self.config.input_audio_codebooks;
        let input_cb_stt = stt_config.input_audio_codebooks;

        sender.send(StreamOut::Ready)?;

        state.set_prepend_condition_lut("first_speaker", first_speaker_str.unwrap());

        while let Ok(in_pcm) = receiver.recv() {
            if in_pcm.is_empty() {
                continue;
            }
            let pcm_len = in_pcm.len();
            sender.send(StreamOut::InputPcm { pcm_len })?;
            let pcms = candle::Tensor::from_vec(in_pcm.clone(), (1, 1, pcm_len), mimi_device)?;

            // Chunk-wise filtering (per frame): zero each frame_size window when its RMS (dB) is below threshold.
            let pcms_for_main = match power_threshold {
                None => pcms.clone(),
                Some(thresh_db) => {
                    let thresh_db = thresh_db as f32;
                    let mut data = pcms.flatten_all()?.to_vec1::<f32>()?;
                    apply_power_threshold_frames(&mut data, frame_size, thresh_db);
                    candle::Tensor::from_vec(data, (1, 1, pcm_len), mimi_device)?
                }
            };
            let pcms_for_stt = &pcms;

            // Always run both encoders on this chunk so they stay aligned.
            let audio_tokens_main = mimi.encode_step(&pcms_for_main.into(), &().into())?;
            let audio_tokens_stt =
                mimi_stt.encode_step(&pcms_for_stt.clone().into(), &().into())?;
            let (audio_tokens_main, audio_tokens_stt) =
                match (audio_tokens_main.as_option(), audio_tokens_stt.as_option()) {
                    (Some(m), Some(s)) => (m, s.clone()),
                    _ => continue, // one or both still buffering; next chunk will keep them in sync
                };
            let steps_main = audio_tokens_main.dims3()?.2;
            let steps_stt = audio_tokens_stt.dims3()?.2;
            let steps = steps_main.min(steps_stt);
            if steps_main != steps_stt {
                tracing::warn!(
                    "Main and STT step count mismatch for same PCM chunk: main={} stt={}, using min",
                    steps_main,
                    steps_stt
                );
            }

            for step in 0..steps {
                if skip_frames > 0 {
                    skip_frames -= 1;
                    continue;
                }

                if let Some(rag_mgr) = rag_manager {
                    if let Some(ref_text) = rag_mgr.try_recv_result() {
                        if ref_text.trim().is_empty() {
                            sender.send(StreamOut::TextByRole {
                                text: "[RET_FAILED]".to_string(),
                                role: TextRole::Model,
                            })?;
                        } else {
                            sender.send(StreamOut::ReferenceText { text: ref_text.clone() })?;
                        }
                        state.set_streaming_sum_condition("reference_with_time", &ref_text);
                    }
                }

                let codes_main = audio_tokens_main
                    .i((0, 0..input_cb_main, step))?
                    .flatten_all()?
                    .to_vec1::<u32>()?;
                let text_token = state.step(prev_text_token, &codes_main, None, None)?;
                let model_text = decode_text_piece(
                    &self.inner_state().text_tokenizer,
                    prev_text_token,
                    text_token,
                    &self.config,
                );
                let (text_token_stt, extra_heads) = {
                    let codes_stt = audio_tokens_stt
                        .i((0, 0..input_cb_stt, step))?
                        .flatten_all()?
                        .to_vec1::<u32>()?;
                    state_stt.step_with_extra_heads(
                        prev_text_token_stt,
                        &codes_stt,
                        None,
                        None,
                        None,
                    )?
                };

                let vad_value = extra_heads
                    .get(VAD_HEAD_INDEX)
                    .and_then(|t: &candle::Tensor| {
                        let t = t.to_dtype(candle::DType::F32).ok()?;
                        let t = candle_nn::ops::softmax_last_dim(&t).ok()?;
                        t.i((0, 0, 0)).ok()?.to_vec0::<f32>().ok()
                    })
                    .unwrap_or(0.0);
                turn_manager.lock().unwrap().update_vad(vad_value);

                if rag_token_id == Some(text_token) {
                    if let Some(rag_mgr) = rag_manager {
                        let tm_ptr = Arc::clone(&turn_manager);
                        rag_mgr.trigger_background_generation(
                            stt_wait_secs,
                            self.inner_state().config.rag_timeout,
                            move || tm_ptr.lock().unwrap().get_context().to_string(),
                        );
                    }
                    sender.send(StreamOut::TextByRole {
                        text: "[RET]".to_string(),
                        role: crate::turn_manager::TextRole::Model,
                    })?;
                }

                if let Some(audio_tokens) = state.last_audio_tokens() {
                    let cb = self.inner_state().config.mimi_num_codebooks;
                    let slice: Vec<u32> = audio_tokens[..cb].to_vec();
                    let audio_tensor = candle::Tensor::from_slice(&slice, (1, cb, 1), mimi_device)?;
                    let pcm = mimi.decode_step(&audio_tensor.into(), &().into())?;
                    if let Some(pcm) = pcm.as_option() {
                        let pcm = pcm.i((0, 0))?.to_vec1::<f32>()?;
                        sender.send(StreamOut::Pcm { pcm })?;
                    }
                }

                let user_text = decode_text_piece(
                    &rag.stt_text_tokenizer,
                    prev_text_token_stt,
                    text_token_stt,
                    &stt_config,
                );
                let outputs = turn_manager
                    .lock()
                    .unwrap()
                    .handle_spoken_text(model_text.as_deref(), user_text.as_deref());
                for (text, role) in outputs {
                    sender.send(StreamOut::TextByRole { text, role })?;
                }

                prev_text_token = text_token;
                prev_text_token_stt = text_token_stt;
            }
        }
        tracing::info!("RAG processing loop finished");
        Ok(())
    }
}

const VAD_HEAD_INDEX: usize = 2;
const BATCH_VAD_HEAD_INDEX: usize = 2;

type Handle = tokio::task::JoinHandle<Result<()>>;

fn spawn_recv_loops(
    mut receiver: SplitStream<ws::WebSocket>,
    sender: std::sync::mpsc::Sender<Vec<f32>>,
    retrieval_switch: Option<Arc<crate::rag_retrieval::RagRetrievalEndpoints>>,
) -> Result<(Handle, Handle)> {
    use tokio::io::AsyncWriteExt;
    use tokio::time::{timeout, Duration};

    let (mut tx, rx) = tokio::io::duplex(100_000);
    let mut pr = ogg::reading::async_api::PacketReader::new(rx);
    let mut decoder = opus::Decoder::new(24000, opus::Channels::Mono)?;
    let handle1 = tokio::spawn({
        async move {
            loop {
                // Enforce an inactivity timeout on incoming WebSocket messages.
                let next_msg = timeout(Duration::from_secs(10), receiver.next()).await;
                let maybe_item = match next_msg {
                    Ok(v) => v,
                    Err(_) => {
                        tracing::warn!("closing websocket due to 10s of inactivity (recv loop)");
                        break;
                    }
                };
                match maybe_item {
                    None => {
                        // The close logic is that if this loop exits, then tx gets dropped so pr
                        // gets closed and the second thread gets dropped resulting in sender
                        // getting dropped.
                        break;
                    }
                    Some(v) => {
                        let v = v?.into_data();
                        if v.is_empty() {
                            continue;
                        }
                        let msg_type = MsgType::from_u8(v[0])?;
                        match msg_type {
                            MsgType::Metadata => {
                                if let Some(ref ep) = retrieval_switch {
                                    if v.len() > 1 {
                                        if let Ok(val) =
                                            serde_json::from_slice::<serde_json::Value>(&v[1..])
                                        {
                                            if let Some(id) = val
                                                .get("retrieval_backend_id")
                                                .and_then(|x| x.as_str())
                                            {
                                                if let Err(e) = ep.set_active_id(id) {
                                                    tracing::warn!(%e, "client retrieval switch rejected");
                                                } else {
                                                    tracing::info!(
                                                        ?id,
                                                        "retrieval backend switched"
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            MsgType::Handshake => {}
                            MsgType::Control => {}
                            MsgType::Text => {}
                            MsgType::Error => {}
                            MsgType::Ping => {}
                            MsgType::Audio => tx.write_all(&v[1..]).await?,
                            MsgType::ColoredText => {}
                            MsgType::ReferenceText => {}
                            MsgType::ColoredReferenceText => {}
                        }
                    }
                }
            }
            tracing::info!("socket closed");
            Ok::<_, anyhow::Error>(())
        }
    });
    let handle2 = tokio::spawn(async move {
        // TODO: dynamic sizing?
        let mut pcm_buf = vec![0f32; 24_000 * 10];
        let mut size_in_buf = 0;
        loop {
            match pr.next().await {
                None => {
                    break;
                }
                Some(packet) => {
                    let packet = packet?;
                    if packet.data.starts_with(b"OpusHead") || packet.data.starts_with(b"OpusTags")
                    {
                        continue;
                    }
                    let read_size = decoder.decode_float(
                        &packet.data,
                        &mut pcm_buf[size_in_buf..],
                        /* Forward Error Correction */ false,
                    )?;
                    size_in_buf += read_size;
                    // flush the data every half timestep
                    if size_in_buf >= 24_000 / 25 {
                        if sender.send(pcm_buf[..size_in_buf].to_vec()).is_err() {
                            break;
                        }
                        size_in_buf = 0;
                    }
                }
            }
        }
        tracing::info!("decoder closed");
        Ok::<_, anyhow::Error>(())
    });
    Ok((handle1, handle2))
}

/// Like spawn_recv_loops but sends InMsg::Init once then InMsg::Audio { pcm } for batched channel pool.
fn spawn_recv_loops_in_msg(
    mut receiver: SplitStream<ws::WebSocket>,
    in_msg_tx: std::sync::mpsc::Sender<crate::batched_channels::InMsg>,
    retrieval_switch: Option<Arc<crate::rag_retrieval::RagRetrievalEndpoints>>,
    slot_id: usize,
) -> Result<(Handle, Handle)> {
    use tokio::io::AsyncWriteExt;
    use tokio::time::{timeout, Duration};

    let (mut tx, rx) = tokio::io::duplex(100_000);
    let mut pr = ogg::reading::async_api::PacketReader::new(rx);
    let mut decoder = opus::Decoder::new(24000, opus::Channels::Mono)?;
    let handle1 = tokio::spawn({
        async move {
            loop {
                // Enforce an inactivity timeout on incoming WebSocket messages for batched mode.
                let next_msg = timeout(Duration::from_secs(10), receiver.next()).await;
                let maybe_item = match next_msg {
                    Ok(v) => v,
                    Err(_) => {
                        tracing::warn!(
                            "closing websocket due to 10s of inactivity (batched recv loop)"
                        );
                        break;
                    }
                };
                match maybe_item {
                    None => break,
                    Some(v) => {
                        let v = v?.into_data();
                        if v.is_empty() {
                            continue;
                        }
                        let msg_type = MsgType::from_u8(v[0])?;
                        match msg_type {
                            MsgType::Metadata => {
                                if let Some(ref ep) = retrieval_switch {
                                    if v.len() > 1 {
                                        if let Ok(val) =
                                            serde_json::from_slice::<serde_json::Value>(&v[1..])
                                        {
                                            if let Some(id) = val
                                                .get("retrieval_backend_id")
                                                .and_then(|x| x.as_str())
                                            {
                                                if let Err(e) =
                                                    ep.set_active_id_for_slot(slot_id, id)
                                                {
                                                    tracing::warn!(%e, "client retrieval switch rejected (batched)");
                                                } else {
                                                    tracing::info!(
                                                        ?id,
                                                        slot_id,
                                                        "retrieval backend switched (batched)"
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            MsgType::Handshake => {}
                            MsgType::Control => {}
                            MsgType::Text => {}
                            MsgType::Error => {}
                            MsgType::Ping => {}
                            MsgType::Audio => tx.write_all(&v[1..]).await?,
                            MsgType::ColoredText => {}
                            MsgType::ReferenceText => {}
                            MsgType::ColoredReferenceText => {}
                        }
                    }
                }
            }
            tracing::info!("socket closed (batched recv)");
            Ok::<_, anyhow::Error>(())
        }
    });
    let handle2 = tokio::spawn(async move {
        if in_msg_tx.send(crate::batched_channels::InMsg::Init).is_err() {
            return Ok::<_, anyhow::Error>(());
        }
        let mut pcm_buf = vec![0f32; 24_000 * 10];
        let mut size_in_buf = 0;
        loop {
            match pr.next().await {
                None => break,
                Some(packet) => {
                    let packet = packet?;
                    if packet.data.starts_with(b"OpusHead") || packet.data.starts_with(b"OpusTags")
                    {
                        continue;
                    }
                    let read_size =
                        decoder.decode_float(&packet.data, &mut pcm_buf[size_in_buf..], false)?;
                    size_in_buf += read_size;
                    if size_in_buf >= 24_000 / 25 {
                        if in_msg_tx
                            .send(crate::batched_channels::InMsg::Audio {
                                pcm: pcm_buf[..size_in_buf].to_vec(),
                            })
                            .is_err()
                        {
                            break;
                        }
                        size_in_buf = 0;
                    }
                }
            }
        }
        tracing::info!("decoder closed (batched)");
        Ok(())
    });
    Ok((handle1, handle2))
}

/// Handle one WebSocket connection when using batched pool (batch_size > 1). Takes a slot and runs recv/send loops.
pub async fn handle_socket_batched(socket: ws::WebSocket, runner: Arc<BatchedState>) -> Result<()> {
    tracing::info!("accepted websocket connection (batched)");
    let (sender_ws, receiver) = socket.split();
    let mut sender = MsgSender::new(sender_ws)?;
    let Some((batch_idx, in_tx, out_rx)) = runner.pool.take_slot() else {
        tracing::warn!("no free channel, sending batch_full error and closing connection");
        // Notify client that the batch is full so it can transition to a waiting page.
        // Best-effort: ignore send errors and just close.
        let _ = sender.send_error("batch_full".to_string()).await;
        return Ok(());
    };
    let session_config = default_session_config_batched();
    let (repetition_penalty_context, repetition_penalty) =
        session_config.repetition_penalty.unwrap_or((32, 1.));
    runner.rag_retrieval.reset_active_slot(batch_idx);
    let (retrieval_backends, retrieval_backend_default) =
        if runner.inner.config.rag_llm_profiles.as_ref().is_some_and(|p| p.len() >= 2) {
            let profs = runner.inner.config.rag_llm_profiles.as_ref().unwrap();
            let backends: Vec<RetrievalBackendMeta> =
                profs.iter().map(|p| RetrievalBackendMeta { id: p.id.clone() }).collect();
            (Some(backends), runner.rag_retrieval.default_id_for_ui_slot(batch_idx))
        } else {
            (None, None)
        };
    let metadata = MetaData {
        text_temperature: session_config.text_temperature,
        text_topk: session_config.text_topk,
        audio_temperature: session_config.audio_temperature,
        audio_topk: session_config.audio_topk,
        pad_mult: session_config.pad_mult.unwrap_or(0.),
        repetition_penalty,
        repetition_penalty_context,
        lm_model_file: runner.inner.config.lm_model_file.clone(),
        mimi_model_file: runner.inner.config.mimi_model_file.clone(),
        build_info: crate::utils::BuildInfo::new(),
        instance_name: runner.inner.config.instance_name.clone(),
        retrieval_backends,
        retrieval_backend_default,
    };
    if runner
        .pool
        .send_to_slot(batch_idx, StreamOut::MetaData { metadata: Box::new(metadata) })
        .is_err()
    {
        tracing::warn!("failed to send metadata to slot {}", batch_idx);
    }
    let (loop1, loop2) =
        spawn_recv_loops_in_msg(receiver, in_tx, Some(runner.rag_retrieval.clone()), batch_idx)?;
    tokio::task::spawn(sender_loop(out_rx, sender));
    let (r1, r2) = (loop1.await?, loop2.await?);
    r1?;
    r2?;
    Ok(())
}

async fn sender_loop(
    mut stream_out_rx: tokio::sync::mpsc::UnboundedReceiver<StreamOut>,
    mut sender: MsgSender,
) -> Result<()> {
    // It is important for the recv here to be an async enabled one. Otherwise this could lead
    // to some weird deadlocks.
    while let Some(v) = stream_out_rx.recv().await {
        match v {
            StreamOut::Pcm { pcm } => sender.send_pcm(pcm).await?,
            StreamOut::Ready => sender.send_ready().await?,
            StreamOut::MetaData { metadata } => sender.send_metadata(metadata).await?,
            StreamOut::Text { text } => sender.send_text(text).await?,
            StreamOut::TextByRole { text, role } => sender.send_text_by_role(text, role).await?,
            StreamOut::ReferenceText { text } => sender.send_reference_text(text).await?,
            StreamOut::InputPcm { .. }
            | StreamOut::StepStart { .. }
            | StreamOut::StepPostSampling { .. } => {}
        }
    }
    Ok::<_, anyhow::Error>(())
}

/// Shared interface for standard and RAG streaming so one socket handler can serve both.
pub trait StreamingRunner: Send {
    fn run_stream(
        &self,
        receiver: std::sync::mpsc::Receiver<Vec<f32>>,
        sender: tokio::sync::mpsc::UnboundedSender<StreamOut>,
        addr: Option<String>,
    ) -> Result<()>;

    /// When set, client `metadata` messages may switch the active retrieval LLM profile.
    fn retrieval_switch_endpoints(
        &self,
    ) -> Option<Arc<crate::rag_retrieval::RagRetrievalEndpoints>> {
        None
    }
}

impl StreamingRunner for StreamingModel {
    fn run_stream(
        &self,
        receiver: std::sync::mpsc::Receiver<Vec<f32>>,
        sender: tokio::sync::mpsc::UnboundedSender<StreamOut>,
        addr: Option<String>,
    ) -> Result<()> {
        match &self.state_variant {
            AppStateVariant::Standard(_) => self.run(receiver, sender, addr),
            AppStateVariant::Rag(_) => self.run_rag(receiver, sender, addr),
            AppStateVariant::Batched(_) => panic!("run_stream not used with Batched"),
        }
    }

    fn retrieval_switch_endpoints(
        &self,
    ) -> Option<Arc<crate::rag_retrieval::RagRetrievalEndpoints>> {
        Some(self.rag_retrieval.clone())
    }
}

pub async fn handle_socket<T: StreamingRunner + Send + 'static>(
    socket: ws::WebSocket,
    sm: T,
    addr: Option<String>,
) -> Result<()> {
    tracing::info!("accepted websocket connection");
    let (sender, receiver) = socket.split();
    let sender = MsgSender::new(sender)?;

    tracing::info!("starting streaming");

    let (in_pcm_tx, in_pcm_rx) = std::sync::mpsc::channel();
    let (stream_out_tx, stream_out_rx) = tokio::sync::mpsc::unbounded_channel();
    let (loop1, loop2) = spawn_recv_loops(receiver, in_pcm_tx, sm.retrieval_switch_endpoints())?;
    std::thread::spawn(move || {
        if let Err(err) = sm.run_stream(in_pcm_rx, stream_out_tx, addr) {
            tracing::error!("{err}")
        }
    });
    tokio::task::spawn(sender_loop(stream_out_rx, sender));
    let (r1, r2) = (loop1.await?, loop2.await?);
    r1?;
    r2?;
    Ok(())
}
