// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::Result;
use axum::extract::ws;
use std::str::FromStr;
use std::sync::Arc;

use crate::{stream_both, StandaloneArgs};

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    cert_dir: String,
    pub static_dir: String,
    addr: String,
    port: u16,
    #[serde(default = "default_true")]
    use_https: bool,

    #[serde(flatten)]
    pub stream: stream_both::Config,
}

fn default_true() -> bool {
    true
}

impl Config {
    pub fn load<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let config = std::fs::read_to_string(p)?;
        let mut config: Self = serde_json::from_str(&config)?;
        config.static_dir = crate::utils::replace_env_vars(&config.static_dir);
        config.cert_dir = crate::utils::replace_env_vars(&config.cert_dir);
        config.stream.log_dir = crate::utils::replace_env_vars(&config.stream.log_dir);
        config.stream.text_tokenizer_file =
            crate::utils::replace_env_vars(&config.stream.text_tokenizer_file);
        config.stream.mimi_model_file =
            crate::utils::replace_env_vars(&config.stream.mimi_model_file);
        config.stream.lm_model_file = crate::utils::replace_env_vars(&config.stream.lm_model_file);
        if let Some(ref mut s) = config.stream.stt_lm_model_file {
            *s = crate::utils::replace_env_vars(s);
        }
        if let Some(ref mut s) = config.stream.stt_text_tokenizer_file {
            *s = crate::utils::replace_env_vars(s);
        }
        if let Some(ref mut s) = config.stream.stt_mimi_model_file {
            *s = crate::utils::replace_env_vars(s);
        }
        if let Some(ref mut s) = config.stream.arc_encoder_tokenizer_path {
            *s = crate::utils::replace_env_vars(s);
        }
        if let Some(ref mut s) = config.stream.arc_encoder_model_file {
            *s = crate::utils::replace_env_vars(s);
        }
        crate::stream_both::parse_retrieval_llms_json(&mut config.stream)?;
        Ok(config)
    }

    pub fn cert_file(&self, name: &str) -> std::path::PathBuf {
        let cert_dir = std::path::PathBuf::from(&self.cert_dir);
        cert_dir.join(name)
    }
}

pub(crate) fn create_device(cpu: bool, gpu_id: usize) -> Result<candle::Device> {
    use candle::Device;
    if cpu {
        Ok(Device::Cpu)
    } else if candle::utils::cuda_is_available() {
        Ok(Device::new_cuda(gpu_id)?)
    } else if candle::utils::metal_is_available() {
        Ok(Device::new_metal(gpu_id)?)
    } else {
        Ok(Device::Cpu)
    }
}

impl stream_both::AppStateInner {
    pub fn new(args: &StandaloneArgs, config: &stream_both::Config) -> Result<Self> {
        let device = create_device(args.cpu, config.moshi_gpu_id)?;
        tracing::info!(
            "Loading Moshi LM on GPU {} (config moshi_gpu_id={}, device={:?})",
            config.moshi_gpu_id,
            config.moshi_gpu_id,
            device
        );

        let stt_device = create_device(args.cpu, config.stt_gpu_id)?;
        tracing::info!(
            "STT models will be loaded on GPU {} (config stt_gpu_id={}, stt_device={:?})",
            config.stt_gpu_id,
            config.stt_gpu_id,
            stt_device
        );

        let dtype = if device.is_cuda() { candle::DType::BF16 } else { candle::DType::F32 };
        let batch_size = if config.batch_size > 1 { config.batch_size } else { 1 };
        // Uses config.use_rag() to choose LM loader.
        let lm_model = if config.use_rag() {
            let arc_encoder_tokenizer_path = config
                .arc_encoder_tokenizer_path
                .as_deref()
                .expect("RAG config requires arc_encoder_tokenizer_path");
            let arc_encoder_weights =
                config.arc_encoder_model_file.as_deref().map(std::path::Path::new);
            moshi::lm::load_streaming_rag_batched(
                batch_size,
                &config.lm_model_file,
                dtype,
                &device,
                arc_encoder_tokenizer_path,
                arc_encoder_weights,
            )?
        } else {
            moshi::lm::load_streaming_batched(batch_size, &config.lm_model_file, dtype, &device)?
        };
        let mimi_device = if config.use_cpu_for_mimi { &candle::Device::Cpu } else { &device };
        tracing::info!(
            "Loading Mimi audio codec on {:?} (use_cpu_for_mimi={})",
            mimi_device,
            config.use_cpu_for_mimi
        );
        let mimi_model = moshi::mimi::load(
            &config.mimi_model_file,
            Some(config.mimi_num_codebooks),
            mimi_device,
        )?;
        let text_tokenizer =
            sentencepiece::SentencePieceProcessor::open(&config.text_tokenizer_file)?;
        // Warm-up code.
        {
            let mut lm_model = lm_model.clone();
            if batch_size <= 1 {
                let (_v, ys) =
                    lm_model.forward(None, vec![None; config.mimi_num_codebooks], &().into())?;
                let mut lp = candle_transformers::generation::LogitsProcessor::new(123, None, None);
                let _ = lm_model.depformer_sample(&ys, None, &[], &mut lp)?;
            } else {
                // Batched warm-up: one forward with all slots active, then one batched DepFormer step.
                let mask = moshi::StreamMask::new(vec![true; config.batch_size], &device)?;
                let start = lm_model.text_start_token();
                let text_ids = candle::Tensor::from_vec(
                    vec![start; config.batch_size],
                    (config.batch_size, 1),
                    &device,
                )?;
                let (_logits, ys) = lm_model.forward(
                    Some(text_ids),
                    vec![None; config.mimi_num_codebooks],
                    &mask,
                )?;
                let mut audio_lp: Vec<_> = (0..config.batch_size)
                    .map(|_| candle_transformers::generation::LogitsProcessor::new(123, None, None))
                    .collect();
                let text_tokens: Vec<Option<u32>> =
                    (0..config.batch_size).map(|_| Some(start)).collect();
                let forced: Vec<Vec<Option<u32>>> =
                    (0..config.batch_size).map(|_| vec![]).collect();
                let _ = lm_model.depformer_sample_batched(
                    &ys,
                    &text_tokens,
                    &forced,
                    &mask,
                    &mut audio_lp,
                )?;
            }
            let mut mimi_model = mimi_model.clone();
            let mimi_config = mimi_model.config();
            let frame_length = (mimi_config.sample_rate / mimi_config.frame_rate).ceil() as usize;
            let fake_pcm =
                candle::Tensor::zeros((1, 1, frame_length), candle::DType::F32, mimi_device)?;
            let codes = mimi_model.encode_step(&fake_pcm.into(), &().into())?;
            let ys = mimi_model.decode_step(&codes, &().into())?;
            if ys.as_option().is_none() {
                anyhow::bail!("Expected Mimi to output some stuff, but nothing came out.");
            }
            device.synchronize()?;
            tracing::info!("model is ready to roll!");
        }
        Ok(Self {
            lm_model,
            mimi_model,
            text_tokenizer,
            device,
            stt_device,
            config: config.clone(),
        })
    }
}

impl stream_both::AppStateRag {
    pub fn new(args: &StandaloneArgs, config: &stream_both::Config) -> Result<Self> {
        if !config.use_stt() {
            anyhow::bail!("RAG config requires stt_lm_model_file, stt_text_tokenizer_file, stt_mimi_model_file");
        }
        // Build main state first.
        let inner = stream_both::AppStateInner::new(args, config)?;
        let inner = std::sync::Arc::new(inner);

        // Determine device for STT models based on stt_gpu_id
        let stt_device = create_device(args.cpu, config.stt_gpu_id)?;
        let stt_dtype = if stt_device.is_cuda() { candle::DType::BF16 } else { candle::DType::F32 };

        let stt_mimi_device =
            if config.use_cpu_for_mimi { &candle::Device::Cpu } else { &stt_device };

        let stt_lm_file = config.stt_lm_model_file.as_ref().unwrap();
        let stt_mimi_file = config.stt_mimi_model_file.as_ref().unwrap();
        let stt_tokenizer_file = config.stt_text_tokenizer_file.as_ref().unwrap();

        tracing::info!("Loading STT LM and STT Mimi on GPU {} (config stt_gpu_id={}, stt_device={:?}, stt_mimi_device={:?}, use_cpu_for_mimi={})",
                       config.stt_gpu_id, config.stt_gpu_id, stt_device, stt_mimi_device, config.use_cpu_for_mimi);
        // STT model is ASR (audio→text); stt-1b-en_fr-candle uses text vocab 8001/8000 and 32 codebooks (config.json n_q=32).
        let stt_lm_model = moshi::lm::load_asr_stt_1b_en_fr(
            config.batch_size,
            stt_lm_file,
            stt_dtype,
            &stt_device,
        )?;
        let stt_mimi_model = moshi::mimi::load(stt_mimi_file, Some(32), stt_mimi_device)?;
        let stt_text_tokenizer = sentencepiece::SentencePieceProcessor::open(stt_tokenizer_file)?;

        tracing::info!("RAG models loaded (main + STT). Main device: {:?}, STT device: {:?}, STT Mimi device: {:?}",
                       inner.device, stt_device, stt_mimi_device);
        Ok(Self { inner, stt_lm_model, stt_mimi_model, stt_text_tokenizer })
    }
}

async fn handle_socket<T: stream_both::StreamingRunner + Send + 'static>(
    socket: ws::WebSocket,
    sm: T,
) {
    if let Err(err) = stream_both::handle_socket(socket, sm, None).await {
        tracing::error!(err = err.to_string(), "handle_socket")
    }
}

pub async fn stream_handler(
    ws: ws::WebSocketUpgrade,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    state: axum::extract::State<std::sync::Arc<stream_both::AppStateVariant>>,
    req: axum::extract::Query<stream_both::SessionConfigReq>,
) -> impl axum::response::IntoResponse {
    tracing::info!(?addr, "received connection");
    match state.0.as_ref() {
        stream_both::AppStateVariant::Standard(inner) => {
            let inner = inner.clone();
            let sm = stream_both::StreamingModel::new(&inner, req.0);
            ws.on_upgrade(move |v| handle_socket(v, sm))
        }
        stream_both::AppStateVariant::Rag(rag) => {
            let rag = rag.clone();
            let sm = stream_both::StreamingModel::new_rag(&rag, req.0);
            ws.on_upgrade(move |v| handle_socket(v, sm))
        }
        stream_both::AppStateVariant::Batched(batched) => {
            let runner = batched.clone();
            ws.on_upgrade(move |v| async move {
                if let Err(err) = stream_both::handle_socket_batched(v, runner).await {
                    tracing::error!(err = err.to_string(), "handle_socket_batched");
                }
            })
        }
    }
}

#[derive(serde::Serialize)]
pub struct AvailabilityResponse {
    pub available: bool,
}

/// Lightweight HTTP endpoint to check whether the batched runner currently has a free slot.
/// This does not reserve a slot; it only reports the instantaneous availability.
pub async fn availability_handler(
    state: axum::extract::State<std::sync::Arc<stream_both::AppStateVariant>>,
) -> impl axum::response::IntoResponse {
    let available = match state.0.as_ref() {
        stream_both::AppStateVariant::Batched(batched) => batched.pool.has_free_slot(),
        // For non-batched configurations, consider the model always available.
        _ => true,
    };
    axum::Json(AvailabilityResponse { available })
}

pub async fn download_from_hub(config: &mut stream_both::Config) -> Result<()> {
    let token = std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN")).ok();

    let api = if let Some(token) = token {
        hf_hub::api::tokio::ApiBuilder::new().with_token(Some(token)).build()?
    } else {
        hf_hub::api::tokio::ApiBuilder::from_env().build()?
    };
    let hf = (!config.hf_repo.is_empty()).then_some(config.hf_repo.as_str());

    for file_path in
        [&mut config.lm_model_file, &mut config.mimi_model_file, &mut config.text_tokenizer_file]
            .iter_mut()
    {
        crate::hf_path::resolve_model_file(&api, file_path, hf).await?;
    }

    if config.use_stt() {
        for file_path in [
            config.stt_lm_model_file.as_mut().unwrap(),
            config.stt_mimi_model_file.as_mut().unwrap(),
            config.stt_text_tokenizer_file.as_mut().unwrap(),
        ] {
            crate::hf_path::resolve_model_file(&api, file_path, hf).await?;
        }
    }

    if config.use_rag() {
        let arc_path = config
            .arc_encoder_tokenizer_path
            .as_mut()
            .expect("RAG requires arc_encoder_tokenizer_path");
        crate::hf_path::resolve_model_file(&api, arc_path, hf).await?;
        if let Some(ref mut path) = config.arc_encoder_model_file {
            crate::hf_path::resolve_model_file(&api, path, hf).await?;
        }
    }
    Ok(())
}

pub async fn run(args: &StandaloneArgs, config: &Config) -> Result<()> {
    let tls_config = if config.use_https {
        let cert_pem = config.cert_file("cert.pem");
        let key_pem = config.cert_file("key.pem");
        if !cert_pem.exists() || !key_pem.exists() {
            let rcgen::CertifiedKey { cert, key_pair } =
                rcgen::generate_simple_self_signed(vec!["localhost".to_string()])?;
            std::fs::write(&cert_pem, cert.pem())?;
            std::fs::write(&key_pem, key_pair.serialize_pem())?;
        }
        Some(axum_server::tls_rustls::RustlsConfig::from_pem_file(cert_pem, key_pem).await?)
    } else {
        None
    };

    let scheme = if config.use_https { "https" } else { "http" };
    let sock_addr = std::net::SocketAddr::from((
        std::net::IpAddr::from_str(config.addr.as_str())
            .unwrap_or(std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)),
        config.port,
    ));
    let state: Arc<stream_both::AppStateVariant> = if config.stream.batch_size > 1 {
        let inner = Arc::new(stream_both::AppStateInner::new(args, &config.stream)?);
        let batched_state = stream_both::BatchedState::new(inner)?;
        Arc::new(stream_both::AppStateVariant::Batched(Arc::new(batched_state)))
    } else if config.stream.use_rag() {
        Arc::new(stream_both::AppStateVariant::Rag(Arc::new(stream_both::AppStateRag::new(
            args,
            &config.stream,
        )?)))
    } else {
        Arc::new(stream_both::AppStateVariant::Standard(Arc::new(stream_both::AppStateInner::new(
            args,
            &config.stream,
        )?)))
    };
    tracing::info!(
        "serving static dir {} (Batch size: {}, RAG: {})",
        config.static_dir,
        config.stream.batch_size,
        config.stream.use_rag()
    );
    let app = axum::Router::new()
        .route("/api/chat", axum::routing::get(stream_handler))
        .route("/api/availability", axum::routing::get(availability_handler))
        .route(
            "/api/session_feedback",
            axum::routing::post(crate::session_feedback::session_feedback_handler),
        )
        .fallback_service(
            tower_http::services::ServeDir::new(&config.static_dir)
                .append_index_html_on_directories(true),
        )
        .layer(tower::ServiceBuilder::new().layer(tower_http::trace::TraceLayer::new_for_http()))
        .with_state(state);
    tracing::info!("standalone worker listening on {}://{}", scheme, sock_addr);
    match tls_config {
        Some(tls_conf) => {
            axum_server::bind_rustls(sock_addr, tls_conf)
                .serve(app.into_make_service_with_connect_info::<std::net::SocketAddr>())
                .await?;
        }
        None => {
            axum_server::bind(sock_addr)
                .serve(app.into_make_service_with_connect_info::<std::net::SocketAddr>())
                .await?;
        }
    }
    Ok(())
}
