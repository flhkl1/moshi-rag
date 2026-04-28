// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use crate::nn::{linear, MaybeQuantizedEmbedding, MaybeQuantizedLinear, MaybeQuantizedVarBuilder};
use crate::{
    batched_transformer,
    transformer::{self, CaSrc},
    NormType, StreamMask,
};
use candle::{DType, Device, IndexOp, Module, Result, Tensor};

thread_local! {
    pub static VERBOSE: bool = {
        match std::env::var("MIMI_VERBOSE") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}
#[derive(Debug, Clone, serde::Deserialize)]
pub struct DepFormerConfig {
    pub transformer: transformer::Config,
    pub num_slices: usize,
    pub low_rank_embeddings: Option<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ExtraHeadsConfig {
    pub num_heads: usize,
    pub dim: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub transformer: transformer::Config,
    pub depformer: Option<DepFormerConfig>,
    pub text_in_vocab_size: usize,
    pub text_out_vocab_size: usize,
    pub audio_vocab_size: usize,
    pub audio_codebooks: usize,
    pub conditioners: Option<crate::conditioner::Config>,
    pub extra_heads: Option<ExtraHeadsConfig>,
}

impl Config {
    fn depformer_cfg(num_slices: usize) -> DepFormerConfig {
        let depformer_cfg = transformer::Config {
            d_model: 1024,
            num_heads: 16,
            num_layers: 6,
            dim_feedforward: 1024 * 4, // dim * hidden_scale
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: num_slices,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::None,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: false,
        };
        DepFormerConfig { num_slices, transformer: depformer_cfg, low_rank_embeddings: None }
    }

    // /lustre/scwpod02/client/kyutai/alex/mimi_exp/xps/af78657c/outputs/hyperparams.json
    // Update 2024-03-19: Sin embeddings -> None, RmsNorm fix, scale factor 4.125
    // Update 2024-05-02: split text_vocab_size into text_in_vocab_size and text_out_vocab_size.
    // embeddings.
    pub fn v0_1() -> Self {
        let lm_cfg = transformer::Config {
            d_model: 4096,
            num_heads: 32,
            num_layers: 32,
            dim_feedforward: 4096 * 4, // dim * hidden_scale
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 3000,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: false,
        };
        Self {
            transformer: lm_cfg,
            depformer: Some(Self::depformer_cfg(8)),
            audio_vocab_size: 2049,
            text_in_vocab_size: 32001,
            text_out_vocab_size: 32000,
            audio_codebooks: 8,
            conditioners: Default::default(),
            extra_heads: None,
        }
    }

    pub fn v0_1_vision() -> Self {
        let lm_cfg = transformer::Config {
            d_model: 4096,
            num_heads: 32,
            num_layers: 32,
            dim_feedforward: 4096 * 4, // dim * hidden_scale
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 3000,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: Some((
                transformer::CrossAttentionGating::ConditionalGatedSigmoid,
                NormType::RmsNorm,
                None,
            )),
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: true,
        };
        Self {
            transformer: lm_cfg,
            depformer: Some(Self::depformer_cfg(8)),
            audio_vocab_size: 2049,
            text_in_vocab_size: 32001,
            text_out_vocab_size: 32000,
            audio_codebooks: 8,
            conditioners: Default::default(),
            extra_heads: None,
        }
    }

    pub fn v0_1_vision_streaming(num_slices: usize) -> Self {
        let mut s = Self::v0_1_vision();
        s.audio_codebooks = 16;
        if let Some(depformer) = s.depformer.as_mut() {
            depformer.num_slices = num_slices;
            depformer.transformer.context = num_slices;
        }
        s
    }

    pub fn v0_1_streaming(num_slices: usize) -> Self {
        let mut s = Self::v0_1();
        s.audio_codebooks = 16;
        if let Some(depformer) = s.depformer.as_mut() {
            depformer.num_slices = num_slices;
            depformer.transformer.context = num_slices;
        }
        s
    }

    /// Same as v0_1_streaming but with conditioners for RAG (first_speaker LUT, reference_with_time ArcEncoder).
    /// Use this when loading a RAG checkpoint that has condition_provider.conditioners.first_speaker and reference_with_time.
    pub fn v0_1_streaming_rag(num_slices: usize, arc_encoder_tokenizer_path: String) -> Self {
        use crate::conditioner::{ArcEncoderConfig, ConditionerConfig, LutConfig};
        let mut s = Self::v0_1_streaming(num_slices);
        let first_speaker = LutConfig {
            n_bins: 2,
            dim: 16,
            possible_values: vec!["model".to_string(), "user".to_string()],
        };
        let reference_with_time = ArcEncoderConfig {
            out_dim: 4096,
            compression_rate: -4,
            tokenizer_path: arc_encoder_tokenizer_path,
        };
        let mut conditioners = std::collections::HashMap::new();
        conditioners.insert("first_speaker".to_string(), ConditionerConfig::Lut(first_speaker));
        conditioners.insert(
            "reference_with_time".to_string(),
            ConditionerConfig::ArcEncoder(reference_with_time),
        );
        s.conditioners = Some(conditioners);
        s
    }

    pub fn v0_1_asr() -> Self {
        let mut s = Self::v0_1();
        s.audio_codebooks = 8;
        if let Some(depformer) = s.depformer.as_mut() {
            depformer.num_slices = 0;
            depformer.transformer.context = 0;
        }
        s
    }

    // /lustre/scwpod02/client/kyutai/neilz/mimi_exp/xps/6bbe4692/outputs/hyperparams.json
    pub fn tts_v0_1() -> Self {
        let lm_cfg = transformer::Config {
            d_model: 2048,
            num_heads: 32,
            num_layers: 48,
            dim_feedforward: 4096 * 2, // dim * hidden_scale
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 4096,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: Some((
                transformer::CrossAttentionGating::Normal,
                NormType::LayerNorm,
                None,
            )),
            gating: None,
            norm: NormType::LayerNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: false,
        };
        Self {
            transformer: lm_cfg,
            depformer: Some(Self::depformer_cfg(16)),
            audio_vocab_size: 2050,
            text_in_vocab_size: 32001,
            text_out_vocab_size: 32001,
            audio_codebooks: 16,
            conditioners: Default::default(),
            extra_heads: None,
        }
    }

    // /lustre/scwpod02/client/kyutai-interns/tomlab/mimi_exp/xps/c879d080/.hydra/config.yaml
    // /lustre/scwpod02/client/kyutai-interns/tomlab/mimi_exp/xps/41e5e07d/.hydra/config.yaml
    pub fn s2s_v0_1() -> Self {
        let lm_cfg = transformer::Config {
            d_model: 2048,
            num_heads: 16,
            num_layers: 16,
            dim_feedforward: 4096 * 2, // dim * hidden_scale
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 3000,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: false,
        };
        Self {
            transformer: lm_cfg,
            depformer: Some(Self::depformer_cfg(16)),
            audio_vocab_size: 2049,
            text_in_vocab_size: 48001,
            text_out_vocab_size: 48000,
            audio_codebooks: 16,
            conditioners: Default::default(),
            extra_heads: None,
        }
    }

    pub fn s2s_v0_1_streaming(num_slices: usize) -> Self {
        let mut s = Self::s2s_v0_1();
        s.audio_codebooks = 16;
        if let Some(depformer) = s.depformer.as_mut() {
            depformer.num_slices = num_slices;
            depformer.transformer.context = num_slices;
        }
        s
    }

    // /lustre/scwpod02/client/kyutai/neilz/mimi_exp/xps/33e476c7/.hydra/config.yaml
    pub fn asr_v0_1_1b() -> Self {
        let lm_cfg = transformer::Config {
            d_model: 2048,
            num_heads: 16,
            num_layers: 16,
            dim_feedforward: 2048 * 4,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 750,
            max_period: 100_000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: false,
        };
        Self {
            transformer: lm_cfg,
            depformer: None,
            audio_vocab_size: 2049,
            text_in_vocab_size: 48001,
            text_out_vocab_size: 48000,
            audio_codebooks: 8,
            conditioners: Default::default(),
            extra_heads: None,
        }
    }

    /// ASR config for stt-1b-en_fr-candle (text vocab 8001/8000). Checkpoint config has n_q=32.
    pub fn asr_stt_1b_en_fr() -> Self {
        let mut cfg = Self::asr_v0_1_1b();
        cfg.audio_codebooks = 32;
        cfg.text_in_vocab_size = 8001;
        cfg.text_out_vocab_size = 8000;
        cfg.extra_heads = Some(ExtraHeadsConfig { num_heads: 3, dim: 6 });
        cfg
    }

    pub fn asr_300m_202501() -> Self {
        let lm_cfg = transformer::Config {
            d_model: 1024,
            num_heads: 8,
            num_layers: 16,
            dim_feedforward: 1024 * 4,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 750,
            max_period: 100_000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: false,
        };
        Self {
            transformer: lm_cfg,
            depformer: None,
            audio_vocab_size: 2049,
            text_in_vocab_size: 48001,
            text_out_vocab_size: 48000,
            audio_codebooks: 32,
            conditioners: Default::default(),
            extra_heads: None,
        }
    }

    // /lustre/scwpod02/client/kyutai/alex/mimi_exp/xps/d50593ae/.hydra/config.yaml
    pub fn tts_202501() -> Self {
        let lm_cfg = transformer::Config {
            d_model: 2048,
            num_heads: 32,
            num_layers: 48,
            dim_feedforward: 2048 * 4, // dim * hidden_scale
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 500,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: Some((
                transformer::CrossAttentionGating::Normal,
                NormType::LayerNorm,
                None,
            )),
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: false,
        };
        Self {
            transformer: lm_cfg,
            depformer: Some(Self::depformer_cfg(32)),
            audio_vocab_size: 2049,
            text_in_vocab_size: 8001,
            text_out_vocab_size: 8000,
            audio_codebooks: 32,
            conditioners: Default::default(),
            extra_heads: None,
        }
    }

    // /lustre/scwpod02/client/kyutai-interns/tomlab/mimi_exp/xps/1d426dfd/.hydra/config.yaml
    pub fn s2s_2b_16rvq_202501() -> Self {
        let lm_cfg = transformer::Config {
            d_model: 2560,
            num_heads: 20,
            num_layers: 24,
            dim_feedforward: 2560 * 4, // dim * hidden_scale
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 3000,
            max_period: 100000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
            shared_cross_attn: false,
        };
        Self {
            transformer: lm_cfg,
            depformer: Some(Self::depformer_cfg(16)),
            audio_vocab_size: 2049,
            text_in_vocab_size: 48001,
            text_out_vocab_size: 48000,
            audio_codebooks: 32,
            conditioners: Default::default(),
            extra_heads: None,
        }
    }
}

#[derive(Debug, Clone)]
struct LowRankEmbeddings {
    embeddings: MaybeQuantizedEmbedding,
    low_rank: Option<MaybeQuantizedLinear>,
}

impl LowRankEmbeddings {
    fn new(
        in_vocab_size: usize,
        dim: usize,
        low_rank_dim: Option<usize>,
        vb: MaybeQuantizedVarBuilder,
    ) -> Result<Self> {
        let (low_rank, embeddings) = match low_rank_dim {
            None => {
                let embeddings = MaybeQuantizedEmbedding::new(in_vocab_size, dim, vb)?;
                (None, embeddings)
            }
            Some(low_rank_dim) => {
                let low_rank = linear(low_rank_dim, dim, false, vb.pp("low_rank"))?;
                let embeddings = MaybeQuantizedEmbedding::new(in_vocab_size, low_rank_dim, vb)?;
                (Some(low_rank), embeddings)
            }
        };
        Ok(Self { embeddings, low_rank })
    }
}

impl Module for LowRankEmbeddings {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let embs = xs.apply(&self.embeddings)?;
        match self.low_rank.as_ref() {
            None => Ok(embs),
            Some(lr) => embs.apply(lr),
        }
    }
}

/// Transformer used inside a DepFormer slice; can be single-batch or batched.
#[derive(Debug, Clone)]
enum DepFormerTransformer {
    Normal(transformer::StreamingTransformer),
    Batched(batched_transformer::StreamingTransformer),
}

impl DepFormerTransformer {
    fn reset_state(&mut self) {
        use crate::streaming::StreamingModule;
        match self {
            DepFormerTransformer::Normal(t) => t.reset_state(),
            DepFormerTransformer::Batched(t) => t.reset_state(),
        }
    }

    fn copy_state(&mut self, from: &Self) -> Result<()> {
        match (self, from) {
            (DepFormerTransformer::Normal(t), DepFormerTransformer::Normal(f)) => t.copy_state(f),
            (DepFormerTransformer::Batched(t), DepFormerTransformer::Batched(f)) => t.copy_state(f),
            _ => candle::bail!("depformer transformer variant mismatch on copy_state"),
        }
    }

    fn forward(&mut self, xs: &Tensor, m: &StreamMask) -> Result<Tensor> {
        match self {
            DepFormerTransformer::Normal(t) => t.forward(xs),
            DepFormerTransformer::Batched(t) => t.forward(xs, m),
        }
    }
}

#[derive(Debug, Clone)]
struct DepFormerSlice {
    transformer: DepFormerTransformer,
    // Note that the embedding for the first slice does not have the same dimension as the
    // embedding for the other slices as it takes a text token as input rather than an audio token.
    emb: LowRankEmbeddings,
    linear_in: MaybeQuantizedLinear,  // depformer_in.{idx}
    linear_out: MaybeQuantizedLinear, // linears.{idx}
}

impl DepFormerSlice {
    fn new(
        in_vocab_size: usize,
        out_vocab_size: usize,
        main_transformer_dim: usize,
        batch_size: Option<usize>,
        cfg: &DepFormerConfig,
        vb: MaybeQuantizedVarBuilder,
    ) -> Result<Self> {
        let dim = cfg.transformer.d_model;
        let transformer = match batch_size {
            None => {
                let t =
                    transformer::StreamingTransformer::new(&cfg.transformer, vb.pp("transformer"))?;
                DepFormerTransformer::Normal(t)
            }
            Some(1) => {
                let t =
                    transformer::StreamingTransformer::new(&cfg.transformer, vb.pp("transformer"))?;
                DepFormerTransformer::Normal(t)
            }
            Some(batch_size) if batch_size > 1 => {
                let t = batched_transformer::StreamingTransformer::new(
                    batch_size,
                    &cfg.transformer,
                    vb.pp("transformer"),
                )?;
                DepFormerTransformer::Batched(t)
            }
            _ => candle::bail!("invalid batch size"),
        };
        let emb =
            LowRankEmbeddings::new(in_vocab_size, dim, cfg.low_rank_embeddings, vb.pp("emb"))?;
        let linear_in = linear(main_transformer_dim, dim, false, vb.pp("linear_in"))?;
        let linear_out = linear(dim, out_vocab_size, false, vb.pp("linear_out"))?;
        Ok(Self { transformer, emb, linear_in, linear_out })
    }
}

#[derive(Debug, Clone)]
pub struct DepFormer {
    slices: Vec<DepFormerSlice>,
}

impl DepFormer {
    pub fn new(
        text_vocab_size: usize,
        audio_vocab_size: usize,
        main_transformer_dim: usize,
        batch_size: Option<usize>,
        cfg: &DepFormerConfig,
        vb: MaybeQuantizedVarBuilder,
    ) -> Result<Self> {
        let mut slices = Vec::with_capacity(cfg.num_slices);
        for slice_idx in 0..cfg.num_slices {
            let in_vs = if slice_idx == 0 { text_vocab_size } else { audio_vocab_size };
            // The depformer cannot predict the audio padding token.
            let slice = DepFormerSlice::new(
                in_vs,
                audio_vocab_size - 1, // The depformer cannot emit an audio padding token.
                main_transformer_dim,
                batch_size,
                cfg,
                vb.pp(slice_idx),
            )?;
            slices.push(slice)
        }
        Ok(Self { slices })
    }

    /// Run a transformer sampling step, getting a token id per codebook.
    /// - `xs` is the previous layer hidden state.
    pub fn sample(
        &mut self,
        xs: &Tensor,
        text_token: Option<u32>,
        forced_audio_tokens: &[Option<u32>],
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Vec<u32>> {
        let dev = xs.device();
        let mask_1 = StreamMask::new(vec![true], dev)?;
        let mut tokens = Vec::with_capacity(self.slices.len());
        let mut last_token = text_token;
        for slice_idx in 0..self.slices.len() {
            if slice_idx == 0 {
                self.slices[slice_idx].transformer.reset_state();
            } else {
                let (lhs, rhs) = self.slices.split_at_mut(slice_idx);
                rhs[0].transformer.copy_state(&lhs[slice_idx - 1].transformer)?
            }
            let slice = &mut self.slices[slice_idx];
            let xs = slice.linear_in.forward(xs)?;
            let xs = match last_token {
                Some(last_token) => {
                    let token_id = Tensor::from_vec(vec![last_token], (1, 1), dev)?;
                    let token_emb = slice.emb.forward(&token_id)?;
                    xs.broadcast_add(&token_emb)?
                }
                None => xs,
            };
            let xs = slice.transformer.forward(&xs, &mask_1)?;
            let logits = xs.apply(&slice.linear_out)?;
            let logits = match logits.dim(0)? {
                1 => logits.i((0, 0))?,
                b_size => candle::bail!("unexpected batch size {b_size}"),
            };
            let token = lp.sample(&logits)?;
            if VERBOSE.with(|v| *v) {
                println!("sampled {token} logits {slice_idx}:\n{logits}");
            }
            tokens.push(token);
            let token_for_next_layer =
                forced_audio_tokens.get(slice_idx).copied().flatten().unwrap_or(token);
            last_token = Some(token_for_next_layer);
        }
        Ok(tokens)
    }

    // Sampling with classifier free guidance.
    pub fn sample_cfg(
        &mut self,
        xs: &Tensor,
        cfg_alpha: f64,
        text_token: Option<u32>,
        forced_audio_tokens: &[Option<u32>],
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Vec<u32>> {
        let dev = xs.device();
        let mask_1 = StreamMask::new(vec![true], dev)?;
        let mut tokens = Vec::with_capacity(self.slices.len());
        let mut last_token = text_token;
        for slice_idx in 0..self.slices.len() {
            if slice_idx == 0 {
                self.slices[slice_idx].transformer.reset_state();
            } else {
                let (lhs, rhs) = self.slices.split_at_mut(slice_idx);
                rhs[0].transformer.copy_state(&lhs[slice_idx - 1].transformer)?
            }
            let slice = &mut self.slices[slice_idx];
            let xs = slice.linear_in.forward(xs)?;
            let xs = match last_token {
                Some(last_token) => {
                    let token_id = Tensor::from_vec(vec![last_token], (1, 1), dev)?;
                    let token_emb = slice.emb.forward(&token_id)?;
                    xs.broadcast_add(&token_emb)?
                }
                None => xs,
            };
            let xs = slice.transformer.forward(&xs, &mask_1)?;
            let logits = xs.apply(&slice.linear_out)?;
            let logits = match logits.dim(0)? {
                2 => ((logits.i((0, 0))? * cfg_alpha)? - (logits.i((1, 0))? * (cfg_alpha - 1.))?)?,
                b_size => candle::bail!("unexpected batch size {b_size}"),
            };
            let token = lp.sample(&logits)?;
            if VERBOSE.with(|v| *v) {
                println!("sampled {token} logits {slice_idx}:\n{logits}");
            }
            tokens.push(token);
            let token_for_next_layer =
                forced_audio_tokens.get(slice_idx).copied().flatten().unwrap_or(token);
            last_token = Some(token_for_next_layer);
        }
        Ok(tokens)
    }

    /// Batched sampling: one forward per slice over the full batch.
    pub fn sample_batched(
        &mut self,
        xs: &Tensor,
        text_tokens: &[Option<u32>],
        forced_audio_tokens: &[impl AsRef<[Option<u32>]>],
        mask: &StreamMask,
        lp: &mut [candle_transformers::generation::LogitsProcessor],
    ) -> Result<Vec<Option<Vec<u32>>>> {
        let (batch_size, _t, _d) = xs.dims3()?;
        if batch_size != text_tokens.len()
            || batch_size != forced_audio_tokens.len()
            || lp.len() != batch_size
            || mask.cpu().is_none_or(|c| c.len() != batch_size)
        {
            candle::bail!(
                "depformer_sample_batched: batch size mismatch xs={} text_tokens={} forced={} lp={}",
                batch_size,
                text_tokens.len(),
                forced_audio_tokens.len(),
                lp.len()
            );
        }
        let dev = xs.device();
        let mut tokens: Vec<Option<Vec<u32>>> = (0..batch_size).map(|_| Some(Vec::new())).collect();
        let mut last_tokens: Vec<Option<u32>> = text_tokens.to_vec();
        for slice_idx in 0..self.slices.len() {
            if slice_idx == 0 {
                self.slices[slice_idx].transformer.reset_state();
            } else {
                let (lhs, rhs) = self.slices.split_at_mut(slice_idx);
                rhs[0].transformer.copy_state(&lhs[slice_idx - 1].transformer)?;
            }
            let slice = &mut self.slices[slice_idx];
            let xs = slice.linear_in.forward(xs)?;

            // Add text embedding of valid last tokens to input embeddings.
            // For inactive streams (mask=false), always use id 0 so we never index
            // embeddings out of range with stale text tokens.
            let token_ids: Vec<u32> = (0..batch_size)
                .map(|b| if !mask.is_active(b) { 0 } else { last_tokens[b].unwrap_or_default() })
                .collect();
            let token_id = Tensor::from_vec(token_ids, (batch_size, 1), dev)?;
            let token_emb = slice.emb.forward(&token_id)?;
            // Per-batch mask indicating which streams have a valid last token and are active.
            // Expand to match the embedding dimension so elementwise mul works without broadcasting.
            let (_b, _t, dim) = token_emb.dims3()?;
            let mut add_mask_vec = Vec::with_capacity(batch_size * dim);
            for lt in last_tokens.iter().take(batch_size) {
                let v = if lt.is_some() { 1.0f32 } else { 0.0f32 };
                for _ in 0..dim {
                    add_mask_vec.push(v);
                }
            }
            let add_mask = Tensor::from_vec(add_mask_vec, (batch_size, 1, dim), dev)?
                .to_dtype(token_emb.dtype())?;
            let xs = xs.broadcast_add(&(token_emb * add_mask)?)?;

            let xs = slice.transformer.forward(&xs, mask)?;
            let logits = xs.apply(&slice.linear_out)?;
            for (b, token_slot) in tokens.iter_mut().enumerate().take(batch_size) {
                if !mask.is_active(b) {
                    continue;
                }
                let logits_b = logits.i((b, 0))?;
                let token = lp[b].sample(&logits_b)?;
                token_slot.as_mut().unwrap().push(token);
                let forced = forced_audio_tokens[b].as_ref().get(slice_idx).copied().flatten();
                let token_for_next_layer = forced.unwrap_or(token);
                last_tokens[b] = Some(token_for_next_layer);
            }
        }
        for (b, token_opt) in tokens.iter_mut().enumerate().take(batch_size) {
            if !mask.is_active(b) {
                *token_opt = None;
            }
        }
        Ok(tokens)
    }
}

#[derive(Debug, Clone)]
enum StreamingTransformer {
    Normal(transformer::StreamingTransformer),
    Batched(batched_transformer::StreamingTransformer),
}

impl crate::StreamingModule for StreamingTransformer {
    fn reset_state(&mut self) {
        match self {
            StreamingTransformer::Normal(t) => t.reset_state(),
            StreamingTransformer::Batched(t) => t.reset_state(),
        }
    }

    fn step(
        &mut self,
        xs: &crate::StreamTensor,
        mask: &crate::StreamMask,
    ) -> Result<crate::StreamTensor> {
        match self {
            StreamingTransformer::Normal(t) => t.step(xs, mask),
            StreamingTransformer::Batched(t) => t.step(xs, mask),
        }
    }
}

impl StreamingTransformer {
    fn batch_size(&self) -> usize {
        match self {
            StreamingTransformer::Normal(_) => 1,
            StreamingTransformer::Batched(t) => t.batch_size(),
        }
    }

    fn reset_batch_idx(&mut self, batch_idx: usize, batch_size: usize) -> Result<()> {
        match self {
            StreamingTransformer::Normal(t) => t.reset_batch_idx(batch_idx, batch_size),
            StreamingTransformer::Batched(t) => t.reset_batch_idx(batch_idx),
        }
    }

    fn maybe_precompute_ca_kv(&self, ca_src: Option<CaSrc>) -> Result<Option<CaSrc>> {
        match self {
            StreamingTransformer::Normal(t) => t.maybe_precompute_ca_kv(ca_src),
            StreamingTransformer::Batched(t) => t.maybe_precompute_ca_kv(ca_src),
        }
    }

    fn forward(&mut self, xs: &Tensor, m: &StreamMask) -> Result<Tensor> {
        match self {
            StreamingTransformer::Normal(t) => t.forward(xs),
            StreamingTransformer::Batched(t) => t.forward(xs, m),
        }
    }

    fn forward_ca(
        &mut self,
        xs: &Tensor,
        ca_src: Option<&CaSrc>,
        m: &StreamMask,
    ) -> Result<Tensor> {
        match self {
            StreamingTransformer::Normal(t) => t.forward_ca(xs, ca_src),
            StreamingTransformer::Batched(t) => t.forward_ca(xs, ca_src, m),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LmModel {
    transformer: StreamingTransformer,
    text_emb: MaybeQuantizedEmbedding,
    audio_embs: Vec<MaybeQuantizedEmbedding>,
    text_linear: MaybeQuantizedLinear,
    out_norm: transformer::Norm,
    depformer: Option<DepFormer>,
    audio_vocab_size: usize,
    text_in_vocab_size: usize,
    condition_provider: Option<crate::conditioner::ConditionProvider>,
    extra_heads: Vec<MaybeQuantizedLinear>,
    dtype: DType,
}

impl LmModel {
    pub fn new(cfg: &Config, vb: MaybeQuantizedVarBuilder) -> Result<Self> {
        Self::new_(None, cfg, vb)
    }

    pub fn batched(batch_size: usize, cfg: &Config, vb: MaybeQuantizedVarBuilder) -> Result<Self> {
        Self::new_(Some(batch_size), cfg, vb)
    }

    pub fn new_(
        batch_size: Option<usize>,
        cfg: &Config,
        vb: MaybeQuantizedVarBuilder,
    ) -> Result<Self> {
        let d_model = cfg.transformer.d_model;
        let depformer = match &cfg.depformer {
            None => None,
            Some(depformer_cfg) => {
                let depformer = DepFormer::new(
                    cfg.text_in_vocab_size,
                    cfg.audio_vocab_size,
                    d_model,
                    batch_size,
                    depformer_cfg,
                    vb.pp("depformer"),
                )?;
                Some(depformer)
            }
        };
        let text_emb =
            MaybeQuantizedEmbedding::new(cfg.text_in_vocab_size, d_model, vb.pp("text_emb"))?;
        let out_norm = transformer::Norm::new(d_model, &cfg.transformer, vb.pp("out_norm"))?;
        let text_linear = linear(d_model, cfg.text_out_vocab_size, false, vb.pp("text_linear"))?;
        let transformer = match batch_size {
            None => {
                let transformer =
                    transformer::StreamingTransformer::new(&cfg.transformer, vb.pp("transformer"))?;
                StreamingTransformer::Normal(transformer)
            }
            Some(batch_size) => {
                let transformer = batched_transformer::StreamingTransformer::new(
                    batch_size,
                    &cfg.transformer,
                    vb.pp("transformer"),
                )?;
                StreamingTransformer::Batched(transformer)
            }
        };
        let vb_e = vb.pp("emb");
        let mut audio_embs = Vec::with_capacity(cfg.audio_codebooks);
        for i in 0..cfg.audio_codebooks {
            let emb = MaybeQuantizedEmbedding::new(cfg.audio_vocab_size, d_model, vb_e.pp(i))?;
            audio_embs.push(emb)
        }
        let dtype = vb.dtype();
        let condition_provider = match cfg.conditioners.as_ref() {
            None => None,
            Some(cfg) => {
                let conditioners = crate::conditioner::ConditionProvider::new(
                    d_model,
                    cfg,
                    vb.pp("condition_provider"),
                )?;
                Some(conditioners)
            }
        };
        let mut extra_heads = vec![];
        if let Some(ExtraHeadsConfig { num_heads, dim }) = cfg.extra_heads {
            for i in 0..num_heads {
                let extra_head = linear(d_model, dim, false, vb.pp("extra_heads").pp(i))?;
                extra_heads.push(extra_head)
            }
        }
        Ok(Self {
            transformer,
            text_emb,
            text_linear,
            audio_embs,
            out_norm,
            depformer,
            text_in_vocab_size: cfg.text_in_vocab_size,
            audio_vocab_size: cfg.audio_vocab_size,
            condition_provider,
            extra_heads,
            dtype,
        })
    }

    pub fn condition_provider(&self) -> Option<&crate::conditioner::ConditionProvider> {
        self.condition_provider.as_ref()
    }

    /// After the main checkpoint load, merge a secondary checkpoint into the named conditioner (when supported).
    pub fn load_conditioner_hf_weights_from_path(
        &mut self,
        conditioner_name: &str,
        path: &std::path::Path,
        dtype: DType,
    ) -> Result<()> {
        let dev = self.device().clone();
        let cp = self
            .condition_provider
            .as_mut()
            .ok_or_else(|| candle::Error::Msg("no condition_provider".into()))?;
        cp.load_hf_weights_overlay(conditioner_name, path, dtype, &dev)
    }

    /// Run the transformer on a prepend sequence to prime the KV cache.
    /// `prepend` shape: [batch_size, T, d_model]. No logits are computed.
    pub fn forward_prepend(
        &mut self,
        prepend: &Tensor,
        mask: Option<&crate::streaming::StreamMask>,
    ) -> Result<()> {
        let (b, t, _) = prepend.dims3()?;
        if t == 0 {
            return Ok(());
        }
        if b == 1 {
            self.transformer.forward(prepend, &().into())?;
            return Ok(());
        }
        if b != self.batch_size() {
            candle::bail!(
                "forward_prepend batch size {} != model batch size {}",
                b,
                self.batch_size()
            );
        }
        if let Some(mask) = mask {
            self.transformer.forward(prepend, mask)?;
        } else {
            candle::bail!("mask is required for batched forward_prepend");
        }
        Ok(())
    }

    /// If the model has a lut conditioner, return the tensor for the text.
    pub fn get_lut_condition(&self, name: &str, text: &str) -> Option<Tensor> {
        let cp = self.condition_provider.as_ref()?;
        let cond = cp.condition_lut(name, text).ok()?;
        match cond {
            crate::conditioner::Condition::AddToInput(t) => {
                let (b, seq, _) = t.dims3().ok()?;
                if b == 1 && seq >= 1 {
                    Some(t)
                } else {
                    None
                }
            }
        }
    }

    /// Rreturn an embedding sequence condition for the text.
    pub fn get_emb_seq_condition(&self, name: &str, text: &str) -> Option<Tensor> {
        let cp = self.condition_provider.as_ref()?;
        cp.condition_emb_seq(name, text, self.device()).ok().flatten()
    }

    pub fn reset_state(&mut self) {
        use crate::streaming::StreamingModule;
        self.transformer.reset_state()
    }

    pub fn in_audio_codebooks(&self) -> usize {
        self.audio_embs.len()
    }

    pub fn audio_pad_token(&self) -> u32 {
        self.audio_vocab_size as u32 - 1
    }

    pub fn text_start_token(&self) -> u32 {
        self.text_in_vocab_size as u32 - 1
    }

    pub fn generated_audio_codebooks(&self) -> usize {
        self.depformer.as_ref().map_or(0, |v| v.slices.len())
    }

    pub fn is_quantized(&self) -> bool {
        match self.text_linear {
            MaybeQuantizedLinear::Quantized(_) => true,
            MaybeQuantizedLinear::Real(_) => false,
        }
    }

    pub fn device(&self) -> &Device {
        self.text_emb.embeddings().device()
    }

    /// Batch size of the model (1 for single-stream, N for batched).
    pub fn batch_size(&self) -> usize {
        self.transformer.batch_size()
    }

    pub fn dtype(&self) -> DType {
        self.text_emb.embeddings().dtype()
    }

    pub fn forward(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        mask: &StreamMask,
    ) -> candle::Result<(Tensor, Tensor)> {
        self.forward_cond(text_ids, audio_ids, None, mask)
    }

    pub fn extra_heads(&self, vs: &Tensor) -> Result<Vec<Tensor>> {
        let mut extra_heads = Vec::with_capacity(self.extra_heads.len());
        for extra_head in self.extra_heads.iter() {
            let extra_head = vs.apply(extra_head)?;
            extra_heads.push(extra_head)
        }
        Ok(extra_heads)
    }

    pub fn forward_cond(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        conditions: Option<&crate::conditioner::Condition>,
        mask: &StreamMask,
    ) -> candle::Result<(Tensor, Tensor)> {
        if VERBOSE.with(|v| *v) {
            print!("text_ids ");
            if let Some(text_ids) = text_ids.as_ref() {
                let text_ids = text_ids.flatten_all()?.to_vec1::<u32>()?;
                println!("{text_ids:?}");
            } else {
                println!("none")
            }
            print!("audio_ids ");
            for audio_id in audio_ids.iter() {
                if let Some(audio_id) = audio_id {
                    let audio_id = audio_id.flatten_all()?.to_vec1::<u32>()?;
                    print!(" {audio_id:?}");
                } else {
                    print!(" none")
                }
            }
            println!();
        }
        let mut emb = match text_ids.as_ref() {
            Some(text_ids) => text_ids.apply(&self.text_emb)?,
            None => {
                let device = self.text_emb.embeddings().device();
                Tensor::zeros((1, 1, self.text_emb.hidden_size()?), self.dtype, device)?
            }
        };

        for (audio_emb, audio_ids) in self.audio_embs.iter().zip(audio_ids.iter()) {
            if let Some(audio_ids) = audio_ids {
                let e = audio_ids.apply(audio_emb)?;
                emb = (emb + e)?
            }
        }
        if let Some(conditions) = conditions {
            match conditions {
                crate::conditioner::Condition::AddToInput(v) => emb = emb.broadcast_add(v)?,
            }
        }
        let ys = self.transformer.forward(&emb, mask)?;
        let ys = ys.apply(&self.out_norm)?;
        let logits = ys.apply(&self.text_linear)?;
        if VERBOSE.with(|v| *v) {
            println!("logits:\n{logits}");
        }
        Ok((logits, ys))
    }

    pub fn maybe_precompute_ca_kv(&self, ca_src: Option<CaSrc>) -> Result<Option<CaSrc>> {
        let ca_src = match ca_src {
            None => None,
            z => self.transformer.maybe_precompute_ca_kv(z)?,
        };
        Ok(ca_src)
    }

    pub fn forward_ca(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        ca_src: &CaSrc,
        conditions: Option<&crate::conditioner::Condition>,
        mask: &StreamMask,
    ) -> candle::Result<(Tensor, Tensor)> {
        if VERBOSE.with(|v| *v) {
            print!("text_ids ");
            if let Some(text_ids) = text_ids.as_ref() {
                let text_ids = text_ids.flatten_all()?.to_vec1::<u32>()?;
                println!("{text_ids:?}");
            } else {
                println!("none")
            }
            print!("audio_ids ");
            for audio_id in audio_ids.iter() {
                if let Some(audio_id) = audio_id {
                    let audio_id = audio_id.flatten_all()?.to_vec1::<u32>()?;
                    print!(" {audio_id:?}");
                } else {
                    print!(" none")
                }
            }
            println!();
        }
        let b_size = match ca_src {
            CaSrc::KeysValues((cak, _)) => cak.dim(0)?,
            CaSrc::Tokens(catoks) => catoks.dim(0)?,
        };
        let mut emb = match text_ids {
            Some(text_ids) => text_ids.apply(&self.text_emb)?,
            None => {
                let device = self.text_emb.embeddings().device();
                Tensor::zeros((b_size, 1, self.text_emb.hidden_size()?), self.dtype, device)?
            }
        };
        for (audio_emb, audio_ids) in self.audio_embs.iter().zip(audio_ids.iter()) {
            if let Some(audio_ids) = audio_ids {
                let e = audio_ids.apply(audio_emb)?;
                emb = emb.broadcast_add(&e)?
            }
        }
        if let Some(conditions) = conditions {
            match conditions {
                crate::conditioner::Condition::AddToInput(v) => emb = emb.broadcast_add(v)?,
            }
        }
        let ys = self.transformer.forward_ca(&emb, Some(ca_src), mask)?;
        let ys = ys.apply(&self.out_norm)?;
        let logits = ys.apply(&self.text_linear)?;
        Ok((logits, ys))
    }

    pub fn depformer_sample(
        &mut self,
        xs: &Tensor,
        text_token: Option<u32>,
        forced_audio_tokens: &[Option<u32>],
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Option<Vec<u32>>> {
        let sample = match self.depformer.as_mut() {
            None => None,
            Some(m) => {
                let sample = m.sample(xs, text_token, forced_audio_tokens, lp)?;
                Some(sample)
            }
        };
        Ok(sample)
    }

    pub fn depformer_sample_cfg(
        &mut self,
        xs: &Tensor,
        cfg_alpha: f64,
        text_token: Option<u32>,
        forced_audio_tokens: &[Option<u32>],
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Option<Vec<u32>>> {
        let sample = match self.depformer.as_mut() {
            None => None,
            Some(m) => {
                let sample = m.sample_cfg(xs, cfg_alpha, text_token, forced_audio_tokens, lp)?;
                Some(sample)
            }
        };
        Ok(sample)
    }

    /// Batched depformer sampling. Use when `batch_size() > 1`.
    /// `lp` must have length equal to batch size (one [candle_transformers::generation::LogitsProcessor] per stream).
    pub fn depformer_sample_batched(
        &mut self,
        xs: &Tensor,
        text_tokens: &[Option<u32>],
        forced_audio_tokens: &[impl AsRef<[Option<u32>]>],
        mask: &StreamMask,
        lp: &mut [candle_transformers::generation::LogitsProcessor],
    ) -> Result<Vec<Option<Vec<u32>>>> {
        match self.depformer.as_mut() {
            None => {
                let b = xs.dim(0)?;
                Ok((0..b).map(|_| None).collect())
            }
            Some(m) => m.sample_batched(xs, text_tokens, forced_audio_tokens, mask, lp),
        }
    }

    pub fn reset_batch_idx(&mut self, batch_idx: usize, batch_size: usize) -> Result<()> {
        self.transformer.reset_batch_idx(batch_idx, batch_size)
    }
}

pub fn load_lm_model<P: AsRef<std::path::Path>>(
    cfg: Config,
    model_file: P,
    dtype: DType,
    dev: &Device,
) -> Result<LmModel> {
    load_lm_model_batched(1, cfg, model_file, dtype, dev)
}

pub fn load_lm_model_batched<P: AsRef<std::path::Path>>(
    batch_size: usize,
    cfg: Config,
    model_file: P,
    dtype: DType,
    dev: &Device,
) -> Result<LmModel> {
    let quantized = model_file.as_ref().extension().is_some_and(|v| v == "gguf");
    let vb = if quantized {
        MaybeQuantizedVarBuilder::Quantized(
            candle_transformers::quantized_var_builder::VarBuilder::from_gguf(model_file, dev)?,
        )
    } else {
        unsafe {
            MaybeQuantizedVarBuilder::Real(candle_nn::VarBuilder::from_mmaped_safetensors(
                &[model_file],
                dtype,
                dev,
            )?)
        }
    };
    let model = if batch_size <= 1 {
        LmModel::new(&cfg, vb)?
    } else {
        LmModel::batched(batch_size, &cfg, vb)?
    };
    Ok(model)
}

pub fn load<P: AsRef<std::path::Path>>(
    model_file: P,
    dtype: DType,
    dev: &Device,
) -> Result<LmModel> {
    let cfg = Config::v0_1();
    load_lm_model(cfg, model_file, dtype, dev)
}

pub fn load_streaming<P: AsRef<std::path::Path>>(
    model_file: P,
    dtype: DType,
    dev: &Device,
) -> Result<LmModel> {
    let cfg = Config::v0_1_streaming(8);
    load_lm_model(cfg, model_file, dtype, dev)
}

pub fn load_streaming_batched<P: AsRef<std::path::Path>>(
    batch_size: usize,
    model_file: P,
    dtype: DType,
    dev: &Device,
) -> Result<LmModel> {
    let cfg = Config::v0_1_streaming(8);
    load_lm_model_batched(batch_size, cfg, model_file, dtype, dev)
}

/// Load streaming LM for RAG.
pub fn load_streaming_rag<P: AsRef<std::path::Path>>(
    model_file: P,
    dtype: DType,
    dev: &Device,
    arc_encoder_tokenizer_path: &str,
    arc_encoder_weights: Option<&std::path::Path>,
) -> Result<LmModel> {
    let cfg = Config::v0_1_streaming_rag(8, arc_encoder_tokenizer_path.to_string());
    let mut model = load_lm_model(cfg, model_file, dtype, dev)?;
    if let Some(path) = arc_encoder_weights {
        model.load_conditioner_hf_weights_from_path("reference_with_time", path, dtype)?;
    }
    Ok(model)
}

/// Load streaming LM for RAG with batched inference.
pub fn load_streaming_rag_batched<P: AsRef<std::path::Path>>(
    batch_size: usize,
    model_file: P,
    dtype: DType,
    dev: &Device,
    arc_encoder_tokenizer_path: &str,
    arc_encoder_weights: Option<&std::path::Path>,
) -> Result<LmModel> {
    let cfg = Config::v0_1_streaming_rag(8, arc_encoder_tokenizer_path.to_string());
    let mut model = load_lm_model_batched(batch_size, cfg, model_file, dtype, dev)?;
    if let Some(path) = arc_encoder_weights {
        model.load_conditioner_hf_weights_from_path("reference_with_time", path, dtype)?;
    }
    Ok(model)
}

pub fn load_streaming_both_ways<P: AsRef<std::path::Path>>(
    model_file: P,
    dtype: DType,
    dev: &Device,
) -> Result<LmModel> {
    let cfg = Config::v0_1_streaming(16);
    load_lm_model(cfg, model_file, dtype, dev)
}

pub fn load_vision<P: AsRef<std::path::Path>>(
    model_file: P,
    override_cross_attention_gating: Option<transformer::CrossAttentionGating>,
    override_cross_attention_in_dim: Option<usize>,
    dtype: DType,
    dev: &Device,
) -> Result<LmModel> {
    // load_vision allows for overriding some hyperparams of the lm from the main config file
    let mut cfg = Config::v0_1_vision_streaming(8);
    cfg.transformer.cross_attention = override_cross_attention_gating
        .map(|v| (v, cfg.transformer.norm, override_cross_attention_in_dim));
    load_lm_model(cfg, model_file, dtype, dev)
}

pub fn load_s2s<P: AsRef<std::path::Path>>(
    model_file: P,
    dtype: DType,
    dev: &Device,
) -> Result<LmModel> {
    let cfg = Config::s2s_2b_16rvq_202501();
    load_lm_model(cfg, model_file, dtype, dev)
}

pub fn load_asr<P: AsRef<std::path::Path>>(
    model_file: P,
    dtype: DType,
    dev: &Device,
) -> Result<LmModel> {
    let cfg = Config::asr_v0_1_1b();
    load_lm_model(cfg, model_file, dtype, dev)
}

/// Load ASR model for stt-1b-en_fr-candle (text vocab 8001/8000).
pub fn load_asr_stt_1b_en_fr<P: AsRef<std::path::Path>>(
    batch_size: usize,
    model_file: P,
    dtype: DType,
    dev: &Device,
) -> Result<LmModel> {
    let cfg = Config::asr_stt_1b_en_fr();
    load_lm_model_batched(batch_size, cfg, model_file, dtype, dev)
}

#[derive(Debug)]
pub struct ForcedAudioTokens {
    acoustic_delay: usize,
    // Tokens that are teacher forced before the acoustic delay.
    pre_delay_tokens: Vec<Option<u32>>,
}

impl ForcedAudioTokens {
    pub fn new(acoustic_delay: usize, audio_pad_token: u32, stream_codebooks: &[usize]) -> Self {
        let mut pre_delay_tokens = vec![];
        for codebooks in stream_codebooks.iter() {
            for c in 0..*codebooks {
                let token = if c == 0 { None } else { Some(audio_pad_token) };
                pre_delay_tokens.push(token);
            }
        }
        Self { acoustic_delay, pre_delay_tokens }
    }

    pub fn forced_tokens(&self, step_idx: usize) -> &[Option<u32>] {
        if step_idx < self.acoustic_delay {
            &self.pre_delay_tokens
        } else {
            &[]
        }
    }
}
