// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use crate::arc_encoder::{ArcEncoderTransformer, EmbProjector};
use crate::nn::{
    linear, MaybeQuantizedEmbedding as Embedding, MaybeQuantizedLinear as Linear,
    MaybeQuantizedVarBuilder as VarBuilder,
};
use candle::{DType, Result, Tensor};
use candle_nn::VarBuilder as CandleVarBuilder;
use std::collections::HashMap;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LutConfig {
    pub n_bins: usize,
    pub dim: usize,
    pub possible_values: Vec<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ContinuousAttributeConfig {
    pub dim: usize,
    pub scale_factor: f32,
    pub max_period: f32,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ArcEncoderConfig {
    pub out_dim: usize,
    pub compression_rate: i32,
    pub tokenizer_path: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "type")]
pub enum ConditionerConfig {
    Lut(LutConfig),
    ContinuousAttribute(ContinuousAttributeConfig),
    ArcEncoder(ArcEncoderConfig),
}

pub type Config = HashMap<String, ConditionerConfig>;

#[derive(Debug, Clone)]
pub struct LutConditioner {
    embed: Embedding,
    output_proj: Linear,
    #[allow(unused)]
    learnt_padding: Tensor,
    possible_values: HashMap<String, usize>,
}

impl LutConditioner {
    pub fn new(output_dim: usize, cfg: &LutConfig, vb: VarBuilder) -> Result<Self> {
        let embed = Embedding::new(cfg.n_bins + 1, cfg.dim, vb.pp("embed"))?;
        let output_proj = linear(cfg.dim, output_dim, false, vb.pp("output_proj"))?;
        let learnt_padding = vb.get_as_tensor((1, 1, output_dim), "learnt_padding")?;
        let possible_values: HashMap<String, usize> =
            cfg.possible_values.iter().enumerate().map(|(i, v)| (v.to_string(), i)).collect();
        Ok(Self { embed, output_proj, learnt_padding, possible_values })
    }

    pub fn condition(&self, value: &str) -> Result<Condition> {
        let idx = match self.possible_values.get(value) {
            None => candle::bail!("unknown value for lut conditioner '{value}'"),
            Some(idx) => *idx,
        };
        let cond = Tensor::from_vec(vec![idx as u32], (1, 1), self.embed.embeddings().device())?
            .apply(&self.embed)?
            .apply(&self.output_proj)?;
        Ok(Condition::AddToInput(cond))
    }
}

#[derive(Debug, Clone)]
pub struct ContinuousAttributeConditioner {
    scale_factor: f32,
    max_period: f32,
    dim: usize,
    output_proj: Linear,
    #[allow(unused)]
    learnt_padding: Tensor,
    device: candle::Device,
}

impl ContinuousAttributeConditioner {
    pub fn new(output_dim: usize, cfg: &ContinuousAttributeConfig, vb: VarBuilder) -> Result<Self> {
        let output_proj = linear(cfg.dim, output_dim, false, vb.pp("output_proj"))?;
        let learnt_padding = vb.get_as_tensor((1, 1, output_dim), "learnt_padding")?;
        Ok(Self {
            scale_factor: cfg.scale_factor,
            max_period: cfg.max_period,
            dim: cfg.dim,
            output_proj,
            learnt_padding,
            device: vb.device().clone(),
        })
    }

    // `positions` should have shape (b, t, 1), the output will be (b, t, dim)
    pub fn create_sin_embeddings(&self, positions: &Tensor, dtype: DType) -> Result<Tensor> {
        let dev = positions.device();
        let half_dim = self.dim / 2;
        let positions = positions.to_dtype(dtype)?;
        let adim: Vec<_> = (0..half_dim)
            .map(|i| 1f32 / self.max_period.powf(i as f32 / (half_dim - 1) as f32))
            .collect();
        let adim = Tensor::from_vec(adim, (1, 1, ()), dev)?;
        let freqs = positions.broadcast_mul(&adim)?;
        let pos_emb = Tensor::cat(&[freqs.cos()?, freqs.sin()?], candle::D::Minus1)?;
        Ok(pos_emb)
    }

    // TODO(laurent): should we support different values per batch element?
    pub fn condition(&self, value: f32) -> Result<Condition> {
        let value = value * self.scale_factor;
        let positions = Tensor::full(value, (1, 1, 1), &self.device)?;
        let cond = self
            .create_sin_embeddings(&positions, DType::F32)?
            .to_dtype(self.output_proj.dtype())?
            .apply(&self.output_proj)?;
        Ok(Condition::AddToInput(cond))
    }
}

/// Full Arc encoder conditioner: ArcEncoderTransformer (26-layer embedder) + EmbProjector (bridge)
#[derive(Clone)]
pub struct ArcEncoderConditioner {
    embedder: ArcEncoderTransformer,
    bridge: EmbProjector,
    out_dim: usize,
    tokenizer: tokenizers::Tokenizer,
    /// Optional: maps bridge out_dim -> out_dim. Loaded when present in checkpoint.
    output_proj: Linear,
    /// Optional: (1, 1, out_dim). Applied where mask is 0. Loaded when present in checkpoint.
    learnt_padding: Tensor,
    compression_rate: i32,
    bridge_out_dim: usize,
}

impl std::fmt::Debug for ArcEncoderConditioner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArcEncoderConditioner")
            .field("out_dim", &self.out_dim)
            .finish_non_exhaustive()
    }
}

impl ArcEncoderConditioner {
    /// Build from config. Loads embedder, bridge_module, output_proj, and learnt_padding.
    /// output_dim: model condition dim (final output). Bridge outputs cfg.out_dim; output_proj maps to output_dim.
    pub fn new(output_dim: usize, cfg: &ArcEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let embedder = if vb.pp("embedder").contains_key("tok_embeddings.weight") {
            ArcEncoderTransformer::new(cfg.compression_rate, vb.pp("embedder"))?
        } else {
            ArcEncoderTransformer::new(
                cfg.compression_rate,
                VarBuilder::Real(CandleVarBuilder::zeros(vb.dtype(), vb.device())),
            )?
        };
        let bridge = if vb.pp("bridge_module").contains_key("layer1.weight") {
            EmbProjector::new(cfg.out_dim, vb.pp("bridge_module"))?
        } else {
            EmbProjector::new(
                cfg.out_dim,
                VarBuilder::Real(CandleVarBuilder::zeros(vb.dtype(), vb.device())),
            )?
        };
        let tokenizer = tokenizers::Tokenizer::from_file(&cfg.tokenizer_path)
            .map_err(|e| candle::Error::Msg(format!("Arc encoder tokenizer load: {e}")))?;
        let output_proj = linear(cfg.out_dim, output_dim, false, vb.pp("output_proj"))?;
        let learnt_padding = vb.get_as_tensor((1, 1, output_dim), "learnt_padding")?;
        Ok(Self {
            embedder,
            bridge,
            out_dim: output_dim,
            tokenizer,
            output_proj,
            learnt_padding,
            compression_rate: cfg.compression_rate,
            bridge_out_dim: cfg.out_dim,
        })
    }

    /// Replace embedder + bridge weights from a merged ``model.safetensors`` (HF layout: ``embedder.*``, ``bridge_module.*``).
    /// Leaves ``output_proj`` / ``learnt_padding`` from the main checkpoint.
    pub fn reload_from_hf(
        &mut self,
        path: &std::path::Path,
        dtype: DType,
        dev: &candle::Device,
    ) -> Result<()> {
        let inner = unsafe { CandleVarBuilder::from_mmaped_safetensors(&[path], dtype, dev)? };
        let vb = VarBuilder::Real(inner);
        self.embedder = ArcEncoderTransformer::new(self.compression_rate, vb.pp("embedder"))?;
        self.bridge = EmbProjector::new(self.bridge_out_dim, vb.pp("bridge_module"))?;
        Ok(())
    }

    /// Condition from text: tokenize then run through embedder + bridge.
    /// Returns tensor (1, _compressed_len, out_dim).
    pub fn condition(&self, text: &str, device: &candle::Device) -> Result<Tensor> {
        let encoding = self
            .tokenizer
            .encode(text.trim(), false)
            .map_err(|e| candle::Error::Msg(format!("tokenizer encode: {e}")))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        if ids.is_empty() {
            return Ok(self.learnt_padding.clone());
        }
        let seq_len = ids.len();
        let tokens = Tensor::from_vec(ids.clone(), (1, seq_len), device)?;
        let mask = Tensor::ones((1, seq_len), DType::U8, device)?;
        self.condition_with_tokens(&tokens, &mask)
    }

    /// Process tokens with mask: extract valid tokens, run through embedder + bridge.
    /// tokens: (1, seq_len), mask: (1, seq_len) bool/u8.
    /// Returns (1, _compressed_len, out_dim) tensor with embeddings.
    pub fn condition_with_tokens(&self, tokens: &Tensor, mask: &Tensor) -> Result<Tensor> {
        use candle::IndexOp;
        let (_batch_size, seq_len) = tokens.dims2()?;
        let dev = tokens.device();

        let mask = mask.i(0)?;
        let tokens = tokens.i(0)?;
        let mask_u8 = mask.to_dtype(DType::U8)?;
        let valid_indices: Vec<usize> = (0..seq_len)
            .filter(|&j| {
                let v = mask_u8.i(j).and_then(|t| t.to_vec0::<u8>()).unwrap_or(0);
                v != 0
            })
            .collect();
        if valid_indices.is_empty() {
            return Ok(self.learnt_padding.clone());
        }
        let valid_tokens: Vec<u32> = valid_indices
            .iter()
            .filter_map(|&j| tokens.i(j).ok().and_then(|t| t.to_vec0::<u32>().ok()))
            .collect();
        let seq_len = valid_tokens.len();
        let ids = Tensor::from_vec(valid_tokens, (seq_len,), dev)?;
        let (embeddings, _embed_seqlens) = self.embedder.forward_embedder(&ids, &[seq_len])?;
        let out = self.bridge.forward(&embeddings)?;
        let (_compressed_len, _dim) = out.dims2()?;
        let result = out.unsqueeze(0)?;
        let result = result.apply(&self.output_proj)?;

        Ok(result)
    }
}

#[derive(Debug, Clone)]
pub enum Conditioner {
    Lut(LutConditioner),
    ContinuousAttribute(ContinuousAttributeConditioner),
    ArcEncoder(Box<ArcEncoderConditioner>),
}

#[derive(Debug, Clone)]
pub struct ConditionProvider {
    conditioners: HashMap<String, Conditioner>,
}

#[derive(Debug, Clone)]
pub enum Condition {
    AddToInput(Tensor),
}

impl ConditionProvider {
    /// After the main LM load, merge weights from a secondary ``model.safetensors`` into the named conditioner (when supported).
    pub fn load_hf_weights_overlay(
        &mut self,
        conditioner_name: &str,
        path: &std::path::Path,
        dtype: DType,
        dev: &candle::Device,
    ) -> Result<()> {
        let c = self.conditioners.get_mut(conditioner_name).ok_or_else(|| {
            candle::Error::Msg(format!("unknown conditioner '{conditioner_name}'"))
        })?;
        match c {
            Conditioner::ArcEncoder(enc) => enc.reload_from_hf(path, dtype, dev),
            _ => {
                candle::bail!("conditioner '{conditioner_name}' does not support HF weight overlay")
            }
        }
    }

    pub fn new(output_dim: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("conditioners");
        let mut conditioners = HashMap::new();
        for (conditioner_name, conditioner_cfg) in cfg.iter() {
            let vb = vb.pp(conditioner_name);
            let conditioner = match conditioner_cfg {
                ConditionerConfig::Lut(cfg) => {
                    Conditioner::Lut(LutConditioner::new(output_dim, cfg, vb)?)
                }
                ConditionerConfig::ContinuousAttribute(cfg) => Conditioner::ContinuousAttribute(
                    ContinuousAttributeConditioner::new(output_dim, cfg, vb)?,
                ),
                ConditionerConfig::ArcEncoder(cfg) => Conditioner::ArcEncoder(Box::new(
                    ArcEncoderConditioner::new(output_dim, cfg, vb)?,
                )),
            };
            conditioners.insert(conditioner_name.to_string(), conditioner);
        }
        Ok(Self { conditioners })
    }

    pub fn condition_lut(&self, name: &str, value: &str) -> Result<Condition> {
        let c = match self.conditioners.get(name) {
            None => candle::bail!("unknown conditioner {name}"),
            Some(Conditioner::Lut(c)) => c,
            Some(_) => candle::bail!("cannot use conditioner with a str value {name}"),
        };
        let cond = c.condition(value)?;
        Ok(cond)
    }

    pub fn condition_cont(&self, name: &str, value: f32) -> Result<Condition> {
        let c = match self.conditioners.get(name) {
            None => candle::bail!("unknown conditioner {name}"),
            Some(Conditioner::ContinuousAttribute(c)) => c,
            Some(_) => candle::bail!("cannot use conditioner with a str value {name}"),
        };
        let cond = c.condition(value)?;
        Ok(cond)
    }

    pub fn learnt_padding(&self, name: &str) -> Result<Condition> {
        let c = match self.conditioners.get(name) {
            None => candle::bail!("unknown conditioner {name}"),
            Some(Conditioner::ContinuousAttribute(c)) => c.learnt_padding.clone(),
            Some(Conditioner::Lut(c)) => c.learnt_padding.clone(),
            Some(Conditioner::ArcEncoder(c)) => c.learnt_padding.clone(),
        };
        Ok(Condition::AddToInput(c))
    }

    pub fn condition_emb_seq(
        &self,
        name: &str,
        value: &str,
        device: &candle::Device,
    ) -> Result<Option<Tensor>> {
        let c = match self.conditioners.get(name) {
            None => candle::bail!("unknown conditioner {name}"),
            Some(Conditioner::ArcEncoder(c)) => c,
            Some(_) => candle::bail!("cannot use conditioner with a str value {name}"),
        };
        c.condition(value, device).map(Some)
    }

    /// Returns true if a conditioner with this name exists.
    pub fn has(&self, name: &str) -> bool {
        self.conditioners.contains_key(name)
    }
}
