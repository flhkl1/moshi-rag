// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use crate::nn::{
    linear, MaybeQuantizedEmbedding as Embedding, MaybeQuantizedLinear as Linear,
    MaybeQuantizedVarBuilder as VarBuilder,
};
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};

// ArcEncoderTransformer constants (match Python ArcEncoderTransformer)
const VOCAB_SIZE: usize = 128256;
const DIM: usize = 3072;
const N_LAYERS: usize = 26;
const HIDDEN_DIM: usize = 8192;
const N_HEADS: usize = 24;
const N_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const NORM_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 500_000.0;
const ROPE_DIM: usize = 128;
const MAX_POS: usize = 128_000;
const BRIDGE_IN_DIM: usize = 3072;
const BRIDGE_HIDDEN_DIM: usize = 2048;

/// Split integer x into chunks, matching the Python `split_integer(x, n)`.
/// Positive n: split into exactly n parts (each ~x/n).
/// Negative n: split into parts of size ~|n| each (number of parts = ceil(x/|n|)).
fn split_integer(x: usize, n: i32) -> Vec<usize> {
    if n > 0 {
        let n = n as usize;
        let base = x / n;
        let remainder = x % n;
        let mut result = vec![base; n];
        for r in result.iter_mut().take(remainder) {
            *r += 1;
        }
        result
    } else {
        let n = (-n) as usize;
        let base = x / n;
        let remainder = x % n;
        if remainder > 0 {
            let chunk_count = base + 1;
            let chunk_size = x / chunk_count;
            let leftover = x % chunk_count;
            let mut result = vec![chunk_size; chunk_count];
            for r in result.iter_mut().take(leftover) {
                *r += 1;
            }
            debug_assert_eq!(result.iter().sum::<usize>(), x);
            result
        } else {
            vec![n; base]
        }
    }
}

#[derive(Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

impl RmsNorm {
    fn new(dim: usize, eps: f32, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_as_tensor((dim,), "weight")?.reshape(dim)?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(x, &self.weight, self.eps)
    }
}

/// Precompute RoPE freqs_cis: shape (MAX_POS, rope_dim/2, 2) with cos and sin.
fn precompute_freqs_cis(dev: &Device) -> Result<Tensor> {
    let half = ROPE_DIM / 2;
    let rope_dim_f = ROPE_DIM as f32;
    let inv_freq: Vec<f32> =
        (0..half).map(|i| 1.0 / ROPE_THETA.powf(2.0 * (i as f32) / rope_dim_f)).collect();
    let inv_freq = Tensor::from_vec(inv_freq, (half,), dev)?;
    let t = Tensor::arange(0u32, MAX_POS as u32, dev)?.to_dtype(DType::F32)?;
    let t = t.reshape((MAX_POS, 1))?;
    let freqs = t.matmul(&inv_freq.reshape((1, half))?)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    Tensor::stack(&[cos, sin], D::Minus1)
}

/// Apply rotary embedding to q or k.
/// x shape (seq_len, n_heads, head_dim), freqs_cis (seq_len, head_dim/2, 2) as cos/sin.
fn apply_rotary_emb(x: &Tensor, freqs_cis: &Tensor) -> Result<Tensor> {
    let (seq_len, n_heads, head_dim) = x.dims3()?;
    let half = head_dim / 2;
    let rope_dtype = x.dtype();

    // Cast to F32 on device (matches Python's .float()), reshape to pairs
    let x_f32 = x.to_dtype(DType::F32)?.reshape((seq_len, n_heads, half, 2))?;
    let x0 = x_f32.i((.., .., .., 0))?; // (seq_len, n_heads, half)
    let x1 = x_f32.i((.., .., .., 1))?; // (seq_len, n_heads, half)

    // freqs_cis is F32 from precompute; extract cos/sin, broadcast to (seq_len, n_heads, half)
    let cos = freqs_cis
        .i((.., .., 0))?
        .unsqueeze(1)?
        .broadcast_as((seq_len, n_heads, half))?
        .contiguous()?;
    let sin = freqs_cis
        .i((.., .., 1))?
        .unsqueeze(1)?
        .broadcast_as((seq_len, n_heads, half))?
        .contiguous()?;
    let x0 = x0.contiguous()?;
    let x1 = x1.contiguous()?;

    // Complex multiply: (x0 + x1*i) * (cos + sin*i) = (x0*cos - x1*sin) + (x0*sin + x1*cos)*i
    let y0 = ((&x0 * &cos)? - (&x1 * &sin)?)?; // (seq_len, n_heads, half)
    let y1 = ((&x0 * &sin)? + (&x1 * &cos)?)?; // (seq_len, n_heads, half)

    // Interleave real/imag back: stack on last dim then flatten
    let y0 = y0.unsqueeze(D::Minus1)?; // (seq_len, n_heads, half, 1)
    let y1 = y1.unsqueeze(D::Minus1)?; // (seq_len, n_heads, half, 1)
    let out = Tensor::cat(&[y0, y1], D::Minus1)?; // (seq_len, n_heads, half, 2)
    out.reshape((seq_len, n_heads, head_dim))?.to_dtype(rope_dtype)
}

/// Repeat KV for GQA: (seq, n_kv_heads, head_dim) -> (seq, n_heads, head_dim)
fn repeat_kv(x: Tensor, repeats: usize) -> Result<Tensor> {
    let (seq, n_kv, d) = x.dims3()?;
    let x = x.unsqueeze(2)?;
    let x = x.broadcast_as((seq, n_kv, repeats, d))?.contiguous()?;
    x.reshape((seq, n_kv * repeats, d))
}

/// Attention with block-diagonal semantics: we process each sequence segment separately (no cross-sequence attention).
#[derive(Clone)]
struct Attention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    repeats: usize,
}

impl Attention {
    fn new(vb: VarBuilder) -> Result<Self> {
        let wq = linear(DIM, N_HEADS * HEAD_DIM, false, vb.pp("wq"))?;
        let wk = linear(DIM, N_KV_HEADS * HEAD_DIM, false, vb.pp("wk"))?;
        let wv = linear(DIM, N_KV_HEADS * HEAD_DIM, false, vb.pp("wv"))?;
        let wo = linear(N_HEADS * HEAD_DIM, DIM, false, vb.pp("wo"))?;
        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            n_heads: N_HEADS,
            n_kv_heads: N_KV_HEADS,
            head_dim: HEAD_DIM,
            repeats: N_HEADS / N_KV_HEADS,
        })
    }

    /// Shared scaled dot-product attention: q,k,v (seq, n_heads, head_dim). Returns (out_seq, dim).
    fn scaled_dot_product_attn(
        &self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        out_seq: usize,
    ) -> Result<Tensor> {
        let q = q.transpose(0, 1)?.contiguous()?;
        let k = k.transpose(0, 1)?.contiguous()?;
        let v = v.transpose(0, 1)?.contiguous()?;
        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let attn = (attn * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(0, 1)?.contiguous()?;
        let out = out.reshape((out_seq, self.n_heads * self.head_dim))?;
        self.wo.forward(&out)
    }

    /// x: (total_seq, dim). Run attention per block; causal=false so full attention within block.
    fn forward(&self, x: &Tensor, freqs_cis: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let (_total, dim) = x.dims2()?;
        let mut outputs = Vec::with_capacity(positions.len());
        let mut offset = 0usize;
        for &seq_len in positions {
            if seq_len == 0 {
                continue;
            }
            let block = x.i(offset..offset + seq_len)?;
            let q = self.wq.forward(&block)?.reshape((seq_len, self.n_heads, self.head_dim))?;
            let k = self.wk.forward(&block)?.reshape((seq_len, self.n_kv_heads, self.head_dim))?;
            let v = self.wv.forward(&block)?.reshape((seq_len, self.n_kv_heads, self.head_dim))?;
            let freqs = freqs_cis.i(offset..offset + seq_len)?;
            let q = apply_rotary_emb(&q, &freqs)?;
            let k = apply_rotary_emb(&k, &freqs)?;
            let k = repeat_kv(k, self.repeats)?;
            let v = repeat_kv(v, self.repeats)?;
            let out = self.scaled_dot_product_attn(q, k, v, seq_len)?;
            outputs.push(out);
            offset += seq_len;
        }
        if outputs.is_empty() {
            return Tensor::zeros((0, dim), DType::F32, x.device());
        }
        Tensor::cat(&outputs, 0)
    }

    /// Cross-attention: Q from q_x (q_seq, dim), K,V from kv_x (kv_seq, dim). Used for compressing layer.
    fn forward_cross(
        &self,
        q_x: &Tensor,
        kv_x: &Tensor,
        freqs_cis_q: &Tensor,
        freqs_cis_kv: &Tensor,
    ) -> Result<Tensor> {
        let (q_seq, _) = q_x.dims2()?;
        let (kv_seq, _) = kv_x.dims2()?;
        let q = self.wq.forward(q_x)?.reshape((q_seq, self.n_heads, self.head_dim))?;
        let k = self.wk.forward(kv_x)?.reshape((kv_seq, self.n_kv_heads, self.head_dim))?;
        let v = self.wv.forward(kv_x)?.reshape((kv_seq, self.n_kv_heads, self.head_dim))?;
        let q = apply_rotary_emb(&q, freqs_cis_q)?;
        let k = apply_rotary_emb(&k, freqs_cis_kv)?;
        let k = repeat_kv(k, self.repeats)?;
        let v = repeat_kv(v, self.repeats)?;
        self.scaled_dot_product_attn(q, k, v, q_seq)
    }
}

/// FeedForward: w2(silu(w1(x)) * w3(x))
#[derive(Clone)]
struct FeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl FeedForward {
    fn new(vb: VarBuilder) -> Result<Self> {
        let w1 = linear(DIM, HIDDEN_DIM, false, vb.pp("w1"))?;
        let w2 = linear(HIDDEN_DIM, DIM, false, vb.pp("w2"))?;
        let w3 = linear(DIM, HIDDEN_DIM, false, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.w3.forward(x)?;
        let gate = (gate * up)?;
        self.w2.forward(&gate)
    }
}

#[derive(Clone)]
struct TransformerBlock {
    attention_norm: RmsNorm,
    attention: Attention,
    ffn_norm: RmsNorm,
    feed_forward: FeedForward,
}

impl TransformerBlock {
    fn new(vb: VarBuilder) -> Result<Self> {
        let attention_norm = RmsNorm::new(DIM, NORM_EPS, vb.pp("attention_norm"))?;
        let attention = Attention::new(vb.pp("attention"))?;
        let ffn_norm = RmsNorm::new(DIM, NORM_EPS, vb.pp("ffn_norm"))?;
        let feed_forward = FeedForward::new(vb.pp("feed_forward"))?;
        Ok(Self { attention_norm, attention, ffn_norm, feed_forward })
    }

    fn forward_inner(&self, x: &Tensor, freqs_cis: &Tensor, seqlens: &[usize]) -> Result<Tensor> {
        let normed = self.attention_norm.forward(x)?;
        let attn_out = self.attention.forward(&normed, freqs_cis, seqlens)?;
        let h = (x + &attn_out)?;
        let ffn_normed = self.ffn_norm.forward(&h)?;
        let ffn_out = self.feed_forward.forward(&ffn_normed)?;
        let out = (&h + &ffn_out)?;
        Ok(out)
    }

    /// Cross-attention block for compressing layer: Q from x (pooled), K,V from other_kv (pre-pooled).
    fn forward_cross(
        &self,
        x: &Tensor,
        other_kv: &Tensor,
        freqs_cis: &Tensor,
        freqs_cis_k: &Tensor,
    ) -> Result<Tensor> {
        let q_normed = self.attention_norm.forward(x)?;
        let kv_normed = self.attention_norm.forward(other_kv)?;
        let attn_out =
            self.attention.forward_cross(&q_normed, &kv_normed, freqs_cis, freqs_cis_k)?;
        let h = (x + &attn_out)?;
        let ffn_normed = self.ffn_norm.forward(&h)?;
        let ffn_out = self.feed_forward.forward(&ffn_normed)?;
        let out = (&h + &ffn_out)?;
        Ok(out)
    }
}

/// Pooling: split each sequence into chunks and mean-pool.
/// comp_rate = -4 means each chunk has ~4 elements (number of chunks = ceil(seq_len/4)).
struct PoolingModule;

impl PoolingModule {
    /// seqlens: lengths per sequence. comp_rate passed directly to split_integer.
    /// Returns (pooled_tensor, new_seqlens). pooled_tensor shape (sum(new_seqlens), dim).
    fn forward(x: &Tensor, comp_rate: i32, seqlens: &[usize]) -> Result<(Tensor, Vec<usize>)> {
        if comp_rate == -1 || comp_rate >= 0 {
            return Ok((x.clone(), seqlens.to_vec()));
        }
        let (_total, _dim) = x.dims2()?;
        let mut new_seqlens = Vec::with_capacity(seqlens.len());
        let mut out_list = Vec::new();
        let mut start = 0usize;
        for &len in seqlens {
            if len == 0 {
                new_seqlens.push(0);
                continue;
            }
            let chunk_sizes = split_integer(len, comp_rate);
            for size in &chunk_sizes {
                let part = x.i(start..start + size)?;
                let mean = part.mean(0)?;
                out_list.push(mean);
                start += size;
            }
            new_seqlens.push(chunk_sizes.len());
        }
        if out_list.is_empty() {
            return Ok((x.clone(), seqlens.to_vec()));
        }
        let out = Tensor::stack(&out_list, 0)?;
        Ok((out, new_seqlens))
    }
}

/// ArcEncoderTransformer: 26 layers, RoPE, optional pooling in last layer.
#[derive(Clone)]
pub struct ArcEncoderTransformer {
    tok_embeddings: Embedding,
    layers: Vec<TransformerBlock>,
    freqs_cis: Tensor,
    compression_rate: i32,
    start_compressing: usize,
}

impl ArcEncoderTransformer {
    pub fn new(compression_rate: i32, vb: VarBuilder) -> Result<Self> {
        let dev = vb.device();
        let tok_embeddings = Embedding::new(VOCAB_SIZE, DIM, vb.pp("tok_embeddings"))?;
        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            let block = TransformerBlock::new(vb.pp("layers").pp(i.to_string()))?;
            layers.push(block);
        }
        let freqs_cis = precompute_freqs_cis(dev)?;
        let start_compressing = N_LAYERS - 1;
        Ok(Self { tok_embeddings, layers, freqs_cis, compression_rate, start_compressing })
    }

    /// input_ids: flat token ids (sum(seqlens) == input_ids.len()). Returns (embeddings, new_seqlens).
    pub fn forward_embedder(
        &self,
        input_ids: &Tensor,
        seqlens: &[usize],
    ) -> Result<(Tensor, Vec<usize>)> {
        let total: usize = seqlens.iter().sum();
        let token_embeds = self.tok_embeddings.forward(input_ids)?;
        let mut h = token_embeds;
        let mut current_seqlens = seqlens.to_vec();
        let positions = positions_from_sizes_i64(seqlens, h.device())?;
        let freqs_cis = self.freqs_cis.index_select(&positions, 0)?;
        let freqs_cis = freqs_cis.reshape((total, ROPE_DIM / 2, 2))?;

        for (i, layer) in self.layers.iter().enumerate() {
            if i >= self.start_compressing {
                let (pooled, new_seqlens) =
                    PoolingModule::forward(&h, self.compression_rate, &current_seqlens)?;
                let new_positions = positions_from_sizes_i64(&new_seqlens, h.device())?;
                let new_freqs = self.freqs_cis.index_select(&new_positions, 0)?;
                let new_total: usize = new_seqlens.iter().sum();
                let new_freqs = new_freqs.reshape((new_total, ROPE_DIM / 2, 2))?;
                // Cross-attention: Q from pooled (3), K,V from pre-pooled h (12). Matches Python other_kv=h.
                h = layer.forward_cross(&pooled, &h, &new_freqs, &freqs_cis)?;
                current_seqlens = new_seqlens;
            } else {
                h = layer.forward_inner(&h, &freqs_cis, &current_seqlens)?;
            }
        }
        Ok((h, current_seqlens))
    }
}

fn positions_from_sizes_i64(seqlens: &[usize], dev: &Device) -> Result<Tensor> {
    let pos: Vec<i64> = seqlens.iter().flat_map(|&len| 0..len as i64).collect();
    let len = pos.len();
    Tensor::from_vec(pos, (len,), dev)
}

/// EmbProjector (bridge): layer1 then layer2, no activation. Matches Python.
#[derive(Clone)]
pub struct EmbProjector {
    layer1: Linear,
    layer2: Linear,
}

impl EmbProjector {
    pub fn new(out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let layer1 = linear(BRIDGE_IN_DIM, BRIDGE_HIDDEN_DIM, false, vb.pp("layer1"))?;
        let layer2 = linear(BRIDGE_HIDDEN_DIM, out_dim, false, vb.pp("layer2"))?;
        Ok(Self { layer1, layer2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.layer1.forward(x)?;
        self.layer2.forward(&x)
    }
}
