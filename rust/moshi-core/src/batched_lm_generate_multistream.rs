// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// Batched multi-stream streaming inference: N independent streams share one LM and one Mimi

use candle::{IndexOp, Tensor};
use candle_transformers::generation::LogitsProcessor;

use crate::lm::ForcedAudioTokens;
use crate::lm_generate_multistream::{Config, UNGENERATED};
use crate::streaming::{StreamMask, StreamTensor};

/// Per-slot state for one stream in the batch (no model reference).
#[derive(Debug)]
pub struct SlotState {
    pub(super) step_idx: usize,
    pub(super) audio_tokens: Vec<Vec<u32>>,
    pub(super) text_tokens: Vec<u32>,
    /// Last STT text token emitted for this slot; used as prev for the STT detokenization (slot-wise memory)
    pub(super) last_stt_text_token: Option<u32>,
    pub(super) forced_audio_tokens: ForcedAudioTokens,
    /// RAG prepend condition (e.g. first_speaker LUT). Applied once at step 0.
    pub(super) prepend_condition: Option<Tensor>,
    /// Pending streaming_sum (1, T, dim). Consumed one step per generation step.
    pub(super) pending_streaming_sum: Option<Tensor>,
    /// Current index into pending_streaming_sum for consumption.
    pub(super) streaming_sum_index: usize,
}

impl SlotState {
    /// Initialize one slot for a stream. Uses the same ring sizes as [crate::lm_generate_multistream::State].
    pub fn new(config: &Config, max_step_idx: usize) -> Self {
        let audio_tokens: Vec<Vec<u32>> = vec![
            vec![UNGENERATED; config.total_audio_codebooks()];
            max_step_idx + config.acoustic_delay
        ];
        let text_tokens = vec![UNGENERATED; max_step_idx + config.acoustic_delay];
        let forced_audio_tokens =
            ForcedAudioTokens::new(config.acoustic_delay, config.audio_pad_token(), &[8, 8]);
        Self {
            step_idx: 0,
            audio_tokens,
            text_tokens,
            last_stt_text_token: None,
            forced_audio_tokens,
            prepend_condition: None,
            pending_streaming_sum: None,
            streaming_sum_index: 0,
        }
    }

    pub fn reset(&mut self, config: &Config) {
        self.step_idx = 0;
        self.audio_tokens.iter_mut().for_each(|row| row.fill(UNGENERATED));
        self.text_tokens.fill(UNGENERATED);
        self.last_stt_text_token = None;
        self.forced_audio_tokens =
            ForcedAudioTokens::new(config.acoustic_delay, config.audio_pad_token(), &[8, 8]);
        self.prepend_condition = None;
        self.pending_streaming_sum = None;
        self.streaming_sum_index = 0;
    }

    pub fn step_idx(&self) -> usize {
        self.step_idx
    }
}

/// Output message for one step, with batch index for demuxing to the right channel.
#[derive(Debug, Clone)]
pub enum StreamingOutMsg {
    /// Decoded PCM for one slot. Backend maps to StreamOut::Pcm.
    Pcm { batch_idx: usize, pcm: Vec<f32> },
    /// Raw text token for one slot; (prev_token, token) will be used for detokenization
    TextToken { batch_idx: usize, prev_token: u32, token: u32, role: TextRole },
}

/// Speaker role for text (user vs model). Kept in core to avoid backend dependency;
/// backend can convert to its own TextRole when demuxing into StreamOut::TextByRole
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextRole {
    User,
    Model,
}

/// Batched streaming state: N streams share one LM and one Mimi.
/// One [SlotState] per batch index; one forward per step over all active slots.
pub struct State {
    model: crate::lm::LmModel,
    mimi: crate::mimi::Mimi,
    slots: Vec<SlotState>,
    /// One [LogitsProcessor] per stream so depformer sampling matches single-stream behavior.
    audio_lp: Vec<LogitsProcessor>,
    text_lp: LogitsProcessor,
    pad_mult: Option<f32>,
    repetition_penalty: Option<(usize, f32)>,
    config: Config,
}

impl State {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: crate::lm::LmModel,
        mimi: crate::mimi::Mimi,
        max_step_idx: usize,
        audio_lp: Vec<LogitsProcessor>,
        text_lp: LogitsProcessor,
        pad_mult: Option<f32>,
        repetition_penalty: Option<(usize, f32)>,
        config: Config,
    ) -> Result<Self, candle::Error> {
        let batch_size = model.batch_size();
        if audio_lp.len() != batch_size {
            candle::bail!("audio_lp len {} != model batch_size {}", audio_lp.len(), batch_size);
        }
        let slots = (0..batch_size).map(|_| SlotState::new(&config, max_step_idx)).collect();
        Ok(Self { model, mimi, slots, audio_lp, text_lp, pad_mult, repetition_penalty, config })
    }

    pub fn batch_size(&self) -> usize {
        self.slots.len()
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Reset a single slot so it can be reused for a new connection.
    pub fn reset_batch_idx(&mut self, batch_idx: usize) -> Result<(), candle::Error> {
        if batch_idx >= self.batch_size() {
            candle::bail!("batch index out of range: {} >= {}", batch_idx, self.batch_size());
        }
        self.slots[batch_idx].reset(&self.config);
        self.model.reset_batch_idx(batch_idx, self.batch_size())?;
        self.mimi.reset_batch_idx(batch_idx, self.batch_size())?;
        Ok(())
    }

    /// Last STT text token for this slot. Like text_tokens[step_idx - 1] for main LM.
    pub fn last_stt_text_token(&self, batch_idx: usize) -> Option<u32> {
        self.slots.get(batch_idx).and_then(|s| s.last_stt_text_token)
    }

    /// Set last STT text token for this slot after emitting a token.
    pub fn set_last_stt_text_token(&mut self, batch_idx: usize, token: u32) {
        if let Some(slot) = self.slots.get_mut(batch_idx) {
            slot.last_stt_text_token = Some(token);
        }
    }

    fn audio_pad_token(&self) -> u32 {
        self.config.audio_pad_token()
    }

    /// Set prepend (LUT) condition for a slot (e.g. first_speaker). Applied once at step 0.
    pub fn set_prepend_condition_lut(
        &mut self,
        batch_idx: usize,
        name: &str,
        text: Option<&str>,
    ) -> Result<(), candle::Error> {
        if batch_idx >= self.batch_size() {
            candle::bail!("batch_idx {} out of range", batch_idx);
        }
        if let Some(text) = text {
            if let Some(tensor) = self.model.get_lut_condition(name, text) {
                self.slots[batch_idx].prepend_condition = Some(tensor);
            }
        }
        Ok(())
    }

    /// Set streaming_sum (ArcEncoder) condition for a slot. Consumed one step per generation step.
    pub fn set_streaming_sum_condition(
        &mut self,
        batch_idx: usize,
        name: &str,
        text: &str,
    ) -> Result<(), candle::Error> {
        if batch_idx >= self.batch_size() {
            candle::bail!("batch_idx {} out of range", batch_idx);
        }
        if let Some(tensor) = self.model.get_emb_seq_condition(name, text) {
            self.slots[batch_idx].pending_streaming_sum = Some(tensor);
            self.slots[batch_idx].streaming_sum_index = 0;
        }
        Ok(())
    }

    /// Build per-step batched condition tensor (batch_size, 1, dim) from per-slot streaming_sum.
    /// Prepend (e.g. first_speaker) is handled separately via `forward_prepend` before step 0.
    /// Returns None if no slot has any streaming_sum condition.
    fn build_per_step_conditions(&mut self, _mask: &StreamMask) -> candle::Result<Option<Tensor>> {
        let batch_size = self.batch_size();
        let template = match self.model.get_emb_seq_condition("reference_with_time", "") {
            Some(t) => t,
            None => return Ok(None),
        };
        let (_, _, cond_dim) = template.dims3()?;
        let model_dtype = self.model.dtype();
        let dev = template.device();
        let empty =
            Tensor::zeros((1, 1, cond_dim), template.dtype(), dev)?.to_dtype(model_dtype)?;
        let mut has_any = false;
        let mut all_conds: Vec<Tensor> = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let slot = &mut self.slots[b];
            // Take one streaming_sum step (if any) as [1, 1, dim] for this slot.
            let streaming_step: Option<Tensor> =
                slot.pending_streaming_sum.as_ref().and_then(|t| {
                    let t_len = t.dim(1).ok()?;
                    if slot.streaming_sum_index < t_len {
                        let step = t
                            .i((.., slot.streaming_sum_index..slot.streaming_sum_index + 1, ..))
                            .ok()?;
                        slot.streaming_sum_index += 1;
                        has_any = true;
                        Some(step.to_dtype(model_dtype).ok()?)
                    } else {
                        None
                    }
                });
            let cond_b = match streaming_step {
                Some(s) => s,
                None => empty.clone(),
            };
            all_conds.push(cond_b);
        }
        if !has_any {
            return Ok(None);
        }
        let batched = Tensor::cat(&all_conds, 0)?;
        Ok(Some(batched))
    }

    /// Apply repetition penalty per batch index using each slot's text history.
    fn apply_repetition_penalty_batched(
        &self,
        logits: &Tensor,
        mask: &StreamMask,
    ) -> candle::Result<Tensor> {
        let (batch_size, _one, vocab_size) = logits.dims3()?;
        let device = logits.device();
        let (context_size, penalty_factor) = match self.repetition_penalty {
            None => return Ok(logits.clone()),
            Some((_, 1.0)) => return Ok(logits.clone()),
            Some((ctx, p)) => (ctx, p),
        };
        let mut out = logits.to_dtype(candle::DType::F32)?.to_vec2::<f32>()?;
        for (b, out_row) in out.iter_mut().enumerate().take(batch_size) {
            if !mask.is_active(b) {
                continue;
            }
            let slot = &self.slots[b];
            let mut non_pad_tokens = 0usize;
            let mut already_seen = std::collections::HashSet::new();
            let max_idx = slot.step_idx.min(slot.text_tokens.len());
            for &token_id in slot.text_tokens[..max_idx].iter().rev() {
                if token_id == self.config.text_pad_token
                    || token_id == self.config.text_eop_token
                    || token_id == self.config.text_start_token
                {
                    continue;
                }
                if non_pad_tokens >= context_size {
                    break;
                }
                non_pad_tokens += 1;
                if already_seen.contains(&token_id) {
                    continue;
                }
                already_seen.insert(token_id);
                let idx = token_id as usize;
                if idx < vocab_size {
                    let v = &mut out_row[idx];
                    if *v >= 0.0 {
                        *v /= penalty_factor;
                    } else {
                        *v *= penalty_factor;
                    }
                }
            }
        }
        Tensor::from_vec(
            out.into_iter().flatten().collect::<Vec<_>>(),
            (batch_size, 1, vocab_size),
            device,
        )
    }

    /// One batched step: encode PCM, run LM steps for each time step, decode and emit messages.
    /// `pcm` shape `(batch_size, 1, frame_len)`; `mask` indicates which slots have valid input.
    pub fn step_pcm(
        &mut self,
        pcm: &Tensor,
        mask: &StreamMask,
    ) -> candle::Result<Vec<StreamingOutMsg>> {
        let dev = self.model.device().clone();
        let batch_size = self.batch_size();
        let mut out = vec![];

        // Apply prepend conditioning before step-0 input (same as single-stream forward_prepend).
        let mut prepend_mask = vec![false; batch_size];
        let mut prepend_list: Vec<Option<Tensor>> = (0..batch_size)
            .map(|b| {
                let slot = &mut self.slots[b];
                if mask.is_active(b) && slot.step_idx == 0 {
                    slot.prepend_condition.take()
                } else {
                    None
                }
            })
            .collect();
        if prepend_list.iter().any(|t| t.is_some()) {
            let (_, t_len, dim) = prepend_list
                .iter()
                .find_map(|t| t.as_ref())
                .and_then(|t| t.dims3().ok())
                .ok_or_else(|| candle::Error::Msg("prepend shape".to_string()))?;
            let dev = self.model.device();
            let empty = Tensor::zeros((1, t_len, dim), candle::DType::BF16, dev)?;
            let batch_tensors: Vec<Tensor> = (0..batch_size)
                .map(|b| {
                    prepend_mask[b] = prepend_list[b].is_some();
                    prepend_list[b].take().unwrap_or_else(|| empty.clone())
                })
                .collect();
            let prepend_batch = Tensor::cat(&batch_tensors, 0)?;
            let prepend_stream_mask = crate::streaming::StreamMask::new(prepend_mask, dev)?;
            self.model.forward_prepend(&prepend_batch, Some(&prepend_stream_mask))?;
        }

        let mut codes = Vec::with_capacity(self.config.total_audio_codebooks());

        let pcm_st = StreamTensor::from_tensor(pcm.clone());
        let input_audio_tokens = self.mimi.encode_step(&pcm_st, mask)?;
        let input_audio_tokens = match input_audio_tokens.as_option() {
            None => return Ok(out),
            Some(t) => t,
        };
        let (_batch, _codebooks, steps) = input_audio_tokens.dims3()?;
        if steps != 1 {
            candle::bail!("Only one step a time for batched generation");
        }

        for b in 0..batch_size {
            if !mask.is_active(b) {
                continue;
            }
            let slot = &mut self.slots[b];
            if slot.step_idx >= slot.audio_tokens.len() {
                candle::bail!("slot {b} step_idx overflow");
            }
            for c in 0..self.config.input_audio_codebooks {
                let t = input_audio_tokens.i((b, c, 0))?.to_vec0::<u32>()?;
                slot.audio_tokens[slot.step_idx][self.config.generated_audio_codebooks + c] = t;
            }
        }

        for codebook in 0..self.config.total_audio_codebooks() {
            let mut vals = vec![self.audio_pad_token(); batch_size];
            for (b, val) in vals.iter_mut().enumerate().take(batch_size) {
                let slot = &self.slots[b];
                let t = if codebook == 0 || codebook == self.config.generated_audio_codebooks {
                    if slot.step_idx == 0 {
                        self.audio_pad_token()
                    } else {
                        slot.audio_tokens[slot.step_idx - 1][codebook]
                    }
                } else if slot.step_idx <= self.config.acoustic_delay {
                    self.audio_pad_token()
                } else {
                    slot.audio_tokens[slot.step_idx - self.config.acoustic_delay - 1][codebook]
                };
                if mask.is_active(b) {
                    *val = t;
                }
                if *val == UNGENERATED {
                    candle::bail!(
                        "ungenerated token at step {} codebook {}",
                        slot.step_idx,
                        codebook
                    );
                }
            }
            let t = Tensor::from_vec(vals.clone(), (batch_size, 1), &dev)?;
            codes.push(Some(t));
        }

        let mut text_vals = vec![self.config.text_start_token; batch_size];
        for (b, tv) in text_vals.iter_mut().enumerate().take(batch_size) {
            if !mask.is_active(b) {
                continue;
            }
            let slot = &self.slots[b];
            *tv = if slot.step_idx == 0 {
                self.config.text_start_token
            } else {
                slot.text_tokens[slot.step_idx - 1]
            };
        }
        let text_token = Tensor::from_vec(text_vals, (batch_size, 1), &dev)?;

        let cond_tensor = self.build_per_step_conditions(mask)?;
        let conditions =
            cond_tensor.as_ref().map(|t| crate::conditioner::Condition::AddToInput(t.clone()));
        let (text_logits, ys) =
            self.model.forward_cond(Some(text_token), codes, conditions.as_ref(), mask)?;
        let text_logits = self.apply_repetition_penalty_batched(&text_logits, mask)?;

        let mut sampled_text = vec![self.config.text_start_token; batch_size];
        for (b, st) in sampled_text.iter_mut().enumerate().take(batch_size) {
            if !mask.is_active(b) {
                continue;
            }
            let logits_b = text_logits.i((b, 0))?;
            let token = self.text_lp.sample_f(&logits_b, |prs| {
                if let Some(pm) = self.pad_mult.as_ref() {
                    let idx = self.config.text_pad_token as usize;
                    if idx < prs.len() {
                        prs[idx] *= pm.exp();
                    }
                }
            })?;
            *st = token;
            let step_idx_b = self.slots[b].step_idx;
            let prev = if step_idx_b == 0 {
                self.config.text_start_token
            } else {
                self.slots[b].text_tokens[step_idx_b - 1]
            };
            self.slots[b].text_tokens[step_idx_b] = token;
            out.push(StreamingOutMsg::TextToken {
                batch_idx: b,
                prev_token: prev,
                token,
                role: TextRole::Model,
            });
        }

        let audio_pad = self.audio_pad_token();
        let forced: Vec<Vec<Option<u32>>> = (0..batch_size)
            .map(|b| {
                self.slots[b].forced_audio_tokens.forced_tokens(self.slots[b].step_idx).to_vec()
            })
            .collect();
        let sampled_text: Vec<Option<u32>> = sampled_text.iter().copied().map(Some).collect();
        let last_audio_per_batch = self.model.depformer_sample_batched(
            &ys,
            &sampled_text,
            &forced,
            mask,
            &mut self.audio_lp,
        )?;
        for (b, last_audio_entry) in last_audio_per_batch.iter().enumerate().take(batch_size) {
            if !mask.is_active(b) {
                continue;
            }
            let slot = &mut self.slots[b];
            let last_audio = match last_audio_entry {
                Some(tokens) => tokens.as_slice(),
                None => continue,
            };
            for c_idx in 0..self.config.generated_audio_codebooks {
                let delay = if c_idx == 0 || c_idx == self.config.generated_audio_codebooks {
                    0
                } else {
                    self.config.acoustic_delay
                };
                let pos = slot.step_idx.saturating_sub(delay);
                let v = last_audio.get(c_idx).copied().unwrap_or(audio_pad);
                slot.audio_tokens[pos][c_idx] = v;
            }
            slot.step_idx += 1;
            if slot.step_idx >= slot.audio_tokens.len() {
                candle::bail!("slot {b} max step_idx reached");
            }
        }

        let need_decode: Vec<usize> = (0..batch_size)
            .filter(|&b| mask.is_active(b) && self.slots[b].step_idx > self.config.acoustic_delay)
            .collect();
        if need_decode.is_empty() {
            return Ok(out);
        }
        let mimi_cb = self.mimi.config().quantizer_n_q;
        let cb = self.config.generated_audio_codebooks.min(mimi_cb);
        let mut decode_codes = vec![0u32; batch_size * cb];
        for &b in &need_decode {
            let slot = &self.slots[b];
            let idx = slot.step_idx - self.config.acoustic_delay - 1;
            let row = &slot.audio_tokens[idx][..cb];
            for (i, &v) in row.iter().enumerate() {
                decode_codes[b * cb + i] = v;
            }
        }
        let decode_tensor = Tensor::from_vec(decode_codes, (batch_size, cb, 1), &dev)?;
        let decode_mask = StreamMask::new(
            (0..batch_size).map(|b| need_decode.contains(&b)).collect::<Vec<_>>(),
            &dev,
        )?;
        let pcm_out = self.mimi.decode_step(&decode_tensor.into(), &decode_mask)?;
        if let Some(pcm_out) = pcm_out.as_option() {
            for b in 0..batch_size {
                if !need_decode.contains(&b) {
                    continue;
                }
                let pcm_vec = pcm_out.i((b, 0))?.to_vec1::<f32>()?;
                out.push(StreamingOutMsg::Pcm { batch_idx: b, pcm: pcm_vec });
            }
        }
        Ok(out)
    }

    /// Run one step with zeros and full mask to warm up the model (e.g. at startup).
    /// `frame_len` should match mimi frame size: `(sample_rate / frame_rate).ceil() as usize`.
    /// The backend calls this once when the batched model loop starts.
    pub fn warmup(&mut self, frame_len: usize) -> candle::Result<()> {
        let batch_size = self.batch_size();
        let dev = self.model.device().clone();
        let pcm = candle::Tensor::zeros((batch_size, 1, frame_len), candle::DType::F32, &dev)?;
        let mask = StreamMask::new(vec![true; batch_size], &dev)?;
        let _ = self.step_pcm(&pcm, &mask)?;
        dev.synchronize()?;
        Ok(())
    }
}
