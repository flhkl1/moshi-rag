// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use std::collections::VecDeque;

/// Role for text display (model vs user).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextRole {
    Model,
    User,
}

const ROLE_PREFIX_MODEL: &str = "moshi: ";
const ROLE_PREFIX_USER: &str = "user: ";

/// Number of steps to keep initial speaker before allowing VAD-based switch to the other.
/// Avoids switching to User too early so model greeting text can be shown on the live transcript.
const VAD_WARMUP_STEPS: usize = 12;

/// Manages VAD state, speaker transitions, text buffering, and context for RAG.
#[derive(Debug, Clone)]
pub struct TurnManager {
    pub vad_window_size: usize,
    pub vad_threshold: f32,
    pub vad_wait_steps: usize,

    vad_history: VecDeque<f32>,
    wait_counter: usize,
    pending_speaker: Option<TextRole>,
    pub active_speaker: TextRole,
    pub conversation_context: String,
    model_text_buffer: Vec<(String, TextRole)>,
    user_text_buffer: Vec<(String, TextRole)>,
    step_count: usize,
}

impl TurnManager {
    pub fn new(
        vad_window_size: usize,
        vad_threshold: f32,
        vad_wait_steps: usize,
        init_active_speaker: TextRole,
    ) -> Self {
        let (model_text_buffer, user_text_buffer, active_speaker) = match init_active_speaker {
            TextRole::User => {
                (vec![], vec![(ROLE_PREFIX_USER.to_string(), TextRole::User)], TextRole::User)
            }
            TextRole::Model => {
                (vec![(ROLE_PREFIX_MODEL.to_string(), TextRole::Model)], vec![], TextRole::Model)
            }
        };
        Self {
            vad_window_size,
            vad_threshold,
            vad_wait_steps,
            vad_history: VecDeque::with_capacity(vad_window_size + 1),
            wait_counter: 0,
            pending_speaker: None,
            active_speaker,
            conversation_context: String::new(),
            model_text_buffer,
            user_text_buffer,
            step_count: 0,
        }
    }

    pub fn get_context(&self) -> &str {
        &self.conversation_context
    }

    pub fn update_vad(&mut self, vad_value: f32) {
        self.step_count = self.step_count.saturating_add(1);
        self.wait_counter = self.wait_counter.saturating_sub(1);
        if self.vad_history.len() >= self.vad_window_size {
            self.vad_history.pop_front();
        }
        self.vad_history.push_back(vad_value);
    }

    fn handle_pending_speaker_switch(&mut self) -> Option<TextRole> {
        let pending = self.pending_speaker?;
        if self.wait_counter == 0 {
            self.pending_speaker = None;
            return Some(pending);
        }
        None
    }

    fn update_active_speaker(&mut self) -> bool {
        if self.vad_history.len() < self.vad_window_size {
            return false;
        }
        // Don't switch from Model to User in the first VAD_WARMUP_STEPS steps so initial model text shows on transcript.
        let past_warmup = self.step_count >= VAD_WARMUP_STEPS;
        let vad_avg = self.vad_history.iter().sum::<f32>() / self.vad_history.len() as f32;
        let vad_pos = vad_avg < self.vad_threshold;
        let vad_neg = self.vad_history.iter().all(|&v| v > self.vad_threshold);

        if past_warmup && vad_pos && self.active_speaker == TextRole::Model {
            // Switch to User right away if average vad is positive for over VAD_WINDOW_SIZE steps.
            if self.pending_speaker != Some(TextRole::User) {
                self.pending_speaker = Some(TextRole::User);
                self.wait_counter = 0;
            }
        } else if vad_neg && self.active_speaker == TextRole::User {
            // Switch to Model if
            // 1. vad is negative (user is silent) for consecutive VAD_WINDOW_SIZE steps, and
            // 2. model text buffer is not empty (i.e. model said something during previous user turn or model is currently speaking)
            // after waiting for self.vad_wait_steps steps (to avoid switching during a user PAUSE).
            if !self.model_text_buffer.is_empty() && self.pending_speaker != Some(TextRole::Model) {
                self.pending_speaker = Some(TextRole::Model);
                self.wait_counter = self.vad_wait_steps;
            }
        }

        if let Some(speaker) = self.handle_pending_speaker_switch() {
            if speaker != self.active_speaker {
                self.active_speaker = speaker;
                return true;
            }
        }
        false
    }

    /// Returns list of (text, role) to send to client.
    pub fn handle_spoken_text(
        &mut self,
        model_text: Option<&str>,
        user_text: Option<&str>,
    ) -> Vec<(String, TextRole)> {
        let mut outputs = Vec::new();
        if self.update_active_speaker() {
            let prefix = match self.active_speaker {
                TextRole::Model => format!("\n{ROLE_PREFIX_MODEL}"),
                TextRole::User => format!("\n{ROLE_PREFIX_USER}"),
            };
            outputs.push((prefix, self.active_speaker));
        }

        if self.active_speaker == TextRole::User {
            outputs.append(&mut self.user_text_buffer);
            if let Some(t) = user_text {
                if !t.is_empty() {
                    outputs.push((t.to_string(), TextRole::User));
                }
            }
            if let Some(t) = model_text {
                if !t.is_empty() {
                    self.model_text_buffer.push((t.to_string(), TextRole::Model));
                }
            }
        } else {
            outputs.append(&mut self.model_text_buffer);
            if let Some(t) = model_text {
                outputs.push((t.to_string(), TextRole::Model));
            }
            if let Some(t) = user_text {
                self.user_text_buffer.push((t.to_string(), TextRole::User));
            }
        }

        for (text, _) in &outputs {
            // Do not include [RAG] symbols in the conversation context.
            let cleaned = text.replace("[RAG]", "");
            if !cleaned.is_empty() {
                self.conversation_context.push_str(&cleaned);
            }
        }
        outputs
    }

    #[allow(dead_code)]
    pub fn reset(&mut self, init_active_speaker: TextRole) {
        self.vad_history.clear();
        self.wait_counter = 0;
        self.pending_speaker = None;
        self.active_speaker = init_active_speaker;
        self.conversation_context.clear();
        self.model_text_buffer.clear();
        self.user_text_buffer.clear();
        self.step_count = 0;
        match init_active_speaker {
            TextRole::User => {
                self.user_text_buffer.push((ROLE_PREFIX_USER.to_string(), TextRole::User));
            }
            TextRole::Model => {
                self.model_text_buffer.push((ROLE_PREFIX_MODEL.to_string(), TextRole::Model));
            }
        }
    }
}
