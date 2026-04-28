// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

//! Channel pool and pre/post-process for batched streaming.
//! Multiple WebSocket connections share one model loop; each connection gets a slot.

use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::UnboundedSender;

use crate::stream_both::StreamOut;

/// Message from the recv loop into the model loop for one channel.
#[derive(Debug, Clone)]
pub enum InMsg {
    /// Channel ready / init; model loop should reset this slot and send Ready to client.
    Init,
    /// Raw PCM chunk (e.g. after opus decode). Append to channel buffer; when we have a full frame, contribute to batch.
    Audio { pcm: Vec<f32> },
}

/// Unique identifier for a channel (so we can detect slot reuse).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChannelId(usize);

impl ChannelId {
    fn new() -> Self {
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

/// One slot in the pool: receives InMsg, buffers PCM, sends StreamOut to the client.
pub struct Channel {
    pub id: ChannelId,
    pub in_rx: Receiver<InMsg>,
    pub out_tx: UnboundedSender<StreamOut>,
    pub data: VecDeque<f32>,
}

impl Channel {
    /// Create a channel. Caller stores it in the pool and wires the WebSocket to in_tx/out_rx.
    pub fn new(in_rx: Receiver<InMsg>, out_tx: UnboundedSender<StreamOut>) -> Self {
        Self { id: ChannelId::new(), in_rx, out_tx, data: VecDeque::new() }
    }

    /// Append PCM; if we have at least one full frame, return it and remove it from the buffer.
    pub fn extend_data(&mut self, frame_size: usize, pcm: Vec<f32>) -> Option<Vec<f32>> {
        self.data.extend(pcm);
        if self.data.len() >= frame_size {
            Some(self.data.drain(..frame_size).collect())
        } else {
            None
        }
    }

    /// Send to client only if channel id still matches (slot not reused).
    pub fn send(&self, msg: StreamOut, ref_channel_id: Option<ChannelId>) -> Result<(), ()> {
        if Some(self.id) != ref_channel_id {
            return Ok(());
        }
        self.out_tx.send(msg).map_err(|_| ())
    }
}

pub type Channels = Arc<Mutex<Vec<Option<Channel>>>>;

/// Holds the channel pool and frame size for pre_process / post_process.
pub struct BatchedStreamingChannels {
    pub channels: Channels,
    pub batch_size: usize,
    pub frame_size: usize,
}

impl BatchedStreamingChannels {
    pub fn new(batch_size: usize, frame_size: usize) -> Self {
        let channels = (0..batch_size).map(|_| None).collect::<Vec<_>>();
        Self { channels: Arc::new(Mutex::new(channels)), batch_size, frame_size }
    }

    /// Returns true if at least one slot in the batch is currently free.
    pub fn has_free_slot(&self) -> bool {
        let ch = self.channels.lock().unwrap();
        ch.iter().any(|slot| slot.is_none())
    }

    /// Assign a channel; returns (batch_idx, in_tx for the recv task to send to, out_rx for the sender task).
    pub fn take_slot(
        &self,
    ) -> Option<(
        usize,
        std::sync::mpsc::Sender<InMsg>,
        tokio::sync::mpsc::UnboundedReceiver<StreamOut>,
    )> {
        let (in_tx, in_rx) = std::sync::mpsc::channel();
        let (out_tx, out_rx) = tokio::sync::mpsc::unbounded_channel();
        let channel = Channel::new(in_rx, out_tx);
        let mut ch = self.channels.lock().unwrap();
        for (bid, slot) in ch.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(channel);
                return Some((bid, in_tx, out_rx));
            }
        }
        None
    }

    /// Send a message to a specific slot (e.g. initial MetaData after take_slot). Ignores ref_channel_id.
    pub fn send_to_slot(&self, batch_idx: usize, msg: StreamOut) -> Result<(), ()> {
        let ch = self.channels.lock().unwrap();
        if let Some(c) = ch.get(batch_idx).and_then(|s| s.as_ref()) {
            c.out_tx.send(msg).map_err(|_| ())
        } else {
            Err(())
        }
    }

    pub fn pre_process(
        &self,
        state: &mut moshi::batched_lm_generate_multistream::State,
    ) -> (Vec<f32>, Vec<bool>, Vec<Option<ChannelId>>, Vec<usize>) {
        enum Todo {
            Reset(usize),
        }
        let frame_size = self.frame_size;
        let mut mask = vec![false; self.batch_size];
        let mut batch_pcm = vec![0f32; self.batch_size * frame_size];
        let mut channels = self.channels.lock().unwrap();
        let channel_ids: Vec<Option<ChannelId>> =
            channels.iter().map(|c| c.as_ref().map(|c| c.id)).collect();

        let todo: Vec<Todo> = batch_pcm
            .chunks_mut(frame_size)
            .zip(channels.iter_mut())
            .zip(mask.iter_mut())
            .enumerate()
            .filter_map(|(bid, ((out_pcm, slot), mask_elem))| {
                let c = slot.as_mut()?;
                if c.out_tx.is_closed() {
                    *slot = None;
                    return None;
                }
                match c.in_rx.try_recv() {
                    Ok(InMsg::Init) => {
                        let _ = c.out_tx.send(StreamOut::Ready);
                        c.data.clear();
                        *mask_elem = false;
                        Some(Todo::Reset(bid))
                    }
                    Ok(InMsg::Audio { pcm }) => {
                        if let Some(frame) = c.extend_data(frame_size, pcm) {
                            out_pcm.copy_from_slice(&frame);
                            *mask_elem = true;
                        }
                        None
                    }
                    Err(TryRecvError::Empty) => {
                        if let Some(frame) = c.extend_data(frame_size, vec![]) {
                            out_pcm.copy_from_slice(&frame);
                            *mask_elem = true;
                        }
                        None
                    }
                    Err(TryRecvError::Disconnected) => {
                        *slot = None;
                        None
                    }
                }
            })
            .collect();

        let reset_slots: Vec<usize> = todo
            .iter()
            .map(|t| match t {
                Todo::Reset(bid) => *bid,
            })
            .collect();
        for t in todo {
            match t {
                Todo::Reset(bid) => {
                    if let Err(e) = state.reset_batch_idx(bid) {
                        tracing::error!(?e, bid, "reset_batch_idx");
                    }
                }
            }
        }

        (batch_pcm, mask, channel_ids, reset_slots)
    }

    /// Demux step output to the right channels and sends.
    pub fn post_process(
        &self,
        msg: StreamOut,
        batch_idx: usize,
        ref_channel_ids: &[Option<ChannelId>],
    ) -> Result<(), anyhow::Error> {
        let mut channels = self.channels.lock().unwrap();
        if let Some(c) = channels[batch_idx].as_ref() {
            if c.send(msg, ref_channel_ids.get(batch_idx).copied().flatten()).is_err() {
                channels[batch_idx] = None;
            }
        };
        Ok(())
    }
}
