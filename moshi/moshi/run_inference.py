# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the LICENSE file in the root directory of this source tree.

"""Batch offline inference

Run from the ``moshi/moshi`` package root, e.g.:

    uv run python -m moshi.run_inference --input-dir ... --output-dir ...
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any
from copy import deepcopy

import torch

from .models import loaders
from .server import ServerState
from .inference_utils import load_models
from .inference_utils.inference_job import InferenceJob
from .inference_utils.utils import get_reference_encoder_url, seed_all, setup_logging
from .stt.local_stt import LocalSpeechToText

logger = logging.getLogger(__name__)


def load_sidecar_gt(wav_path: Path) -> dict[str, Any]:
    p = wav_path.with_suffix(".json")
    out: dict[str, Any] = {}
    if not p.is_file():
        return out
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.error("failed to read sidecar %s: %s", p, e)
        return out
    if "topic" in data:
        out["gt_user_text"] = data["topic"]
    if "gt_reference_text" in data:
        out["gt_reference_text"] = data["gt_reference_text"]
    if "answer" in data:
        out["answer"] = data["answer"]
    return out


async def _run_wav_job(state: ServerState, stts: list[LocalSpeechToText], args: argparse.Namespace, wav: Path) -> None:
    """Wait for a free batch slot, run one ``InferenceJob``, then release the slot."""
    side = load_sidecar_gt(wav)
    out_path = args.output_dir / f"{wav.stem}.json"

    job = InferenceJob(
        state,
        wav,
        out_path,
        stop_on_end_of_input=args.stop_on_end_of_input,
        use_gt_reference=args.use_gt_reference,
        max_tail_silence=None if args.stop_on_end_of_input else args.max_consecutive_silence_frames,
        sidecar=side,
    )
    slot_idx = await state.wait_acquire_slot(job)
    setattr(job, "slot_idx", slot_idx)
    setattr(job, "stt", stts[slot_idx])
    try:
        async with asyncio.TaskGroup() as jtg:
            await job.run(jtg)
    except BaseException as eg:
        first_err: BaseException | None = None
        for e in eg.exceptions:
            if isinstance(e, asyncio.CancelledError):
                continue
            if first_err is None:
                first_err = e
            logger.error("job %s failed: %s", wav, e, exc_info=e)
    finally:
        if job.slot_idx >= 0:
            await state.release_slot(job.slot_idx)


async def _async_main(
    state: ServerState,
    stt: LocalSpeechToText,
    args: argparse.Namespace,
) -> None:
    step_task = asyncio.create_task(state._step_loop(), name="batched-step-loop")
    stts = [deepcopy(stt) for _ in range(state.batch_size)]
    try:
        wavs = sorted(args.input_dir.glob("*.wav"), key=lambda p: str(p).lower())
        if not wavs:
            logger.warning("no .wav files under %s", args.input_dir)
            return
        await asyncio.gather(*[_run_wav_job(state, stts, args, wav) for wav in wavs])
    finally:
        step_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await step_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--moshi-weight", type=str, default=None)
    parser.add_argument("--mimi-weight", type=str, default=None)
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_RAG_REPO)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--cfg-coef", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--stt-wait-time", type=float, default=0.5)
    parser.add_argument("--half", action="store_const", const=torch.float16, default=torch.bfloat16, dest="dtype")
    parser.add_argument("--rag-timeout", type=float, default=10.0)
    parser.add_argument("--max-reference-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vad-window-size", type=int, default=4)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--power-threshold", type=int, default=-65)
    parser.add_argument("--init-active-speaker", type=str, default="user", choices=["model", "user"])
    parser.add_argument("--stop-on-end-of-input", action="store_true")
    parser.add_argument("--use-gt-reference", action="store_true")
    parser.add_argument("--max-consecutive-silence-frames", type=int, default=None)
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )

    args = parser.parse_args()

    # Configure logging with colored output and centisecond timestamps.
    setup_logging(getattr(logging, args.log_level.upper()))

    if not args.stop_on_end_of_input and args.max_consecutive_silence_frames is None:
        parser.error("--max-consecutive-silence-frames is required unless --stop-on-end-of-input is set")

    # Get and validate REFERENCE_ENCODER_URL environment variable
    reference_encoder_url = get_reference_encoder_url()
    logger.info(f"Using Reference Encoder service at: {reference_encoder_url}")

    seed_all(42424242)

    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all models
    mimi, text_tokenizer, lm_gen = load_models(args)
    stt = LocalSpeechToText(deepcopy(mimi))

    state = ServerState(
        mimi=mimi,
        text_tokenizer=text_tokenizer,
        lm_gen=lm_gen,
        reference_encoder_url=reference_encoder_url,
        stt_wait_time=args.stt_wait_time,
        gradium_stt=False,
        device=args.device,
        rag_timeout=args.rag_timeout,
        max_reference_tokens=args.max_reference_tokens,
        batch_size=args.batch_size,
        vad_window_size=args.vad_window_size,
        vad_threshold=args.vad_threshold,
        init_active_speaker=args.init_active_speaker,
        power_threshold=args.power_threshold,
    )
    logger.info("warming up the model")
    state.warmup()

    with torch.no_grad():
        asyncio.run(_async_main(state, stt, args))


if __name__ == "__main__":
    main()
