# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from calflops import calculate_flops
from accelerate import init_empty_weights
from scipy.io import wavfile

from .judge import Judge, LLMJudge, KeywordLLMJudge
from .utils import (
    compute_asr_ops,
    tokens_to_plain_text,
    average,
    get_keyword_step_delay,
    get_time_aligned_transcription,
)


MIMI_FRAME_RATE = 12.5

model_id = "google/gemma-3-27b-it"
gemma_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
gemma_tokenizer = AutoTokenizer.from_pretrained(model_id)
with init_empty_weights():
    gemma_model = AutoModelForCausalLM.from_config(gemma_config, trust_remote_code=True)
gemma_model = gemma_model.to_empty(device="cpu")


@dataclass
class ScoreResult:
    # Raw data
    input_path: Path = Path("")
    path: Path = Path("")
    rag_trigger_step: int = -1
    retrieval_step: int = -1
    conditioning_step: int = -1
    question_end_step: int = -1
    reference_text: str = ""
    user_text: Optional[List[str]] | str = None
    model_text: Optional[List[str]] | str = None
    gt_user_text: str = ""
    gt_reference_text: str = ""
    answer: Any = None
    query: Any = None
    # Profiling
    elapsed: float = -1
    first_audio_token_latency: float = -1
    profile: Dict[str, Any] = field(default_factory=dict)

    # Processed data
    processed_user_text: str = ""
    processed_model_text: str = ""
    processed_answer: Optional[List[str]] = None
    keyword_step: int = -1
    timestamps: List[Dict[str, Any]] = field(default_factory=list)

    # Final scores
    rag_trigger_delay: float = -1
    # This is actually retrieval_delay_minus_rag_trigger_delay, but we keep its name as retrieval_delay for backward compatibility
    # The naming will be corrected in summarize_folder
    retrieval_delay: float = -1
    keyword_delay: int = -1
    insertion: int = -1
    removal: int = -1
    substitution: int = -1
    wer: float = -1
    word_count: int = -1
    reference_correctness: int = -1
    correctness: int = -1
    duplex_scores: Dict[str, Any] = field(default_factory=dict)
    keyword: str = ""
    total_tflops: int = -1
    rag_tflops: int = -1
    tflops_per_second: float = -1
    rag_tflops_per_second: float = -1
    wav_duration_seconds: float = -1

    @classmethod
    def from_dict(cls, data: Dict) -> ScoreResult:
        return cls(**{subkey: data[key][subkey] for key in data.keys() for subkey in data.get(key, {})})

    def get_raw_data(self) -> Dict:
        return {
            "path": str(self.path),
            "rag_trigger_step": self.rag_trigger_step,
            "retrieval_step": self.retrieval_step,
            "conditioning_step": self.conditioning_step,
            "question_end_step": self.question_end_step,
            "reference_text": self.reference_text,
            "user_text": self.user_text,
            "model_text": self.model_text,
            "gt_user_text": self.gt_user_text,
            "gt_reference_text": self.gt_reference_text,
            "answer": self.answer,
            "query": self.query,
            "elapsed": self.elapsed,
            "first_audio_token_latency": self.first_audio_token_latency,
            "profile": self.profile,
        }

    def get_processed_data(self) -> Dict:
        return {
            "processed_user_text": self.processed_user_text,
            "processed_model_text": self.processed_model_text,
            "processed_answer": self.processed_answer,
            "keyword_step": self.keyword_step,
            "timestamps": self.timestamps,
        }

    def get_scores(self) -> Dict:
        return {
            "path": str(self.path),
            "rag_trigger_delay": self.rag_trigger_delay,
            "retrieval_delay": self.retrieval_delay,
            "insertion": self.insertion,
            "removal": self.removal,
            "substitution": self.substitution,
            "wer": self.wer,
            "word_count": self.word_count,
            "reference_correctness": self.reference_correctness,
            "correctness": self.correctness,
            "keyword": self.keyword,
            "keyword_delay": self.keyword_delay,
            "duplex_scores": self.duplex_scores,
            "first_audio_token_latency": self.first_audio_token_latency,
            "total_tflops": self.total_tflops,
            "rag_tflops": self.rag_tflops,
            "tflops_per_second": self.tflops_per_second,
            "rag_tflops_per_second": self.rag_tflops_per_second,
            "wav_duration_seconds": self.wav_duration_seconds,
        }

    def process_data(self) -> None:
        self.user_text = self.user_text or []
        self.model_text = self.model_text or []
        self.processed_user_text = (
            self.user_text if isinstance(self.user_text, str) else tokens_to_plain_text(self.user_text)
        )
        self.processed_model_text = (
            self.model_text if isinstance(self.model_text, str) else tokens_to_plain_text(self.model_text)
        )
        self.processed_answer = extract_answer_variants(self.answer)

    def get_timestamps(self) -> None:
        self.timestamps = get_time_aligned_transcription(str(self.path.with_suffix(".wav")))

    def compute_scores(
        self, keyword_judge: KeywordLLMJudge | None, correctness_judge: LLMJudge | None, duplex_judge: Judge | None
    ) -> None:
        (
            self.insertion,
            self.removal,
            self.substitution,
            self.wer,
            self.word_count,
        ) = compute_asr_ops(self.gt_user_text, self.processed_user_text)

        if correctness_judge is not None:
            self.reference_correctness = correctness_judge(
                self.gt_user_text, self.reference_text, self.processed_answer
            )
            self.correctness = correctness_judge(self.gt_user_text, self.processed_model_text, self.processed_answer)
        else:
            self.reference_correctness = -1
            self.correctness = -1

        if duplex_judge is not None:
            self.duplex_scores = duplex_judge(self.input_path, self.path, self.timestamps)
        else:
            self.duplex_scores = {}

        # Default values
        self.keyword_step = -1
        self.keyword_delay = -1
        self.keyword = ""
        if keyword_judge is not None:
            keyword = keyword_judge(self.gt_user_text, self.processed_model_text, self.processed_answer)
            if keyword:
                assert self.model_text is not None
                if isinstance(self.model_text, list):
                    keyword_step, _, keyword = get_keyword_step_delay(tokens=self.model_text, keyword=keyword)
                    keyword_delay = keyword_step - self.question_end_step
                elif isinstance(self.model_text, str):
                    keyword_step, keyword_delay, keyword = get_keyword_step_delay(
                        timestamps=self.timestamps, keyword=keyword
                    )
                    keyword_delay = int(keyword_delay * MIMI_FRAME_RATE)
                if keyword:
                    assert keyword_step != -1 or keyword_delay != -1
                    self.keyword_delay = keyword_delay
                    self.keyword_step = keyword_step
                    self.keyword = keyword

        if self.retrieval_step != -1 and self.rag_trigger_step != -1:
            self.rag_trigger_delay = self.retrieval_step - self.rag_trigger_step
        else:
            self.rag_trigger_delay = -1
        if self.conditioning_step != -1 and self.retrieval_step != -1:
            self.retrieval_delay = self.conditioning_step - self.retrieval_step
        else:
            self.retrieval_delay = -1

        flops_profile = self.profile or {}
        self.total_tflops = 0
        self.rag_tflops = 0
        for key, value in flops_profile.items():
            if key == "rag":
                gemma_seq_len = value["tokens"]
                rag_flops, _, _ = calculate_flops(
                    gemma_model,
                    input_shape=(1, gemma_seq_len),
                    transformer_tokenizer=gemma_tokenizer,
                    output_as_string=False,
                    print_results=False,
                )
                self.rag_tflops = rag_flops / 1e12
                self.total_tflops += self.rag_tflops
            elif isinstance(value, dict) and "flops" in value:
                self.total_tflops += value["flops"] / 1e12
        wav_path = self.path.with_suffix(".wav")
        sr, wav = wavfile.read(wav_path)
        wav_duration = len(wav) / sr
        self.tflops_per_second = self.total_tflops / wav_duration
        self.rag_tflops_per_second = self.rag_tflops / wav_duration
        self.wav_duration_seconds = wav_duration


def score_file(
    path: Path,
    output_path: Path,
    keyword_judge: KeywordLLMJudge | None,
    correctness_judge: LLMJudge | None,
    duplex_judge: Judge | None,
) -> Optional[ScoreResult]:
    try:
        with path.open() as f:
            entry = json.load(f)
    except Exception as err:
        logging.warning("Failed to parse %s: %s", path, err)
        return None

    score_result = ScoreResult(**entry, path=path)
    score_result.process_data()
    score_result.get_timestamps()
    score_result.compute_scores(keyword_judge, correctness_judge, duplex_judge)

    result_data = {
        "scores": score_result.get_scores(),
        "raw_data": score_result.get_raw_data(),
        "processed_data": score_result.get_processed_data(),
    }
    output_path.write_text(json.dumps(result_data, indent=2))
    return score_result


def summarize_folder(folder: Path, records: List[ScoreResult], model_name: str) -> Dict:
    summary = {
        "folder": str(folder),
        "num_examples": len(records),
    }

    total_rag_trigger_delay: List[float] = []
    total_retrieval_delay: List[float] = []
    total_insertion: List[int] = []
    total_removal: List[int] = []
    total_substitution: List[int] = []
    total_word_count: List[int] = []
    total_reference_correctness: List[int] = []
    total_correctness: List[int] = []
    total_keyword_delay: List[float] = []
    total_duplex_scores: Dict[str, Any] = {}
    total_rag_count: int = 0
    total_first_audio_token_latency: List[float] = []
    total_tflops: List[float] = []
    total_rag_tflops: List[float] = []
    total_wav_duration_seconds: List[float] = []

    for rec in records:
        if not any(
            getattr(rec, key) == -1
            for key in [
                "insertion",
                "removal",
                "substitution",
                "word_count",
            ]
        ):
            total_insertion.append(rec.insertion)
            total_removal.append(rec.removal)
            total_substitution.append(rec.substitution)
            total_word_count.append(rec.word_count)
        if rec.rag_trigger_step is not None and rec.rag_trigger_delay >= 0:
            total_rag_trigger_delay.append(rec.rag_trigger_delay / MIMI_FRAME_RATE)
        if rec.retrieval_step is not None and rec.retrieval_delay >= 0:
            total_retrieval_delay.append(rec.retrieval_delay / MIMI_FRAME_RATE)
        if rec.reference_correctness is not None and rec.reference_correctness >= 0:
            total_reference_correctness.append(rec.reference_correctness)
        if rec.correctness is not None and rec.correctness >= 0:
            total_correctness.append(rec.correctness)

        if (
            rec.keyword_step is not None
            and rec.keyword_delay is not None
            and rec.keyword_step >= 0
            and rec.keyword_delay >= 0
        ):
            total_keyword_delay.append(rec.keyword_delay / MIMI_FRAME_RATE)

        total_rag_count += rec.reference_text != ""

        for key, value in rec.duplex_scores.items():
            if key not in total_duplex_scores:
                total_duplex_scores[key] = []
            if value is not None and value >= 0:
                total_duplex_scores[key].append(value)

        if rec.first_audio_token_latency is not None and rec.first_audio_token_latency >= 0:
            total_first_audio_token_latency.append(rec.first_audio_token_latency)

        if rec.total_tflops is not None and rec.total_tflops >= 0:
            total_tflops.append(rec.total_tflops)
        if rec.rag_tflops is not None and rec.rag_tflops >= 0:
            total_rag_tflops.append(rec.rag_tflops)
        if rec.wav_duration_seconds is not None and rec.wav_duration_seconds >= 0:
            total_wav_duration_seconds.append(rec.wav_duration_seconds)

    summary["rag_trigger_delay"] = {
        "avg": average(total_rag_trigger_delay),
        "num_examples": len(total_rag_trigger_delay),
    }
    summary["retrieval_delay_minus_rag_trigger_delay"] = {
        "avg": average(total_retrieval_delay),
        "num_examples": len(total_retrieval_delay),
    }
    if average(total_rag_trigger_delay) is not None and average(total_retrieval_delay) is not None:
        summary["retrieval_delay"] = average(total_retrieval_delay) + average(total_rag_trigger_delay)
    else:
        summary["retrieval_delay"] = None
    insertions = sum(total_insertion)
    removals = sum(total_removal)
    substitutions = sum(total_substitution)
    word_count = sum(total_word_count)
    wer = (insertions + removals + substitutions) / word_count if word_count else None
    summary["wer"] = {"avg": wer, "num_examples": len(total_word_count)}

    summary["reference_correctness"] = {
        "avg": average(total_reference_correctness),
        "num_examples": len(total_reference_correctness),
    }
    summary["correctness"] = {"avg": average(total_correctness), "num_examples": len(total_correctness)}
    summary["keyword_delay"] = {"avg": average(total_keyword_delay), "num_examples": len(total_keyword_delay)}
    summary["duplex_scores"] = {
        key: {"avg": average(total_duplex_scores[key]), "num_examples": len(total_duplex_scores[key])}
        for key in total_duplex_scores.keys()
    }
    summary["rag_percentage"] = total_rag_count / len(records)
    summary["first_audio_token_latency"] = {
        "avg": average(total_first_audio_token_latency),
        "num_examples": len(total_first_audio_token_latency),
    }
    if sum(total_wav_duration_seconds) > 0:
        summary["tflops_per_second"] = sum(total_tflops) / sum(total_wav_duration_seconds)
    summary["tflops_per_sample"] = average(total_tflops)
    summary["rag_tflops_per_sample"] = average(total_rag_tflops)

    output_path = folder / f"score_summary.{model_name}.json"
    output_path.write_text(json.dumps(summary, indent=2))
    return summary


def extract_answer_variants(answer_field) -> List[str]:
    variants: List[str] = []
    if isinstance(answer_field, str):
        variants.append(answer_field)
    elif isinstance(answer_field, list):
        variants.extend(str(item) for item in answer_field if item)
    elif isinstance(answer_field, dict):
        for key in ("normalized_aliases", "aliases"):
            values = answer_field.get(key)
            if isinstance(values, list):
                variants.extend(values)
        for key in ("normalized_value", "value"):
            value = answer_field.get(key)
            if value:
                variants.append(value)
    else:
        logging.debug("Unsupported answer format (%s)", type(answer_field))

    variants = [v for v in variants if v]
    deduped = list(dict.fromkeys(filter(None, variants)))
    return deduped
