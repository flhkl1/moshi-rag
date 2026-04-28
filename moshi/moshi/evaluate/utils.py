# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import math
import unicodedata
from typing import List, Tuple, Iterable, Optional, Dict, Any

from jiwer import (
    Compose,
    RemoveEmptyStrings,
    RemoveWhiteSpace,
    RemoveMultipleSpaces,
    RemovePunctuation,
    ReduceToListOfListOfWords,
    Strip,
    ToLowerCase,
    compute_measures,
)
import nemo.collections.asr as nemo_asr


SPECIAL_TOKEN_PATTERN = re.compile(r"^<[^>]+>$")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
WHITESPACE_RE = re.compile(r"\s+")

JIWER_TRANSFORM = Compose(
    [
        ToLowerCase(),
        RemovePunctuation(),
        RemoveWhiteSpace(replace_by_space=True),
        RemoveMultipleSpaces(),
        Strip(),
        RemoveEmptyStrings(),
        ReduceToListOfListOfWords(),
    ]
)

parakeet_asr_model = None


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = NON_ALNUM_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokens_to_plain_text(tokens: List[str]) -> str:
    parts: List[str] = []
    for token in tokens or []:
        if not token or SPECIAL_TOKEN_PATTERN.match(token):
            continue
        parts.append(token.replace("\u2581", " "))
    text = "".join(parts)
    return WHITESPACE_RE.sub(" ", text).strip()


def average(values: Iterable[Optional[float]]) -> Optional[float]:
    data = [v for v in values if v is not None and not math.isnan(v)]
    if not data:
        return None
    return sum(data) / len(data)


def compute_asr_ops(reference: str, hypothesis: str) -> Tuple[int, int, int, float, int]:
    try:
        measures = compute_measures(
            reference,
            hypothesis,
            truth_transform=JIWER_TRANSFORM,
            hypothesis_transform=JIWER_TRANSFORM,
        )
    except Exception as err:
        logging.warning("Failed to compute jiwer measures: %s", err)
        return -1, -1, -1, -1.0, -1

    insertions = int(round(measures.get("insertions", -1)))
    deletions = int(round(measures.get("deletions", -1)))
    substitutions = int(round(measures.get("substitutions", -1)))
    wer = float(measures.get("wer", -1.0))
    word_count = len(measures.get("truth")[0]) if "truth" in measures else -1

    if any(value == -1 for value in [insertions, deletions, substitutions, word_count]) or wer == -1:
        return -1, -1, -1, -1.0, -1

    return insertions, deletions, substitutions, wer, word_count


def get_keyword_step_delay(
    tokens: List[str] = [], timestamps: List[Dict[str, Any]] = [], keyword: str = ""
) -> Tuple[int, float, str]:
    if not tokens and not timestamps:
        return -1, -1, ""
    if not keyword:
        return -1, -1, ""

    if timestamps:
        tokens = [" " + timestamp["text"] for timestamp in timestamps]

    cumulative = ""
    # Find the index of the first token that matches any of the answers
    for idx, token in enumerate(tokens):
        if not token or SPECIAL_TOKEN_PATTERN.match(token):
            continue
        cumulative += token.replace("\u2581", " ")
        normalized = normalize_text(cumulative)
        if keyword in normalized:
            break

    # Count back from the index we found until before the answer
    idx_after_answer = idx
    cumulative = ""
    for idx in range(idx_after_answer, -1, -1):
        token = tokens[idx]
        if not token or SPECIAL_TOKEN_PATTERN.match(token):
            continue
        cumulative = token.replace("\u2581", " ") + cumulative
        normalized = normalize_text(cumulative)
        if keyword in normalized:
            if not timestamps:
                # Return the index of the token that matches the beginning of the keyword
                return idx, -1, keyword
            else:
                # Return the timestamp of the token that matches the beginning of the keyword
                if normalize_text(timestamps[idx]["text"]) in normalize_text(keyword):
                    return idx, timestamps[idx]["timestamp"][0], keyword
                else:
                    return -1, -1, ""

    return -1, -1, ""


def get_time_aligned_transcription(audio_path: str) -> List[Dict[str, Any]]:
    # Load the pretrained NeMo ASR model and move to GPU
    global parakeet_asr_model
    if parakeet_asr_model is None:
        parakeet_asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2").cuda()

        # Enable local attention and chunking for subsampling module to make large audio files work
        parakeet_asr_model.change_attention_model("rel_pos_local_attn", [128, 128])  # local attn
        parakeet_asr_model.change_subsampling_conv_chunking_factor(
            1
        )  # chunking for subsampling module; 1 = auto select

    try:
        asr_outputs = parakeet_asr_model.transcribe([audio_path], timestamps=True)
    except Exception as err:
        logging.warning("Failed to transcribe audio: %s", err)
        return []

    # Take the first (and only) result
    result = asr_outputs[0][0]
    word_timestamps = result.timestep["word"]

    # Build the output dict
    timestamps = []
    for w in word_timestamps:
        start_time = w["start"]
        end_time = w["end"]
        word = w["word"]

        timestamps.append(
            {
                "text": word,
                "timestamp": [start_time, end_time],
            }
        )

    return timestamps
