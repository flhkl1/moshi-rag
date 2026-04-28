# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from glob import glob

from ...llm import LLMClient, get_llm
from .base import Judge

import torchaudio
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps
from scipy.spatial.distance import jensenshannon
from scipy.interpolate import interp1d


def check_TO(timestamps: List[Dict[str, Any]], duration_threshold: float, num_words_threshold: int) -> bool:
    if len(timestamps) == 0:
        return False

    # check if the timestamps are valid
    prev_end = -1
    for timestamp_dict in timestamps:
        assert timestamp_dict["timestamp"][0] >= prev_end
        assert timestamp_dict["timestamp"][1] >= timestamp_dict["timestamp"][0]
        prev_end = timestamp_dict["timestamp"][1]

    duration = timestamps[-1]["timestamp"][1] - timestamps[0]["timestamp"][0]
    if duration >= duration_threshold:
        return True
    if len(timestamps) >= num_words_threshold:
        return True
    return False


class BackChannelJudge(Judge):
    WINDOW_SIZE = 0.2
    EPSILON = 1e-10
    TURN_DURATION_THRESHOLD = 1
    TURN_NUM_WORDS_THRESHOLD = 3

    def __init__(self):
        with open("moshi/moshi/evaluate/judge/assets/icc_gt_distribution.json", "r") as f:
            self.gt_distribution = json.load(f)
        self.vad_model = load_silero_vad()

    def __call__(self, input_path: Path, path: Path, timestamps: List[Dict[str, Any]]) -> Dict[str, Any]:
        out_wav_path = path.with_suffix(".wav")

        wav, sr = torchaudio.load(
            out_wav_path,
        )
        # silero_vad supports 16kHz sampling rate
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
        max_end_time = wav.shape[-1] / 16000

        segments = get_speech_timestamps(
            wav,
            self.vad_model,
            return_seconds=True,
        )

        backchannel_prediction = []

        TO = False
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]

            # by the time-aligned trancription, get the timestamps of this segment
            segment_timestamps = []
            for timestamp_dict in timestamps:
                # Get timestamps, handling potential None/null values
                t_start = timestamp_dict["timestamp"][0]
                t_end = timestamp_dict["timestamp"][1]

                # Skip segments with invalid timestamps
                if t_start is None or (t_end is None and t_start > end_time):
                    continue

                # If end is None but start is valid, treat as potentially relevant
                if t_end is None:
                    t_end = t_start  # Assume some duration for null-ended segments

                # Check for any overlap with the target time range
                if t_start >= start_time and t_end <= end_time:
                    pass
                elif t_start <= end_time and t_end > end_time:
                    t_end = end_time
                elif t_start <= start_time and t_end > start_time:
                    t_start = start_time
                else:
                    continue

                segment_timestamps.append({"text": timestamp_dict["text"], "timestamp": [t_start, t_end]})

            this_TO = check_TO(segment_timestamps, self.TURN_DURATION_THRESHOLD, self.TURN_NUM_WORDS_THRESHOLD)
            if not this_TO:
                backchannel_prediction.append((start_time, end_time))

            TO = TO or this_TO

        freq = len(backchannel_prediction) / max_end_time

        # Get ground truth distribution
        spk = path.stem.split("_")[-1]
        gt_distribution = self.gt_distribution[spk]
        js_divergence = self.get_js_divergence(backchannel_prediction, max_end_time, gt_distribution)

        return {
            "TO": TO,
            "js_divergence": js_divergence,
            "freq": freq,
        }

    def get_js_divergence(
        self,
        backchannel_prediction: List[Tuple[float, float]],
        max_end_time: float,
        gt_distribution: Dict[str, List[float]],
    ) -> float:
        if len(backchannel_prediction) == 0:
            js_divergence = 1
        else:
            time_intervals = [0 for i in range(int(max_end_time / self.WINDOW_SIZE) + 1)]

            # Count occurrences in time intervals
            for interval in backchannel_prediction:
                start = int(interval[0] / self.WINDOW_SIZE)
                end = int(interval[1] / self.WINDOW_SIZE)
                for i in range(start, end + 1):
                    if i < len(time_intervals):
                        time_intervals[i] += 1

            # Normalize the time intervals
            time_intervals = np.array(time_intervals)
            time_intervals = time_intervals + self.EPSILON  # Avoid division by zero
            time_intervals = time_intervals / np.sum(time_intervals)
            time_intervals = list(time_intervals)

            # Ensure lengths match using interpolation
            x_gt = np.linspace(0, 1, len(gt_distribution))
            x_pred = np.linspace(0, 1, len(time_intervals))

            interp_func = interp1d(x_gt, gt_distribution, kind="linear", fill_value="extrapolate")
            gt_dist_resized = interp_func(x_pred)

            # Convert to numpy arrays
            hist1 = np.array(time_intervals)
            hist2 = np.array(gt_dist_resized)

            # Calculate Jensen-Shannon divergence
            js_divergence = jensenshannon(hist1, hist2)

        return js_divergence


class PauseHandlingJudge(Judge):
    TURN_DURATION_THRESHOLD = 1
    TURN_NUM_WORDS_THRESHOLD = 3

    def __call__(self, input_path: Path, path: Path, timestamps: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "TO": check_TO(timestamps, self.TURN_DURATION_THRESHOLD, self.TURN_NUM_WORDS_THRESHOLD),
        }


class TurnTakingJudge(Judge):
    TURN_DURATION_THRESHOLD = 1
    TURN_NUM_WORDS_THRESHOLD = 3

    def __call__(self, input_path: Path, path: Path, timestamps: List[Dict[str, Any]]) -> Dict[str, Any]:
        with open(input_path, "r") as f:
            input_turn = json.load(f)["turn_taking"]

        # This follows the original setup of FullDuplexBench-v1.0
        # It might make sense to exlude the timestamps before input_end_time
        TO = check_TO(timestamps, self.TURN_DURATION_THRESHOLD, self.TURN_NUM_WORDS_THRESHOLD)
        latency = -1
        if TO:
            input_end_time = input_turn[0]["timestamp"][0]
            latency = max(0, timestamps[0]["timestamp"][0] - input_end_time)

        return {
            "TO": TO,
            "latency": latency,
        }


class UserInterruptionJudge(Judge):
    TURN_DURATION_THRESHOLD = 1
    TURN_NUM_WORDS_THRESHOLD = 3
    SYSTEM_PROMPT = """
   The scenario is that the user and AI are talking in the spoken conversation.
   The user first speaks, then the AI responds. But when AI is speaking, the user interrupts the AI's turn.
   Your task is to rate the quality of AI's response after the user interrupt the turn.


   Below is the rating guideline (from 0 to 5, 0 is the worst and 5 is the best):
   - 0: The AI's response is totally unrelated to the user's interrupting turn.
   - 1: The AI's response is not related to the user's interrupting turn.
   - 2: The AI's response is slightly related to the user's interrupting turn.
   - 3: The AI's response is related to the user's interrupting turn.
   - 4: The AI's response is highly related to the user's interrupting turn.
   - 5: The AI's response is perfectly related to the user's interrupting turn.


   Firstly, briefly analyze the user's interrupting turn and the AI's response
   Then, you must return the overall output as the following format:
   Analysis: [Your analysis].
   I would rate the AI's response as [Rating].
   """

    def __init__(self, llm: LLMClient | None = None):
        assert llm is None
        self.llm = get_llm(system_prompt=self.SYSTEM_PROMPT, prompt_type="empty")
        self.llm.model_name = "gpt-4-turbo"
        self.max_retries = 1
        self.stop_on_new_line = False

    def __call__(self, input_path: Path, path: Path, timestamps: List[Dict[str, Any]]) -> Dict[str, Any]:
        with open(input_path, "r") as f:
            metadata = json.load(f)["interrupt"]

            in_interrupt_text = metadata[0]["interrupt"]
            in_before_interrupt_text = metadata[0]["context"]
            input_end_time = metadata[0]["timestamp"][1]

        timestamps_after_interrupt = [
            timestamp_dict for timestamp_dict in timestamps if timestamp_dict["timestamp"][0] >= input_end_time
        ]
        out_after_interrupt_text = " ".join([timestamp_dict["text"] for timestamp_dict in timestamps_after_interrupt])

        TO = check_TO(timestamps_after_interrupt, self.TURN_DURATION_THRESHOLD, self.TURN_NUM_WORDS_THRESHOLD)
        latency = -1
        score = -1
        if TO:
            latency = max(0, timestamps_after_interrupt[0]["timestamp"][0] - input_end_time)

            prompt = f"""
            - Contextual user turn: {in_before_interrupt_text}
            - User interrupting turn: {in_interrupt_text}
            - AI's response: {out_after_interrupt_text}
            """

            response = self.llm.generate(prompt=prompt, context="", max_new_tokens=512, seed=0, stop_token=None)
            score = self._parse_response(response)

        return {
            "TO": TO,
            "latency": latency,
            "score": score,
        }

    def _parse_response(self, response: str | None) -> int:
        if response is None:
            return -1
        example_pattern = re.compile(r"Analysis:\s*(.*?)\nI would rate the AI's response as (\d+)", re.DOTALL)

        # Parse the response
        rating = -1
        for match in example_pattern.finditer(response):
            rating = int(match.group(2).strip())

        return rating


class BehaviorJudge(Judge):
    VALID_CATEGORIES = ["C_RESPOND", "C_RESUME", "C_UNCERTAIN_HANDLING", "C_UNKNOWN"]
    DEFAULT_CATEGORY = "C_UNKNOWN"

    def __init__(self, llm: LLMClient | None = None):
        assert llm is None
        with open("moshi/moshi/evaluate/judge/assets/behavior_prompt.txt", "r") as f:
            system_prompt = f.read()
        self.llm = get_llm(system_prompt=system_prompt, prompt_type="empty")
        self.llm.model_name = "gpt-4o-2024-08-06"
        self.max_retries = 1
        self.stop_on_new_line = False

    @staticmethod
    def json_list_to_compact_text(json_list: List[Dict[str, Any]]) -> str:
        return json.dumps(json_list, separators=(",", ":"), ensure_ascii=False)

    def __call__(self, input_path: Path, path: Path, timestamps: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Skip the noisy version and do all analysis on the clean version
        if "clean" not in path.name:
            return {key: -1 for key in self.VALID_CATEGORIES}

        with open(input_path, "r") as f:
            input_data = json.load(f)
            input_noisy = input_data["input"]["chunks"]
            input_clean = input_data["clean_input"]["chunks"]
        output_noisy_path = glob(f"{path.with_name(path.name.replace('clean_', '')).with_suffix('')}.*score*json")[0]
        with open(output_noisy_path, "r") as f:
            output_noisy_data = json.load(f)
            output_noisy = output_noisy_data["processed_data"]["timestamps"]
        output_clean = timestamps

        prompt = f"""
        {{
            "input_clean": {self.json_list_to_compact_text(input_clean)},
            "input_noisy": {self.json_list_to_compact_text(input_noisy)},
            "output_clean": {self.json_list_to_compact_text(output_clean)},
            "output_noisy": {self.json_list_to_compact_text(output_noisy)}
        }}
        """

        response = self.llm.generate(prompt=prompt, context="", max_new_tokens=512, seed=0, stop_token=None)
        category = self._parse_response(response)

        return {key: int(category == key) for key in self.VALID_CATEGORIES}

    def _parse_response(self, response: str | None) -> str:
        if response is None:
            return self.DEFAULT_CATEGORY

        decoder = json.JSONDecoder()
        pos = response.find("{")
        while pos != -1:
            try:
                obj, end = decoder.raw_decode(response, pos)
                if "behavior" in obj:
                    if obj["behavior"][0] in self.VALID_CATEGORIES:
                        return obj["behavior"][0]
                    else:
                        return self.DEFAULT_CATEGORY
                pos = response.find("{", end)
            except json.JSONDecodeError:
                pos = response.find("{", pos + 1)
        return self.DEFAULT_CATEGORY
