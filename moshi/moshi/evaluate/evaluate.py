# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Score Moshi RAG experiment outputs.

This utility walks through `/results/*/*` directories and computes:
- rag_trigger_delay: retrieval_step - rag_trigger_step (equals to stt_wait_time in server_rag.py)
- retrieval_delay_minus_rag_trigger_delay: conditioning_step - retrieval_step
- retrieval_delay: rag_trigger_delay + retrieval_delay_minus_rag_trigger_delay
- ASR statistics (insertions/removals/substitutions, WER) between gt_user_text and user_text
- reference/model correctness (boolean) wrt gold answers
- keyword_delay: keyword_step - question_end_step
- duplex_scores: based on Full-Duplex-Bench 1.0 and 1.5
- rag_percentage: fraction of samples using RAG
- first_audio_token_latency: time (seconds) from question_end_step to first generated audio token
- tflops_per_sample: average total FLOPs per sample from profiling
- rag_tflops_per_sample: average FLOPs attributed to the RAG (retrieval) model per sample

Keyword detection uses an LLM.
Dataset-dependent scoring functions in moshi/moshi/evaluate/judge/.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List

from tqdm import tqdm

from .judge import (
    KeywordLLMJudge,
    SimpleQALLMJudge,
    MathQALLMJudge,
    TriviaQAJudge,
    LLamaQuestionsJudge,
    WebQuestionsJudge,
    BackChannelJudge,
    PauseHandlingJudge,
    TurnTakingJudge,
    UserInterruptionJudge,
    BehaviorJudge,
)
from .score import ScoreResult, score_file, summarize_folder


DATASET_CORRECTNESS_JUDGE_MAP = {
    "halueval": (SimpleQALLMJudge, "gemma3"),
    "trivia_qa": (TriviaQAJudge, "gpt4o"),
    "llama_questions": (LLamaQuestionsJudge, "gpt4o"),
    "web_questions": (WebQuestionsJudge, "gpt4o"),
    "addsub": (MathQALLMJudge, "gemma3"),
    "gsm8k": (MathQALLMJudge, "gemma3"),
    "multiarith": (MathQALLMJudge, "gemma3"),
    "singleq": (MathQALLMJudge, "gemma3"),
    "svamp": (MathQALLMJudge, "gemma3"),
}
DATASET_DUPLEX_JUDGE_MAP = {
    "candor_pause_handling": PauseHandlingJudge,
    "icc_backchannel": BackChannelJudge,
    "synthetic_user_interruption": UserInterruptionJudge,
    "candor_turn_taking": TurnTakingJudge,
    "synthetic_pause_handling": PauseHandlingJudge,
    "background_speech": BehaviorJudge,
    "talking_to_other": BehaviorJudge,
    "user_backchannel": BehaviorJudge,
    "user_interruption": BehaviorJudge,
}


def iter_target_folders(root: Path) -> List[Path]:
    folders = [p for p in sorted(root.glob("*")) if p.is_dir()]
    return folders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/lm.rag.base"),
        help="Root directory that contains experiment folders (default: %(default)s).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only load the results and summarize them, do not score or save the results.",
    )
    parser.add_argument(
        "--force-reeval",
        action="store_true",
        help="Force re-evaluation of all files.",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        required=False,
        help="Names of the datasets to evaluate, separated by space.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_root = args.results_root.expanduser().resolve()

    if not results_root.exists():
        logging.error("Results root %s does not exist.", results_root)
        sys.exit(1)

    folders = iter_target_folders(results_root)

    keyword_judge = None
    if not args.summary_only:
        keyword_judge = KeywordLLMJudge()

    for folder in folders:
        if args.dataset_names and folder.name not in args.dataset_names:
            continue
        correctness_judge = None
        correctness_judge_class, judge_model_name = DATASET_CORRECTNESS_JUDGE_MAP.get(folder.name, (None, None))
        if not args.summary_only and keyword_judge is not None:
            try:
                if correctness_judge_class is not None:
                    if judge_model_name == "gemma3":
                        correctness_judge = correctness_judge_class(keyword_judge.llm)
                        correctness_judge.llm.model_name = "google/gemma-3-27b-it"
                    elif judge_model_name == "gpt4o":
                        correctness_judge = correctness_judge_class()
                    else:
                        raise ValueError(f"Invalid judge model name: {judge_model_name}")
                else:
                    correctness_judge = None
            except Exception as err:
                logging.error("Failed to initialize correctness judge: %s", err)
                sys.exit(1)

        duplex_judge = None
        duplex_judge_class = DATASET_DUPLEX_JUDGE_MAP.get(folder.name, None)
        if duplex_judge_class is not None:
            duplex_judge = duplex_judge_class()

        logging.info("Processing %s", folder)
        records: List[ScoreResult] = []
        files = sorted([path for path in folder.rglob("*/*.json") if "score" not in path.name])
        for path in tqdm(files):
            output_path = path.with_name(f"{path.stem}.score.{judge_model_name}.json")
            if (args.summary_only or output_path.exists()) and not args.force_reeval:
                with output_path.open() as f:
                    entry = json.load(f)
                score: ScoreResult | None = ScoreResult.from_dict(entry)
            else:
                score = score_file(path, output_path, keyword_judge, correctness_judge, duplex_judge)
            if score:
                records.append(score)

        summary = summarize_folder(folder, records, judge_model_name)
        logging.info(
            "Summary %s -> %d examples, ref_acc=%.3f, model_acc=%.3f",
            folder.name,
            summary["num_examples"],
            summary.get("reference_correctness") or 0.0,
            summary.get("correctness") or 0.0,
        )
        logging.info("Wrote folder summary to %s", folder)


if __name__ == "__main__":
    main()
