# Moshi RAG Evaluation

This package scores experiment outputs from speech language models. It expects the **standard output format** produced by `moshi.moshi.run_inference]]`. The same evaluator also works for **other speech LMs** as long as their inference code writes results in this format.

The evaluator walks result directories, loads per-sample JSON and optional WAV files, runs dataset-specific correctness and duplex judges, and writes per-file score JSON plus a folder-level summary.

## Code structure

```
evaluate/
├── evaluate.py      # CLI entrypoint
├── score.py         # ScoreResult dataclass, score_file(), summarize_folder()
├── utils.py
├── profiler.py      # DeepSpeed FLOPs profiler wrapper for compute metrics
├── judge/
│   ├── __init__.py
│   ├── base.py
│   ├── simple_qa.py      # SimpleQALLMJudge (Correct/Incorrect)
│   ├── math_qa.py        # MathQALLMJudge (Yes/No)
│   ├── open_audio_bench.py # TriviaQA, LLamaQuestions, WebQuestions
│   ├── duplex.py         # BackChannel, PauseHandling, TurnTaking, UserInterruption, Behavior
│   ├── assets/           # Files for Full Duplex Bench
│   │   ├── icc_gt_distribution.json
│   │   └── behavior_prompt.txt
│   └── ...
└── README.md
```

- **evaluate.py**: Parses CLI, selects judges from dataset name, loops over `results_root` subfolders and `*/*.json` (excluding `*score*`), calls `score_file` and `summarize_folder`.
- **score.py**: Defines `ScoreResult`, loads JSON into it, runs `process_data` / `get_timestamps` / `compute_scores` (ASR, correctness, keyword, duplex, delays, FLOPs), writes per-sample output `{stem}.score.{judge_model}.json` and folder (dataset) level summarization `score_summary.{model}.json`.
- **judge/**: Abstract `Judge` and LLM-backed judges (Open Router GPT-4o or vLLM Gemma) for QA correctness, and Full-Duplex-Bench Judges.

## How to run

From the project root:

```bash
python -m moshi.moshi.evaluate.evaluate --results-root /path/to/results [OPTIONS]
```

**Options:**

| Option            | Description                                                                                                                                              |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--results-root`  | Root directory containing one folder per dataset (e.g. `llama_questions`, `gsm8k`).                                                                      |
| `--summary-only`  | Only load existing `*.score.*.json` and write folder summaries; do not run judges or save new scores.                                                    |
| `--force-reeval`  | Re-run scoring and overwrite existing score files. If this is not set the script will skip existing score files.                                         |
| `--dataset-names` | Space-separated list of dataset folder names to process (e.g. `llama_questions gsm8k`). If omitted, all subfolders under `--results-root` are processed. |

**Examples:**

```bash
# Score all datasets under default results root
python -m moshi.moshi.evaluate.evaluate

# Score only llama_questions and gsm8k
python -m moshi.moshi.evaluate.evaluate --dataset-names llama_questions gsm8k
```

Expected layout under `--results-root` before scorigg:

- One directory per dataset (e.g. `llama_questions/`, `gsm8k/`).
- Under each dataset directory: raw json output files `**/*.json` (generated with `moshi.moshi.run_inference_rag`) that are **not** named with `*score*` (e.g. `llama_questions/llama_questions_23.json`).
- Each such JSON is treated as one sample (see **Expected input structure** below). A co-located `.wav` with the same stem is used for ASR timestamps and duplex when needed.

## Supported datasets

Correctness and duplex judges are selected by **dataset folder name** (the name of the subfolder under `--results-root`).

### Correctness (QA) – judge and default model

| Dataset folder                                                                       | Judge class                                             | Judge model |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------- | ----------- |
| `halueval` | SimpleQALLMJudge                                        | vllm_gemma3 |
| `trivia_qa`, `llama_questions`, `web_questions`                                      | TriviaQAJudge / LLamaQuestionsJudge / WebQuestionsJudge | gpt4o       |
| `addsub`, `gsm8k`, `multiarith`, `singleq`, `svamp`                                  | MathQALLMJudge                                          | vllm_gemma3 |

Any other folder name: no correctness judge (only ASR, keyword, delays, and duplex if mapped).

### Duplex (Full-Duplex-Bench style)

| Dataset folder                                              | Judge class           |
| ----------------------------------------------------------- | --------------------- |
| `candor_pause_handling`, `synthetic_pause_handling`         | PauseHandlingJudge    |
| `icc_backchannel`                                           | BackChannelJudge      |
| `synthetic_user_interruption`, `user_interruption`          | UserInterruptionJudge |
| `candor_turn_taking`                                        | TurnTakingJudge       |
| `background_speech`, `talking_to_other`, `user_backchannel` | BehaviorJudge         |

Duplex judges require a co-located **input metadata JSON** path in `ScoreResult.input_path` for turn-taking, interrupt, or behavior metadata (see **Expected input structure**).

## Expected input structure

Input files are produced by `moshi.moshi.run_inference` or by inference scripts of other speech LMs, as long as they emit the same schema. Each **input** file is a single JSON object (one sample) with keys that match (or are a subset of) the `ScoreResult` fields. The evaluator uses `path` as the path of the current JSON file; a co-located `.wav` is used when present.

**Required / commonly used keys:**

| Key              | Type                      | Description                                                                                                                |
| ---------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `gt_user_text`   | string                    | Ground-truth user utterance (for WER and judge prompts).                                                                   |
| `user_text`      | list of strings or string | Model’s view of user input (token list or text). Used for WER vs `gt_user_text`.                                           |
| `model_text`     | list of strings or string | Model’s reply (token list or text). Used for correctness and keyword.                                                      |
| `answer`         | string, list, or dict     | Gold answer(s). String or list used as-is; dict can have `value`/`normalized_value` and/or `aliases`/`normalized_aliases`. |
| `reference_text` | string or null            | Reference (e.g. RAG) response; optional. Used for reference_correctness and rag_percentage (non-empty = RAG used).         |

**Optional keys (delays, profiling, duplex):**

| Key                         | Type           | Description                                                                             |
| --------------------------- | -------------- | --------------------------------------------------------------------------------------- |
| `rag_trigger_step`          | int            | Step at which retrieval was triggered.                                                  |
| `retrieval_step`            | int            | Step when retrieval started.                                                            |
| `conditioning_step`         | int            | Step when conditioning was applied.                                                     |
| `question_end_step`         | int            | Step when the user question ended (for keyword_delay).                                  |
| `elapsed`                   | float          | Elapsed time (e.g. seconds).                                                            |
| `first_audio_token_latency` | float          | Seconds from question end to first audio token.                                         |
| `profile`                   | dict           | Profiling data.                                                                         |
| `input_path`                | string         | Path to input metadata JSON for duplex judges (turn_taking, interrupt, behavior, etc.). |
| `gt_reference_text`         | string or null | Optional ground-truth reference.                                                        |

**Example (minimal) input JSON:**

```json
{
  "rag_trigger_step": 40,
  "question_end_step": 41,
  "reference_text": " The largest waterfall in the world is Angel Falls in Venezuela. It has a height of nine hundred seventy nine meters.",
  "retrieval_step": 46,
  "conditioning_step": 52,
  "model_text": ["<pad>", "<0x00>", "▁That", "'", "s", "▁a", "▁great", "▁question", "!", "▁The", "▁Angel", "▁Falls", "..."],
  "user_text": ["<pad>", "▁What", "▁is", "▁the", "▁name", "▁of", "▁the", "▁large", "s", "t", "▁water", "f", "all", "..."],
  "gt_user_text": "What is the name of the largest waterfall in the world?",
  "gt_reference_text": null,
  "answer": "Victoria Falls"
}
```

For duplex datasets, the same directory (or a path given in `input_path`) should contain the corresponding metadata (e.g. `turn_taking`, `interrupt`, or `input`/`clean_input` chunks) expected by the duplex judge.