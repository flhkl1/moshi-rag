# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re

from .base import LLMJudge


class SimpleQALLMJudge(LLMJudge):
    PROMPT_TEMPLATE = '\'\n## Background\nYou are a professional QA evaluation expert. You need to assess whether the model\'s answer is correct based on the standard answer.\n\n\n## Scoring Criteria\nCorrect: The answer matches or is equivalent to the standard answer, or contains the same core concept. \n\nIncorrect: The answer is wrong or irrelevant to the question \n\n\n## Evaluation Guidelines\n1. The expression of answers can be flexible, not requiring exact matches. For example: \n\n   - Numbers can be expressed in either Arabic numerals or words \n\n   - Differences in punctuation or simple spelling mistakes can be ignored \n\n2. Focus on whether the core meaning of the answer is correct \n\n## Output Format\nProvide the reasoning for your score, then generate the result in "[]" format and make sure it contains "the score is [Correct]" or "the score is [Incorrect]", for example:\n\nThe answer is correct and equivalent to the standard answer, the score is [Correct]\n\nor\n\nThe answer is incorrect and does not match the standard answer, the score is [Incorrect]\n\n\n\n## Question:\n{question}\n## Standard Answer:\n{valid_answers}\n## Model\'s Answer:\n{answer}\n\''

    def _parse_response(self, response: str | None) -> str | int | bool | None:
        if response is None:
            return -1
        try:
            parsed_response = re.findall(r"[Tt]he score is \[(Correct|Incorrect)\]", response)[0]
            if parsed_response == "Correct":
                return 1
            elif parsed_response == "Incorrect":
                return 0
            return -1
        except Exception:
            if "incorrect" in str(response).lower():
                return 0
            elif "correct" in str(response).lower():
                return 1
            return -1
