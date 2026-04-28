# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re

from .base import LLMJudge
from ...llm import LLMClient


class OpenAudioBenchLLMJudge(LLMJudge):
    def __init__(self, llm: LLMClient | None = None):
        super().__init__(llm)
        self.max_retries = 3
        self.stop_on_new_line = False
        # Follow the original setup of OpenAudioBench
        if isinstance(self.llm, LLMClient):
            self.llm.model_name = "gpt-4o-2024-08-06"
            self.llm.system_prompt = ""
            self.llm.temperature = 0.0
            self.llm.top_p = 1.0


class LLamaQuestionsJudge(OpenAudioBenchLLMJudge):
    PROMPT_TEMPLATE = '\'\n## Background\nYou are a professional QA evaluation expert. You need to assess whether the model\'s answer is correct based on the standard answer.\n\n\n## Scoring Criteria\nCorrect: The answer matches or is equivalent to the standard answer \n\nIncorrect: The answer is wrong or irrelevant to the question \n\n\n## Evaluation Guidelines\n1. The expression of answers can be flexible, not requiring exact matches. For example: \n\n   - Numbers can be expressed in either Arabic numerals or words \n\n   - Proper nouns can be in either English or Chinese \n\n   - Differences in punctuation can be ignored \n\n2. Focus on whether the core meaning of the answer is correct \n\n## Output Format\nProvide the reasoning for your score, then generate the result in "[]" format and make sure it contains "the score is [Correct]" or "the score is [Incorrect]", for example:\n\nThe answer is correct and equivalent to the standard answer, the score is [Correct]\n\nor\n\nThe answer is incorrect and does not match the standard answer, the score is [Incorrect]\n\n\n\n## Question:\n{question}\n## Standard Answer:\n{valid_answers}\n## Model\'s Answer:\n{answer}\n\''

    def _parse_response(self, response: str | None) -> str | int | bool | None:
        if response is None:
            return -1
        try:
            parsed_response = re.findall(r"[Tt]he score is \[(Correct|Incorrect)\]", response)
            if len(parsed_response) > 0 and parsed_response[0] == "Correct":
                return 1
            elif len(parsed_response) > 0 and parsed_response[0] == "Incorrect":
                return 0
            elif "incorrect" in response.lower():
                return 0
            elif "correct" in response.lower():
                return 1
            else:
                return -1
        except Exception:
            return -1


class TriviaQAJudge(OpenAudioBenchLLMJudge):
    PROMPT_TEMPLATE = (
        "\nYour will be given a question, the reference answers to that question, and an answer to be judged. Your tasks is to judge whether the answer to be judged is correct, given the question and reference answers. An answer considered correct expresses or contains the same meaning as at least **one of** the reference answers. The format and the tone of the response does not matter.  \nYou should respond in JSON format. First provide a one-sentence concise analysis for the judgement in field 'analysis', then your judgment in field 'judgment'. For example, \n'''json \n"
        '{{"analysis": "<a one-sentence concise analysis for the judgement>", "judgment": < your final judgment, "correct" or "incorrect">}} \n\'\'\' \n# Question \n{question}  \n# Reference Answer \n{valid_answers}  \n# Answer To Be Judged \n{answer}\n'
    )

    def _parse_response(self, response: str | None) -> str | int | bool | None:
        if response is None:
            return -1
        try:
            eval_js = eval(response[7:-3])
        except Exception:
            eval_js = eval(response)
        assert "analysis" in eval_js and "judgment" in eval_js and eval_js["judgment"]
        result = eval_js["judgment"]
        if result == "correct":
            return 1
        elif result == "incorrect":
            return 0
        else:
            return -1


class WebQuestionsJudge(TriviaQAJudge):
    pass
