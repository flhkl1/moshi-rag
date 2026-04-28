# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import List, Optional, Any
from abc import ABC, abstractmethod

from ...llm import LLMClient, get_llm
from ..utils import normalize_text


class Judge(ABC):
    """Abstract interface for judges."""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Run the judge on the given data."""
        pass


class LLMJudge(Judge):
    """Abstract interface for LLM-based judges."""

    PROMPT_TEMPLATE: str = "PLACEHOLDER"

    def __init__(self, llm: LLMClient | None = None):
        if llm is not None:
            self.llm = llm
        else:
            # use GPT4 by default
            self.llm = get_llm(system_prompt="", prompt_type="empty")
            self.llm.model_name = "gpt-4o-2024-08-06"

        self.max_retries = 3
        self.stop_on_new_line = True

    def __call__(
        self,
        question_text: str,
        answer_text: str,
        gt_answers: Optional[List[str]] = None,
    ) -> str | int | bool | None:
        """Run the judge on question/answer data."""
        if not answer_text or not question_text:
            return None

        prompt = self.PROMPT_TEMPLATE.format(
            question=question_text,
            answer=answer_text,
            valid_answers=str(gt_answers) if gt_answers else "",
        )

        result = None
        for i in range(self.max_retries):
            try:
                response = self.llm.generate(
                    prompt=prompt,
                    context="",
                    max_new_tokens=512,
                    stop_token=None,
                )
                assert response is not None and response.strip() != "", (
                    f"response: '{response}' is None or empty for {prompt}"
                )
                result = self._parse_response(response)
                break
            except Exception:
                continue
        return result

    def _parse_response(self, response: str | None) -> str | int | bool | None:
        return response


class KeywordLLMJudge(LLMJudge):
    PROMPT_TEMPLATE = (
        "You are helping evaluate a question answering model.\n"
        "Identify the single keyword or short phrase in the model answer that directly expresses any of the answer aliases. "
        "If the model answer does not directly express any of the answer aliases, return the keyword/phrase that the model intends to answer the question. "
        "Respond with that keyword/phrase only."
        "\nExample:\n"
        "Question: Give me a capital of an European country?\n"
        "Model answer: Berlin is the capital of Germany.\n"
        'Valid answer aliases: ["Paris", "Madrid", "Budapest", "Lisbon"]\n'
        "Response: Berlin"
        "\nInput:\n"
        "Question: {question}\n"
        "Model answer: {answer}\n"
        "Valid answer aliases: {valid_answers}\n"
        "Response: "
    )

    def __init__(self, llm: LLMClient | None = None):
        super().__init__(llm)
        if llm is None:
            self.llm = get_llm(system_prompt="", prompt_type="empty")
            self.llm.model_name = "google/gemma-3-27b-it"

    def _parse_response(self, response: str | None) -> str | int | bool | None:
        if response is None:
            return None
        lines = response.strip().splitlines()
        if len(lines) > 0:
            return normalize_text(lines[0].strip())
        else:
            return None
