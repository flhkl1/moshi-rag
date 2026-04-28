# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from ..utils import normalize_text
from .base import LLMJudge


class MathQALLMJudge(LLMJudge):
    PROMPT_TEMPLATE = (
        "You are helping evaluate a mathematical question answering model.\n"
        "Determine whether the model's answer contains the provided Correct Answer. Ignore intermediate calculations or reasoning steps and focus on the numerical correctness of the model's answer.\n\n"
        "Instructions:\n"
        "1. Compare the value in the model's answer to the Correct Answer.\n"
        "2. The comparison must be based on numerical equivalence (e.g., 5.0 should match 5). Ignore rounding errors or other small differences.\n"
        '3. Your response must be only the word "Yes" or "No".\n\n'
        "Example 1: Correct Match\n"
        "Question: If a train travels at 60 mph for 2 hours, how far does it travel?\n"
        "Correct Answer: 120.0\n"
        "Model Answer: The total distance is 120 miles.\n"
        "Response: Yes\n\n"
        "Example 2: Incorrect Match\n"
        "Question: John had 10 apples and ate 3. How many are left?\n"
        "Correct Answer: 7\n"
        "Model Answer: He has 8 apples left.\n"
        "Response: No\n\n"
        "Input:\n"
        "Question: {question}\n"
        "Correct Answer: {valid_answers}\n"
        "Model Answer: {answer}\n"
        "Response: "
    )

    def _parse_response(self, response: str | None) -> int:
        if response is None:
            return -1
        response = normalize_text(response.strip().splitlines()[0].strip())
        if not response:
            return -1
        if response == "yes":
            return 1
        if response == "no":
            return 0
        return -1
