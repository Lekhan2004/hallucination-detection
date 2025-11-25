# run_generation.py

import os
import json
from dataclasses import asdict
from typing import List, Optional

from datastructures import ProblemExample
from load_data import load_gsm8k

# pick one of these:
# from mistral_api_inference import get_single_model_answer
from mistral_local_inference import get_single_model_answer


def run_generation(
    out_path: str,
    max_examples: Optional[int] = None,
    split: str = "train",
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    all_examples: List[ProblemExample] = load_gsm8k(split=split)

    if max_examples is not None:
        all_examples = all_examples[:max_examples]

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            ans = get_single_model_answer(ex)
            f.write(json.dumps(asdict(ans), ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    run_generation(
        out_path="outputs/mistral_gsm8k_single_answer.jsonl",
        max_examples=200,  # set None to run full split
        split="train",
    )
