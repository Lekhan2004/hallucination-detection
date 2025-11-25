# load_data.py

from typing import List
from datasets import load_dataset

from datastructures import ProblemExample, normalize_answer_str, GSM_ANSWER_RE


def load_gsm8k(split: str = "train") -> List[ProblemExample]:
    """
    Load GSM8K from Hugging Face and normalize.
    Dataset: openai/gsm8k, config "main"
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)
    examples: List[ProblemExample] = []

    for i, row in enumerate(ds):
        question = row["question"]
        answer = row["answer"]

        m = GSM_ANSWER_RE.search(answer)
        if m:
            gold = m.group(1)
        else:
            gold = answer.strip().split("\n")[-1]

        examples.append(
            ProblemExample(
                id=f"gsm8k-{split}-{i}",
                dataset="gsm8k",
                split=split,
                problem=question,
                gold_answer_raw=gold,
                gold_answer_norm=normalize_answer_str(gold),
            )
        )
    return examples
