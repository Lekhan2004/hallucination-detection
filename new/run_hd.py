import json
import os
from typing import Dict, Any, Optional, List

from hallucination_features import (
    compute_objective_correctness,
    compute_cosine_similarity,
)
from llm_judges import deepseek_judge, mistral_self_judge


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

# The original write_jsonl function is removed as we will write line-by-line


def enrich_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given one record from the Mistral GSM8K JSONL, compute multiple
    hallucination signals and return an augmented dict.
    """
    augmented = dict(rec)  # shallow copy

    # 1) Objective correctness based on gold vs our_final_answer
    try:
        objective_correct = compute_objective_correctness(rec)
    except Exception:
        objective_correct = False

    augmented["objective_correct"] = objective_correct

    # 2) Cosine similarity feature
    try:
        cos_sim = compute_cosine_similarity(rec)
    except Exception:
        cos_sim = None

    augmented["cosine_sim_ref_vs_model"] = cos_sim

    # 3) DeepSeek R1 judge (if configured)
    ds_result: Optional[Dict[str, Any]] = None
    try:
        ds_result = deepseek_judge(rec)
    except Exception:
        ds_result = None

    if ds_result is not None:
        augmented["deepseek_correct"] = ds_result.get("correct")
        augmented["deepseek_hallucination"] = ds_result.get("hallucination")
        augmented["deepseek_hallucination_score"] = ds_result.get("hallucination_score")
        augmented["deepseek_reason"] = ds_result.get("reason")
    else:
        augmented["deepseek_correct"] = None
        augmented["deepseek_hallucination"] = None
        augmented["deepseek_hallucination_score"] = None
        augmented["deepseek_reason"] = None

    # 4) Mistral self-judge via Ollama (faithfulness-based)
    mj_result: Optional[Dict[str, Any]] = None
    try:
        mj_result = mistral_self_judge(rec)
    except Exception:
        mj_result = None

    if mj_result is not None:
        # Core summary fields
        augmented["mistral_self_hallucination"] = mj_result.get("hallucination")
        augmented["mistral_self_hallucination_score"] = mj_result.get(
            "hallucination_score"
        )
        augmented["mistral_self_reason"] = mj_result.get("reason")

        # Faithfulness-specific fields from the new prompt
        augmented["mistral_self_faithfulness_score"] = mj_result.get(
            "faithfulness_score"
        )
        augmented["mistral_self_claims"] = mj_result.get("claims")
    else:
        augmented["mistral_self_hallucination"] = None
        augmented["mistral_self_hallucination_score"] = None
        augmented["mistral_self_reason"] = None
        augmented["mistral_self_faithfulness_score"] = None
        augmented["mistral_self_claims"] = None
    return augmented


def main():
    input_path = "outputs/mistral_gsm8k_single_answer_train.jsonl"
    output_path = "outputs/mistral_gsm8k_hallucination_annotated_4.jsonl"

    records = read_jsonl(input_path)

    # If you only want to process first 200:
    records = records[:2]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Open the output file in append mode ('a') to write line by line
    with open(output_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            print("Processing record {} / {}".format(i + 1, len(records)))
            augmented = enrich_record(rec)

            # Write the completed record immediately to the file
            f.write(json.dumps(augmented, ensure_ascii=False) + "\n")

    print("Finished processing and writing annotated records to {}".format(output_path))


if __name__ == "__main__":
    main()
