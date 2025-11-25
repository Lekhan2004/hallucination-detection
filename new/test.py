# test_mistral_self_judge.py

from typing import Dict, Any
from llm_judges import mistral_self_judge

def get_sample_record() -> Dict[str, Any]:
    return {
        "id": "gsm8k-train-1",
        "dataset": "gsm8k",
        "split": "train",
        "model": "mistral:7b-instruct",
        "problem": (
            "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes "
            "of babysitting. How much did she earn?"
        ),
        "gold_answer_raw": "10",
        "gold_answer_norm": "10",
        "our_final_answer_raw": "$9.999999999996 (rounded to the nearest cent)",
        "our_final_answer_norm": "$9.999999999996 (rounded to the nearest cent)",
        "full_response": (
            " First, let's convert the given time into hours since there are 60 minutes in an hour.\n"
            "50 minutes = 50 / 60 = 0.833333333333 hours.\n"
            "Since Weng earns $12 per hour, her earnings for yesterday can be calculated by "
            "multiplying the number of hours she worked by her hourly wage:\n"
            "Earnings = Hours worked * Hourly wage\n"
            "Earnings = 0.833333333333 * $12\n\n"
            "Final answer: $9.999999999996 (rounded to the nearest cent)"
        ),
        "is_correct": False,
        "objective_correct": False,
        "cosine_sim_ref_vs_model": 0.8504077196121216,
        "deepseek_correct": False,
        "deepseek_hallucination": False,
        "deepseek_hallucination_score": 0,
        "deepseek_reason": (
            "The model's reasoning is sound and follows a logical path derived from the problem. "
            "The final answer is incorrect due to a floating-point precision error in the calculation "
            "(0.8333... * 12), not due to any hallucinated facts or steps."
        ),
        "mistral_self_correct": None,
        "mistral_self_hallucination": None,
        "mistral_self_hallucination_score": None,
        "mistral_self_reason": None,
    }

def main():
    rec = get_sample_record()
    print("Calling mistral_self_judge on sample record...")
    result = mistral_self_judge(rec)

    print("\n=== Judge result object ===")
    print(result)

    if result is not None:
        print("\ncorrect:", result.get("correct"))
        print("hallucination:", result.get("hallucination"))
        print("hallucination_score:", result.get("hallucination_score"))
        print("reason:", result.get("reason"))
        # if you added a "raw" field in that function, also:
        print("raw response:", result.get("raw"))

if __name__ == "__main__":
    main()
