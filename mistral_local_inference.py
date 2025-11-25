# mistral_local_inference.py

import re
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from datastructures import (
    ModelAnswer,
    ProblemExample,
    FINAL_ANSWER_RE,
    GSM_ANSWER_RE,
    normalize_answer_str,
    answers_equal,
)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)


def build_prompt(problem: str) -> str:
    system = (
        "You are a careful math tutor. Solve the problem step by step. "
        "At the end, output a single line of the form 'Final answer: <answer>'. "
        "Use only one final numeric or algebraic answer."
    )
    prompt = f"<s>[INST] {system}\n\n{problem} [/INST]"
    return prompt


def extract_final_answer_from_response(text: str) -> Optional[str]:
    m = FINAL_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()

    m2 = GSM_ANSWER_RE.search(text)
    if m2:
        return m2.group(1).strip()

    nums = re.findall(r"([+-]?\d+(?:\.\d+)?)", text)
    if nums:
        return nums[-1]

    return None


def get_single_model_answer(example: ProblemExample) -> ModelAnswer:
    prompt = build_prompt(example.problem)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # remove prompt prefix if present
    if full_text.startswith(prompt):
        response_text = full_text[len(prompt):].strip()
    else:
        response_text = full_text.strip()

    if response_text:
        final_raw = extract_final_answer_from_response(response_text)
    else:
        final_raw = None

    if final_raw is not None:
        correct = answers_equal(example.gold_answer_norm, final_raw)
        final_norm = normalize_answer_str(final_raw)
    else:
        correct = None
        final_norm = None

    return ModelAnswer(
        id=example.id,
        dataset=example.dataset,
        split=example.split,
        model=MODEL_NAME,
        problem=example.problem,
        gold_answer_raw=example.gold_answer_raw,
        gold_answer_norm=example.gold_answer_norm,
        our_final_answer_raw=final_raw,
        our_final_answer_norm=final_norm,
        full_response=response_text,
        is_correct=correct,
    )
