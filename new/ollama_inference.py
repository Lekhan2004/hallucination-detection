# ollama_inference.py

import re
from typing import Optional, Tuple

import requests

from datastructures import (
    ModelAnswer,
    ProblemExample,
    FINAL_ANSWER_RE,
    GSM_ANSWER_RE,
    normalize_answer_str,
    answers_equal,
)

# Ollama model name; "mistral" is Mistral 7B Instruct by default
OLLAMA_MODEL_NAME = "mistral:7b-instruct"
# Use /api/generate for older / default Ollama
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def build_prompt(problem: str) -> Tuple[str, str]:
    system = (
        "You are a careful math tutor. Solve the problem step by step. "
        "At the end, output a single line of the form 'Final answer: <answer>'. "
        "Use only one final numeric or algebraic answer."
    )
    return system, problem


def extract_final_answer_from_response(text: str) -> Optional[str]:
    """
    Extract final answer from model response.
    Priority:
      1. 'Final answer: ...'
      2. GSM8K style '#### 42'
      3. Last number in the text
    """
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


def call_ollama_generate(system_prompt: str, user_prompt: str) -> str:
    # Turn system + user into a single prompt string for /api/generate
    prompt = (
        system_prompt
        + "\n\nProblem:\n"
        + user_prompt
        + "\n\nRemember to end with a line like:\nFinal answer: <answer>\n"
    )

    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
        },
    }

    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()
    # /api/generate returns the full text in "response"
    content = data.get("response", "")
    return content


def get_single_model_answer(example: ProblemExample) -> ModelAnswer:
    system_prompt, user_prompt = build_prompt(example.problem)
    content = call_ollama_generate(system_prompt, user_prompt)

    if content:
        final_raw = extract_final_answer_from_response(content)
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
        model=OLLAMA_MODEL_NAME,
        problem=example.problem,
        gold_answer_raw=example.gold_answer_raw,
        gold_answer_norm=example.gold_answer_norm,
        our_final_answer_raw=final_raw,
        our_final_answer_norm=final_norm,
        full_response=content,
        is_correct=correct,
    )