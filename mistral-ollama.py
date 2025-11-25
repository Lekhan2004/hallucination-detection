# ollama_inference.py

import re
from typing import Optional

import requests

from datastructures import (
    ModelAnswer,
    ProblemExample,
    FINAL_ANSWER_RE,
    GSM_ANSWER_RE,
    normalize_answer_str,
    answers_equal,
)

OLLAMA_MODEL_NAME = "mistral"  # this is the name Ollama uses by default
OLLAMA_URL = "http://localhost:11434/api/chat"


def build_prompt(problem: str) -> str:
    system = (
        "You are a careful math tutor. Solve the problem step by step. "
        "At the end, output a single line of the form 'Final answer: <answer>'. "
        "Use only one final numeric or algebraic answer."
    )
    # We will feed system and user separately to Ollama chat API
    return system, problem


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


def call_ollama_chat(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.7,
        },
    }

    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()
    # Ollama chat API returns choices like OpenAI style
    content = data["message"]["content"]
    return content


def get_single_model_answer(example: ProblemExample) -> ModelAnswer:
    system_prompt, user_prompt = build_prompt(example.problem)
    content = call_ollama_chat(system_prompt, user_prompt)

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
