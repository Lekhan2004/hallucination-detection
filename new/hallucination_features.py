# hallucination_features.py

from typing import Dict, Any

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from datastructures import answers_equal

# Load a small sentence embedding model
# It will download the first time you run it.
_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def compute_objective_correctness(rec: Dict[str, Any]) -> bool:
    """
    Recompute correctness using gold_answer_raw vs our_final_answer_raw.
    Does not trust any existing 'is_correct' field in the JSON.
    """
    gold = rec.get("gold_answer_raw", "")
    pred = rec.get("our_final_answer_raw", "")
    if not gold or not pred:
        return False
    return answers_equal(gold, pred)


def compute_cosine_similarity(rec: Dict[str, Any]) -> float:
    """
    Compute a cosine similarity feature between a 'reference' text
    and the model's answer + explanation.

    For math GSM8K we build:
      ref_text  = "Problem: ... Correct answer: <gold>"
      model_text = "Problem: ... Model answer: <our_final_answer> Explanation: <full_response>"

    Returns a float in [-1, 1] (normally 0..1 for this model).
    """
    problem = rec.get("problem", "")
    gold = rec.get("gold_answer_raw", "")
    model_ans = rec.get("our_final_answer_raw", "")
    full_resp = rec.get("full_response", "")

    ref_text = "Problem: {}\nCorrect answer: {}".format(problem, gold)
    model_text = "Problem: {}\nModel answer: {}\nExplanation: {}".format(
        problem, model_ans, full_resp
    )

    embeddings = _EMBED_MODEL.encode([ref_text, model_text])
    emb_ref = embeddings[0].reshape(1, -1)
    emb_model = embeddings[1].reshape(1, -1)

    sim = cosine_similarity(emb_ref, emb_model)[0][0]
    # Ensure it's a plain float
    return float(sim)
