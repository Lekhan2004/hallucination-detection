import os
import json
from typing import Dict, Any, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import requests
# ==========================
# 1. The Wrapper Class
# ==========================

class GeminiWrapper:
    def __init__(self, api_key: str = None, model_name: str = "gemini-1.5-pro"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API Key not found. Set GEMINI_API_KEY env var.")
            
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Default config - we will override this for the judge
        self.generation_config = genai.GenerationConfig(
            temperature=0.0, # Low temperature for factual judging
            max_output_tokens=8192,
            response_mime_type="text/plain" 
        )

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        # Adjust config based on mode
        if json_mode:
            self.generation_config.response_mime_type = "application/json"
        else:
            self.generation_config.response_mime_type = "text/plain"

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API Error: {str(e)}")

# ==========================
# 2. Judge Logic
# ==========================

def _build_judge_prompt(rec: Dict[str, Any]) -> str:
    """Constructs the prompt text."""
    problem = rec.get("problem", "")
    gold = rec.get("gold_answer_raw", "")
    model_ans = rec.get("our_final_answer_raw", "")
    full_resp = rec.get("full_response", "")

    return (
        "You are a strict hallucination detector for math word problems.\n\n"
        "You are given:\n"
        "1) The problem statement.\n"
        "2) The reference correct final numeric answer.\n"
        "3) A model's full chain-of-thought (explanation) and its final answer.\n\n"
        "You must evaluate whether the model's final answer is correct, and whether the "
        "explanation contains hallucinations: unsupported numbers, steps, or claims not "
        "grounded in the problem.\n\n"
        "Return a SINGLE JSON object with the following keys:\n"
        "  correct: boolean (true if final answer matches the reference answer)\n"
        "  hallucination: boolean (true if you see any hallucination in the reasoning)\n"
        "  hallucination_score: number between 0 and 1 (0 = no hallucination, 1 = very hallucinated)\n"
        "  reason: short natural language explanation\n\n"
        f"Problem:\n{problem}\n\n"
        f"Reference final answer: {gold}\n\n"
        f"Model final answer: {model_ans}\n\n"
        f"Model chain-of-thought:\n{full_resp}\n"
    )

def _parse_json_from_content(content: str) -> Optional[Dict[str, Any]]:
    """ robust parsing in case the model wraps JSON in markdown blocks """
    try:
        return json.loads(content)
    except Exception:
        pass
    
    # Fallback: extract substring between first { and last }
    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        return json.loads(content[start:end])
    except Exception:
        return None

def deepseek_judge(rec: Dict[str, Any], api_key="AIzaSyBMg6v31a6eCYgALJ0W3cVww6xdMZY-vzE") -> Dict[str, Any]:
    """
    LLM-as-judge using the GeminiWrapper.
    """
    record_id = rec.get("id", "unknown")
    print(f"=== Gemini judge for id: {record_id} ===")

    try:
        # 1. Initialize Wrapper
        # We prefer passing the key via env var, but support argument passing
        llm = GeminiWrapper(api_key=api_key, model_name="gemini-2.5-pro")

        # 2. Build Prompt
        prompt = _build_judge_prompt(rec)

        # 3. Call API (Enable JSON Mode natively)
        # JSON mode guarantees the structure, so we don't need regex parsing usually
        raw_response = llm.generate(prompt, json_mode=True)
        
        print(f"Gemini raw response preview: {raw_response[:200]}...")

        # 4. Parse Response
        parsed_data = _parse_json_from_content(raw_response)
        
        if parsed_data:
            parsed_data["raw"] = raw_response
            return parsed_data
        else:
            return {"error": "json_parse_failed", "raw": raw_response}

    except Exception as e:
        print(f"Gemini Judge Failed: {e}")
        return {"error": "api_exception", "details": str(e)}

from typing import Dict, Any

def build_self_eval_prompt(rec: Dict[str, Any]) -> str:
    """
    Build a self-eval / faithfulness prompt for Mistral via Ollama.

    We treat:
      - Source Context  := GSM8K problem + reference final answer
      - Generated Answer := model's full_response (its chain-of-thought + final answer)

    The model is asked to do claim-level faithfulness analysis and then summarize it
    as a JSON object for our pipeline.
    """
    problem = rec.get("problem", "")
    gold = rec.get("gold_answer_raw", "")
    generated = rec.get("full_response", "")

    prompt = (
        "You are a fact-checking AI focused on detecting hallucinations.\n\n"
        "SOURCE CONTEXT:\n"
        "The following is the *only* information you are allowed to treat as ground truth.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Reference final answer:\n{gold}\n\n"
        "GENERATED ANSWER (to evaluate):\n"
        f"{generated}\n\n"
        "Perform a faithfulness analysis of the GENERATED ANSWER with respect to the SOURCE CONTEXT.\n"
        "Follow these steps *internally*, then output ONLY a single JSON object:\n"
        "1. Extract each distinct factual claim or step from the generated answer.\n"
        "2. For each claim, check if it is directly supported by the source context.\n"
        "3. Label each claim with one of:\n"
        '   - \"SUPPORTED\" (explicitly present in the source context)\n'
        '   - \"PARTIALLY_SUPPORTED\" (some parts match, but not fully)\n'
        '   - \"UNSUPPORTED\" (not stated or contradicted by the source context)\n'
        "4. Highlight any information that appears fabricated or inferred without an explicit basis.\n"
        "5. Compute a faithfulness score as the percentage of claims that are fully SUPPORTED "
        "(0 to 1, not 0 to 100).\n"
        "   For example, if 8 out of 10 claims are SUPPORTED, faithfulness_score = 0.8.\n\n"
        "Be extremely rigorous: if a detail is not explicitly present in the source context, "
        "mark the claim as UNSUPPORTED or PARTIALLY_SUPPORTED.\n\n"
        "Additionally, derive these summary judgments:\n"
        "- correct: boolean, true if the final numeric answer in the generated answer matches the "
        "reference final answer exactly.\n"
        "- hallucination: boolean, true if there is at least one UNSUPPORTED claim.\n"
        "- hallucination_score: number between 0 and 1, where 0 means no hallucinations "
        "and 1 means heavily hallucinated. A simple default is hallucination_score = 1 - faithfulness_score.\n"
        "- reason: a short natural language summary of the main issues you found.\n\n"
        "IMPORTANT OUTPUT FORMAT:\n"
        "Return ONLY a single JSON object with this structure (no extra text before or after):\n"
        "{\n"
        "  \"claims\": [\n"
        "    {\n"
        "      \"text\": \"...\",           // the claim text\n"
        "      \"label\": \"SUPPORTED\" | \"PARTIALLY_SUPPORTED\" | \"UNSUPPORTED\"\n"
        "    },\n"
        "    ...\n"
        "  ],\n"
        "  \"faithfulness_score\": <number between 0 and 1>,\n"
        "  \"hallucination\": <true or false>,\n"
        "  \"hallucination_score\": <number between 0 and 1>,\n"
        "  \"reason\": \"short explanation\"\n"
        "}\n"
        "Do not include any commentary outside of this JSON.\n"
    )

    return prompt


# ========================== # Mistral Self-Judge via Ollama # ========================== 

OLLAMA_MODEL_NAME = "mistral:7b-instruct" 
OLLAMA_URL = "http://127.0.0.1:11434/api/generate" 

def mistral_self_judge(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt = build_self_eval_prompt(rec)

    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
        },
    }
    print(payload, "payload")

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    print(data, "data")

    content = data.get("response", "")

    try:
        return json.loads(content)
    except Exception:
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            obj_str = content[start:end]
            return json.loads(obj_str)
        except Exception:
            return None
